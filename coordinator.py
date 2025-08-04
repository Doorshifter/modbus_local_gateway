"""
Coordinator module for Modbus Local Gateway custom integration.

This module provides the core logic for polling Modbus devices efficiently.
It defines the `ModbusCoordinator` which leverages a `ReactiveBatchManager`
to handle data fetching. The coordinator pre-computes and caches entity
groupings and batch managers at startup to minimize CPU load, while the
manager itself can dynamically re-group batches upon failure.

Classes:
    ModbusEntityBlock: A simple data class representing a contiguous block of
                       registers for a single entity.
    ReactiveBatchManager: Manages the grouping and batching of Modbus reads,
                          dynamically adjusting request sizes and batches.
    ModbusCoordinator: The main orchestrator that schedules updates, manages
                       the batching process, and provides data to entities.

Author: Doorshifter
Date: 2025-07-12
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from homeassistant.core import HomeAssistant
from homeassistant.helpers import event
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .context import ModbusContext
from .conversion import Conversion
from .entity_management.const import ModbusDataType
from .const import DOMAIN
from .statistics import EntityStatisticsTracker, StatisticMetric

_LOGGER = logging.getLogger(__name__)


class ModbusEntityBlock:
    """Represents a contiguous block of Modbus registers for a single entity."""

    def __init__(self, start: int, count: int, entity: Any):
        self.start = start
        self.count = count
        self.end = start + count - 1
        self.entity = entity

    def __repr__(self) -> str:
        return f"Block(entity={self.entity.desc.key}, start={self.start}, count={self.count})"


class ReactiveBatchManager:
    """Manages Modbus read batches with reactive congestion control."""

    def __init__(self, blocks: List[ModbusEntityBlock], max_read_size: int):
        self.blocks = sorted(blocks, key=lambda b: b.start)
        self.max_read_size = max_read_size
        self.congestion_window = min(max_read_size // 2, 10)
        self.slow_start_threshold = max_read_size
        self.consecutive_successes = 0
        self.last_update_timestamp = None  
        self.last_update_attempt_timestamp = None
        self.last_update_success = False
        self.last_value_change_timestamp = None  
        self.batches = self._initial_grouping()
        self.log_batch_statistics()

        _LOGGER.debug(
            "ReactiveBatchManager initialized: max_read=%d, initial_cwnd=%d",
            max_read_size,
            self.congestion_window,
        )
        _LOGGER.debug("ReactiveBatchManager initial batches created: %s", self.batches)

    def log_batch_statistics(self):
        """Log compact statistics about batches for easier reading in Home Assistant logs."""
        if not self.batches:
            _LOGGER.info("No batches created")
            return

        total_registers = 0
        total_entities = 0
        batch_sizes = []

        for batch_blocks in self.batches:
            if not batch_blocks:
                continue
            batch_start = batch_blocks[0].start
            batch_end = batch_blocks[-1].end
            batch_size = batch_end - batch_start + 1
            batch_sizes.append(batch_size)
            batch_entities = len(batch_blocks)
            total_entities += batch_entities
            batch_registers = sum(block.count for block in batch_blocks)
            total_registers += batch_registers

        _LOGGER.info(
            "BATCH SUMMARY: %d batches, %d entities, %d registers",
            len(self.batches), total_entities, total_registers
        )

        if not batch_sizes:
            return

        _LOGGER.info(
            "BATCH STATISTICS: avg=%.1f, max=%d, min=%d, efficiency=%.1f%%",
            sum(batch_sizes) / len(batch_sizes), max(batch_sizes), min(batch_sizes),
            (total_registers / sum(batch_sizes) * 100) if sum(batch_sizes) else 0
        )

        size_ranges = {
            "1": 0,
            "2-5": 0,
            "6-10": 0,
            "11-20": 0,
            "21+": 0
        }
        for size in batch_sizes:
            if size == 1:
                size_ranges["1"] += 1
            elif 2 <= size <= 5:
                size_ranges["2-5"] += 1
            elif 6 <= size <= 10:
                size_ranges["6-10"] += 1
            elif 11 <= size <= 20:
                size_ranges["11-20"] += 1
            else:
                size_ranges["21+"] += 1

        _LOGGER.info(
            "BATCH SIZE DISTRIBUTION: 1:%d, 2-5:%d, 6-10:%d, 11-20:%d, 21+:%d",
            size_ranges["1"], size_ranges["2-5"], size_ranges["6-10"],
            size_ranges["11-20"], size_ranges["21+"]
        )

    def _initial_grouping(self) -> List[List[ModbusEntityBlock]]:
        if not self.blocks:
            return []

        batches = []
        current_batch = [self.blocks[0]]
        batch_start = self.blocks[0].start

        for i in range(1, len(self.blocks)):
            prev_block = self.blocks[i - 1]
            current_block = self.blocks[i]
            potential_batch_size = current_block.end - batch_start + 1

            if (
                current_block.start == prev_block.end + 1
                and potential_batch_size <= self.max_read_size
            ):
                current_batch.append(current_block)
            else:
                batches.append(current_batch)
                current_batch = [current_block]
                batch_start = current_block.start

        if current_batch:
            batches.append(current_batch)

        return batches

    def get_batches(self) -> List[Tuple[int, int, List[ModbusEntityBlock]]]:
        result = []
        _LOGGER.debug("Getting batches to read from: %s", self.batches)
        for batch_blocks in self.batches:
            if not batch_blocks:
                continue

            batch_start = batch_blocks[0].start
            batch_end = batch_blocks[-1].end
            total_count = batch_end - batch_start + 1

            current_pos = 0
            while current_pos < total_count:
                chunk_size = min(self.congestion_window, total_count - current_pos)
                read_start_addr = batch_start + current_pos

                result.append((read_start_addr, chunk_size, batch_blocks))

                current_pos += chunk_size

        _LOGGER.debug("Batches to be read this cycle: %s", result)
        return result

    def feedback(self, batch_results: List[Tuple[List[ModbusEntityBlock], bool]]):
        failures = sum(1 for _, success in batch_results if not success)

        if failures > 0:
            self.slow_start_threshold = max(self.congestion_window // 2, 1)
            self.congestion_window = self.slow_start_threshold
            self.consecutive_successes = 0
            _LOGGER.warning(
                "Batch failures detected, reducing congestion window to %d and re-grouping.",
                self.congestion_window,
            )

            new_batches = []
            for batch, success in batch_results:
                if success or len(batch) == 1:
                    new_batches.append(batch)
                else:
                    _LOGGER.debug("Splitting failed batch: %s", batch)
                    for block in batch:
                        new_batches.append([block])
            self.batches = new_batches
        else:
            self.consecutive_successes += 1
            if self.congestion_window < self.slow_start_threshold:
                self.congestion_window = min(
                    self.congestion_window * 2, self.max_read_size
                )
            elif self.consecutive_successes >= 5:
                self.congestion_window = min(
                    self.congestion_window + 1, self.max_read_size
                )
                self.consecutive_successes = 0


class ModbusCoordinator(DataUpdateCoordinator):
    """Coordinates updates, batching, and data conversion for Modbus entities."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: Any,
        gateway: str,
        # Modbus/TCP has a theoretical maximum of 125 registers per transaction 120 to be safe
        max_read_size: int = 120,
        max_registers_per_second: int | None = None,
        gateway_device: Optional[DeviceInfo] = None,
    ):
        super().__init__(
            hass,
            _LOGGER,
            name=f"Modbus Coordinator - {gateway}",
            update_interval=None,
        )
        self.gateway = gateway
        self.gateway_device = gateway_device
        self.client = client
        self.entities: List[ModbusContext] = []
        self._max_read_size = max_read_size
        self._max_registers_per_second = max_registers_per_second
        self._batch_managers: Dict[Tuple[int, str], ReactiveBatchManager] = {}
        self._initialized = False
        self._polling_jobs: List[Callable[[], None]] = []
        self._update_lock = asyncio.Lock()
        self._ready_event_fired = False
        self.last_update_timestamp = None
        self.last_update_attempt_timestamp = None
        self.last_update_success = False

    def add_entity(self, entity: ModbusContext) -> None:
        self.entities.append(entity)
        _LOGGER.debug(
            "Adding entity to coordinator: %s with scan_interval=%s",
            entity.desc.key,
            getattr(entity, "scan_interval", "NOT SET"),
        )
        if not self._initialized and len(self.entities) == 1:
            _LOGGER.debug("First entity added - will initialize on next update")

    def _initialize_coordination_data(self) -> None:
        """Initialize coordination data and ensure all entities have statistics trackers."""
        if not self.entities:
            _LOGGER.warning(
                "Coordinator initialization called with 0 entities - deferring until entities are added"
            )
            return
    
        _LOGGER.info(
            "Initializing coordinator %s: Grouping %d entities for polling.",
            self.gateway,
            len(self.entities),
        )
    
        # Initialize statistics trackers for all entities
        for entity in self.entities:
            try:
                if not hasattr(entity, "statistics") or entity.statistics is None:
                    entity.statistics = EntityStatisticsTracker(entity.scan_interval)
                    _LOGGER.debug(
                        "Created statistics tracker for entity %s with scan_interval=%d",
                        entity.desc.key if hasattr(entity, "desc") else "unknown",
                        entity.scan_interval
                    )
            except Exception as ex:
                _LOGGER.warning(
                    "Failed to initialize statistics for entity: %s",
                    ex
                )
    
        self._batch_managers.clear()
    
        grouped = self._group_entities_by_slave_and_type(self.entities)
        _LOGGER.debug("Grouped entities for batching: %s", grouped)
    
        for (slave_id, data_type), group_entities in grouped.items():
            bm_key = (slave_id, data_type)
            blocks = [
                ModbusEntityBlock(e.desc.register_address, e.desc.register_count, e)
                for e in group_entities
            ]
            self._batch_managers[bm_key] = ReactiveBatchManager(
                blocks, self._max_read_size
            )
            _LOGGER.debug(
                "Cached batch manager for slave %d, type %s with %d blocks.",
                slave_id,
                data_type,
                len(blocks),
            )
    
        total_batches = sum(len(bm.batches) for bm in self._batch_managers.values())
        total_entities = len(self.entities)
        _LOGGER.info(
            "OPTIMIZATION SUMMARY: %d entities optimized into %d batches (%.1f entities/batch)",
            total_entities,
            total_batches,
            total_entities / total_batches if total_batches else 0
        )
    
        # Add info about statistics tracking
        _LOGGER.info(
            "Statistical polling optimization enabled for %d entities. "
            "Check entity attributes after 24 hours for optimization recommendations.",
            sum(1 for e in self.entities if hasattr(e, "statistics") and e.statistics is not None)
        )
    
        self._initialized = True
        self._schedule_dynamic_updates()
        # NOTE: Do NOT fire coordinator_ready here! Only after first successful poll.

    def _signal_coordinator_ready(self) -> None:
        if not self._ready_event_fired and self._batch_managers:
            self.hass.bus.async_fire(
                f"{DOMAIN}_coordinator_ready",
                {"gateway_key": self.gateway}
            )
            self._ready_event_fired = True
            _LOGGER.info("🚀 Coordinator %s is fully ready with %d batch managers",
                         self.gateway, len(self._batch_managers))

    def _schedule_dynamic_updates(self) -> None:
        for unsub in self._polling_jobs:
            unsub()
        self._polling_jobs.clear()

        interval_groups: Dict[int, List[ModbusContext]] = {}
        for entity in self.entities:
            interval = getattr(entity, "scan_interval", None)
            if interval and interval > 0:
                if interval not in interval_groups:
                    interval_groups[interval] = []
                interval_groups[interval].append(entity)

        for interval, entities_in_group in interval_groups.items():
            _LOGGER.info(
                "Scheduling dynamic update for %d entities with a %d-second interval.",
                len(entities_in_group),
                interval,
            )

            async def update_task(_now: datetime, entities=entities_in_group) -> None:
                _LOGGER.debug(
                    "Polling job fired: interval=%d, entities=%d",
                    interval, len(entities)
                )
                await self._dynamic_update(entities)

            unsub = event.async_track_time_interval(
                self.hass, update_task, timedelta(seconds=interval)
            )
            self._polling_jobs.append(unsub)
        _LOGGER.debug(
            "Total polling jobs scheduled: %d", len(self._polling_jobs)
        )

    async def _dynamic_update(self, entities_to_update: List[ModbusContext]) -> None:
        """Update specific entities with statistics tracking and response time capture."""
        async with self._update_lock:
            start_time = time.time()
            # Update attempt timestamp (always recorded)
            self.last_update_attempt_timestamp = datetime.now(timezone.utc).isoformat()
            
            entity_keys = {e.desc.key for e in entities_to_update}
            _LOGGER.debug(
                "Dynamic update triggered for %d entities: %s",
                len(entities_to_update),
                entity_keys,
            )
            
            try:
                # Connection handling
                if not self.client.connected:
                    _LOGGER.info("Client not connected - attempting to connect")
                    await self.client.connect()
                    if not self.client.connected:
                        _LOGGER.error("Could not connect to Modbus device, skipping update")
                        return
    
                new_data = self.data.copy() if self.data else {}
                conversion = Conversion(type(self.client))
                any_value_changed = False
                successful_reads = 0
                failed_reads = 0
    
                grouped_for_update = self._group_entities_by_slave_and_type(entities_to_update)
    
                for (slave_id, data_type), group_entities in grouped_for_update.items():
                    bm = self._batch_managers.get((slave_id, data_type))
                    if not bm:
                        _LOGGER.warning(
                            "No batch manager found for slave %d, type %s. Skipping update.",
                            slave_id,
                            data_type,
                        )
                        continue
    
                    func = self.client._data_type_function_mapping[data_type]
                    blocks_for_update = [
                        ModbusEntityBlock(e.desc.register_address, e.desc.register_count, e)
                        for e in group_entities
                    ]
    
                    temp_bm = ReactiveBatchManager(blocks_for_update, self._max_read_size)
                    batches_to_read = temp_bm.get_batches()
    
                    for read_start_addr, read_count, original_batch_blocks in batches_to_read:
                        try:
                            # ✅ ENHANCED: Get response AND timing from client
                            resp, response_time = await self.client.read_data(
                                func, read_start_addr, read_count, slave_id, self._max_read_size
                            )
                            
                            if resp and not getattr(resp, "isError", lambda: False)():
                                successful_reads += 1
                                registers = getattr(resp, "registers", []) or getattr(resp, "bits", [])
                                for block in original_batch_blocks:
                                    if block.start >= read_start_addr and block.end < read_start_addr + read_count:
                                        idx = block.start - read_start_addr
                                        regs = registers[idx: idx + block.count]
                                        if len(regs) == block.count:
                                            old_value = new_data.get(block.entity.desc.key)
                                            value = conversion.convert_from_response(
                                                block.entity.desc, regs
                                            )
                                            new_data[block.entity.desc.key] = value
                                            
                                            # Track value changes for statistics
                                            value_changed = old_value != value
                                            if value_changed:
                                                any_value_changed = True
                                                
                                                # Store current value in context for future comparison
                                                if hasattr(block.entity, "last_value"):
                                                    block.entity.last_value = value
                                            
                                            # ✅ ENHANCED: Store response time in context
                                            if response_time is not None:
                                                block.entity.last_response_time = response_time
                                            
                                            # ✅ ENHANCED: Update statistics with REAL response time
                                            if hasattr(block.entity, "statistics"):
                                                block.entity.statistics.record_poll(
                                                    value_changed=value_changed,
                                                    timestamp=time.time(),
                                                    error=False,
                                                    response_time=response_time,
                                                    value=value
                                                )
                            else:
                                failed_reads += 1
                                # ✅ ENHANCED: Record errors with response time (if available)
                                for block in original_batch_blocks:
                                    if hasattr(block.entity, "statistics"):
                                        block.entity.statistics.record_poll(
                                            value_changed=False,
                                            timestamp=time.time(),
                                            error=True,
                                            response_time=response_time,
                                            value=None
                                        )
                                _LOGGER.debug(
                                    "Read failed for slave=%s, start=%s, count=%s",
                                    slave_id, read_start_addr, read_count
                                )
                        
                        except asyncio.CancelledError:
                            _LOGGER.debug(
                                "Batch read cancelled by HA for slave=%s, start=%s",
                                slave_id, read_start_addr
                            )
                            continue
                        
                        except Exception as ex:
                            failed_reads += 1
                            # ✅ ENHANCED: Record exceptions in statistics
                            for block in original_batch_blocks:
                                if hasattr(block.entity, "statistics"):
                                    block.entity.statistics.record_poll(
                                        value_changed=False,
                                        timestamp=time.time(),
                                        error=True,
                                        response_time=None,
                                        value=None
                                    )
                            _LOGGER.debug(
                                "Error reading batch for slave=%s, start=%s: %s",
                                slave_id, read_start_addr, ex
                            )
    
                # Update data and timestamps
                self.async_set_updated_data(new_data)
                
                if successful_reads > 0:
                    self.last_update_success = True
                    self.last_update_timestamp = datetime.now(timezone.utc).isoformat()
                    if any_value_changed:
                        self.last_value_change_timestamp = datetime.now(timezone.utc).isoformat()
                else:
                    self.last_update_success = False
    
                _LOGGER.debug(
                    "Dynamic update completed in %.2fs. Success: %d, Failed: %d, Changes: %s",
                    time.time() - start_time,
                    successful_reads,
                    failed_reads,
                    any_value_changed
                )
    
            except Exception as exc:
                self.last_update_success = False
                _LOGGER.exception(
                    "Error during dynamic update for %s: %s", self.gateway, exc
                )
                
    async def async_write_register(self, address: int, value: int, slave_id: int) -> bool:
        _LOGGER.debug(
            "Writing register: slave=%s, address=%s, value=%s", slave_id, address, value
        )
        try:
            # ✅ ENHANCED: Get response time from write operation
            resp, response_time = await self.client.write_data(
                address=address, value=value, slave=slave_id
            )

            if resp and not resp.isError():
                _LOGGER.debug(
                    "Success writing register in %.2fms. Requesting immediate refresh.",
                    response_time or 0
                )
                await self.async_request_refresh()
                return True
            else:
                _LOGGER.error("Failed to write register. Response: %s", resp)
                return False

        except Exception as exc:
            _LOGGER.exception("Exception during write_register: %s", exc)
            return False

    async def async_write_coil(self, address: int, value: bool, slave_id: int) -> bool:
        _LOGGER.debug(
            "Writing coil: slave=%s, address=%s, value=%s", slave_id, address, value
        )
        try:
            # ✅ ENHANCED: Get response time from coil write operation
            resp, response_time = await self.client.write_coil_data(
                address=address, value=value, slave=slave_id
            )
            if resp and not resp.isError():
                _LOGGER.debug(
                    "Success writing coil in %.2fms. Requesting immediate refresh.",
                    response_time or 0
                )
                await self.async_request_refresh()
                return True
            else:
                _LOGGER.error("Failed to write coil. Response: %s", resp)
                return False
        except Exception as exc:
            _LOGGER.exception("Exception during write_coil: %s", exc)
            return False

    async def async_write_text(self, address: int, text: str, slave_id: int) -> bool:
        _LOGGER.debug(
            "Writing text: slave=%s, address=%s, text='%s'", slave_id, address, text
        )
        try:
            conversion = Conversion(type(self.client))
            encoded_registers = conversion.encode_text(text)

            if not encoded_registers:
                _LOGGER.warning("Text encoding resulted in no registers to write.")
                return False

            # ✅ ENHANCED: Get response time from text write operation
            resp, response_time = await self.client.write_data(
                address=address, value=encoded_registers, slave=slave_id
            )

            if resp and not resp.isError():
                _LOGGER.debug(
                    "Success writing text in %.2fms. Requesting immediate refresh.",
                    response_time or 0
                )
                await self.async_request_refresh()
                return True
            else:
                _LOGGER.error("Failed to write text. Response: %s", resp)
                return False

        except Exception as exc:
            _LOGGER.exception("Exception during write_text: %s", exc)
            return False

    async def _async_update_data(self) -> Dict[str, Any]:
        """Fetch data from Modbus devices with statistics tracking and response time capture."""
        if not self._initialized and self.entities:
            _LOGGER.info("Coordinator not initialized - initializing with %d entities", len(self.entities))
            self._initialize_coordination_data()
        
        if not self.entities:
            _LOGGER.debug("No entities configured - returning empty data")
            return {}
        
        if not self._initialized:
            _LOGGER.warning("Coordinator failed to initialize - returning empty data")
            return {}
        
        async with self._update_lock:
            start_time = time.time()
            successful_reads = 0
            failed_reads = 0
            entities_updated = 0
            any_value_changed = False
    
            try:
                # Always record attempt
                self.last_update_attempt_timestamp = datetime.now(timezone.utc).isoformat()
                
                if not self.client.connected:
                    _LOGGER.info("Client not connected - attempting to connect")
                    await self.client.connect()
                    if not self.client.connected:
                        _LOGGER.error("Failed to connect to Modbus device")
                        return self.data or {}
    
                data: Dict[str, Any] = self.data.copy() if self.data else {}
                original_data_count = len(data)
                conversion = Conversion(type(self.client))
    
                _LOGGER.debug("Starting full update cycle. Batch managers: %s", list(self._batch_managers.keys()))
    
                for (slave_id, data_type), bm in self._batch_managers.items():
                    _LOGGER.debug("Processing batch for slave=%s, type=%s", slave_id, data_type)
                    func = self.client._data_type_function_mapping[data_type]
                    batches_to_read = bm.get_batches()
                    all_batch_results = []
    
                    for read_start_addr, read_count, original_batch_blocks in batches_to_read:
                        try:
                            # ✅ ENHANCED: Get response AND timing from client
                            resp, response_time = await self.client.read_data(
                                func, read_start_addr, read_count, slave_id, self._max_read_size
                            )
                            success = resp and not getattr(resp, "isError", lambda: False)()
                            all_batch_results.append((original_batch_blocks, success))
    
                            if not success:
                                failed_reads += 1
                                # ✅ ENHANCED: Record failed reads with timing info
                                for block in original_batch_blocks:
                                    if hasattr(block.entity, "statistics"):
                                        block.entity.statistics.record_poll(
                                            value_changed=False,
                                            timestamp=time.time(),
                                            error=True,
                                            response_time=response_time,
                                            value=None
                                        )
                                _LOGGER.debug(
                                    "Batch read failed for slave=%s, start=%s, count=%s",
                                    slave_id, read_start_addr, read_count
                                )
                                continue
    
                            successful_reads += 1
                            registers = getattr(resp, "registers", []) or getattr(resp, "bits", [])
                            if not registers:
                                _LOGGER.debug("No data returned for slave=%s, start=%s", slave_id, read_start_addr)
                                continue
    
                            for block in original_batch_blocks:
                                if block.start >= read_start_addr and block.end < read_start_addr + read_count:
                                    idx = block.start - read_start_addr
                                    regs = registers[idx: idx + block.count]
                                    if len(regs) == block.count:
                                        try:
                                            old_value = data.get(block.entity.desc.key)
                                            value = conversion.convert_from_response(
                                                block.entity.desc, regs
                                            )
                                            data[block.entity.desc.key] = value
                                            entities_updated += 1
                                            
                                            # Track value changes for statistics
                                            value_changed = old_value != value
                                            if value_changed:
                                                any_value_changed = True
                                                if _LOGGER.isEnabledFor(logging.DEBUG):
                                                    _LOGGER.debug("Value changed: %s: %s → %s",
                                                                block.entity.desc.key, old_value, value)
                                                
                                                # Store current value in context for future comparison
                                                if hasattr(block.entity, "last_value"):
                                                    block.entity.last_value = value
                                            
                                            # ✅ ENHANCED: Store response time in context
                                            if response_time is not None:
                                                block.entity.last_response_time = response_time
                                            
                                            # ✅ ENHANCED: Update statistics with REAL response time
                                            if hasattr(block.entity, "statistics"):
                                                block.entity.statistics.record_poll(
                                                    value_changed=value_changed,
                                                    timestamp=time.time(),
                                                    error=False,
                                                    response_time=response_time,
                                                    value=value
                                                )
                                        except Exception as e:
                                            _LOGGER.exception(
                                                "Data conversion error for key %s: %s",
                                                block.entity.desc.key, e
                                            )
    
                        except Exception as ex:
                            failed_reads += 1
                            # ✅ ENHANCED: Record exceptions in statistics
                            for block in original_batch_blocks:
                                if hasattr(block.entity, "statistics"):
                                    block.entity.statistics.record_poll(
                                        value_changed=False,
                                        timestamp=time.time(),
                                        error=True,
                                        response_time=None,
                                        value=None
                                    )
                            _LOGGER.debug(
                                "Exception during batch read for slave=%s, start=%s: %s",
                                slave_id, read_start_addr, ex
                            )
                            all_batch_results.append((original_batch_blocks, False))
    
                    if all_batch_results:
                        bm.feedback(all_batch_results)
    
                elapsed = time.time() - start_time
                _LOGGER.info(
                    "Update cycle completed in %.2fs. Success: %d, Failed: %d, Entities: %d, Changes: %s",
                    elapsed, successful_reads, failed_reads, entities_updated, any_value_changed
                )
    
                # Timestamp handling (keep for internal operation but reduce UI impact)
                if successful_reads > 0:
                    self.last_update_success = True
                    self.last_update_timestamp = datetime.now(timezone.utc).isoformat()
                    if any_value_changed:
                        self.last_value_change_timestamp = datetime.now(timezone.utc).isoformat()
                else:
                    self.last_update_success = False
    
                # Coordinator ready check
                if not self._ready_event_fired and successful_reads > 0:
                    self._signal_coordinator_ready()
    
                return data
    
            except Exception as exc:
                self.last_update_success = False
                _LOGGER.exception("Critical error in coordinator update: %s", exc)
                raise UpdateFailed(f"Error updating Modbus Coordinator: {exc}") from exc

    def _group_entities_by_slave_and_type(
        self, entities: List[ModbusContext]
    ) -> Dict[Tuple[int, str], List[ModbusContext]]:
        grouped = {}
        for entity in entities:
            key = (entity.slave_id, entity.desc.data_type)
            grouped.setdefault(key, []).append(entity)
        for key in grouped:
            grouped[key].sort(key=lambda ent: ent.desc.register_address)
        return grouped
        
    def _update_entity_statistics(self, entity, old_value, new_value, timestamp=None):
        """Update statistics for an entity."""
        if not timestamp:
            timestamp = time.time()
            
        value_changed = old_value != new_value
        
        # Update statistics if available
        if hasattr(entity, "statistics"):
            entity.statistics.record_poll(
                value_changed=value_changed,
                timestamp=timestamp
            )
        return value_changed

    # === Diagnostic helper methods for Home Assistant sensors ===

    def get_connection_status(self) -> str:
        """Return 'connected' or 'disconnected' for the Modbus client."""
        client = getattr(self, "client", None)
        return "connected" if client and getattr(client, "connected", False) else "disconnected"

    def get_health_score(self) -> float:
        """Calculate and return an overall health score percentage."""
        score = 0.0
        max_score = 100.0
        if getattr(self, "client", None) and getattr(self.client, "connected", False):
            score += 25.0
        if getattr(self, "last_update_success", False):
            score += 25.0
        if self.entities:
            score += min(len(self.entities) * 2, 25.0)
        if getattr(self, "_initialized", False) and getattr(self, "_batch_managers", {}):
            total_batches = sum(len(bm.batches) for bm in self._batch_managers.values() if bm)
            if total_batches > 0:
                efficiency = min((len(self.entities) / total_batches) * 5, 25.0)
                score += efficiency
        return round((score / max_score) * 100, 0)

    def get_batch_efficiency(self) -> float:
        """Return batch efficiency percentage."""
        if not getattr(self, "_initialized", False):
            return 0.0
        if not self.entities or not getattr(self, "_batch_managers", {}):
            return 0.0
        total_entities = len(self.entities)
        total_batches = sum(len(bm.batches) for bm in self._batch_managers.values() if bm)
        if total_batches == 0:
            return 0.0
        efficiency = min((total_entities / total_batches) * 10, 100.0)
        return round(efficiency, 1)

    def get_active_batch_managers(self) -> int:
        """Return number of active batch managers."""
        if not getattr(self, "_initialized", False):
            return 0
        return sum(1 for bm in getattr(self, "_batch_managers", {}).values() if bm and getattr(bm, "batches", []))

    def get_total_register_reads(self) -> int:
        """Return total number of entities with data (proxy for total reads)."""
        return len(self.data) if getattr(self, "data", None) else 0

    def get_coordinator_status(self) -> str:
        """Return coordinator status text."""
        client = getattr(self, "client", None)
        if not client:
            return "No Client"
        if not getattr(client, "connected", False):
            return "Disconnected"
        if not getattr(self, "_initialized", False):
            return "Initializing"
        if not self.entities:
            return "No Entities"
        if getattr(self, "last_update_success", None) is False:
            return "Update Failed"
        if getattr(self, "last_update_success", None) is True:
            return "Working"
        return "Unknown"

    def get_batch_manager_types(self) -> str:
        """Return a human-readable list of active batch manager types."""
        MODBUS_TYPE_NAMES = {
            "HOLDING_REGISTER": "Holding Registers",
            "INPUT_REGISTER": "Input Registers",
            "COIL": "Coils",
            "DISCRETE_INPUT": "Discrete Inputs",
        }
        if not getattr(self, "_batch_managers", {}):
            return "None"
        manager_types = []
        for key in self._batch_managers.keys():
            if isinstance(key, tuple) and len(key) >= 2:
                slave_id, data_type = key[0], key[1]
                friendly_name = MODBUS_TYPE_NAMES.get(str(data_type), str(data_type))
                manager_types.append(f"{friendly_name} (Slave {slave_id})")
        if not manager_types:
            return "Unknown"
        return ", ".join(sorted(set(manager_types)))

    def get_batch_manager_debug_details(self) -> dict:
        """Return debug details of batch managers."""
        MODBUS_TYPE_NAMES = {
            "HOLDING_REGISTER": "Holding Registers",
            "INPUT_REGISTER": "Input Registers",
            "COIL": "Coils",
            "DISCRETE_INPUT": "Discrete Inputs",
        }
        details = {}
        for key, bm in getattr(self, "_batch_managers", {}).items():
            if isinstance(key, tuple) and len(key) >= 2:
                slave_id, data_type = key[0], key[1]
                friendly_name = MODBUS_TYPE_NAMES.get(str(data_type), str(data_type))
                manager_key = f"{friendly_name} (Slave {slave_id})"
                batches = getattr(bm, "batches", [])
                batch_count = len(batches)
                details[manager_key] = {
                    "batches": batch_count,
                    "congestion_window": getattr(bm, "congestion_window", "unknown"),
                }
        return details