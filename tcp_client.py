"""
TCP Client for Modbus Local Gateway with Adaptive Timeouts and Rate Limiting.

Features:
- Adaptive RTT-based timeouts
- Connection reset on timeout/error
- Request batching and locking
- Configurable per-second register read rate limiting (None/unlimited = RTT only)
"""

import asyncio
import logging
import time
from collections import deque
from typing import Any, Callable, Coroutine, List, Optional

from pymodbus.client import AsyncModbusTcpClient
from pymodbus.exceptions import ModbusException
from pymodbus.framer import FramerType
from pymodbus.pdu import ModbusPDU

from .const import (
    ADAPTIVE_TIMEOUT_ALPHA,
    ADAPTIVE_TIMEOUT_BETA,
    ADAPTIVE_TIMEOUT_INITIAL_RTT,
    ADAPTIVE_TIMEOUT_INITIAL_VAR,
    ADAPTIVE_TIMEOUT_MAX,
    ADAPTIVE_TIMEOUT_MIN,
)
from .context import ModbusContext
from .entity_management.const import ModbusDataType
from .transaction import MyTransactionManager

_LOGGER: logging.Logger = logging.getLogger(__name__)

class DummyCommParams:
    """Fallback for pymodbus TransactionManager."""
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

class AsyncModbusTcpClientGateway(AsyncModbusTcpClient):
    """
    Custom Modbus TCP client with adaptive timeouts, rate limiting, and robust error recovery.
    """

    _CLIENT: dict[str, "AsyncModbusTcpClientGateway"] = {}

    def __init__(
        self,
        host: str,
        port: int = 502,
        framer: FramerType = FramerType.SOCKET,
        source_address: Optional[tuple[str, int]] = None,
        max_registers_per_second: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._data_type_function_mapping: dict[str, Callable] = {
            ModbusDataType.HOLDING_REGISTER: self.read_holding_registers,
            ModbusDataType.INPUT_REGISTER: self.read_input_registers,
            ModbusDataType.COIL: self.read_coils,
            ModbusDataType.DISCRETE_INPUT: self.read_discrete_inputs,
        }
        self.lock = asyncio.Lock()
        # Adaptive timeout setup
        self._smoothed_rtt: float = ADAPTIVE_TIMEOUT_INITIAL_RTT
        self._rtt_var: float = ADAPTIVE_TIMEOUT_INITIAL_VAR
        self._alpha: float = ADAPTIVE_TIMEOUT_ALPHA
        self._beta: float = ADAPTIVE_TIMEOUT_BETA
        self.dynamic_timeout: float = self._smoothed_rtt + 4 * self._rtt_var

        # Rate limiting setup
        self.max_registers_per_second = max_registers_per_second
        self._register_read_history = deque()  # (timestamp, count)

        kwargs.pop("timeout", None)
        super().__init__(
            host=host, port=port, framer=framer, source_address=source_address, **kwargs
        )

        params = getattr(self, "comm_params", None) or DummyCommParams(host, port)
        self.context = MyTransactionManager(
            params=params,
            framer=framer,
            retries=kwargs.get("retries", 3),
            is_server=False,
            trace_connect=kwargs.get("trace_connect", False),
            trace_packet=kwargs.get("trace_packet", False),
            trace_pdu=kwargs.get("trace_pdu", False),
        )

    async def _reset_connection(self) -> None:
        """Force-close and re-establish the connection with transaction cleanup."""
        _LOGGER.warning(
            "Resetting Modbus connection to %s:%s to recover from error. "
            "This may cause transaction ID mismatches for in-flight requests.",
            self.comm_params.host,
            self.comm_params.port,
        )
        if self.connected:
            self.close()
        
        # Clear transaction manager state to prevent ID mismatches
        if hasattr(self, 'context') and hasattr(self.context, 'reset'):
            try:
                self.context.reset()
                _LOGGER.debug("Transaction manager reset to clear pending transactions.")
            except Exception as e:
                _LOGGER.debug("Failed to reset transaction manager: %s", e)
        

        # Introduce a small delay to allow any late, in-flight packets to be
        # discarded by the OS while the socket is closed. This makes the
        # subsequent reconnect cleaner and less prone to transaction ID mismatches.
        await asyncio.sleep(0.25)
        
        await self.connect()

    def _update_timeout(self, rtt_sample: float) -> None:
        """Update the dynamic timeout based on a new RTT sample."""
        self._rtt_var = (1 - self._beta) * self._rtt_var + self._beta * abs(
            rtt_sample - self._smoothed_rtt
        )
        self._smoothed_rtt = (
            (1 - self._alpha) * self._smoothed_rtt + self._alpha * rtt_sample
        )
        self.dynamic_timeout = max(
            ADAPTIVE_TIMEOUT_MIN,
            min(self._smoothed_rtt + 4 * self._rtt_var, ADAPTIVE_TIMEOUT_MAX),
        )

    async def _rate_limit_register_reads(self, n_registers: int):
        """
        Enforce max_registers_per_second if set; else do nothing (RTT only).
        """
        if not self.max_registers_per_second:
            return
        now = time.monotonic()
        # Remove old records (>1s ago)
        while self._register_read_history and self._register_read_history[0][0] < now - 1:
            self._register_read_history.popleft()
        used = sum(count for t, count in self._register_read_history)
        if used + n_registers > self.max_registers_per_second:
            sleep_for = 1 - (now - self._register_read_history[0][0])
            await asyncio.sleep(max(0.01, sleep_for))
            return await self._rate_limit_register_reads(n_registers)
        self._register_read_history.append((now, n_registers))

    async def _execute_transaction(
        self, transaction: Coroutine[Any, Any, Any], operation_name: str, address: int
    ) -> tuple[Any | None, float | None]:
        """
        Execute a single Modbus transaction with locking, timeout, and recovery.
        
        Returns:
            tuple: (response, response_time_ms) where response_time_ms is in milliseconds
        """
        async with self.lock:
            if not self.connected:
                _LOGGER.info("Client not connected. Attempting to connect.")
                await self.connect()
                if not self.connected:
                    _LOGGER.warning("Failed to connect to gateway for %s.", operation_name)
                    return None, None
            
            try:
                start_time = time.monotonic()
                response = await asyncio.wait_for(
                    transaction,
                    timeout=self.dynamic_timeout,
                )
                rtt_sample = time.monotonic() - start_time
                self._update_timeout(rtt_sample)
                
                # Convert to milliseconds for statistics
                response_time_ms = rtt_sample * 1000
                
                _LOGGER.debug(
                    "Modbus %s at address %s completed in %.2fms", 
                    operation_name, address, response_time_ms
                )
                
                return response, response_time_ms
            
            except asyncio.CancelledError:
                _LOGGER.warning(
                    "🔄 [CANCELLATION] Modbus %s at address %s was cancelled by Home Assistant. "
                    "Transaction ID may be orphaned.",
                    operation_name,
                    address,
                )
                # ✅ IMPROVED: Don't reset connection on cancellation - just log warning
                # Connection reset can orphan other transactions and cause ID mismatches
                raise  # Re-raise so HA/core can finish cancelling
            
            except asyncio.TimeoutError:
                _LOGGER.error(
                    "⏰ [TIMEOUT] Modbus %s at address %s timed out after %.2fs (dynamic timeout). "
                    "This usually means the device did not respond in time.",
                    operation_name,
                    address,
                    self.dynamic_timeout,
                )
                # ✅ IMPROVED: Only reset on actual timeouts, not cancellations
                await self._reset_connection()
                return None, None
            
            except ModbusException as exc:
                # Handle wrapped cancellation specifically
                if "Request cancelled outside pymodbus" in str(exc):
                    _LOGGER.warning(
                        "🔄 [CANCELLATION] Modbus %s at address %s was cancelled by Home Assistant (wrapped)",
                        operation_name,
                        address,
                    )
                    # ✅ IMPROVED: Don't reset connection - just handle gracefully
                    raise asyncio.CancelledError from exc
                else:
                    _LOGGER.error(
                        "🔥 [MODBUS ERROR] Exception during Modbus %s at address %s: %s. Resetting connection.",
                        operation_name,
                        address,
                        exc,
                    )
                    await self._reset_connection()
                    return None, None
            
            except Exception:
                _LOGGER.exception(
                    "❗ [UNEXPECTED ERROR] During Modbus %s at address %s. Resetting connection.",
                    operation_name,
                    address
                )
                await self._reset_connection()
                return None, None

    async def read_data(
        self, func: Callable, address: int, count: int, slave: int, max_read_size: int
    ) -> tuple[ModbusPDU | None, float | None]:
        """
        Read registers or coils with adaptive timeouts and automatic recovery.
        Respects both batch size and (optionally) per-second register limit.
        
        Returns:
            tuple: (response, avg_response_time_ms) where avg_response_time_ms is the average
                   response time in milliseconds for all sub-requests, or None if any failed
        """
        is_register_func: bool = func in [
            self.read_holding_registers,
            self.read_input_registers,
        ]
        response: ModbusPDU | None = None
        remaining: int = count
        current_address: int = address
        response_times: list[float] = []
    
        while remaining > 0:
            read_count: int = min(max_read_size, remaining)
            await self._rate_limit_register_reads(read_count)
            transaction = func(address=current_address, count=read_count, slave=slave)
            
            temp_response, response_time = await self._execute_transaction(
                transaction, func.__name__, current_address
            )
            
            if temp_response is None or temp_response.isError() or not hasattr(
                temp_response, "registers" if is_register_func else "bits"
            ):
                _LOGGER.error(
                    "Invalid or error response for %s at address %s: %s",
                    func.__name__,
                    current_address,
                    temp_response,
                )
                return None, None
    
            # Collect response time for averaging
            if response_time is not None:
                response_times.append(response_time)
    
            remaining -= read_count
            current_address += read_count
    
            if response is None:
                response = temp_response
            else:
                if is_register_func:
                    response.registers += temp_response.registers
                else:
                    response.bits += temp_response.bits
    
        # Calculate average response time
        avg_response_time = sum(response_times) / len(response_times) if response_times else None
        
        if avg_response_time is not None:
            _LOGGER.debug(
                "Completed %s read of %d registers starting at %d in %.2fms (avg of %d requests)",
                func.__name__, count, address, avg_response_time, len(response_times)
            )
    
        return response, avg_response_time

    async def write_data(self, address: int, value: int, slave: int) -> tuple[Any | None, float | None]:
        """
        Write a single value to a holding register with robust error handling.
        
        Returns:
            tuple: (response, response_time_ms) where response_time_ms is in milliseconds
        """
        transaction = self.write_register(address=address, value=value, slave=slave)
        response, response_time = await self._execute_transaction(
            transaction, "write_register", address
        )
        
        if response_time is not None:
            _LOGGER.debug(
                "Completed write_register to address %d in %.2fms", 
                address, response_time
            )
        
        return response, response_time

    async def write_coil_data(self, address: int, value: bool, slave: int) -> tuple[Any | None, float | None]:
        """
        Write a single value to a coil with robust error handling.
        
        Returns:
            tuple: (response, response_time_ms) where response_time_ms is in milliseconds
        """
        transaction = self.write_coil(address=address, value=value, slave=slave)
        response, response_time = await self._execute_transaction(
            transaction, "write_coil", address
        )
        
        if response_time is not None:
            _LOGGER.debug(
                "Completed write_coil to address %d in %.2fms", 
                address, response_time
            )
        
        return response, response_time

    async def _write_single_register(self, address: int, value: int, slave: int) -> float | None:
        """
        Write a single value to a Modbus register.
        
        Returns:
            float | None: Response time in milliseconds, or None if failed
        """
        result, response_time = await self._execute_transaction(
            self.write_register(address=address, value=value, slave=slave),
            "write_register", 
            address
        )
        
        if result and result.isError():
            _LOGGER.error(
                "Failed to write value %d to address %d: %s", value, address, result
            )
            return None
        
        return response_time
    
    async def _write_multiple_registers(
        self, address: int, values: List[int], slave: int
    ) -> float | None:
        """
        Attempt to write multiple values to Modbus registers in a single command.
        
        Returns:
            float | None: Response time in milliseconds, or None if failed
        """
        result, response_time = await self._execute_transaction(
            self.write_registers(address=address, values=values, slave=slave),
            "write_registers",
            address
        )
        
        if result and result.isError():
            _LOGGER.warning(
                "Failed to write multiple values: %s. Falling back to individual writes.",
                result,
            )
            # Fall back to individual writes and return average time
            individual_times = []
            for i, value in enumerate(values):
                time_taken = await self._write_single_register(address + i, value, slave)
                if time_taken is not None:
                    individual_times.append(time_taken)
            
            return sum(individual_times) / len(individual_times) if individual_times else None
        
        return response_time
    
    async def _write_registers_individually(
        self, address: int, values: List[int], slave: int
    ) -> float | None:
        """
        Fallback method to write multiple values to Modbus registers one by one.
        
        Returns:
            float | None: Average response time in milliseconds, or None if all failed
        """
        response_times = []
        for i, value in enumerate(values):
            time_taken = await self._write_single_register(address + i, value, slave)
            if time_taken is not None:
                response_times.append(time_taken)
        
        return sum(response_times) / len(response_times) if response_times else None

    @classmethod
    def async_get_client_connection(
        cls, host: str, port: int, max_registers_per_second: Optional[int] = None
    ) -> "AsyncModbusTcpClientGateway":
        """
        Get a shared Modbus client instance for a given host and port.
        """
        key: str = f"{host}:{port}"
        if key not in cls._CLIENT:
            _LOGGER.debug("Creating new client for gateway %s", key)
            cls._CLIENT[key] = cls(
                host=host, port=port, framer=FramerType.SOCKET, retries=5,
                max_registers_per_second=max_registers_per_second,
            )
        return cls._CLIENT[key]