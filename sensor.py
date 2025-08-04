"""Sensor platform for Modbus Local Gateway."""

from __future__ import annotations

import asyncio
import logging

from typing import Set

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .context import ModbusContext
from .coordinator import ModbusCoordinator
from .entity_management.base import ModbusSensorEntityDescription
from .entity_management.const import ControlType
from .entity_management.coordinator_entity import ModbusCoordinatorEntity
from .helpers import async_setup_entities, get_gateway_key
from .const import DOMAIN

_LOGGER: logging.Logger = logging.getLogger(__name__)

_DIAGNOSTIC_SETUP_COMPLETE: Set[str] = set()

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Modbus sensors from device files."""
    await async_setup_entities(
        hass=hass,
        config_entry=config_entry,
        async_add_entities=async_add_entities,
        control=ControlType.SENSOR,
        entity_class=ModbusSensorEntity,
    )

    gateway_key = get_gateway_key(config_entry, with_slave=True)

    @callback
    def coordinator_ready_callback(event):
        """Handle coordinator ready event."""
        _LOGGER.warning("DIAG DEBUG: coordinator_ready_callback fired for gateway_key=%s event.data=%s", gateway_key, event.data)
        if event.data.get("gateway_key") == gateway_key:
            if gateway_key in _DIAGNOSTIC_SETUP_COMPLETE:
                _LOGGER.warning("DIAG DEBUG: Diagnostics already set up for gateway_key=%s, skipping.", gateway_key)
                return
            hass.async_create_task(
                _setup_diagnostic_sensors(hass, config_entry, async_add_entities, gateway_key)
            )

    hass.bus.async_listen(f"{DOMAIN}_coordinator_ready", coordinator_ready_callback)

    if gateway_key not in _DIAGNOSTIC_SETUP_COMPLETE:
        hass.async_create_task(
            _delayed_diagnostic_setup_check(hass, config_entry, async_add_entities, gateway_key)
        )

async def _delayed_diagnostic_setup_check(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
    gateway_key: str,
) -> None:
    _LOGGER.warning("DIAG DEBUG: _delayed_diagnostic_setup_check running for gateway_key=%s", gateway_key)
    await asyncio.sleep(3)

    if gateway_key in _DIAGNOSTIC_SETUP_COMPLETE:
        _LOGGER.warning("DIAG DEBUG: Diagnostics already set up in delayed check for gateway_key=%s, skipping.", gateway_key)
        return

    if gateway_key in hass.data.get(DOMAIN, {}):
        coordinator = hass.data[DOMAIN][gateway_key]
        initialized = getattr(coordinator, '_initialized', False)

        if initialized and gateway_key not in _DIAGNOSTIC_SETUP_COMPLETE:
            await _setup_diagnostic_sensors(hass, config_entry, async_add_entities, gateway_key)
        elif not initialized:
            _LOGGER.warning("DIAG DEBUG: Delayed check: Coordinator for %s NOT initialized", gateway_key)
    else:
        _LOGGER.warning("DIAG DEBUG: Delayed check: Coordinator for %s not found in hass.data[DOMAIN]", gateway_key)

async def _setup_diagnostic_sensors(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
    gateway_key: str,
) -> None:
    _LOGGER.warning("DIAG DEBUG: _setup_diagnostic_sensors entered for gateway_key=%s", gateway_key)
    if gateway_key in _DIAGNOSTIC_SETUP_COMPLETE:
        _LOGGER.warning("DIAG DEBUG: Diagnostics already set up for gateway_key=%s, exiting.", gateway_key)
        return

    try:
        from .diagnostic_sensor import DIAGNOSTIC_SENSORS, ModbusDiagnosticSensor, build_device_info
    except ImportError as err:
        _LOGGER.error("DIAG DEBUG: Diagnostic sensor module not available: %r", err)
        if gateway_key in _DIAGNOSTIC_SETUP_COMPLETE:
            _DIAGNOSTIC_SETUP_COMPLETE.remove(gateway_key)
        return

    try:
        if gateway_key not in hass.data.get(DOMAIN, {}):
            _LOGGER.warning("DIAG DEBUG: Coordinator not found for diagnostic sensors: %s", gateway_key)
            return

        coordinator: ModbusCoordinator = hass.data[DOMAIN][gateway_key]

        if not getattr(coordinator, '_initialized', False):
            _LOGGER.warning("DIAG DEBUG: Coordinator for %s is NOT initialized", gateway_key)
            return

        device_info = getattr(coordinator, "device_info", None)
        if not device_info:
            _LOGGER.warning("DIAG DEBUG: device_info missing for coordinator %s", gateway_key)
            return

        device_info_dict = build_device_info(device_info, config_entry)
        device_info_dict["name"] = f"{device_info_dict.get('name', 'Device')} Diagnostics"
        entities_to_add = [
            ModbusDiagnosticSensor(
                coordinator=coordinator,
                description=description,
                config_entry=config_entry,
                device_info=device_info_dict,
            )
            for description in DIAGNOSTIC_SENSORS
        ]

        _LOGGER.warning(
            "About to add %d diagnostic sensor entities for gateway %s: %s",
            len(entities_to_add), gateway_key, [e.entity_description.key for e in entities_to_add]
        )
        async_add_entities(entities_to_add, True)
        _DIAGNOSTIC_SETUP_COMPLETE.add(gateway_key)
        _LOGGER.info(
            "Added %d diagnostic sensor entities for gateway %s",
            len(entities_to_add), gateway_key
        )

    except Exception as exc:
        if gateway_key in _DIAGNOSTIC_SETUP_COMPLETE:
            _DIAGNOSTIC_SETUP_COMPLETE.remove(gateway_key)
        _LOGGER.exception("Failed to set up diagnostic sensors: %s", exc)

class ModbusSensorEntity(ModbusCoordinatorEntity, SensorEntity):
    """Sensor entity for Modbus gateway."""

    def __init__(
        self,
        coordinator: ModbusCoordinator,
        modbus_context: ModbusContext,
        device,
    ) -> None:
        """Initialize the Modbus sensor."""
        super().__init__(
            coordinator=coordinator,
            modbus_context=modbus_context,
            device_info=device
        )
        if not isinstance(modbus_context.desc, ModbusSensorEntityDescription):
            raise TypeError(f"Invalid description type: {type(modbus_context.desc)}")
        
        # Store the entity description as both properties
        self._attr_entity_description = modbus_context.desc
        self.entity_description = modbus_context.desc

    def _get_current_value(self):
        """Get current sensor value."""
        return self.native_value

    @property
    def native_value(self):
        """Return the native value from coordinator data."""
        # Access entity key safely
        key = getattr(self.entity_description, "key", None) if hasattr(self, "entity_description") else None
        if key is None and hasattr(self, "_attr_entity_description"):
            key = getattr(self._attr_entity_description, "key", None)
        
        # Fallback to other identifiers if needed
        if key is None:
            key = getattr(self, "unique_id", None)
            
        return self.coordinator.data.get(key) if key else None