"""Modbus Local Gateway Switch Entity

Provides on/off control for Modbus-connected devices via both coils and holding registers.
Entities are always set up using the async_setup_entities helper, and device info is always
sourced from the coordinator.

Author: Doorshifter
"""

import asyncio
import logging

from typing import Any

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .context import ModbusContext
from .coordinator import ModbusCoordinator
from .entity_management.coordinator_entity import ModbusCoordinatorEntity
from .entity_management.base import ModbusSwitchEntityDescription
from .entity_management.const import ControlType, ModbusDataType
from .helpers import async_setup_entities

_LOGGER: logging.Logger = logging.getLogger(__name__)
INVALID_DATA_TYPE = "Invalid data_type for switch"

async def async_setup_entry(
    hass,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Modbus switch entities for the config entry."""
    await async_setup_entities(
        hass=hass,
        config_entry=config_entry,
        async_add_entities=async_add_entities,
        control=ControlType.SWITCH,
        entity_class=ModbusSwitchEntity,
    )

class ModbusSwitchEntity(ModbusCoordinatorEntity, SwitchEntity):
    """
    Switch entity for Modbus gateway.

    Provides on/off control for Modbus-based devices connected via coils or holding registers.
    Uses the ModbusCoordinator for efficient polling and state management.
    Automatically exposes extra Modbus diagnostic attributes via smart caching.
    """

    def __init__(
        self,
        coordinator: ModbusCoordinator,
        modbus_context: ModbusContext,
        device,
    ) -> None:
        """Initialize the Modbus switch."""
        super().__init__(
            coordinator=coordinator,
            modbus_context=modbus_context,
            device_info=device
        )
        if not isinstance(modbus_context.desc, ModbusSwitchEntityDescription):
            raise TypeError("Invalid description type")
            
        # Set both properties for consistency
        self._attr_entity_description = modbus_context.desc
        self.entity_description = modbus_context.desc

    def _get_current_value(self):
        """Get current switch state."""
        return self.is_on

    @property
    def native_value(self):
        """Return the native value from coordinator data."""
        # Safe access to entity key
        key = getattr(self.entity_description, "key", None) if hasattr(self, "entity_description") else None
        if key is None and hasattr(self, "_attr_entity_description"):
            key = getattr(self._attr_entity_description, "key", None)
        return self.coordinator.data.get(key) if key else None

    @property
    def is_on(self) -> bool | None:
        """Return true if the switch is on."""
        value = self.native_value
        if value is None:
            return None
            
        # Safe access to entity description
        on_value = None
        if hasattr(self, "entity_description"):
            on_value = getattr(self.entity_description, "on", True)
        elif hasattr(self, "_attr_entity_description"):
            on_value = getattr(self._attr_entity_description, "on", True)
        return value == on_value

    def _generate_extra_attributes(self) -> dict:
        """Add switch-specific attributes."""
        attrs = super()._generate_extra_attributes()
        
        # Add switch-specific attributes
        desc = getattr(self._modbus_context, "desc", None)
        if desc:
            attrs.update({
                "on_value": getattr(desc, "on", True),
                "off_value": getattr(desc, "off", False),
            })
        
        return attrs

    def turn_on(self, **kwargs: Any) -> None:
        """Synchronously turn the entity on (calls async)."""
        future = asyncio.run_coroutine_threadsafe(self.async_turn_on(**kwargs), self.hass.loop)
        future.result()

    def turn_off(self, **kwargs: Any) -> None:
        """Synchronously turn the entity off (calls async)."""
        future = asyncio.run_coroutine_threadsafe(self.async_turn_off(**kwargs), self.hass.loop)
        future.result()

    async def async_turn_on(self, **kwargs: Any) -> None:
        """
        Asynchronously turn the entity on.
        Writes the 'on' value to the Modbus device using the coordinator's write method.
        """
        # Safe access to entity description
        if hasattr(self, "entity_description"):
            desc = self.entity_description
        elif hasattr(self, "_attr_entity_description"):
            desc = self._attr_entity_description
        else:
            _LOGGER.error("Missing entity description for %s", self.name)
            return
            
        if desc.data_type == ModbusDataType.COIL:
            value_to_write = bool(desc.on)
            await self.coordinator.async_write_coil(
                address=desc.register_address,
                value=value_to_write,
                slave_id=self._modbus_context.slave_id,
            )
        elif desc.data_type == ModbusDataType.HOLDING_REGISTER:
            value_to_write = int(desc.on)
            await self.coordinator.async_write_register(
                address=desc.register_address,
                value=value_to_write,
                slave_id=self._modbus_context.slave_id,
            )
        else:
            raise ValueError(INVALID_DATA_TYPE)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """
        Asynchronously turn the entity off.
        Writes the 'off' value to the Modbus device using the coordinator's write method.
        """
        # Safe access to entity description
        if hasattr(self, "entity_description"):
            desc = self.entity_description
        elif hasattr(self, "_attr_entity_description"):
            desc = self._attr_entity_description
        else:
            _LOGGER.error("Missing entity description for %s", self.name)
            return
            
        if desc.data_type == ModbusDataType.COIL:
            value_to_write = bool(desc.off)
            await self.coordinator.async_write_coil(
                address=desc.register_address,
                value=value_to_write,
                slave_id=self._modbus_context.slave_id,
            )
        elif desc.data_type == ModbusDataType.HOLDING_REGISTER:
            value_to_write = int(desc.off)
            await self.coordinator.async_write_register(
                address=desc.register_address,
                value=value_to_write,
                slave_id=self._modbus_context.slave_id,
            )
        else:
            raise ValueError(INVALID_DATA_TYPE)