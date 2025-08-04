"""Modbus Local Gateway Number Entity"""

from __future__ import annotations

import asyncio
import logging

from typing import Any

from homeassistant.components.number import NumberEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .context import ModbusContext
from .coordinator import ModbusCoordinator
from .entity_management.base import ModbusNumberEntityDescription
from .entity_management.const import ControlType, ModbusDataType
from .entity_management.coordinator_entity import ModbusCoordinatorEntity
from .helpers import async_setup_entities

_LOGGER: logging.Logger = logging.getLogger(__name__)

async def async_setup_entry(
    hass,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Modbus number entities for the config entry."""
    await async_setup_entities(
        hass=hass,
        config_entry=config_entry,
        async_add_entities=async_add_entities,
        control=ControlType.NUMBER,
        entity_class=ModbusNumberEntity,
    )

class ModbusNumberEntity(ModbusCoordinatorEntity, NumberEntity):
    """Number entity for Modbus gateway."""

    def __init__(
        self,
        coordinator: ModbusCoordinator,
        modbus_context: ModbusContext,
        device,
    ) -> None:
        """Initialize the Modbus number entity."""
        super().__init__(
            coordinator=coordinator,
            modbus_context=modbus_context,
            device_info=device
        )
        if not isinstance(modbus_context.desc, ModbusNumberEntityDescription):
            raise TypeError("Invalid description type")
        
        # Set both properties for consistency
        self._attr_entity_description = modbus_context.desc
        self.entity_description = modbus_context.desc

    def _get_current_value(self):
        """Get current number value."""
        return self.native_value

    @property
    def native_value(self):
        """Return the native value from coordinator data."""
        key = getattr(self.entity_description, "key", None) if hasattr(self, "entity_description") else None
        if key is None and hasattr(self, "_attr_entity_description"):
            key = getattr(self._attr_entity_description, "key", None)
        return self.coordinator.data.get(key) if key else None

    def _generate_extra_attributes(self) -> dict:
        """Add number-specific attributes."""
        attrs = super()._generate_extra_attributes()
        
        # Add number-specific attributes
        desc = getattr(self._modbus_context, "desc", None)
        if desc:
            attrs.update({
                "precision": getattr(desc, "precision", 0),
                "min_value": getattr(desc, "native_min_value", None),
                "max_value": getattr(desc, "native_max_value", None),
                "step": getattr(desc, "native_step", None),
            })
        
        return attrs

    def set_native_value(self, value: float) -> None:
        """Set the value of the Modbus number entity (sync wrapper)."""
        future = asyncio.run_coroutine_threadsafe(self.async_set_native_value(value), self.hass.loop)
        future.result()

    async def async_set_native_value(self, value: float) -> None:
        """Set the value of the Modbus number entity (async)."""
        # Safe access to entity description
        if hasattr(self, "entity_description"):
            desc = self.entity_description
        elif hasattr(self, "_attr_entity_description"):
            desc = self._attr_entity_description
        else:
            _LOGGER.error("Missing entity description for %s", self.name)
            return
            
        _LOGGER.debug(
            "Setting value for %s to %s (address: %s)",
            self.name,
            value,
            desc.register_address,
        )
        raw_value = int(value * (10 ** getattr(desc, "precision", 0)))
        if desc.data_type == ModbusDataType.HOLDING_REGISTER:
            await self.coordinator.async_write_register(
                address=desc.register_address,
                value=raw_value,
                slave_id=self._modbus_context.slave_id,
            )
        else:
            _LOGGER.error("Cannot write to %s: Not a holding register", self.name)