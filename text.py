"""Modbus Local Gateway Text Entity"""

from __future__ import annotations

import asyncio
import logging

from homeassistant.components.text import TextEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .context import ModbusContext
from .coordinator import ModbusCoordinator
from .entity_management.base import ModbusTextEntityDescription
from .entity_management.const import ControlType, ModbusDataType
from .entity_management.coordinator_entity import ModbusCoordinatorEntity
from .helpers import async_setup_entities

_LOGGER: logging.Logger = logging.getLogger(__name__)

async def async_setup_entry(
    hass,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Modbus text entities for the config entry."""
    await async_setup_entities(
        hass=hass,
        config_entry=config_entry,
        async_add_entities=async_add_entities,
        control=ControlType.TEXT,
        entity_class=ModbusTextEntity,
    )

class ModbusTextEntity(ModbusCoordinatorEntity, TextEntity):
    """Text entity for Modbus gateway."""

    def __init__(
        self,
        coordinator: ModbusCoordinator,
        modbus_context: ModbusContext,
        device,
    ) -> None:
        """Initialize the Modbus text entity."""
        super().__init__(
            coordinator=coordinator,
            modbus_context=modbus_context,
            device_info=device
        )
        if not isinstance(modbus_context.desc, ModbusTextEntityDescription):
            raise TypeError("Invalid description type")
            
        # Set both properties for consistency
        self._attr_entity_description = modbus_context.desc
        self.entity_description = modbus_context.desc

    def _get_current_value(self):
        """Get current text value."""
        return self.native_value

    @property
    def native_value(self) -> str | None:
        """Return the value of the text entity."""
        # Safe access to entity key
        key = getattr(self.entity_description, "key", None) if hasattr(self, "entity_description") else None
        if key is None and hasattr(self, "_attr_entity_description"):
            key = getattr(self._attr_entity_description, "key", None)
            
        value = self.coordinator.data.get(key) if key else None
        return str(value) if value is not None else None

    def _generate_extra_attributes(self) -> dict:
        """Add text-specific attributes."""
        attrs = super()._generate_extra_attributes()
        
        # Add text-specific attributes
        desc = getattr(self._modbus_context, "desc", None)
        if desc:
            attrs.update({
                "max_length": getattr(desc, "native_max", None),
                "min_length": getattr(desc, "native_min", None),
            })
        
        return attrs
    
    def set_value(self, value: str) -> None:
        """Synchronously set the value of the Modbus text entity."""
        future = asyncio.run_coroutine_threadsafe(self.async_set_value(value), self.hass.loop)
        future.result()

    async def async_set_value(self, value: str) -> None:
        """Asynchronously set the value of the Modbus text entity."""
        # Safe access to entity description
        if hasattr(self, "entity_description"):
            desc = self.entity_description
        elif hasattr(self, "_attr_entity_description"):
            desc = self._attr_entity_description
        else:
            _LOGGER.error("Missing entity description for %s", self.name)
            return
            
        _LOGGER.debug(
            "Setting text for %s to '%s' (address: %s)",
            self.name,
            value,
            desc.register_address,
        )
        if desc.data_type == ModbusDataType.HOLDING_REGISTER:
            if hasattr(self.coordinator, "async_write_text"):
                await self.coordinator.async_write_text(
                    address=desc.register_address,
                    text=value,
                    slave_id=self._modbus_context.slave_id,
                )
            else:
                _LOGGER.error("Coordinator is missing `async_write_text` method.")
        else:
            _LOGGER.error("Cannot write text to %s: Not a holding register", self.name)