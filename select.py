"""Modbus Local Gateway Select Entity"""

from __future__ import annotations

import asyncio
import logging

from typing import Any

from homeassistant.components.select import SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .context import ModbusContext
from .coordinator import ModbusCoordinator
from .entity_management.base import ModbusSelectEntityDescription
from .entity_management.const import ControlType, ModbusDataType
from .entity_management.coordinator_entity import ModbusCoordinatorEntity
from .helpers import async_setup_entities

_LOGGER: logging.Logger = logging.getLogger(__name__)

async def async_setup_entry(
    hass,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Modbus select entities for the config entry."""
    await async_setup_entities(
        hass=hass,
        config_entry=config_entry,
        async_add_entities=async_add_entities,
        control=ControlType.SELECT,
        entity_class=ModbusSelectEntity,
    )

class ModbusSelectEntity(ModbusCoordinatorEntity, SelectEntity):
    """Select entity for Modbus gateway."""

    def __init__(
        self,
        coordinator: ModbusCoordinator,
        modbus_context: ModbusContext,
        device,
    ) -> None:
        """Initialize the Modbus select entity."""
        super().__init__(
            coordinator=coordinator,
            modbus_context=modbus_context,
            device_info=device
        )
        desc = modbus_context.desc
        if not isinstance(desc, ModbusSelectEntityDescription) or not desc.select_options:
            raise TypeError("Invalid description type or missing select options")
        
        # Set both properties for consistency
        self._attr_entity_description = desc
        self.entity_description = desc
        
        self._attr_options: list[str] = list(desc.select_options.values())

    def _get_current_value(self):
        """Get current selected option."""
        return self.current_option

    @property
    def native_value(self):
        """Return the native value from coordinator data."""
        key = getattr(self.entity_description, "key", None) if hasattr(self, "entity_description") else None
        if key is None and hasattr(self, "_attr_entity_description"):
            key = getattr(self._attr_entity_description, "key", None)
        return self.coordinator.data.get(key) if key else None

    @property
    def current_option(self) -> str | None:
        """Return the currently selected option."""
        value = self.native_value
        if value is None:
            return None
            
        # Safe access to select options
        select_options = None
        if hasattr(self, "entity_description"):
            select_options = getattr(self.entity_description, "select_options", {})
        elif hasattr(self, "_attr_entity_description"):
            select_options = getattr(self._attr_entity_description, "select_options", {})
        
        return select_options.get(int(value)) if select_options else None

    def _generate_extra_attributes(self) -> dict:
        """Add select-specific attributes."""
        attrs = super()._generate_extra_attributes()
        
        # Add select-specific attributes
        desc = getattr(self._modbus_context, "desc", None)
        if desc:
            attrs.update({
                "select_options": getattr(desc, "select_options", {}),
                "available_options": list(getattr(desc, "select_options", {}).values()),
            })
        
        return attrs

    def select_option(self, option: str) -> None:
        """Synchronously select an option."""
        future = asyncio.run_coroutine_threadsafe(self.async_select_option(option), self.hass.loop)
        future.result()

    async def async_select_option(self, option: str) -> None:
        """Asynchronously change the selected option."""
        # Safe access to entity description
        if hasattr(self, "entity_description"):
            desc = self.entity_description
        elif hasattr(self, "_attr_entity_description"):
            desc = self._attr_entity_description
        else:
            _LOGGER.error("Missing entity description for %s", self.name)
            return
            
        if not desc.select_options:
            raise ValueError(f"No options defined for select entity {self.name}")

        value_keys = list(desc.select_options.keys())
        value_values = list(desc.select_options.values())
        if option not in value_values:
            raise ValueError(f"Invalid option: {option}")

        value_to_write: int = value_keys[value_values.index(option)]

        _LOGGER.debug(
            "Setting select entity %s to '%s' (value: %s, address: %s)",
            self.name,
            option,
            value_to_write,
            desc.register_address,
        )

        if desc.data_type == ModbusDataType.HOLDING_REGISTER:
            await self.coordinator.async_write_register(
                address=desc.register_address,
                value=value_to_write,
                slave_id=self._modbus_context.slave_id,
            )
        else:
            _LOGGER.error("Cannot write to %s: Not a holding register", self.name)