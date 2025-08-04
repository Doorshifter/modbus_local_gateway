"""Modbus Local Gateway binary sensors."""

from __future__ import annotations

import logging

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .context import ModbusContext
from .coordinator import ModbusCoordinator
from .entity_management.base import ModbusBinarySensorEntityDescription
from .entity_management.const import ControlType
from .entity_management.coordinator_entity import ModbusCoordinatorEntity
from .helpers import async_setup_entities

_LOGGER: logging.Logger = logging.getLogger(__name__)

async def async_setup_entry(
    hass,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Modbus Local Gateway binary sensors."""
    await async_setup_entities(
        hass=hass,
        config_entry=config_entry,
        async_add_entities=async_add_entities,
        control=ControlType.BINARY_SENSOR,
        entity_class=ModbusBinarySensorEntity,
    )

class ModbusBinarySensorEntity(ModbusCoordinatorEntity, BinarySensorEntity):
    """Binary sensor entity for Modbus gateway."""

    def __init__(
        self,
        coordinator: ModbusCoordinator,
        modbus_context: ModbusContext,
        device,
    ) -> None:
        """Initialize the Modbus binary sensor."""
        super().__init__(
            coordinator=coordinator,
            modbus_context=modbus_context,
            device_info=device
        )
        if not isinstance(modbus_context.desc, ModbusBinarySensorEntityDescription):
            raise TypeError("Invalid description type")
        
        # Set both properties for consistency
        self._attr_entity_description = modbus_context.desc
        self.entity_description = modbus_context.desc

    def _get_current_value(self):
        """Get current binary sensor state."""
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
        """Return true if the binary sensor is on."""
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
        """Add binary sensor specific attributes."""
        attrs = super()._generate_extra_attributes()
        
        # Add binary sensor specific attributes
        desc = getattr(self._modbus_context, "desc", None)
        if desc:
            attrs["on_value"] = getattr(desc, "on", True)
        
        return attrs