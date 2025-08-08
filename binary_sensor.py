"""Modbus Local Gateway binary sensors."""

import logging

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .context import ModbusContext
from .coordinator import ModbusCoordinator
from .entity_management.base import ModbusBinarySensorEntityDescription
from .entity_management.const import ControlType
from .entity_management.coordinator_entity import ModbusCoordinatorEntity
from .helpers import async_setup_entities

_LOGGER = logging.getLogger(__name__)
DOMAIN = "modbus_local_gateway"


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
):
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
        device: dict,
    ):
        """Initialize the Modbus binary sensor."""
        super().__init__(
            coordinator=coordinator,
            modbus_context=modbus_context,
            device_info=device,
        )

        if not isinstance(modbus_context.desc, ModbusBinarySensorEntityDescription):
            raise TypeError("Invalid description type for ModbusBinarySensorEntity")

        self.entity_description = modbus_context.desc
        self._attr_is_on = None  # Initialize state

    def _update_from_coordinator(self):
        """Update the entity's state from the coordinator's data."""
        key = self.entity_description.key
        value = self.coordinator.data.get(key)

        if value is None:
            self._attr_is_on = None
            return

        on_value = getattr(self.entity_description, "on", True)
        self._attr_is_on = value == on_value

    def _generate_extra_attributes(self):
        """Add binary sensor specific attributes."""
        attrs = super()._generate_extra_attributes()
        desc = getattr(self._modbus_context, "desc", None)
        if desc:
            attrs["on_value"] = getattr(desc, "on", True)
        return attrs
