"""Modbus Local Gateway Text Entity"""

import logging

from homeassistant.components.text import TextEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .context import ModbusContext
from .coordinator import ModbusCoordinator
from .entity_management.base import ModbusTextEntityDescription
from .entity_management.const import ControlType, ModbusDataType
from .entity_management.coordinator_entity import ModbusCoordinatorEntity
from .helpers import async_setup_entities

_LOGGER = logging.getLogger(__name__)
DOMAIN = "modbus_local_gateway"


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
):
    """Set up the text platform."""
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
        device: dict,
    ):
        """Initialize the Modbus text entity."""
        super().__init__(
            coordinator=coordinator,
            modbus_context=modbus_context,
            device_info=device,
        )

        if not isinstance(modbus_context.desc, ModbusTextEntityDescription):
            raise TypeError("Invalid description type for ModbusTextEntity")

        self.entity_description = modbus_context.desc
        self._attr_native_value = None  # Initialize state

    def _update_from_coordinator(self):
        """Update the entity's state from the coordinator's data."""
        key = self.entity_description.key
        value = self.coordinator.data.get(key)
        self._attr_native_value = str(value) if value is not None else None

    async def async_set_value(self, value: str) -> None:
        """Asynchronously set the value of the Modbus text entity."""
        desc = self.entity_description
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
            
    def _generate_extra_attributes(self):
        """Add text-specific attributes."""
        attrs = super()._generate_extra_attributes()
        desc = getattr(self._modbus_context, "desc", None)
        if desc:
            attrs.update({
                "max_length": getattr(desc, "native_max", None),
                "min_length": getattr(desc, "native_min", None),
            })
        
        return attrs