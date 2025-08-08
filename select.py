"""Modbus Local Gateway Select Entity"""

import logging

from homeassistant.components.select import SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .context import ModbusContext
from .coordinator import ModbusCoordinator
from .entity_management.base import ModbusSelectEntityDescription
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
    """Set up the select platform."""
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
        device: dict,
    ):
        """Initialize the Modbus select entity."""
        super().__init__(
            coordinator=coordinator,
            modbus_context=modbus_context,
            device_info=device,
        )

        desc = modbus_context.desc
        if not isinstance(desc, ModbusSelectEntityDescription) or not desc.select_options:
            raise TypeError("Invalid description type or missing select options for ModbusSelectEntity")

        self.entity_description = desc
        self._attr_options = list(desc.select_options.values())
        self._attr_current_option = None  # Initialize state

    def _update_from_coordinator(self):
        """Update the entity's state from the coordinator's data."""
        key = self.entity_description.key
        value = self.coordinator.data.get(key)
        
        if value is None:
            self._attr_current_option = None
            return
            
        select_options = self.entity_description.select_options
        # The value from Modbus is the key, we need to find the corresponding option text
        self._attr_current_option = select_options.get(int(value))

    async def async_select_option(self, option: str) -> None:
        """Asynchronously change the selected option."""
        desc = self.entity_description
        
        if not desc.select_options:
            raise ValueError(f"No options defined for select entity {self.name}")

        # Find the key corresponding to the selected option string
        value_to_write = None
        for key, value in desc.select_options.items():
            if value == option:
                value_to_write = key
                break
        
        if value_to_write is None:
            raise ValueError(f"Invalid option: {option}")

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
            
    def _generate_extra_attributes(self):
        """Add select-specific attributes."""
        attrs = super()._generate_extra_attributes()
        desc = getattr(self._modbus_context, "desc", None)
        if desc:
            attrs.update({
                "select_options": getattr(desc, "select_options", {}),
            })
        return attrs
