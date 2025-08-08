"""Modbus Local Gateway Switch Entity"""

import logging

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .context import ModbusContext
from .coordinator import ModbusCoordinator
from .entity_management.base import ModbusSwitchEntityDescription
from .entity_management.const import ControlType, ModbusDataType
from .entity_management.coordinator_entity import ModbusCoordinatorEntity
from .helpers import async_setup_entities

_LOGGER = logging.getLogger(__name__)
DOMAIN = "modbus_local_gateway"
INVALID_DATA_TYPE = "Invalid data_type for switch"


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
):
    """Set up the switch platform."""
    await async_setup_entities(
        hass=hass,
        config_entry=config_entry,
        async_add_entities=async_add_entities,
        control=ControlType.SWITCH,
        entity_class=ModbusSwitchEntity,
    )


class ModbusSwitchEntity(ModbusCoordinatorEntity, SwitchEntity):
    """Switch entity for Modbus gateway."""

    def __init__(
        self,
        coordinator: ModbusCoordinator,
        modbus_context: ModbusContext,
        device: dict,
    ):
        """Initialize the Modbus switch."""
        super().__init__(
            coordinator=coordinator,
            modbus_context=modbus_context,
            device_info=device,
        )

        if not isinstance(modbus_context.desc, ModbusSwitchEntityDescription):
            raise TypeError("Invalid description type for ModbusSwitchEntity")

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

    async def async_turn_on(self, **kwargs) -> None:
        """Asynchronously turn the entity on."""
        desc = self.entity_description
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

    async def async_turn_off(self, **kwargs) -> None:
        """Asynchronously turn the entity off."""
        desc = self.entity_description
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
            
    def _generate_extra_attributes(self):
        """Add switch-specific attributes."""
        attrs = super()._generate_extra_attributes()
        desc = getattr(self._modbus_context, "desc", None)
        if desc:
            attrs.update({
                "on_value": getattr(desc, "on", True),
                "off_value": getattr(desc, "off", False),
            })
        
        return attrs
