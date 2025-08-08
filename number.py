"""Modbus Local Gateway Number Entity"""

import logging

from homeassistant.components.number import NumberEntity, NumberMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .context import ModbusContext
from .coordinator import ModbusCoordinator
from .entity_management.base import ModbusNumberEntityDescription
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
    """Set up the number platform."""
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
        device: dict,
    ):
        """Initialize the Modbus number entity."""
        super().__init__(
            coordinator=coordinator,
            modbus_context=modbus_context,
            device_info=device,
        )

        if not isinstance(modbus_context.desc, ModbusNumberEntityDescription):
            raise TypeError("Invalid description type for ModbusNumberEntity")

        self.entity_description = modbus_context.desc
        self._attr_native_value = None  # Initialize state
        self._attr_mode = NumberMode.BOX # Set default mode

    def _update_from_coordinator(self):
        """Update the entity's state from the coordinator's data."""
        key = self.entity_description.key
        self._attr_native_value = self.coordinator.data.get(key)

    async def async_set_native_value(self, value: float) -> None:
        """Set the value of the Modbus number entity."""
        desc = self.entity_description
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
            
    def _generate_extra_attributes(self):
        """Add number-specific attributes."""
        attrs = super()._generate_extra_attributes()
        desc = getattr(self._modbus_context, "desc", None)
        if desc:
            attrs.update({
                "precision": getattr(desc, "precision", 0),
                "min_value": getattr(desc, "native_min_value", None),
                "max_value": getattr(desc, "native_max_value", None),
                "step": getattr(desc, "native_step", None),
            })
        
        return attrs
