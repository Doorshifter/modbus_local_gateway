"""Modbus Local Gateway Switch Entity"""

import logging

# Minimal top-level imports
_LOGGER = logging.getLogger(__name__)
DOMAIN = "modbus_local_gateway"

# Constants
INVALID_DATA_TYPE = "Invalid data_type for switch"

async def async_setup_entry(hass, config_entry, async_add_entities):
    """Set up the switch platform."""
    # Import modules only when needed
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.helpers.entity_platform import AddEntitiesCallback
    
    from .entity_management.const import ControlType
    from .helpers import async_setup_entities
    
    await async_setup_entities(
        hass=hass,
        config_entry=config_entry,
        async_add_entities=async_add_entities,
        control=ControlType.SWITCH,
        entity_class=ModbusSwitchEntity,
    )

class ModbusSwitchEntity:
    """Switch entity for Modbus gateway."""

    def __init__(
        self,
        coordinator,
        modbus_context,
        device,
    ):
        """Initialize the Modbus switch."""
        # Import needed modules inside the method
        import asyncio
        from homeassistant.components.switch import SwitchEntity
        from .entity_management.coordinator_entity import ModbusCoordinatorEntity
        from .entity_management.base import ModbusSwitchEntityDescription
        
        # Multiple inheritance using class objects dynamically loaded
        self.__class__ = type(
            self.__class__.__name__,
            (ModbusCoordinatorEntity, SwitchEntity),
            {}
        )
        
        # Initialize parent classes using super()
        ModbusCoordinatorEntity.__init__(
            self,
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
    def is_on(self):
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

    def _generate_extra_attributes(self):
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

    def turn_on(self, **kwargs):
        """Synchronously turn the entity on (calls async)."""
        import asyncio
        future = asyncio.run_coroutine_threadsafe(self.async_turn_on(**kwargs), self.hass.loop)
        future.result()

    def turn_off(self, **kwargs):
        """Synchronously turn the entity off (calls async)."""
        import asyncio
        future = asyncio.run_coroutine_threadsafe(self.async_turn_off(**kwargs), self.hass.loop)
        future.result()

    async def async_turn_on(self, **kwargs):
        """Asynchronously turn the entity on."""
        # Import needed modules inside the method
        from .entity_management.const import ModbusDataType
        
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

    async def async_turn_off(self, **kwargs):
        """Asynchronously turn the entity off."""
        # Import needed modules inside the method
        from .entity_management.const import ModbusDataType
        
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