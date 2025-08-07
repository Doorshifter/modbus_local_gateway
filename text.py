"""Modbus Local Gateway Text Entity"""

import logging

# Minimal top-level imports
_LOGGER = logging.getLogger(__name__)
DOMAIN = "modbus_local_gateway"

async def async_setup_entry(hass, config_entry, async_add_entities):
    """Set up the text platform."""
    # Import modules only when needed
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.helpers.entity_platform import AddEntitiesCallback
    
    from .entity_management.const import ControlType
    from .helpers import async_setup_entities
    
    await async_setup_entities(
        hass=hass,
        config_entry=config_entry,
        async_add_entities=async_add_entities,
        control=ControlType.TEXT,
        entity_class=ModbusTextEntity,
    )

class ModbusTextEntity:
    """Text entity for Modbus gateway."""

    def __init__(
        self,
        coordinator,
        modbus_context,
        device,
    ):
        """Initialize the Modbus text entity."""
        # Import needed modules inside the method
        import asyncio
        from homeassistant.components.text import TextEntity
        from .entity_management.coordinator_entity import ModbusCoordinatorEntity
        from .entity_management.base import ModbusTextEntityDescription
        
        # Multiple inheritance using class objects dynamically loaded
        self.__class__ = type(
            self.__class__.__name__,
            (ModbusCoordinatorEntity, TextEntity),
            {}
        )
        
        # Initialize parent classes using super()
        ModbusCoordinatorEntity.__init__(
            self,
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
    def native_value(self):
        """Return the value of the text entity."""
        # Safe access to entity key
        key = getattr(self.entity_description, "key", None) if hasattr(self, "entity_description") else None
        if key is None and hasattr(self, "_attr_entity_description"):
            key = getattr(self._attr_entity_description, "key", None)
            
        value = self.coordinator.data.get(key) if key else None
        return str(value) if value is not None else None

    def _generate_extra_attributes(self):
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
    
    def set_value(self, value):
        """Synchronously set the value of the Modbus text entity."""
        import asyncio
        future = asyncio.run_coroutine_threadsafe(self.async_set_value(value), self.hass.loop)
        future.result()

    async def async_set_value(self, value):
        """Asynchronously set the value of the Modbus text entity."""
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