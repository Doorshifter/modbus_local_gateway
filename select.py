"""Modbus Local Gateway Select Entity"""

import logging

# Minimal top-level imports
_LOGGER = logging.getLogger(__name__)
DOMAIN = "modbus_local_gateway"

async def async_setup_entry(hass, config_entry, async_add_entities):
    """Set up the select platform."""
    # Import modules only when needed
    from homeassistant.components.select import SelectEntity
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.helpers.entity_platform import AddEntitiesCallback
    
    from .entity_management.const import ControlType
    from .helpers import async_setup_entities
    
    # Set up entities
    await async_setup_entities(
        hass=hass,
        config_entry=config_entry,
        async_add_entities=async_add_entities,
        control=ControlType.SELECT,
        entity_class=ModbusSelectEntity,
    )

class ModbusSelectEntity:
    """Select entity for Modbus gateway."""

    def __init__(
        self,
        coordinator,
        modbus_context,
        device,
    ):
        """Initialize the Modbus select entity."""
        # Import needed modules inside the method
        import asyncio
        from homeassistant.components.select import SelectEntity
        from .entity_management.coordinator_entity import ModbusCoordinatorEntity
        from .entity_management.base import ModbusSelectEntityDescription
        
        # Multiple inheritance using class objects dynamically loaded
        self.__class__ = type(
            self.__class__.__name__,
            (ModbusCoordinatorEntity, SelectEntity),
            {}
        )
        
        # Initialize parent classes using super()
        ModbusCoordinatorEntity.__init__(
            self,
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
        
        self._attr_options = list(desc.select_options.values())

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
    def current_option(self):
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

    def _generate_extra_attributes(self):
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

    def select_option(self, option):
        """Synchronously select an option."""
        import asyncio
        future = asyncio.run_coroutine_threadsafe(self.async_select_option(option), self.hass.loop)
        future.result()

    async def async_select_option(self, option):
        """Asynchronously change the selected option."""
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
            
        if not desc.select_options:
            raise ValueError(f"No options defined for select entity {self.name}")

        value_keys = list(desc.select_options.keys())
        value_values = list(desc.select_options.values())
        if option not in value_values:
            raise ValueError(f"Invalid option: {option}")

        value_to_write = value_keys[value_values.index(option)]

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