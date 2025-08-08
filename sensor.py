"""Sensor platform for Modbus Local Gateway."""

import logging
import asyncio

# Import required Home Assistant components and base classes
from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback

# Import from custom component
from .entity_management.const import ControlType
from .helpers import async_setup_entities, get_gateway_key
from .entity_management.coordinator_entity import ModbusCoordinatorEntity
from .entity_management.base import ModbusSensorEntityDescription
from .context import ModbusContext
from .coordinator import ModbusCoordinator

_LOGGER = logging.getLogger(__name__)
DOMAIN = "modbus_local_gateway"

# Dictionary to track setup status for each gateway key
_DIAGNOSTIC_SETUP_STATUS = {}

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddEntitiesCallback):
    """Set up the sensor platform."""
    await async_setup_entities(
        hass=hass,
        config_entry=config_entry,
        async_add_entities=async_add_entities,
        control=ControlType.SENSOR,
        entity_class=ModbusSensorEntity,
    )

    gateway_key = get_gateway_key(config_entry, with_slave=True)

    # Initialize the setup status dictionary if it doesn't exist
    if not hasattr(hass.data[DOMAIN], "_DIAGNOSTIC_SETUP_STATUS"):
        hass.data[DOMAIN]["_DIAGNOSTIC_SETUP_STATUS"] = {}

    # We'll use a lock to prevent concurrent setup attempts
    setup_lock = asyncio.Lock()

    @callback
    def coordinator_ready_callback(event):
        """Handle coordinator ready event."""
        _LOGGER.debug("Coordinator ready event fired for gateway_key=%s event.data=%s", gateway_key, event.data)
        if event.data.get("gateway_key") == gateway_key:
            # Check if setup is already complete or in progress
            setup_status = hass.data[DOMAIN]["_DIAGNOSTIC_SETUP_STATUS"].get(gateway_key)
            if setup_status == "complete" or setup_status == "in_progress":
                _LOGGER.debug("Diagnostics already %s for gateway_key=%s, skipping.", 
                             setup_status, gateway_key)
                return
                
            # Mark as in progress and start setup
            hass.data[DOMAIN]["_DIAGNOSTIC_SETUP_STATUS"][gateway_key] = "in_progress"
            hass.async_create_task(
                _setup_diagnostic_sensors(hass, config_entry, async_add_entities, gateway_key, setup_lock)
            )

    hass.bus.async_listen(f"{DOMAIN}_coordinator_ready", coordinator_ready_callback)

    # Initialize delayed setup only if not already set up or in progress
    setup_status = hass.data[DOMAIN].get("_DIAGNOSTIC_SETUP_STATUS", {}).get(gateway_key)
    if setup_status != "complete" and setup_status != "in_progress":
        hass.async_create_task(
            _delayed_diagnostic_setup_check(hass, config_entry, async_add_entities, gateway_key, setup_lock)
        )

async def _delayed_diagnostic_setup_check(hass, config_entry, async_add_entities, gateway_key, setup_lock):
    """Delayed check for diagnostic sensor setup."""
    _LOGGER.debug("Delayed diagnostic setup check running for gateway_key=%s", gateway_key)
    await asyncio.sleep(3)

    # Check if setup has already happened or is in progress
    setup_status = hass.data[DOMAIN].get("_DIAGNOSTIC_SETUP_STATUS", {}).get(gateway_key)
    if setup_status == "complete" or setup_status == "in_progress":
        _LOGGER.debug("Diagnostics already %s in delayed check for gateway_key=%s, skipping.", 
                      setup_status, gateway_key)
        return

    if gateway_key in hass.data.get(DOMAIN, {}):
        coordinator = hass.data[DOMAIN][gateway_key]
        initialized = getattr(coordinator, '_initialized', False)

        if initialized:
            # Mark as in progress before we start
            hass.data[DOMAIN].setdefault("_DIAGNOSTIC_SETUP_STATUS", {})[gateway_key] = "in_progress"
            await _setup_diagnostic_sensors(hass, config_entry, async_add_entities, gateway_key, setup_lock)
        else:
            _LOGGER.debug("Delayed check: Coordinator for %s NOT initialized", gateway_key)
    else:
        _LOGGER.debug("Delayed check: Coordinator for %s not found in hass.data[DOMAIN]", gateway_key)

async def _setup_diagnostic_sensors(hass, config_entry, async_add_entities, gateway_key, setup_lock):
    """Set up diagnostic sensors."""
    # Use the lock to prevent concurrent setup for the same gateway key
    async with setup_lock:
        # Double-check if setup is already complete
        setup_status = hass.data[DOMAIN].get("_DIAGNOSTIC_SETUP_STATUS", {}).get(gateway_key)
        if setup_status == "complete":
            _LOGGER.debug("Diagnostics already set up for gateway_key=%s, exiting.", gateway_key)
            return
            
        # Mark as in progress (in case it wasn't already)
        hass.data[DOMAIN].setdefault("_DIAGNOSTIC_SETUP_STATUS", {})[gateway_key] = "in_progress"
        
        _LOGGER.debug("Setting up diagnostic sensors for gateway_key=%s", gateway_key)

        try:
            # Import diagnostic sensor module when needed, not at module load time
            from .diagnostic_sensor import DIAGNOSTIC_SENSORS, ModbusDiagnosticSensor, build_device_info
        except ImportError as err:
            _LOGGER.error("Diagnostic sensor module not available: %r", err)
            # Reset status since setup failed
            hass.data[DOMAIN]["_DIAGNOSTIC_SETUP_STATUS"][gateway_key] = "failed"
            return

        try:
            if gateway_key not in hass.data.get(DOMAIN, {}):
                _LOGGER.warning("Coordinator not found for diagnostic sensors: %s", gateway_key)
                hass.data[DOMAIN]["_DIAGNOSTIC_SETUP_STATUS"][gateway_key] = "failed"
                return

            coordinator = hass.data[DOMAIN][gateway_key]

            if not getattr(coordinator, '_initialized', False):
                _LOGGER.warning("Coordinator for %s is NOT initialized", gateway_key)
                hass.data[DOMAIN]["_DIAGNOSTIC_SETUP_STATUS"][gateway_key] = "pending"
                return

            device_info = getattr(coordinator, "device_info", None)
            if not device_info:
                _LOGGER.warning("Device info missing for coordinator %s", gateway_key)
                hass.data[DOMAIN]["_DIAGNOSTIC_SETUP_STATUS"][gateway_key] = "failed"
                return

            # Create device info dictionary
            device_info_dict = build_device_info(device_info, config_entry)
            
            # Generate a stable unique ID hash - use a consistent property from config_entry
            config_id = config_entry.entry_id

            # Create sensor entities with the config ID included for unique ID generation
            entities_to_add = []
            for description in DIAGNOSTIC_SENSORS:
                try:
                    entity = ModbusDiagnosticSensor(
                        coordinator=coordinator,
                        description=description,
                        config_entry=config_entry,
                        device_info=device_info,
                    )
                    entities_to_add.append(entity)
                except Exception as e:
                    _LOGGER.warning("Error creating diagnostic sensor %s: %s", description.key, e)

            if entities_to_add:
                _LOGGER.info(
                    "Adding %d diagnostic sensor entities for gateway %s: %s",
                    len(entities_to_add), gateway_key, [e.entity_description.key for e in entities_to_add]
                )
                
                # Check for existing entity IDs to prevent duplicates
                unique_ids = set()
                entities_to_add_deduped = []
                
                for entity in entities_to_add:
                    if entity._attr_unique_id not in unique_ids:
                        unique_ids.add(entity._attr_unique_id)
                        entities_to_add_deduped.append(entity)
                    else:
                        _LOGGER.warning(
                            "Duplicate unique ID detected and skipped: %s", entity._attr_unique_id
                        )
                
                async_add_entities(entities_to_add_deduped, True)
                _LOGGER.info(
                    "Added %d diagnostic sensor entities for gateway %s",
                    len(entities_to_add_deduped), gateway_key
                )
            
            # Mark setup as complete only after adding entities
            hass.data[DOMAIN]["_DIAGNOSTIC_SETUP_STATUS"][gateway_key] = "complete"

        except Exception as exc:
            _LOGGER.exception("Failed to set up diagnostic sensors: %s", exc)
            hass.data[DOMAIN]["_DIAGNOSTIC_SETUP_STATUS"][gateway_key] = "failed"

class ModbusSensorEntity(ModbusCoordinatorEntity, SensorEntity):
    """Sensor entity for Modbus gateway."""

    def __init__(
        self,
        coordinator: ModbusCoordinator,
        modbus_context: ModbusContext,
        device: dict,
    ):
        """Initialize the Modbus sensor."""
        super().__init__(
            coordinator=coordinator,
            modbus_context=modbus_context,
            device_info=device
        )
        
        if not isinstance(modbus_context.desc, ModbusSensorEntityDescription):
            raise TypeError(f"Invalid description type: {type(modbus_context.desc)}")
        
        # Store the entity description and initialize the native value
        self.entity_description = modbus_context.desc
        self._attr_native_value = None

    def _update_from_coordinator(self):
        """Update the entity's state from the coordinator's data."""
        # This method is called by the base class when new data is available.
        key = self.entity_description.key
        
        if self.coordinator.data and key in self.coordinator.data:
            self._attr_native_value = self.coordinator.data[key]
        else:
            # Set to None if the key is not in the data, which results in an "unknown" state
            self._attr_native_value = None