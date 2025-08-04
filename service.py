"""Services for Modbus Local Gateway."""
import logging
import voluptuous as vol
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.config_entries import ConfigEntry

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

async def async_setup_services(hass: HomeAssistant):
    """Set up services for Modbus Local Gateway integration."""
    
    async def set_scan_interval(call: ServiceCall):
        """Set scan interval for a Modbus entity or group of entities."""
        entity_id = call.data.get("entity_id")
        scan_interval = call.data.get("scan_interval")
        
        if not entity_id or not scan_interval:
            _LOGGER.error("Both entity_id and scan_interval are required")
            return
        
        # Get all gateway coordinators
        coordinators = {}
        for key, data in hass.data.get(DOMAIN, {}).items():
            if hasattr(data, "entities"):
                coordinators[key] = data
        
        # Find the entity in available coordinators
        entity_found = False
        for gateway_key, coordinator in coordinators.items():
            for entity in coordinator.entities:
                if getattr(entity, "entity_id", None) == entity_id or (
                    isinstance(entity_id, list) and entity.entity_id in entity_id
                ):
                    entity_found = True
                    old_interval = entity.scan_interval
                    entity.scan_interval = scan_interval
                    _LOGGER.info(
                        "Updated scan interval for %s from %s to %s seconds",
                        entity.entity_id, old_interval, scan_interval
                    )
                    
            # Reschedule polling jobs if we found entities for this coordinator
            if entity_found:
                coordinator._schedule_dynamic_updates()
        
        if not entity_found:
            _LOGGER.warning("No entities found matching %s", entity_id)
    
    hass.services.async_register(
        DOMAIN,
        "set_scan_interval",
        set_scan_interval,
        schema=vol.Schema({
            vol.Required("entity_id"): cv.entity_id,
            vol.Required("scan_interval"): vol.All(vol.Coerce(int), vol.Range(min=1, max=3600)),
        }),
    )
    
    return True