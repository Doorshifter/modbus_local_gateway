"""
Modbus Local Gateway Integration - Robust Initialization & Synchronization

Ensures that device info and the ModbusCoordinator are fully set up and available
before any platform or diagnostic entity is initialized. Diagnostics are only
set up after a 'coordinator_ready' event is fired.
"""

import os
import logging
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_FILENAME
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady

from .const import (
    DOMAIN,
    PLATFORMS,
    CONF_SLAVE_ID,
    DEFAULT_SCAN_INTERVAL,
    CONF_SCAN_INTERVAL,
)
from .coordinator import ModbusCoordinator
from .entity_management.device_loader import create_device_info
from .tcp_client import AsyncModbusTcpClientGateway
from .statistics import EntityStatisticsTracker
from .service import async_setup_services

_LOGGER: logging.Logger = logging.getLogger(__name__)

def get_gateway_key(entry: ConfigEntry, with_slave: bool = True) -> str:
    """Generate consistent gateway key for coordinator lookup."""
    try:
        if with_slave:
            return f"{entry.data[CONF_HOST]}:{entry.data[CONF_PORT]}:{entry.data[CONF_SLAVE_ID]}"
        return f"{entry.data[CONF_HOST]}:{entry.data[CONF_PORT]}"
    except KeyError as e:
        _LOGGER.error("Missing required config entry data: %s", e)
        raise ConfigEntryNotReady(f"Missing configuration: {e}") from e

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up the Modbus Local Gateway integration from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    # STEP 1: Load device info from YAML (blocking)
    try:
        device_info = await hass.async_add_executor_job(
            create_device_info, hass, entry.data[CONF_FILENAME]
        )
        if not device_info:
            raise ConfigEntryNotReady("Device info YAML loaded but returned empty")
    except FileNotFoundError as e:
        _LOGGER.error(
            "Failed to load device info YAML from %s (cwd: %s): %s",
            entry.data[CONF_FILENAME],
            os.getcwd(),
            e
        )
        raise ConfigEntryNotReady(f"Device YAML not found: {e}") from e
    except Exception as e:
        _LOGGER.exception("Unexpected error loading device info YAML: %s", e)
        raise ConfigEntryNotReady(f"Error loading device info: {e}") from e

    # STEP 2: Create or get the Modbus TCP client instance
    try:
        client = AsyncModbusTcpClientGateway.async_get_client_connection(
            host=entry.data[CONF_HOST],
            port=entry.data[CONF_PORT]
        )
        if not client:
            raise ConfigEntryNotReady("Failed to create Modbus client")
    except Exception as e:
        _LOGGER.exception("Error creating Modbus client: %s", e)
        raise ConfigEntryNotReady(f"Modbus client setup failed: {e}") from e

    # STEP 3: Create and store ModbusCoordinator, attach device info
    gateway_key = get_gateway_key(entry)
    try:
        coordinator = ModbusCoordinator(
            hass=hass,
            client=client,
            gateway=gateway_key,
            max_read_size=getattr(device_info, "max_read_size", 64),
        )
        coordinator.device_info = device_info
        hass.data[DOMAIN][gateway_key] = coordinator
    except Exception as e:
        _LOGGER.exception("Error creating ModbusCoordinator: %s", e)
        await client.close()
        raise ConfigEntryNotReady(f"Coordinator setup failed: {e}") from e

    # STEP 4: Forward setup to all platforms (sensor, switch, etc.)
    try:
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    except Exception as e:
        _LOGGER.exception("Error setting up platforms: %s", e)
        await async_unload_entry(hass, entry)
        raise ConfigEntryNotReady(f"Platform setup failed: {e}") from e

    # STEP 4.5: Set up services for polling optimization
    try:
        await async_setup_services(hass)
    except Exception as e:
        _LOGGER.exception("Error setting up services: %s", e)
        # Non-critical error, continue setup

    # STEP 4.6: Trigger coordinator's first refresh
    try:
        await coordinator.async_config_entry_first_refresh()
    except Exception as e:
        _LOGGER.exception("Initial coordinator refresh failed: %s", e)
        await async_unload_entry(hass, entry)
        raise ConfigEntryNotReady(f"Initial refresh failed: {e}") from e

    # STEP 5: Mark coordinator as initialized and fire ready event
    coordinator._initialized = True
    try:
        hass.bus.async_fire(
            f"{DOMAIN}_coordinator_ready",
            {"gateway_key": gateway_key}
        )
        _LOGGER.info("Successfully initialized coordinator for %s", gateway_key)
    except Exception as e:
        _LOGGER.error("Error firing coordinator_ready event: %s", e)
        # Non-critical error, continue setup
        
    # Show polling optimization information message
    _LOGGER.info(
        "🔍 Modbus Local Gateway now includes polling optimization statistics! "
        "After 24 hours of operation, check entity attributes for 'polling_efficiency' "
        "and 'recommended_scan_interval' to optimize your configuration."
    )
    
    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    gateway_key = get_gateway_key(entry)
    if gateway_key not in hass.data.get(DOMAIN, {}):
        return True

    # Unload platforms first
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    
    # Then clean up coordinator
    coordinator = hass.data[DOMAIN].pop(gateway_key, None)
    if coordinator:
        if hasattr(coordinator, "close"):
            try:
                await coordinator.close()
            except Exception as e:
                _LOGGER.error("Error closing coordinator: %s", e)
                unload_ok = False

    return unload_ok