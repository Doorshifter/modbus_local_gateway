"""
Modbus Local Gateway Integration - Lightweight initialization module.

This module is designed to minimize blocking operations during import.
Heavy initialization is deferred until setup.
"""

import logging
import importlib

# Define constants at module level for quick access
DOMAIN = "modbus_local_gateway"
PLATFORMS = ["sensor", "switch", "binary_sensor", "number", "select"]

_LOGGER = logging.getLogger(__name__)

async def _async_import_platforms(hass, platforms):
    """Import platform modules in executor to avoid blocking."""
    for platform in platforms:
        module_path = f"custom_components.{DOMAIN}.{platform}"
        try:
            await hass.async_add_executor_job(importlib.import_module, module_path)
            _LOGGER.debug(f"Pre-loaded platform module: {platform}")
        except Exception as e:
            _LOGGER.error(f"Error pre-loading platform {platform}: {e}")

async def async_setup_entry(hass, entry):
    """Set up from a config entry."""
    # Import all modules inside the function
    from pathlib import Path
    from homeassistant.const import CONF_HOST, CONF_PORT, CONF_FILENAME
    from homeassistant.exceptions import ConfigEntryNotReady
    
    from .const import CONF_SLAVE_ID, DEFAULT_SCAN_INTERVAL, CONF_SCAN_INTERVAL
    
    # Helper function defined inside to avoid top-level imports
    def get_gateway_key(entry_data, with_slave=True):
        """Generate consistent gateway key for coordinator lookup."""
        try:
            if with_slave:
                return f"{entry_data[CONF_HOST]}:{entry_data[CONF_PORT]}:{entry_data[CONF_SLAVE_ID]}"
            return f"{entry_data[CONF_HOST]}:{entry_data[CONF_PORT]}"
        except KeyError as e:
            _LOGGER.error(f"Missing required config entry data: {e}")
            raise ConfigEntryNotReady(f"Missing configuration: {e}") from e

    # Initialize data structure
    hass.data.setdefault(DOMAIN, {})
    
    # Get gateway key early to minimize duplicated code
    try:
        gateway_key = get_gateway_key(entry.data)
    except Exception as e:
        _LOGGER.exception(f"Error generating gateway key: {e}")
        raise ConfigEntryNotReady(f"Invalid configuration: {e}") from e

    # STEP 1: Initialize storage first
    try:
        from .statistics.persistent_statistics import PERSISTENT_STATISTICS_MANAGER
        
        config_dir = hass.config.path()
        storage_path = Path(config_dir) / "modbus_state"
        
        # Initialize storage in executor to avoid blocking
        await hass.async_add_executor_job(PERSISTENT_STATISTICS_MANAGER.initialize, storage_path)
        _LOGGER.info(f"Initialized statistics storage at {storage_path}")
        
        # Initialize other statistics components
        from .statistics.validation import ValidationManager
        validation_manager = ValidationManager.get_instance()
        validation_manager.set_hass(hass)
        
        # Initialize parameter manager
        from .statistics.adaptive_parameters import PARAMETER_MANAGER
        await hass.async_add_executor_job(PARAMETER_MANAGER.initialize_storage, storage_path)
        
        # Initialize self-healing system
        from .statistics.self_healing import SELF_HEALING_SYSTEM
        await hass.async_add_executor_job(SELF_HEALING_SYSTEM.initialize, storage_path)
        
        # Initialize statistics manager
        from .statistics.manager import STATISTICS_MANAGER
        STATISTICS_MANAGER.hass = hass
        await hass.async_add_executor_job(STATISTICS_MANAGER.ensure_initialized)
        
    except Exception as e:
        _LOGGER.warning(f"Failed to initialize statistics system: {e}")
        # Non-critical error, continue setup

    # STEP 2: Load device info
    from .entity_management.device_loader import create_device_info
    import os
    
    try:
        device_info = await hass.async_add_executor_job(
            create_device_info, hass, entry.data[CONF_FILENAME]
        )
        if not device_info:
            raise ConfigEntryNotReady("Device info YAML loaded but returned empty")
    except FileNotFoundError as e:
        _LOGGER.error(
            f"Failed to load device info YAML from {entry.data[CONF_FILENAME]} (cwd: {os.getcwd()}): {e}"
        )
        raise ConfigEntryNotReady(f"Device YAML not found: {e}") from e
    except Exception as e:
        _LOGGER.exception(f"Unexpected error loading device info YAML: {e}")
        raise ConfigEntryNotReady(f"Error loading device info: {e}") from e

    # STEP 3: Measure system capabilities
    try:
        from .statistics.resource_adaptation import RESOURCE_ADAPTER
        
        # Measure system capabilities using async method
        await RESOURCE_ADAPTER.async_measure_system_capabilities(hass)
        capabilities = await hass.async_add_executor_job(RESOURCE_ADAPTER.get_capabilities)
        throughput = await hass.async_add_executor_job(RESOURCE_ADAPTER.get_throughput_recommendation)
        max_registers_per_second = throughput.get("polls_per_second", None)
        
        _LOGGER.info("System capabilities measured: %s", capabilities)
    except Exception as e:
        _LOGGER.warning("Failed to measure system capabilities: %s", e)
        max_registers_per_second = None
        # Non-critical error, continue setup

    # STEP 4: Create or get the Modbus TCP client instance
    try:
        from .tcp_client import AsyncModbusTcpClientGateway
        
        client = AsyncModbusTcpClientGateway.async_get_client_connection(
            host=entry.data[CONF_HOST],
            port=entry.data[CONF_PORT],
            max_registers_per_second=max_registers_per_second
        )
        if not client:
            raise ConfigEntryNotReady("Failed to create Modbus client")
    except Exception as e:
        _LOGGER.exception("Error creating Modbus client: %s", e)
        raise ConfigEntryNotReady(f"Modbus client setup failed: {e}") from e

    # STEP 5: Create and store ModbusCoordinator, attach device info
    try:
        from .coordinator import ModbusCoordinator
        
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

    # CRITICAL FIX: Pre-import platform modules in executor to prevent blocking
    await _async_import_platforms(hass, PLATFORMS)

    # STEP 6: Forward setup to all platforms (sensor, switch, etc.)
    try:
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    except Exception as e:
        _LOGGER.exception("Error setting up platforms: %s", e)
        await async_unload_entry(hass, entry)
        raise ConfigEntryNotReady(f"Platform setup failed: {e}") from e

    # STEP 7: Set up services for polling optimization
    try:
        from .services import async_setup_services
        await async_setup_services(hass)
    except Exception as e:
        _LOGGER.exception("Error setting up services: %s", e)
        # Non-critical error, continue setup

    # STEP 8: Trigger coordinator's first refresh
    try:
        await coordinator.async_config_entry_first_refresh()
    except Exception as e:
        _LOGGER.exception("Initial coordinator refresh failed: %s", e)
        await async_unload_entry(hass, entry)
        raise ConfigEntryNotReady(f"Initial refresh failed: {e}") from e

    # STEP 9: Mark coordinator as initialized and fire ready event
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
        
    # STEP 10: Initialize statistics collection for this coordinator
    try:
        from .statistics import STATISTICS_MANAGER
        
        # Use async_add_executor_job to prevent blocking
        await hass.async_add_executor_job(
            STATISTICS_MANAGER.register_coordinator, 
            gateway_key, 
            coordinator
        )
        _LOGGER.info("Statistics collection initialized for %s", gateway_key)
    except Exception as e:
        _LOGGER.warning("Failed to initialize statistics collection: %s", e)
        # Non-critical error, continue setup
        
    # Show polling optimization information message
    _LOGGER.info(
        "🔍 Modbus Local Gateway now includes polling optimization statistics! "
        "After 24 hours of operation, check entity attributes for 'polling_efficiency' "
        "and 'recommended_scan_interval' to optimize your configuration."
    )
    
    return True

async def async_unload_entry(hass, entry):
    """Unload a config entry."""
    # Import needed modules only when this function is called
    from homeassistant.const import CONF_HOST, CONF_PORT
    from .const import CONF_SLAVE_ID
    
    # Define helper function inside to avoid top-level import
    def get_gateway_key(entry_data, with_slave=True):
        try:
            if with_slave:
                return f"{entry_data[CONF_HOST]}:{entry_data[CONF_PORT]}:{entry_data[CONF_SLAVE_ID]}"
            return f"{entry_data[CONF_HOST]}:{entry_data[CONF_PORT]}"
        except KeyError:
            return None

    gateway_key = get_gateway_key(entry.data)
    if not gateway_key or gateway_key not in hass.data.get(DOMAIN, {}):
        return True

    # Unload platforms first
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    
    # Clean up coordinator
    coordinator = hass.data[DOMAIN].pop(gateway_key, None)
    if coordinator:
        # Try to unregister from statistics
        try:
            from .statistics.manager import STATISTICS_MANAGER
            await hass.async_add_executor_job(STATISTICS_MANAGER.unregister_coordinator, gateway_key)
        except Exception as e:
            _LOGGER.warning(f"Error unregistering from statistics: {e}")
        
        # Shut down coordinator
        try:
            await coordinator.async_shutdown()
        except Exception as e:
            _LOGGER.error(f"Error shutting down coordinator: {e}")
            unload_ok = False

    return unload_ok

async def async_setup(hass, config):
    """Set up the Modbus Local Gateway component."""
    # This function is kept as lightweight as possible
    return True