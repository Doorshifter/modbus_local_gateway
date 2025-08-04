from __future__ import annotations

"""
Helper functions for Modbus Local Gateway integration.

Provides utility functions and robust entity setup logic. Entities always use
device info from the coordinator (which is attached in __init__.py).
"""

import logging
from typing import Any, Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.exceptions import ConfigEntryNotReady

from .const import (
    CONF_PREFIX,
    CONF_SCAN_INTERVAL,
    CONF_SLAVE_ID,
    DEFAULT_SCAN_INTERVAL,
    DOMAIN,
)
from .context import ModbusContext
from .coordinator import ModbusCoordinator
from .entity_management.const import ControlType

_LOGGER: logging.Logger = logging.getLogger(__name__)

def get_gateway_key(entry: ConfigEntry, with_slave: bool = True) -> str:
    """Generate gateway key for coordinator lookup with validation."""
    try:
        host = entry.data["host"]  # Required field
        port = entry.data.get("port", 502)
        if with_slave:
            slave_id = entry.data[CONF_SLAVE_ID]  # Required field
            return f"{host}:{port}:{slave_id}"
        return f"{host}:{port}"
    except KeyError as e:
        _LOGGER.error("Missing required configuration key: %s", e)
        raise ConfigEntryNotReady(f"Missing configuration: {e}") from e

async def async_setup_entities(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
    control: ControlType,
    entity_class: type,
) -> None:
    """
    Set up Modbus Local Gateway entities for a given control type.
    
    Args:
        hass: HomeAssistant instance
        config_entry: Configuration entry
        async_add_entities: Callback to add entities
        control: Control type (sensor, switch, etc.)
        entity_class: Entity class to instantiate
    """
    try:
        gateway_key = get_gateway_key(config_entry, with_slave=True)
        coordinator: Optional[ModbusCoordinator] = hass.data.get(DOMAIN, {}).get(gateway_key)
        
        if not coordinator:
            _LOGGER.error("Coordinator not found for key: %s", gateway_key)
            raise ConfigEntryNotReady(f"Coordinator not ready for {gateway_key}")

        # Validate coordinator and device info
        device_info = getattr(coordinator, "device_info", None)
        if not device_info:
            _LOGGER.error("Device info not set on coordinator")
            raise ConfigEntryNotReady("Device info missing from coordinator")

        # Build device info dictionary with validation
        try:
            device = {
                "identifiers": {(DOMAIN, f"{device_info.model}_{config_entry.data[CONF_SLAVE_ID]}")},
                "name": " ".join(
                    part for part in [
                        config_entry.data.get(CONF_PREFIX),
                        device_info.manufacturer,
                        device_info.model
                    ] if part
                ).strip(),
                "manufacturer": device_info.manufacturer or "Unknown",
                "model": device_info.model or "Unknown",
            }
        except AttributeError as e:
            _LOGGER.error("Invalid device info structure: %s", e)
            raise ConfigEntryNotReady("Invalid device info format") from e

        # Get scan interval with fallbacks
        default_scan_interval = (
            config_entry.data.get(CONF_SCAN_INTERVAL) or
            config_entry.options.get(CONF_SCAN_INTERVAL) if config_entry.options else None or
            DEFAULT_SCAN_INTERVAL
        )

        # Load and validate entity descriptions
        entity_descriptions = []
        if hasattr(device_info, 'entity_descriptions'):
            descs = device_info.entity_descriptions
            if isinstance(descs, (list, tuple)):
                entity_descriptions = list(descs)
            else:
                _LOGGER.warning(
                    "Expected list/tuple for entity_descriptions, got %s. Using empty list.",
                    type(descs).__name__
                )
        else:
            _LOGGER.error(
                "device_info has no entity_descriptions attribute. Check device YAML config."
            )

        if not entity_descriptions:
            _LOGGER.warning(
                "No entity descriptions found for device %s. Skipping setup.",
                getattr(device_info, 'model', 'unknown')
            )
            return

        # Filter descriptions by control type
        filtered_descriptions = [
            desc for desc in entity_descriptions
            if getattr(desc, "control_type", None) == control
        ]

        if not filtered_descriptions:
            _LOGGER.debug(
                "No %s entities found for device %s",
                control.value,
                getattr(device_info, 'model', 'unknown')
            )
            return

        # Create entities
        entities = []
        for desc in filtered_descriptions:
            try:
                context_scan_interval = getattr(desc, "scan_interval", None) or default_scan_interval
                context = ModbusContext(
                    slave_id=config_entry.data[CONF_SLAVE_ID],
                    desc=desc,
                    scan_interval=context_scan_interval,
                )
                
                coordinator.add_entity(context)
                entity = entity_class(
                    coordinator=coordinator,
                    modbus_context=context,
                    device=device,
                )
                entities.append(entity)
            except Exception as err:
                _LOGGER.error(
                    "Error creating entity for %s: %s",
                    getattr(desc, "key", "unknown"),
                    err,
                    exc_info=True
                )

        if entities:
            _LOGGER.info(
                "Adding %d %s entities for gateway %s",
                len(entities),
                control.value,
                gateway_key
            )
            async_add_entities(entities, update_before_add=False)
            
    except Exception as e:
        _LOGGER.exception("Failed to setup entities for control type %s: %s", control, e)
        raise ConfigEntryNotReady(f"Entity setup failed: {e}") from e