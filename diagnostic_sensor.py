"""Diagnostic sensors for Modbus Local Gateway."""

from __future__ import annotations

import logging
from typing import Any, Dict
from dataclasses import dataclass

from homeassistant.components.sensor import (
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.const import PERCENTAGE, EntityCategory
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import ModbusCoordinator
from .helpers import get_gateway_key

_LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class ModbusDiagnosticSensorEntityDescription(SensorEntityDescription):
    """Extended sensor entity description for diagnostic sensors."""
    diagnostic_func: str = ""

# Diagnostic sensor definitions
DIAGNOSTIC_SENSORS = [
    # Connection status sensor
    ModbusDiagnosticSensorEntityDescription(
        key="diagnostic_connection",
        name="Connection Status",
        icon="mdi:lan-connect",
        entity_category=EntityCategory.DIAGNOSTIC,
        diagnostic_func="get_connection_status",
    ),
    # Health Score
    ModbusDiagnosticSensorEntityDescription(
        key="diagnostic_health",
        name="Health Score",
        icon="mdi:heart-pulse",
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=PERCENTAGE,
        entity_category=EntityCategory.DIAGNOSTIC,
        diagnostic_func="get_health_score",
    ),
    # Batch Efficiency
    ModbusDiagnosticSensorEntityDescription(
        key="diagnostic_eff",
        name="Batch Efficiency",
        icon="mdi:gauge",
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=PERCENTAGE,
        entity_category=EntityCategory.DIAGNOSTIC,
        diagnostic_func="get_batch_efficiency",
    ),
    # Active Batch Managers
    ModbusDiagnosticSensorEntityDescription(
        key="diagnostic_batches",
        name="Active Batch Managers",
        icon="mdi:cog-box",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        diagnostic_func="get_active_batch_managers",
    ),
    # Total Register Reads
    ModbusDiagnosticSensorEntityDescription(
        key="diagnostic_reads",
        name="Total Register Reads",
        icon="mdi:counter",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        diagnostic_func="get_total_register_reads",
    ),
    # Coordinator Status
    ModbusDiagnosticSensorEntityDescription(
        key="diagnostic_status",
        name="Coordinator Status",
        icon="mdi:check-circle-outline",
        entity_category=EntityCategory.DIAGNOSTIC,
        diagnostic_func="get_coordinator_status",
    ),
]

def build_device_info(device_info_obj, config_entry):
    """Build a Home Assistant device_info dict from ModbusDeviceInfo object."""
    VALID_DEVICE_INFO_KEYS = {
        "identifiers",
        "connections",
        "manufacturer",
        "model",
        "name",
        "sw_version",
        "hw_version",
        "serial_number",
        "via_device",
        "suggested_area",
        "entry_type",
        "configuration_url",
    }
    if isinstance(device_info_obj, dict):
        return {k: v for k, v in device_info_obj.items() if k in VALID_DEVICE_INFO_KEYS}
    try:
        manufacturer = getattr(device_info_obj, "manufacturer", None)
        if not manufacturer:
            manufacturer = getattr(device_info_obj, "brand", None)
    except Exception:
        manufacturer = "Unknown"
    try:
        model = device_info_obj.model
    except Exception:
        model = "Unknown"
    gateway_key = get_gateway_key(config_entry, with_slave=True)
    return {
        "identifiers": {(DOMAIN, gateway_key)},
        "manufacturer": manufacturer,
        "model": model,
        "name": f"{manufacturer} {model}",
    }

class ModbusDiagnosticSensor(CoordinatorEntity[ModbusCoordinator], SensorEntity):
    """Hybrid diagnostic sensor entity for Modbus Local Gateway."""

    def __init__(
        self,
        coordinator: ModbusCoordinator,
        description: ModbusDiagnosticSensorEntityDescription,
        config_entry,
        device_info,
    ) -> None:
        """Initialize the diagnostic sensor."""
        super().__init__(coordinator)
        self.entity_description = description
        self._config_entry = config_entry

        # Prepare device info
        self._device_info = build_device_info(device_info, config_entry)
        manufacturer = self._device_info.get("manufacturer", "Unknown")
        model = self._device_info.get("model", "Modbus Device")
        name = self._device_info.get("name", f"{manufacturer} {model}")

        gateway_key = get_gateway_key(config_entry, with_slave=True)
        self._attr_name = f"{name} {description.name}"
        self._attr_unique_id = f"{gateway_key}_{description.key}"
        self._attr_device_info = self._device_info
        self._attr_entity_category = description.entity_category

    @property
    def native_value(self) -> Any:
        func_name = self.entity_description.diagnostic_func
        # Connection status as special case for string value
        if func_name == "get_connection_status":
            client = getattr(self.coordinator, "client", None)
            if client and getattr(client, "connected", False):
                return "connected"
            return "disconnected"
        # Generic: call the coordinator method if exists
        func = getattr(self.coordinator, func_name, None)
        if callable(func):
            return func()
        return None

    @property
    def extra_state_attributes(self) -> Dict[str, Any] | None:
        """Return additional state attributes for diagnostics."""
        attrs = {
            "device_model": self._device_info.get("model"),
            "device_manufacturer": self._device_info.get("manufacturer"),
            "yaml_file": getattr(self.coordinator.device_info, "fname", None),
            "gateway_key": get_gateway_key(self._config_entry, with_slave=True),
            "last_update_success": getattr(self.coordinator, "last_update_success", False),
            "last_update_timestamp": getattr(self.coordinator, "last_update_timestamp", None),
            "last_update_attempt_timestamp": getattr(self.coordinator, "last_update_attempt_timestamp", None),
        }
        
        

        
        
        # Connection info attributes
        if self.entity_description.key == "diagnostic_connection":
            client = getattr(self.coordinator, "client", None)
            if client:
                comm_params = getattr(client, "comm_params", None)
                if comm_params:
                    attrs["host"] = getattr(comm_params, "host", None)
                    attrs["port"] = getattr(comm_params, "port", None)
                attrs["protocol"] = getattr(client, "protocol", "tcp")
            # If available, add slave_id (if single-slave, else via context)
            # attrs["slave_id"] = ... (add if relevant in your context)
        # Batch manager details as attributes for active batch managers
        if self.entity_description.key == "diagnostic_batches":
            if hasattr(self.coordinator, "get_batch_manager_types"):
                attrs["manager_types"] = self.coordinator.get_batch_manager_types()
            if hasattr(self.coordinator, "get_batch_manager_debug_details"):
                attrs["batch_details"] = self.coordinator.get_batch_manager_debug_details()
        return attrs