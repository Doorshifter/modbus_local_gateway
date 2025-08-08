"""Diagnostic sensors for Modbus Local Gateway."""

from __future__ import annotations

import logging
import hashlib
import time
import asyncio
from typing import Any, Dict, ClassVar, Optional
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

# Import statistics components lazily to avoid initialization issues
_LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class ModbusDiagnosticSensorEntityDescription(SensorEntityDescription):
    """Extended sensor entity description for diagnostic sensors."""
    diagnostic_func: str = ""

# Diagnostic sensor definitions - unchanged
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
    # New sensors for optimization metrics
    ModbusDiagnosticSensorEntityDescription(
        key="diagnostic_pattern",
        name="Current Pattern",
        icon="mdi:chart-bell-curve",
        entity_category=EntityCategory.DIAGNOSTIC,
        diagnostic_func="get_current_pattern",
    ),
    ModbusDiagnosticSensorEntityDescription(
        key="diagnostic_prediction_accuracy",
        name="Prediction Accuracy",
        icon="mdi:bullseye-arrow",
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=PERCENTAGE,
        entity_category=EntityCategory.DIAGNOSTIC,
        diagnostic_func="get_prediction_accuracy",
    ),
    ModbusDiagnosticSensorEntityDescription(
        key="diagnostic_optimization_level",
        name="Optimization Level",
        icon="mdi:speedometer",
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=PERCENTAGE,
        entity_category=EntityCategory.DIAGNOSTIC,
        diagnostic_func="get_optimization_level",
    ),
    ModbusDiagnosticSensorEntityDescription(
        key="diagnostic_analysis_tasks",
        name="Analysis Tasks",
        icon="mdi:clock-time-four-outline",
        entity_category=EntityCategory.DIAGNOSTIC,
        diagnostic_func="get_analysis_tasks",
    ),
    ModbusDiagnosticSensorEntityDescription(
        key="diagnostic_storage_status",
        name="Storage Status",
        icon="mdi:database-check",
        entity_category=EntityCategory.DIAGNOSTIC,
        diagnostic_func="get_storage_status",
    ),
    # Health Dashboard sensor
    ModbusDiagnosticSensorEntityDescription(
        key="diagnostic_system_health",
        name="System Health",
        icon="mdi:heart-pulse",
        entity_category=EntityCategory.DIAGNOSTIC,
        diagnostic_func="get_system_health",
    ),
    
    # === New Advanced Interval System Sensors ===
    ModbusDiagnosticSensorEntityDescription(
        key="diagnostic_interval_system_load",
        name="Interval System Load",
        icon="mdi:gauge",
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=PERCENTAGE,
        entity_category=EntityCategory.DIAGNOSTIC,
        diagnostic_func="get_interval_system_load",
    ),
    ModbusDiagnosticSensorEntityDescription(
        key="diagnostic_polling_efficiency",
        name="Polling Efficiency",
        icon="mdi:chart-areaspline",
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=PERCENTAGE,
        entity_category=EntityCategory.DIAGNOSTIC,
        diagnostic_func="get_polling_efficiency",
    ),
    ModbusDiagnosticSensorEntityDescription(
        key="diagnostic_active_clusters",
        name="Active Polling Clusters",
        icon="mdi:sitemap",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        diagnostic_func="get_active_clusters",
    ),
    ModbusDiagnosticSensorEntityDescription(
        key="diagnostic_avg_interval",
        name="Average Scan Interval",
        icon="mdi:timer-outline",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        diagnostic_func="get_average_scan_interval",
    ),
]

# Track if statistics modules are properly initialized
_STATISTICS_INITIALIZED = False
_STATISTICS_COMPONENTS = {}  # Cache for loaded components

def _lazy_import_statistics():
    """Import statistics modules lazily to avoid initialization issues."""
    global _STATISTICS_INITIALIZED, _STATISTICS_COMPONENTS
    
    if _STATISTICS_INITIALIZED:
        return True
        
    try:
        # Import components without initializing them
        from .statistics.persistent_statistics import PERSISTENT_STATISTICS_MANAGER
        from .statistics.health_dashboard import HEALTH_DASHBOARD
        from .statistics.advanced_interval_manager import ADVANCED_INTERVAL_MANAGER
        from .statistics.interval_visualization import IntervalVisualizationTool
        
        # Store in cache
        _STATISTICS_COMPONENTS["PERSISTENT_STATISTICS_MANAGER"] = PERSISTENT_STATISTICS_MANAGER
        _STATISTICS_COMPONENTS["HEALTH_DASHBOARD"] = HEALTH_DASHBOARD
        _STATISTICS_COMPONENTS["ADVANCED_INTERVAL_MANAGER"] = ADVANCED_INTERVAL_MANAGER
        _STATISTICS_COMPONENTS["IntervalVisualizationTool"] = IntervalVisualizationTool
        
        _STATISTICS_INITIALIZED = True
        return True
    except ImportError:
        _LOGGER.warning("Statistics modules could not be imported - some diagnostic features will be limited")
        return False
    except Exception as e:
        _LOGGER.error("Error importing statistics modules: %s", e)
        return False

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

def extract_device_specific_id(device_info_obj, config_entry) -> str:
    """Extract a device-specific ID string to differentiate devices.
    
    This creates a simple string based on device model, manufacturer, and config entry.
    """
    # Get basic device properties
    model = getattr(device_info_obj, "model", "unknown")
    manufacturer = getattr(device_info_obj, "manufacturer", 
                          getattr(device_info_obj, "brand", "unknown"))
    
    # Include entry_id to differentiate between entries
    entry_id = getattr(config_entry, "entry_id", "unknown")
    
    # Include filename if available as this uniquely identifies the device
    filename = getattr(device_info_obj, "fname", "")
    
    # Create a simple string
    return f"{model}_{manufacturer}_{filename}_{entry_id}"

class ModbusDiagnosticSensor(CoordinatorEntity[ModbusCoordinator], SensorEntity):
    """Hybrid diagnostic sensor entity for Modbus Local Gateway."""

    # Class variables for tracking created entities
    _created_unique_ids: ClassVar[set] = set()
    _id_counter: ClassVar[dict] = {}
    _initialization_complete: ClassVar[bool] = False
    
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
        self._device_info_obj = device_info
        
        # Caches for expensive operations
        self._storage_info_cache = None
        self._system_health_cache = None
        self._last_storage_update = 0
        self._last_health_update = 0
        self._cache_ttl = 30
        
        # Create simpler attributes first for better performance during init
        self._attr_name = None
        self._attr_unique_id = None
        self._attr_device_info = None
        
        # Prepare device info
        self._device_info = build_device_info(device_info, config_entry)
        manufacturer = self._device_info.get("manufacturer", "Unknown")
        model = self._device_info.get("model", "Modbus Device")
        name = self._device_info.get("name", f"{manufacturer} {model}")
        
        # Generate a unique identifier - use a simple approach
        gateway_key = get_gateway_key(config_entry, with_slave=True)
        device_id = extract_device_specific_id(device_info, config_entry)
        
        # Hash it to create a short, stable identifier
        device_hash = hashlib.md5(device_id.encode()).hexdigest()[:8]
        base_unique_id = f"{gateway_key}_{description.key}_{device_hash}"
        
        # Check for duplicates
        if base_unique_id in self._created_unique_ids:
            counter = self._id_counter.get(base_unique_id, 0) + 1
            self._id_counter[base_unique_id] = counter
            self._attr_unique_id = f"{base_unique_id}_{counter}"
        else:
            self._created_unique_ids.add(base_unique_id)
            self._attr_unique_id = base_unique_id
        
        # Set other entity attributes
        self._attr_name = f"{name} {description.name}"
        self._attr_device_info = self._device_info
        self._attr_entity_category = description.entity_category

    async def async_added_to_hass(self) -> None:
        """Run when entity is added to hass."""
        await super().async_added_to_hass()
        
        _LOGGER.debug("Added diagnostic sensor '%s' with unique_id '%s'", 
                     self._attr_name, self._attr_unique_id)
        
        # Don't pre-load anything during startup - wait for first update

    async def _async_update_storage_info(self) -> None:
        """Update storage info asynchronously."""
        if not self.hass or not _lazy_import_statistics():
            return
            
        stats_mgr = _STATISTICS_COMPONENTS.get("PERSISTENT_STATISTICS_MANAGER")
        if not stats_mgr:
            return
            
        try:
            if hasattr(stats_mgr, "async_get_storage_info"):
                storage_info = await stats_mgr.async_get_storage_info(self.hass)
                self._storage_info_cache = storage_info
                self._last_storage_update = time.time()
                self.async_write_ha_state()
            else:
                # Fallback to executor method
                storage_info = await self.hass.async_add_executor_job(
                    stats_mgr.get_storage_info)
                self._storage_info_cache = storage_info
                self._last_storage_update = time.time()
                self.async_write_ha_state()
        except Exception as e:
            _LOGGER.error("Error updating storage info asynchronously: %s", e)
            
    async def _async_update_system_health(self) -> None:
        """Update system health data asynchronously."""
        if not self.hass or not _lazy_import_statistics():
            return
            
        health_dashboard = _STATISTICS_COMPONENTS.get("HEALTH_DASHBOARD")
        if not health_dashboard:
            return
            
        try:
            # Check if the health dashboard is initialized
            if hasattr(health_dashboard, 'is_initialized'):
                if not health_dashboard.is_initialized():
                    _LOGGER.debug("Health dashboard not initialized yet, deferring health update")
                    return
            
            # Make sure the health dashboard has a hass reference
            if hasattr(health_dashboard, 'set_hass') and health_dashboard.hass is None:
                health_dashboard.set_hass(self.hass)
            
            # Use async method directly instead of async_add_executor_job
            if hasattr(health_dashboard, 'async_get_dashboard_data'):
                health_data = await health_dashboard.async_get_dashboard_data(None, True)
            else:
                # Fall back to executor job if async method not available
                health_data = await self.hass.async_add_executor_job(
                    health_dashboard.get_dashboard_data, None, True)
                
            self._system_health_cache = health_data
            self._last_health_update = time.time()
            self.async_write_ha_state()
        except Exception as e:
            _LOGGER.error("Error updating system health data asynchronously: %s", e)

    @property
    def native_value(self) -> Any:
        """Return the native value of the sensor."""
        func_name = self.entity_description.diagnostic_func
        
        # Connection status is always safe to call - it only accesses properties
        if func_name == "get_connection_status":
            client = getattr(self.coordinator, "client", None)
            if client and getattr(client, "connected", False):
                return "connected"
            return "disconnected"
        
        # Safe methods that read from coordinator directly
        if func_name in ["get_health_score", "get_batch_efficiency", 
                         "get_active_batch_managers", "get_total_register_reads",
                         "get_coordinator_status"]:
            func = getattr(self.coordinator, func_name, None)
            if callable(func):
                return func()
        
        # Methods that need statistics - check initialization first
        if not _lazy_import_statistics():
            return "Statistics Unavailable"
        
        # Get components from cache
        stats_mgr = _STATISTICS_COMPONENTS.get("PERSISTENT_STATISTICS_MANAGER")
        health_dashboard = _STATISTICS_COMPONENTS.get("HEALTH_DASHBOARD")
        interval_mgr = _STATISTICS_COMPONENTS.get("ADVANCED_INTERVAL_MANAGER")
        
        # Handle special cases
        if func_name == "get_storage_status":
            # Skip update during initialization to avoid blocking
            if self.hass and not ModbusDiagnosticSensor._initialization_complete:
                ModbusDiagnosticSensor._initialization_complete = True
                return "Initializing..."
                
            # Schedule an update if needed and stats_mgr is available
            if self.hass and stats_mgr and hasattr(stats_mgr, "async_get_storage_info"):
                current_time = time.time()
                if current_time - self._last_storage_update > self._cache_ttl:
                    # Create task but don't await it
                    self.hass.async_create_task(self._async_update_storage_info())
            
            # Return cached data if available
            if self._storage_info_cache:
                return f"{len(self._storage_info_cache.get('files', {}))} files"
                
            return "Loading..."
                
        if func_name == "get_system_health":
            # Skip update during initialization
            if self.hass and not ModbusDiagnosticSensor._initialization_complete:
                return "Initializing..."
                
            # Schedule an update if needed and health_dashboard is available
            if self.hass and health_dashboard:
                current_time = time.time()
                if current_time - self._last_health_update > self._cache_ttl:
                    # Create task but don't await it
                    self.hass.async_create_task(self._async_update_system_health())
            
            # Return cached data if available
            if self._system_health_cache:
                data = self._system_health_cache
                return f"{data.get('status_emoji', '❓')} {data.get('status', 'unknown')}"
                
            return "Loading..."
        
        # Safe pattern detector functions
        if func_name == "get_current_pattern":
            try:
                from .statistics import STATISTICS_MANAGER
                pattern_info = STATISTICS_MANAGER._pattern_detector.get_current_pattern_info()
                pattern_id = pattern_info.get("pattern_id")
                if pattern_id is not None:
                    return f"Pattern {pattern_id}"
                return "No active pattern"
            except Exception:
                return "Unknown"
                
        # Handle interval system functions
        if func_name == "get_interval_system_load" and interval_mgr:
            try:
                stats = interval_mgr.get_statistics()
                return stats.get("current_load", 0)
            except Exception:
                return 0
                
        if func_name == "get_polling_efficiency" and interval_mgr:
            try:
                stats = interval_mgr.get_statistics()
                return stats.get("efficiency", 0)
            except Exception:
                return 0
                
        if func_name == "get_active_clusters" and interval_mgr:
            try:
                stats = interval_mgr.get_statistics()
                return stats.get("active_clusters", 0)
            except Exception:
                return 0
                
        if func_name == "get_average_scan_interval" and interval_mgr:
            try:
                stats = interval_mgr.get_statistics()
                interval_dist = stats.get("interval_distribution", {})
                return interval_dist.get("avg", 0)
            except Exception:
                return 0
                
        # Handle other functions
        return self._get_specific_diagnostic_value(func_name)
    
    def _get_specific_diagnostic_value(self, func_name: str) -> Any:
        """Get values for specific diagnostic sensors with proper error handling."""
        try:
            if not _STATISTICS_INITIALIZED:
                return "Statistics Initializing"
                
            if func_name == "get_prediction_accuracy":
                from .statistics.prediction_evaluation import PredictionEvaluator
                evaluator = PredictionEvaluator.get_instance()
                metrics = evaluator.calculate_overall_metrics()
                return metrics.get("average_accuracy", 0)
                
            elif func_name == "get_optimization_level":
                from .statistics import STATISTICS_MANAGER
                system_stats = STATISTICS_MANAGER.get_system_statistics()
                return system_stats.get("optimization_level", 0)
                
            elif func_name == "get_analysis_tasks":
                from .statistics.analysis_scheduler import AnalysisScheduler
                scheduler = AnalysisScheduler.get_instance()
                next_task = scheduler.get_next_task()
                return next_task or "No scheduled tasks"
                
        except ImportError:
            return "Module Unavailable"
        except Exception as e:
            _LOGGER.debug("Error in _get_specific_diagnostic_value for %s: %s", func_name, e)
            return "Error"
                
        return None

    @property
    def extra_state_attributes(self) -> Dict[str, Any] | None:
        """Return additional state attributes for diagnostics."""
        # Always include these basic attributes that don't require heavy operations
        attrs = {
            "device_model": self._device_info.get("model"),
            "device_manufacturer": self._device_info.get("manufacturer"),
            "yaml_file": getattr(self._device_info_obj, "fname", None),
            "gateway_key": get_gateway_key(self._config_entry, with_slave=True),
            "last_update_success": getattr(self.coordinator, "last_update_success", False),
            "last_update_timestamp": getattr(self.coordinator, "last_update_timestamp", None),
            "last_update_attempt_timestamp": getattr(self.coordinator, "last_update_attempt_timestamp", None),
            "unique_id": self._attr_unique_id,
        }
        
        # Don't proceed with heavy attributes during initialization
        if not ModbusDiagnosticSensor._initialization_complete:
            attrs["status"] = "initializing"
            return attrs
        
        # Connection info attributes
        if self.entity_description.key == "diagnostic_connection":
            client = getattr(self.coordinator, "client", None)
            if client:
                comm_params = getattr(client, "comm_params", None)
                if comm_params:
                    attrs["host"] = getattr(comm_params, "host", None)
                    attrs["port"] = getattr(comm_params, "port", None)
                attrs["protocol"] = getattr(client, "protocol", "tcp")
                
        # Batch manager details
        if self.entity_description.key == "diagnostic_batches":
            if hasattr(self.coordinator, "get_batch_manager_types"):
                attrs["manager_types"] = self.coordinator.get_batch_manager_types()
            if hasattr(self.coordinator, "get_batch_manager_debug_details"):
                attrs["batch_details"] = self.coordinator.get_batch_manager_debug_details()
        
        # Statistics-dependent attributes - make sure modules are available
        if not _lazy_import_statistics():
            attrs["statistics_available"] = False
            return attrs
            
        attrs["statistics_available"] = True
        
        try:
            # Add sensor-specific attributes
            if self.entity_description.key == "diagnostic_pattern":
                from .statistics import STATISTICS_MANAGER
                pattern_info = STATISTICS_MANAGER._pattern_detector.get_current_pattern_info()
                attrs.update({
                    "pattern_id": pattern_info.get("pattern_id"),
                    "pattern_confidence": pattern_info.get("pattern_confidence"),
                    "active_since": pattern_info.get("active_since"),
                    "predicted_end": pattern_info.get("predicted_end"),
                    "predicted_next_pattern": pattern_info.get("predicted_next_pattern"),
                    "defining_criteria": pattern_info.get("defining_criteria"),
                })
                    
            elif self.entity_description.key == "diagnostic_prediction_accuracy":
                from .statistics.prediction_evaluation import PredictionEvaluator
                evaluator = PredictionEvaluator.get_instance()
                metrics = evaluator.calculate_overall_metrics()
                attrs.update({
                    "average_accuracy": metrics.get("average_accuracy"),
                    "best_performer": metrics.get("best_performer"),
                    "worst_performer": metrics.get("worst_performer"),
                    "total_predictions": metrics.get("total_predictions"),
                    "entities_tracked": metrics.get("entities_tracked"),
                    "last_updated": metrics.get("last_updated"),
                })
                
                # Add improvement suggestions
                suggestions = evaluator.get_prediction_improvement_suggestions()
                if suggestions:
                    attrs["improvement_suggestions"] = suggestions
                    
            elif self.entity_description.key == "diagnostic_optimization_level":
                from .statistics import STATISTICS_MANAGER
                system_stats = STATISTICS_MANAGER.get_system_statistics()
                attrs.update({
                    "total_entities": system_stats.get("total_entities", 0),
                    "optimized_entities": system_stats.get("optimized_entities", 0),
                    "savings_percent": system_stats.get("polling_savings_percent", 0),
                    "polling_efficiency": system_stats.get("polling_efficiency", 0),
                })
                    
            elif self.entity_description.key == "diagnostic_analysis_tasks":
                from .statistics.analysis_scheduler import AnalysisScheduler
                scheduler = AnalysisScheduler.get_instance()
                attrs.update(scheduler.get_task_stats())
                    
            elif self.entity_description.key == "diagnostic_storage_status" and self._storage_info_cache:
                # Use cached storage info if available
                storage_info = self._storage_info_cache
                attrs.update({
                    "storage_location": storage_info.get("storage_location"),
                    "version": storage_info.get("version"),
                    "created": storage_info.get("created"),
                    "last_update": storage_info.get("last_update"),
                    "stats_count": storage_info.get("stats_count"),
                    "patterns_count": storage_info.get("patterns_count"),
                    "clusters_count": storage_info.get("clusters_count"),
                    "files": storage_info.get("files"),
                    "cached_data": bool(self._storage_info_cache),
                    "cache_age_seconds": int(time.time() - self._last_storage_update) if self._last_storage_update else None,
                })
                    
            # Health Dashboard attributes
            elif self.entity_description.key == "diagnostic_system_health" and self._system_health_cache:
                dashboard_data = self._system_health_cache
                
                # Extract sensor attributes from dashboard data
                if hasattr(_STATISTICS_COMPONENTS.get("HEALTH_DASHBOARD"), "get_sensor_attributes"):
                    try:
                        sensor_attrs = _STATISTICS_COMPONENTS["HEALTH_DASHBOARD"].get_sensor_attributes()
                        attrs.update(sensor_attrs)
                    except Exception:
                        # Fall back to basic attributes
                        attrs.update({
                            "status": dashboard_data.get("status"),
                            "status_emoji": dashboard_data.get("status_emoji"),
                            "score": dashboard_data.get("score"),
                            "issues_count": len(dashboard_data.get("issues", [])),
                            "warnings_count": len(dashboard_data.get("warnings", [])),
                            "cached_data": True,
                            "cache_age_seconds": int(time.time() - self._last_health_update) if self._last_health_update else None,
                        })
                else:
                    # Fallback to extracting basic info from dashboard data
                    attrs.update({
                        "status": dashboard_data.get("status"),
                        "status_emoji": dashboard_data.get("status_emoji"),
                        "score": dashboard_data.get("score"),
                        "issues_count": len(dashboard_data.get("issues", [])),
                        "warnings_count": len(dashboard_data.get("warnings", [])),
                        "cached_data": True,
                        "cache_age_seconds": int(time.time() - self._last_health_update) if self._last_health_update else None,
                    })
                    
            # === Advanced Interval System attributes ===
            elif self.entity_description.key == "diagnostic_interval_system_load":
                interval_tool = _STATISTICS_COMPONENTS.get("IntervalVisualizationTool")
                if interval_tool:
                    try:
                        # Use visualization tool to get rich data
                        overview = interval_tool.generate_system_overview()
                        attrs.update({
                            "current_load": overview["system_load"]["current"],
                            "target_load": overview["system_load"]["target"],
                            "load_color": overview["system_load"]["color"],
                            "interval_distribution": overview["interval_distribution"],
                            "efficiency": overview["efficiency"],
                            "total_entities": overview["total_entities"],
                            "total_polls": overview["total_polls"],
                            "total_changes": overview.get("total_changes", 0),
                        })
                    except Exception as e:
                        attrs["error"] = str(e)
                    
            elif self.entity_description.key == "diagnostic_polling_efficiency":
                interval_mgr = _STATISTICS_COMPONENTS.get("ADVANCED_INTERVAL_MANAGER")
                if interval_mgr:
                    try:
                        stats = interval_mgr.get_statistics()
                        attrs.update({
                            "efficiency": stats.get("efficiency", 0),
                            "total_polls": stats.get("total_polls", 0),
                            "total_changes": stats.get("total_changes", 0),
                            "entity_count": stats.get("entity_count", 0),
                            "registers_per_second_limit": stats.get("registers_per_second_limit", 0),
                            "target_utilization": stats.get("target_utilization", 0),
                        })
                    except Exception as e:
                        attrs["error"] = str(e)
                    
            elif self.entity_description.key == "diagnostic_active_clusters":
                interval_mgr = _STATISTICS_COMPONENTS.get("ADVANCED_INTERVAL_MANAGER")
                interval_tool = _STATISTICS_COMPONENTS.get("IntervalVisualizationTool")
                if interval_mgr and interval_tool:
                    try:
                        # Get active clusters info
                        stats = interval_mgr.get_statistics()
                        
                        # Get list of clusters
                        active_clusters = []
                        cluster_ids = set()
                        for entity_data in interval_mgr._entities.values():
                            if entity_data.cluster_id:
                                cluster_ids.add(entity_data.cluster_id)
                        
                        # Get simplified cluster data
                        for cluster_id in list(cluster_ids)[:5]:  # Limit to 5 clusters to avoid huge attributes
                            try:
                                cluster_viz = interval_tool.generate_cluster_visualization(cluster_id)
                                active_clusters.append({
                                    "id": cluster_id,
                                    "entity_count": cluster_viz.get("entity_count", 0),
                                    "avg_interval": cluster_viz.get("avg_interval", 0),
                                    "coordination_level": cluster_viz.get("coordination_level", "unknown"),
                                })
                            except Exception:
                                pass
                                
                        attrs.update({
                            "active_clusters": stats.get("active_clusters", 0),
                            "cluster_details": active_clusters,
                            "pattern_integration": stats.get("pattern_integration", False),
                            "correlation_integration": stats.get("correlation_integration", False),
                        })
                    except Exception as e:
                        attrs["error"] = str(e)
                    
            elif self.entity_description.key == "diagnostic_avg_interval":
                interval_mgr = _STATISTICS_COMPONENTS.get("ADVANCED_INTERVAL_MANAGER")
                if interval_mgr:
                    try:
                        # Get interval distribution data
                        stats = interval_mgr.get_statistics()
                        interval_dist = stats.get("interval_distribution", {})
                        
                        attrs.update({
                            "min_interval": interval_dist.get("min", 0),
                            "max_interval": interval_dist.get("max", 0),
                            "avg_interval": interval_dist.get("avg", 0),
                            "distribution": interval_dist.get("count_by_range", {}),
                        })
                        
                        # Add donut chart data for web view
                        interval_dist = stats.get("interval_distribution", {}).get("count_by_range", {})
                        attrs["interval_chart"] = {
                            "labels": list(interval_dist.keys()),
                            "data": list(interval_dist.values()),
                        }
                    except Exception as e:
                        attrs["error"] = str(e)
        except Exception as e:
            attrs["error"] = str(e)
                
        return attrs