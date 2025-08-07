"""Diagnostic sensors for Modbus Local Gateway."""

from __future__ import annotations

import logging
from typing import Any, Dict
from dataclasses import dataclass
import json

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
from .statistics.persistent_statistics import PERSISTENT_STATISTICS_MANAGER
from .statistics.health_dashboard import HEALTH_DASHBOARD
from .statistics.advanced_interval_manager import ADVANCED_INTERVAL_MANAGER
from .statistics.interval_visualization import IntervalVisualizationTool

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
        
        # New diagnostic functions
        if func_name == "get_current_pattern":
            try:
                # Try to get pattern from statistics manager
                from .statistics import STATISTICS_MANAGER
                pattern_info = STATISTICS_MANAGER._pattern_detector.get_current_pattern_info()
                pattern_id = pattern_info.get("pattern_id")
                if pattern_id is not None:
                    return f"Pattern {pattern_id}"
                return "No active pattern"
            except Exception:
                return "Unknown"
                
        if func_name == "get_prediction_accuracy":
            try:
                # Try to get prediction accuracy metrics
                from .statistics.prediction_evaluation import PredictionEvaluator
                evaluator = PredictionEvaluator.get_instance()
                metrics = evaluator.calculate_overall_metrics()
                return metrics.get("average_accuracy", 0)
            except Exception:
                return 0
                
        if func_name == "get_optimization_level":
            try:
                # Get optimization level from statistics
                from .statistics import STATISTICS_MANAGER
                system_stats = STATISTICS_MANAGER.get_system_statistics()
                return system_stats.get("optimization_level", 0)
            except Exception:
                return 0
                
        if func_name == "get_analysis_tasks":
            try:
                # Get next analysis task
                from .statistics.analysis_scheduler import AnalysisScheduler
                scheduler = AnalysisScheduler.get_instance()
                next_task = scheduler.get_next_task()
                return next_task or "No scheduled tasks"
            except Exception:
                return "Unknown"
                
        if func_name == "get_storage_status":
            try:
                # Get storage status
                storage_info = PERSISTENT_STATISTICS_MANAGER.get_storage_info()
                return f"{len(storage_info.get('files', {}))} files"
            except Exception:
                return "Unknown"
                
        if func_name == "get_system_health":
            try:
                # Get health dashboard status
                dashboard = HEALTH_DASHBOARD
                data = dashboard.get_dashboard_data()
                return f"{data.get('status_emoji', '❓')} {data.get('status', 'unknown')}"
            except Exception:
                return "Unknown"
                
        # === Advanced Interval System functions ===
        if func_name == "get_interval_system_load":
            try:
                # Get system load from advanced interval manager
                stats = ADVANCED_INTERVAL_MANAGER.get_statistics()
                return stats.get("current_load", 0)
            except Exception:
                return 0
                
        if func_name == "get_polling_efficiency":
            try:
                # Get polling efficiency from advanced interval manager
                stats = ADVANCED_INTERVAL_MANAGER.get_statistics()
                return stats.get("efficiency", 0)
            except Exception:
                return 0
                
        if func_name == "get_active_clusters":
            try:
                # Get active clusters from advanced interval manager
                stats = ADVANCED_INTERVAL_MANAGER.get_statistics()
                return stats.get("active_clusters", 0)
            except Exception:
                return 0
                
        if func_name == "get_average_scan_interval":
            try:
                # Get average interval from advanced interval manager
                stats = ADVANCED_INTERVAL_MANAGER.get_statistics()
                interval_dist = stats.get("interval_distribution", {})
                return interval_dist.get("avg", 0)
            except Exception:
                return 0
        
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
                
        # Batch manager details as attributes for active batch managers
        if self.entity_description.key == "diagnostic_batches":
            if hasattr(self.coordinator, "get_batch_manager_types"):
                attrs["manager_types"] = self.coordinator.get_batch_manager_types()
            if hasattr(self.coordinator, "get_batch_manager_debug_details"):
                attrs["batch_details"] = self.coordinator.get_batch_manager_debug_details()
                
        # Add attributes for new diagnostic sensors
        if self.entity_description.key == "diagnostic_pattern":
            try:
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
            except Exception:
                pass
                
        if self.entity_description.key == "diagnostic_prediction_accuracy":
            try:
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
            except Exception:
                pass
                
        if self.entity_description.key == "diagnostic_optimization_level":
            try:
                from .statistics import STATISTICS_MANAGER
                system_stats = STATISTICS_MANAGER.get_system_statistics()
                attrs.update({
                    "total_entities": system_stats.get("total_entities", 0),
                    "optimized_entities": system_stats.get("optimized_entities", 0),
                    "savings_percent": system_stats.get("polling_savings_percent", 0),
                    "polling_efficiency": system_stats.get("polling_efficiency", 0),
                })
            except Exception:
                pass
                
        if self.entity_description.key == "diagnostic_analysis_tasks":
            try:
                from .statistics.analysis_scheduler import AnalysisScheduler
                scheduler = AnalysisScheduler.get_instance()
                attrs.update(scheduler.get_task_stats())
            except Exception:
                pass
                
        if self.entity_description.key == "diagnostic_storage_status":
            try:
                storage_info = PERSISTENT_STATISTICS_MANAGER.get_storage_info()
                attrs.update({
                    "storage_location": storage_info.get("storage_location"),
                    "version": storage_info.get("version"),
                    "created": storage_info.get("created"),
                    "last_update": storage_info.get("last_update"),
                    "stats_count": storage_info.get("stats_count"),
                    "patterns_count": storage_info.get("patterns_count"),
                    "clusters_count": storage_info.get("clusters_count"),
                    "files": storage_info.get("files"),
                })
            except Exception:
                pass
                
        # Health Dashboard attributes
        if self.entity_description.key == "diagnostic_system_health":
            try:
                dashboard = HEALTH_DASHBOARD
                sensor_attrs = dashboard.get_sensor_attributes()
                attrs.update(sensor_attrs)
            except Exception as e:
                attrs["error"] = str(e)
                
        # === Advanced Interval System attributes ===
        if self.entity_description.key == "diagnostic_interval_system_load":
            try:
                # Use visualization tool to get rich data
                overview = IntervalVisualizationTool.generate_system_overview()
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
                
        if self.entity_description.key == "diagnostic_polling_efficiency":
            try:
                stats = ADVANCED_INTERVAL_MANAGER.get_statistics()
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
                
        if self.entity_description.key == "diagnostic_active_clusters":
            try:
                # Get active clusters info
                stats = ADVANCED_INTERVAL_MANAGER.get_statistics()
                
                # Get list of clusters
                active_clusters = []
                cluster_ids = set()
                for entity_data in ADVANCED_INTERVAL_MANAGER._entities.values():
                    if entity_data.cluster_id:
                        cluster_ids.add(entity_data.cluster_id)
                
                # Get simplified cluster data
                for cluster_id in list(cluster_ids)[:5]:  # Limit to 5 clusters to avoid huge attributes
                    try:
                        cluster_viz = IntervalVisualizationTool.generate_cluster_visualization(cluster_id)
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
                
        if self.entity_description.key == "diagnostic_avg_interval":
            try:
                # Get interval distribution data
                stats = ADVANCED_INTERVAL_MANAGER.get_statistics()
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
                
        return attrs