"""Centralized statistics management system.

This module acts as the orchestrator for all Modbus statistics, optimization, 
and adaptive parameter management. It integrates entity tracking, pattern detection,
correlation analysis, and scan interval optimization for the Modbus integration,
and exposes a unified API for both internal and Home Assistant-facing consumers.

Logging follows best practices:
- INFO: User-visible operational milestones and successful operations.
- WARNING: Suboptimal, unexpected, but non-fatal system states.
- ERROR: Failures that break core functionality or prevent operation.
- DEBUG: Detailed system internals and diagnostics for development.

Sensitive data (such as entity values) is never logged.
All log messages are formatted as:
    _LOGGER.level("ModbusStatisticsManager.function_name: ...", ...)
"""

import time
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta

from .statistics_tracker import StatisticsTracker, StatisticMetric
from .correlation import EntityCorrelationManager
from .pattern_detection import PatternDetector
from .interval_manager import INTERVAL_MANAGER
from .persistent_statistics import PERSISTENT_STATISTICS_MANAGER
from .adaptive_parameters import PARAMETER_MANAGER, ContextType
from .advanced_interval_manager import ADVANCED_INTERVAL_MANAGER

_LOGGER = logging.getLogger(__name__)

class ModbusStatisticsManager:
    """
    Centralized manager for Modbus statistics, patterns, and correlation analysis.

    Responsibilities:
    - Registers/unregisters entity trackers and gateway coordinators.
    - Orchestrates periodic and real-time statistics updates and analysis.
    - Integrates and applies adaptive parameter profiles.
    - Provides a unified statistics API for all subsystems and for Home Assistant.
    - Handles error conditions and logs system state according to best practices.
    """

    _instance = None

    @classmethod
    def get_instance(cls, hass=None):
        """
        Singleton accessor for the ModbusStatisticsManager.

        Parameters:
            hass (optional): Home Assistant instance for integration.

        Returns:
            ModbusStatisticsManager: Singleton instance.
        """
        if cls._instance is None:
            cls._instance = ModbusStatisticsManager(hass)
            _LOGGER.debug("ModbusStatisticsManager.get_instance: Created singleton instance")
        return cls._instance

    def __init__(self, hass=None):
        """
        Initialize statistics manager and all required subsystem references.

        Parameters:
            hass (optional): Home Assistant instance for integration.
        """
        self._hass = hass
        self.hass = hass  # Compatibility for __init__.py
        self._entity_trackers: Dict[str, StatisticsTracker] = {}
        self._correlation_manager = PERSISTENT_STATISTICS_MANAGER.correlation_manager
        self._pattern_detector = PERSISTENT_STATISTICS_MANAGER.pattern_detector
        self._interval_manager = INTERVAL_MANAGER
        self._storage = PERSISTENT_STATISTICS_MANAGER.storage
        self._advanced_interval_manager = ADVANCED_INTERVAL_MANAGER
        self._last_daily_analysis: float = 0
        self._cached_entity_stats: Dict[str, Dict[str, Any]] = {}
        self._last_stats_update: float = 0
        self.coordinators: Dict[str, Any] = {}
        self._statistics: Dict[str, Dict[str, Any]] = {}
        self._initialized = False

        self._initialize_adaptive_parameters()
        if self._pattern_detector and self._correlation_manager:
            self._initialize_pattern_correlation_integration()

        _LOGGER.info("ModbusStatisticsManager.__init__: Initialized statistics manager instance")

    def ensure_initialized(self) -> bool:
        """
        Ensure all critical subcomponents are initialized.

        Returns:
            bool: True if all required subsystems are initialized.
        """
        if self._initialized:
            return True
        try:
            if not self._pattern_detector:
                self._pattern_detector = PERSISTENT_STATISTICS_MANAGER.pattern_detector
            if not self._correlation_manager:
                self._correlation_manager = PERSISTENT_STATISTICS_MANAGER.correlation_manager
            self._initialized = True
            _LOGGER.info("ModbusStatisticsManager.ensure_initialized: Manager fully initialized")
            return True
        except Exception as e:
            _LOGGER.error("ModbusStatisticsManager.ensure_initialized: Initialization failed: %s", e)
            return False

    def register_coordinator(self, gateway_key: str, coordinator) -> None:
        """
        Register a coordinator for statistics tracking.

        Parameters:
            gateway_key (str): Unique key for the gateway.
            coordinator: Coordinator object to register.
        """
        if not self._initialized and not self.ensure_initialized():
            _LOGGER.warning("ModbusStatisticsManager.register_coordinator: Cannot register coordinator, manager not initialized")
            return
        try:
            self.coordinators[gateway_key] = coordinator
            self._statistics[gateway_key] = {
                "register_reads": 0,
                "register_writes": 0,
                "failed_reads": 0,
                "failed_writes": 0,
                "last_read_time": 0,
                "last_write_time": 0,
                "read_times": [],
                "write_times": [],
                "error_count": 0,
                "entities": set(),
                "first_seen": datetime.utcnow().isoformat(),
                "last_updated": datetime.utcnow().isoformat()
            }
            _LOGGER.info("ModbusStatisticsManager.register_coordinator: Registered coordinator for gateway '%s'", gateway_key)
        except Exception as e:
            _LOGGER.error("ModbusStatisticsManager.register_coordinator: Error registering coordinator: %s", e)

    def unregister_coordinator(self, gateway_key: str) -> None:
        """
        Unregister a previously registered coordinator.

        Parameters:
            gateway_key (str): Unique key for the gateway.
        """
        try:
            if gateway_key in self.coordinators:
                self.coordinators.pop(gateway_key)
                _LOGGER.info("ModbusStatisticsManager.unregister_coordinator: Unregistered coordinator for gateway '%s'", gateway_key)
            if gateway_key in self._statistics:
                self._statistics.pop(gateway_key)
        except Exception as e:
            _LOGGER.error("ModbusStatisticsManager.unregister_coordinator: Error unregistering coordinator: %s", e)

    def register_entity_tracker(self, entity_id: str, tracker: StatisticsTracker, register_count: int = 1) -> None:
        """
        Register an entity tracker to the manager.

        Parameters:
            entity_id (str): Entity identifier.
            tracker (StatisticsTracker): Tracker object.
            register_count (int): Number of registers for the entity.
        """
        self._entity_trackers[entity_id] = tracker
        if hasattr(tracker, "current_scan_interval"):
            self._interval_manager.register_entity(
                entity_id,
                register_count=register_count,
                initial_interval=tracker.current_scan_interval
            )
        _LOGGER.debug("ModbusStatisticsManager.register_entity_tracker: Registered tracker for entity '%s'", entity_id)

    def process_entity_update(self, entity_id: str, value: Any, value_changed: bool, timestamp: float = None) -> None:
        """
        Process an entity value update for all statistics subsystems.

        Parameters:
            entity_id (str): Entity identifier.
            value (Any): New value.
            value_changed (bool): If value has changed since last update.
            timestamp (float): Update timestamp.
        """
        if timestamp is None:
            timestamp = time.time()
        self._correlation_manager.add_entity_value(entity_id, value, timestamp)
        self._pattern_detector.update_entity_value(entity_id, value, timestamp)
        self._interval_manager.record_poll(entity_id, timestamp, value_changed, value)
        self._advanced_interval_manager.record_poll(entity_id, timestamp, value_changed, value)
        self._periodic_analysis(timestamp)
        _LOGGER.debug("ModbusStatisticsManager.process_entity_update: Processed update for entity '%s', value_changed=%s", entity_id, value_changed)

    def _periodic_analysis(self, current_time: float) -> None:
        """
        Run periodic analyses (e.g., daily correlation, parameter adaptation).

        Parameters:
            current_time (float): Current time (timestamp).
        """
        if current_time - self._last_daily_analysis > 86400:
            _LOGGER.info("ModbusStatisticsManager._periodic_analysis: Running scheduled daily statistical analysis")
            self._last_daily_analysis = current_time
            clusters = self._correlation_manager.analyze_correlations()
            if clusters:
                self._interval_manager.update_clusters(clusters)
            self._cached_entity_stats = {}
            self._run_parameter_adaptation()
            if hasattr(PERSISTENT_STATISTICS_MANAGER, "perform_integration_analysis"):
                integration_results = PERSISTENT_STATISTICS_MANAGER.perform_integration_analysis(current_time)
                _LOGGER.info(
                    "ModbusStatisticsManager._periodic_analysis: Pattern-correlation integration completed: %d relationships",
                    integration_results.get("total_relationships", 0)
                )

    def _run_parameter_adaptation(self) -> None:
        """
        Run adaptive parameter adaptation routine and apply new parameters.
        """
        try:
            metrics = self._get_performance_metrics()
            result = PARAMETER_MANAGER.adapt_parameters()
            if result.get("adapted", False):
                _LOGGER.info("ModbusStatisticsManager._run_parameter_adaptation: Activated profile '%s' (score: %.2f)",
                             result.get("new_profile"), result.get("match_score", 0))
                self._apply_current_parameters()
            time_period = {
                "start": datetime.utcnow() - timedelta(days=1),
                "end": datetime.utcnow()
            }
            learn_result = PARAMETER_MANAGER.learn_from_performance(metrics, time_period)
            if "learned_profile" in learn_result:
                _LOGGER.info("ModbusStatisticsManager._run_parameter_adaptation: Created new learned profile '%s'",
                             learn_result["learned_profile"])
        except Exception as e:
            _LOGGER.error("ModbusStatisticsManager._run_parameter_adaptation: Error during parameter adaptation: %s", e)

    def _get_performance_metrics(self) -> Dict[str, float]:
        """
        Gather current system performance metrics for parameter learning.

        Returns:
            Dict[str, float]: Dictionary of key system metrics.
        """
        metrics = {}
        system_stats = self.get_system_statistics()
        metrics["error_rate"] = system_stats.get("error_rate", 0)
        metrics["polling_efficiency"] = system_stats.get("polling_efficiency", 100)
        if "avg_response_time_ms" in system_stats:
            metrics["response_time_ms"] = system_stats["avg_response_time_ms"]
        if "successful_polls" in system_stats:
            metrics["successful_polls"] = system_stats["successful_polls"]
        if "total_polls" in system_stats:
            metrics["total_polls"] = system_stats["total_polls"]
        _LOGGER.debug("ModbusStatisticsManager._get_performance_metrics: Gathered metrics: %s", metrics)
        return metrics

    def get_entity_statistics(self, entity_id: str) -> Dict[str, Any]:
        """
        Retrieve full statistics for a specific entity.

        Parameters:
            entity_id (str): Entity identifier.

        Returns:
            Dict[str, Any]: Statistics for the entity.
        """
        if entity_id in self._entity_trackers:
            stats = self._entity_trackers[entity_id].get_statistics()
            cluster = self._correlation_manager.get_cluster_for_entity(entity_id)
            if cluster:
                stats['cluster'] = cluster
            pattern_info = self._pattern_detector.get_current_pattern_info()
            stats.update({
                'current_pattern': pattern_info.get('pattern_id'),
                'pattern_confidence': pattern_info.get('pattern_confidence'),
                'pattern_since': pattern_info.get('active_since'),
                'predicted_pattern_end': pattern_info.get('predicted_end'),
                'next_predicted_pattern': pattern_info.get('predicted_next_pattern'),
                'pattern_criteria': pattern_info.get('defining_criteria'),
            })
            legacy_interval = self._interval_manager.get_recommended_interval(entity_id, time.time())
            advanced_interval = self._advanced_interval_manager.get_recommended_interval(entity_id, time.time())
            stats['legacy_scan_interval'] = legacy_interval
            stats['advanced_scan_interval'] = advanced_interval
            stats['dynamic_scan_interval'] = advanced_interval
            advanced_stats = self._advanced_interval_manager.get_entity_statistics(entity_id)
            if advanced_stats:
                stats['interval_metrics'] = advanced_stats
            if hasattr(PERSISTENT_STATISTICS_MANAGER, "get_integrated_insights"):
                integrated_insights = PERSISTENT_STATISTICS_MANAGER.get_integrated_insights(entity_id)
                if "integrated_insights" in integrated_insights:
                    stats["integrated_insights"] = integrated_insights["integrated_insights"]
            PERSISTENT_STATISTICS_MANAGER.save_entity_stats(entity_id, stats)
            _LOGGER.debug("ModbusStatisticsManager.get_entity_statistics: Retrieved stats for entity '%s'", entity_id)
            return stats
        result = self._storage.get_entity_stats(entity_id) or {}
        _LOGGER.debug("ModbusStatisticsManager.get_entity_statistics: Retrieved stats from storage for entity '%s'", entity_id)
        return result

    def set_scan_interval(self, entity_id: str, interval: int) -> bool:
        """
        Set the scan interval for a specific entity.

        Parameters:
            entity_id (str): Entity identifier.
            interval (int): New scan interval (seconds).

        Returns:
            bool: True if successfully set.
        """
        if entity_id in self._entity_trackers:
            tracker = self._entity_trackers[entity_id]
            if hasattr(tracker, "current_scan_interval"):
                tracker.current_scan_interval = interval
            if hasattr(self, "_interval_manager"):
                self._interval_manager.register_entity(entity_id, initial_interval=interval)
                _LOGGER.info("ModbusStatisticsManager.set_scan_interval: Set scan interval for entity '%s' to %ds", entity_id, interval)
                return True
        _LOGGER.warning("ModbusStatisticsManager.set_scan_interval: Could not set scan interval for entity '%s'", entity_id)
        return False

    def _initialize_pattern_correlation_integration(self) -> None:
        """
        Bi-directionally integrate pattern detection and correlation systems.
        """
        if hasattr(self._pattern_detector, "set_correlation_manager"):
            self._pattern_detector.set_correlation_manager(self._correlation_manager)
            _LOGGER.info("ModbusStatisticsManager._initialize_pattern_correlation_integration: Pattern detector enhanced with correlation awareness")
        if hasattr(PERSISTENT_STATISTICS_MANAGER, "initialize_with_components"):
            PERSISTENT_STATISTICS_MANAGER.initialize_with_components(self._pattern_detector, self._correlation_manager)
            _LOGGER.info("ModbusStatisticsManager._initialize_pattern_correlation_integration: Pattern-correlation integration system initialized")

    def _initialize_adaptive_parameters(self) -> None:
        """
        Integrate adaptive parameters system with needed context providers.
        """
        PARAMETER_MANAGER.register_context_provider(ContextType.ERROR_RATE, self._get_system_error_rate)
        PARAMETER_MANAGER.register_context_provider(ContextType.ENTITY_COUNT, lambda: len(self._entity_trackers))
        PARAMETER_MANAGER.register_context_provider(ContextType.CURRENT_PATTERN, self._get_current_pattern)
        self._apply_current_parameters()
        _LOGGER.info("ModbusStatisticsManager._initialize_adaptive_parameters: Adaptive parameters system initialized and integrated")

    def _get_system_error_rate(self) -> float:
        """Get current system-wide error rate (as float)."""
        stats = self.get_system_statistics()
        return stats.get("error_rate", 0.0)

    def _get_current_pattern(self) -> Optional[int]:
        """Get current detected system pattern, if available."""
        if self._pattern_detector:
            return self._pattern_detector.current_pattern
        return None

    def _apply_current_parameters(self) -> None:
        """
        Apply current adaptive parameters to all managed subsystems.
        """
        params = PARAMETER_MANAGER.get_all_parameters()
        if self._interval_manager:
            if "min_scan_interval" in params:
                self._interval_manager.min_interval = params["min_scan_interval"]
            if "max_scan_interval" in params:
                self._interval_manager.max_interval = params["max_scan_interval"]
            if "registers_per_second_limit" in params:
                self._interval_manager.max_registers_per_second = params["registers_per_second_limit"]
        if self._pattern_detector:
            if "pattern_detection_sensitivity" in params:
                setattr(self._pattern_detector, "_adaptive_sensitivity", params["pattern_detection_sensitivity"])
            if "pattern_stability_threshold" in params:
                setattr(self._pattern_detector, "_adaptive_threshold", params["pattern_stability_threshold"])
        if self._correlation_manager:
            if "correlation_threshold" in params:
                self._correlation_manager.threshold = params["correlation_threshold"]
        _LOGGER.debug("ModbusStatisticsManager._apply_current_parameters: Applied adaptive parameters to system components")

    # ---- Remaining methods: Add logging at appropriate points as per best practice ----

    def get_gateway_statistics(self, gateway_key: str) -> Dict[str, Any]:
        """Get statistics for a gateway."""
        stats = self._statistics.get(gateway_key, {})
        _LOGGER.debug("ModbusStatisticsManager.get_gateway_statistics: Retrieved stats for gateway '%s'", gateway_key)
        return stats

    def get_interval_system_overview(self) -> Dict[str, Any]:
        """Get system-wide interval optimization overview."""
        from .interval_visualization import IntervalVisualizationTool
        overview = IntervalVisualizationTool.generate_system_overview()
        _LOGGER.debug("ModbusStatisticsManager.get_interval_system_overview: Generated system overview")
        return overview

    def get_entity_interval_visualization(self, entity_id: str) -> Dict[str, Any]:
        """Get interval visualization data for a specific entity."""
        from .interval_visualization import IntervalVisualizationTool
        data = IntervalVisualizationTool.generate_entity_visualization(entity_id)
        _LOGGER.debug("ModbusStatisticsManager.get_entity_interval_visualization: Generated visualization for entity '%s'", entity_id)
        return data

    def get_cluster_interval_visualization(self, cluster_id: str) -> Dict[str, Any]:
        """Get interval visualization data for a cluster."""
        from .interval_visualization import IntervalVisualizationTool
        data = IntervalVisualizationTool.generate_cluster_visualization(cluster_id)
        _LOGGER.debug("ModbusStatisticsManager.get_cluster_interval_visualization: Generated visualization for cluster '%s'", cluster_id)
        return data

    def export_interval_visualization_data(self) -> str:
        """Export all interval visualization data as JSON."""
        from .interval_visualization import IntervalVisualizationTool
        data = IntervalVisualizationTool.export_visualization_data()
        _LOGGER.debug("ModbusStatisticsManager.export_interval_visualization_data: Exported interval visualization data")
        return data

    def analyze_correlations(self, force: bool = True) -> Dict[str, List[str]]:
        """Force correlation analysis and update clusters."""
        clusters = self._correlation_manager.analyze_correlations(force=force)
        if clusters:
            self._interval_manager.update_clusters(clusters)
            _LOGGER.info("ModbusStatisticsManager.analyze_correlations: Correlation analysis updated clusters")
        return clusters

    def get_correlation(self, entity_a: str, entity_b: str) -> Optional[float]:
        """Get correlation between two entities."""
        correlation = self._correlation_manager.get_correlation(entity_a, entity_b)
        _LOGGER.debug("ModbusStatisticsManager.get_correlation: Correlation between '%s' and '%s' is %s", entity_a, entity_b, correlation)
        return correlation

    def get_cluster_for_entity(self, entity_id: str) -> Optional[str]:
        """Get the cluster containing an entity."""
        cluster = self._correlation_manager.get_cluster_for_entity(entity_id)
        _LOGGER.debug("ModbusStatisticsManager.get_cluster_for_entity: Entity '%s' is in cluster '%s'", entity_id, cluster)
        return cluster

    def get_clusters(self) -> Dict[str, List[str]]:
        """Get all entity clusters."""
        clusters = self._correlation_manager.get_clusters()
        _LOGGER.debug("ModbusStatisticsManager.get_clusters: Retrieved all clusters")
        return clusters

    def get_pattern_information(self) -> Dict[str, Any]:
        """Get current pattern detection information."""
        pattern_info = self._pattern_detector.get_current_pattern_info()
        pattern_stats = self._pattern_detector.get_statistics()
        pattern_info.update(pattern_stats)
        _LOGGER.debug("ModbusStatisticsManager.get_pattern_information: Retrieved pattern information")
        return pattern_info

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system-wide statistics about polling optimization."""
        stats = self._interval_manager.get_statistics()
        _LOGGER.debug("ModbusStatisticsManager.get_system_statistics: Retrieved system statistics")
        return stats

    def optimize_all_intervals(self) -> Dict[str, int]:
        """Force optimization of all scan intervals."""
        current_time = time.time()
        optimized_intervals = {}
        for entity_id in self._entity_trackers:
            interval = self._interval_manager.get_recommended_interval(
                entity_id, current_time, force_recalculate=True
            )
            optimized_intervals[entity_id] = interval
        _LOGGER.info("ModbusStatisticsManager.optimize_all_intervals: Optimized scan intervals for all entities")
        return optimized_intervals

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the storage system."""
        info = PERSISTENT_STATISTICS_MANAGER.get_storage_info()
        _LOGGER.debug("ModbusStatisticsManager.get_storage_info: Fetched storage system info")
        return info

    def export_data(self) -> Dict[str, Any]:
        """Export all statistics data."""
        data = PERSISTENT_STATISTICS_MANAGER.export_data()
        _LOGGER.info("ModbusStatisticsManager.export_data: Exported all statistics data")
        return data

    # --- Adaptive parameter and integration methods below ---
    # (Add logging at INFO or ERROR as appropriate for public API/side effects.)

    def run_integration_analysis(self) -> Dict[str, Any]:
        """Run integration analysis between pattern and correlation systems."""
        if hasattr(PERSISTENT_STATISTICS_MANAGER, "perform_integration_analysis"):
            result = PERSISTENT_STATISTICS_MANAGER.perform_integration_analysis()
            _LOGGER.info("ModbusStatisticsManager.run_integration_analysis: Ran integration analysis")
            return result
        _LOGGER.warning("ModbusStatisticsManager.run_integration_analysis: Integration analysis not available")
        return {"error": "Integration analysis not available"}

    def get_integrated_insights(self, entity_id: str) -> Dict[str, Any]:
        """Get integrated insights for an entity."""
        if hasattr(PERSISTENT_STATISTICS_MANAGER, "get_integrated_insights"):
            result = PERSISTENT_STATISTICS_MANAGER.get_integrated_insights(entity_id)
            _LOGGER.debug("ModbusStatisticsManager.get_integrated_insights: Fetched integrated insights for entity '%s'", entity_id)
            return result
        _LOGGER.warning("ModbusStatisticsManager.get_integrated_insights: Integrated insights not available for entity '%s'", entity_id)
        return {"error": "Integrated insights not available"}

    def get_pattern_correlation_insights(self, pattern_id: Optional[int] = None) -> Dict[str, Any]:
        """Get correlation insights for patterns."""
        if hasattr(PERSISTENT_STATISTICS_MANAGER, "get_pattern_correlation_insights"):
            result = PERSISTENT_STATISTICS_MANAGER.get_pattern_correlation_insights(pattern_id)
            _LOGGER.debug("ModbusStatisticsManager.get_pattern_correlation_insights: Retrieved pattern correlation insights")
            return result
        _LOGGER.warning("ModbusStatisticsManager.get_pattern_correlation_insights: Pattern correlation insights not available")
        return {"error": "Pattern correlation insights not available"}

    # --- Parameter profile management API ---

    def get_parameter_profiles(self, profile_name: Optional[str] = None) -> Dict[str, Any]:
        """Get parameter profiles (all or by name)."""
        if profile_name:
            profile = PARAMETER_MANAGER.get_profile(profile_name) or {"error": "Profile not found"}
            _LOGGER.debug("ModbusStatisticsManager.get_parameter_profiles: Fetched profile '%s'", profile_name)
            return profile
        all_profiles = PARAMETER_MANAGER.get_all_profiles()
        _LOGGER.debug("ModbusStatisticsManager.get_parameter_profiles: Fetched all profiles")
        return all_profiles

    def create_parameter_profile(self, name: str, description: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new parameter profile."""
        if not name:
            _LOGGER.error("ModbusStatisticsManager.create_parameter_profile: Profile name is required")
            return {"success": False, "error": "Profile name is required"}
        try:
            profile = PARAMETER_MANAGER.create_profile(name, description, "user")
            if parameters:
                for param, value in parameters.items():
                    profile.update_parameter(param, value)
            PARAMETER_MANAGER._save_data()
            _LOGGER.info("ModbusStatisticsManager.create_parameter_profile: Created profile '%s'", name)
            return {"success": True, "profile": profile.to_dict(), "message": f"Created profile '{name}'"}
        except Exception as e:
            _LOGGER.error("ModbusStatisticsManager.create_parameter_profile: Error creating profile: %s", e)
            return {"success": False, "error": str(e)}

    def update_parameter_profile(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a parameter profile."""
        if not name:
            _LOGGER.error("ModbusStatisticsManager.update_parameter_profile: Profile name is required")
            return {"success": False, "error": "Profile name is required"}
        try:
            profile_dict = PARAMETER_MANAGER.get_profile(name)
            if not profile_dict:
                _LOGGER.warning("ModbusStatisticsManager.update_parameter_profile: Profile '%s' not found", name)
                return {"success": False, "error": f"Profile '{name}' not found"}
            profile = PARAMETER_MANAGER.profiles.get(name)
            if "description" in data:
                profile.description = data["description"]
            if "parameters" in data and isinstance(data["parameters"], dict):
                for param, value in data["parameters"].items():
                    profile.update_parameter(param, value)
            if "context_match" in data and isinstance(data["context_match"], dict):
                for context_type_str, match_data in data["context_match"].items():
                    try:
                        context_type = ContextType(context_type_str)
                        profile.update_context_match(context_type, match_data)
                    except ValueError:
                        _LOGGER.warning("ModbusStatisticsManager.update_parameter_profile: Invalid context type: %s", context_type_str)
            PARAMETER_MANAGER._save_data()
            _LOGGER.info("ModbusStatisticsManager.update_parameter_profile: Updated profile '%s'", name)
            return {"success": True, "profile": profile.to_dict(), "message": f"Updated profile '{name}'"}
        except Exception as e:
            _LOGGER.error("ModbusStatisticsManager.update_parameter_profile: Error updating profile: %s", e)
            return {"success": False, "error": str(e)}

    def delete_parameter_profile(self, name: str) -> Dict[str, Any]:
        """Delete a parameter profile."""
        if not name:
            _LOGGER.error("ModbusStatisticsManager.delete_parameter_profile: Profile name is required")
            return {"success": False, "error": "Profile name is required"}
        try:
            if name == PARAMETER_MANAGER.active_profile_name:
                _LOGGER.warning("ModbusStatisticsManager.delete_parameter_profile: Cannot delete active profile '%s'", name)
                return {"success": False, "error": "Cannot delete active profile"}
            if name not in PARAMETER_MANAGER.profiles:
                _LOGGER.warning("ModbusStatisticsManager.delete_parameter_profile: Profile '%s' not found", name)
                return {"success": False, "error": f"Profile '{name}' not found"}
            success = PARAMETER_MANAGER.delete_profile(name)
            if success:
                _LOGGER.info("ModbusStatisticsManager.delete_parameter_profile: Deleted profile '%s'", name)
                return {"success": True, "message": f"Deleted profile '{name}'"}
            else:
                _LOGGER.error("ModbusStatisticsManager.delete_parameter_profile: Failed to delete profile '%s'", name)
                return {"success": False, "error": f"Failed to delete profile '{name}'"}
        except Exception as e:
            _LOGGER.error("ModbusStatisticsManager.delete_parameter_profile: Error deleting profile: %s", e)
            return {"success": False, "error": str(e)}

    def activate_parameter_profile(self, name: str) -> Dict[str, Any]:
        """Activate a parameter profile."""
        if not name:
            _LOGGER.error("ModbusStatisticsManager.activate_parameter_profile: Profile name is required")
            return {"success": False, "error": "Profile name is required"}
        try:
            if name not in PARAMETER_MANAGER.profiles:
                _LOGGER.warning("ModbusStatisticsManager.activate_parameter_profile: Profile '%s' not found", name)
                return {"success": False, "error": f"Profile '{name}' not found"}
            success = PARAMETER_MANAGER.activate_profile(name)
            if success:
                self._apply_current_parameters()
                _LOGGER.info("ModbusStatisticsManager.activate_parameter_profile: Activated profile '%s'", name)
                return {"success": True, "message": f"Activated profile '{name}'", "parameters": PARAMETER_MANAGER.get_all_parameters()}
            else:
                _LOGGER.error("ModbusStatisticsManager.activate_parameter_profile: Failed to activate profile '%s'", name)
                return {"success": False, "error": f"Failed to activate profile '{name}'"}
        except Exception as e:
            _LOGGER.error("ModbusStatisticsManager.activate_parameter_profile: Error activating profile: %s", e)
            return {"success": False, "error": str(e)}

    def adapt_parameters(self, force: bool = False) -> Dict[str, Any]:
        """Adapt parameters to current context."""
        try:
            result = PARAMETER_MANAGER.adapt_parameters(force=force)
            if result.get("adapted", False):
                self._apply_current_parameters()
                _LOGGER.info("ModbusStatisticsManager.adapt_parameters: Parameters adapted to current context")
            return result
        except Exception as e:
            _LOGGER.error("ModbusStatisticsManager.adapt_parameters: Error adapting parameters: %s", e)
            return {"success": False, "error": str(e)}

    def get_parameter_status(self) -> Dict[str, Any]:
        """Get parameter system status."""
        try:
            status = PARAMETER_MANAGER.get_status()
            status["current_parameters"] = PARAMETER_MANAGER.get_all_parameters()
            _LOGGER.debug("ModbusStatisticsManager.get_parameter_status: Retrieved parameter status")
            return status
        except Exception as e:
            _LOGGER.error("ModbusStatisticsManager.get_parameter_status: Error getting parameter status: %s", e)
            return {"error": str(e)}

    def set_parameter(self, name: str, value: Any) -> Dict[str, Any]:
        """Set a specific parameter value."""
        try:
            if not name:
                _LOGGER.error("ModbusStatisticsManager.set_parameter: Parameter name is required")
                return {"success": False, "error": "Parameter name is required"}
            success = PARAMETER_MANAGER.set_parameter(name, value)
            if success:
                self._apply_current_parameters()
                _LOGGER.info("ModbusStatisticsManager.set_parameter: Updated parameter '%s' to %s", name, value)
                return {"success": True, "message": f"Updated parameter '{name}' to {value}",
                        "parameter": name, "value": PARAMETER_MANAGER.get_parameter(name)}
            else:
                _LOGGER.error("ModbusStatisticsManager.set_parameter: Failed to update parameter '%s'", name)
                return {"success": False, "error": f"Failed to update parameter '{name}'"}
        except Exception as e:
            _LOGGER.error("ModbusStatisticsManager.set_parameter: Error setting parameter: %s", e)
            return {"success": False, "error": str(e)}

# Singleton instance for import
STATISTICS_MANAGER = ModbusStatisticsManager.get_instance()