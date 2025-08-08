"""Centralized statistics management system."""

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
    """Centralized manager for Modbus statistics and correlations."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls, hass=None):
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = ModbusStatisticsManager(hass)
        return cls._instance
    
    def __init__(self, hass=None):
        """Initialize statistics manager."""
        self._hass = hass
        self.hass = hass  # Added for compatibility with __init__.py
        self._entity_trackers: Dict[str, StatisticsTracker] = {}
        
        # Use persistent components from the persistent statistics manager
        self._correlation_manager = PERSISTENT_STATISTICS_MANAGER.correlation_manager
        self._pattern_detector = PERSISTENT_STATISTICS_MANAGER.pattern_detector
        self._interval_manager = INTERVAL_MANAGER
        self._storage = PERSISTENT_STATISTICS_MANAGER.storage

        # Use the advanced interval manager
        self._advanced_interval_manager = ADVANCED_INTERVAL_MANAGER
        
        # Last analysis time
        self._last_daily_analysis: float = 0
        
        # Cache for entity statistics
        self._cached_entity_stats: Dict[str, Dict[str, Any]] = {}
        self._last_stats_update: float = 0
        
        # Added for coordinator tracking
        self.coordinators: Dict[str, Any] = {}
        self._statistics: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
        
        # Set up adaptive parameters integration
        self._initialize_adaptive_parameters()
        
        # Initialize pattern-correlation integration
        if self._pattern_detector and self._correlation_manager:
            self._initialize_pattern_correlation_integration()

    def ensure_initialized(self) -> bool:
        """Ensure the manager is fully initialized."""
        if self._initialized:
            return True
            
        try:
            # Initialize components that might not be ready
            if not self._pattern_detector:
                self._pattern_detector = PERSISTENT_STATISTICS_MANAGER.pattern_detector
                
            if not self._correlation_manager:
                self._correlation_manager = PERSISTENT_STATISTICS_MANAGER.correlation_manager
            
            # Set initialization flag
            self._initialized = True
            _LOGGER.info("Statistics manager fully initialized")
            return True
        except Exception as e:
            _LOGGER.error("Error initializing statistics manager: %s", e)
            return False
            
    def register_coordinator(self, gateway_key: str, coordinator) -> None:
        """Register a coordinator for statistics tracking."""
        if not self._initialized and not self.ensure_initialized():
            _LOGGER.warning("Cannot register coordinator - statistics manager not initialized")
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
            _LOGGER.info(f"Registered coordinator for gateway {gateway_key}")
        except Exception as e:
            _LOGGER.error("Error registering coordinator: %s", e)
            
    def unregister_coordinator(self, gateway_key: str) -> None:
        """Unregister a coordinator."""
        try:
            if gateway_key in self.coordinators:
                self.coordinators.pop(gateway_key)
                _LOGGER.info(f"Unregistered coordinator for gateway {gateway_key}")
                
            if gateway_key in self._statistics:
                self._statistics.pop(gateway_key)
        except Exception as e:
            _LOGGER.error("Error unregistering coordinator: %s", e)
            
    def get_gateway_statistics(self, gateway_key: str) -> Dict[str, Any]:
        """Get statistics for a gateway."""
        if gateway_key in self._statistics:
            return self._statistics[gateway_key]
        return {}

    def get_interval_system_overview(self) -> Dict[str, Any]:
        """Get system-wide interval optimization overview.
        
        Returns:
            Dictionary with visualization data
        """
        from .interval_visualization import IntervalVisualizationTool
        return IntervalVisualizationTool.generate_system_overview()
    
    def get_entity_interval_visualization(self, entity_id: str) -> Dict[str, Any]:
        """Get interval visualization data for a specific entity.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Dictionary with entity visualization data
        """
        from .interval_visualization import IntervalVisualizationTool
        return IntervalVisualizationTool.generate_entity_visualization(entity_id)
    
    def get_cluster_interval_visualization(self, cluster_id: str) -> Dict[str, Any]:
        """Get interval visualization data for a cluster.
        
        Args:
            cluster_id: Cluster identifier
            
        Returns:
            Dictionary with cluster visualization data
        """
        from .interval_visualization import IntervalVisualizationTool
        return IntervalVisualizationTool.generate_cluster_visualization(cluster_id)
    
    def export_interval_visualization_data(self) -> str:
        """Export all interval visualization data as JSON.
        
        Returns:
            JSON string with visualization data
        """
        from .interval_visualization import IntervalVisualizationTool
        return IntervalVisualizationTool.export_visualization_data()
        
    def _initialize_pattern_correlation_integration(self) -> None:
        """Initialize integration between pattern detection and correlation systems."""
        # Set up bi-directional integration
        if hasattr(self._pattern_detector, "set_correlation_manager"):
            self._pattern_detector.set_correlation_manager(self._correlation_manager)
            _LOGGER.info("Pattern detector enhanced with correlation awareness")
        
        # Initialize pattern-correlation system in persistent statistics manager
        if hasattr(PERSISTENT_STATISTICS_MANAGER, "initialize_with_components"):
            PERSISTENT_STATISTICS_MANAGER.initialize_with_components(
                self._pattern_detector, self._correlation_manager)
            _LOGGER.info("Pattern-correlation integration system initialized")
            
    def _initialize_adaptive_parameters(self) -> None:
        """Initialize adaptive parameters system integration."""
        # Register context providers
        PARAMETER_MANAGER.register_context_provider(
            ContextType.ERROR_RATE,
            self._get_system_error_rate
        )
        
        PARAMETER_MANAGER.register_context_provider(
            ContextType.ENTITY_COUNT,
            lambda: len(self._entity_trackers)
        )
        
        PARAMETER_MANAGER.register_context_provider(
            ContextType.CURRENT_PATTERN,
            self._get_current_pattern
        )
        
        # Initial parameter application
        self._apply_current_parameters()
        
        _LOGGER.info("Adaptive parameters system initialized and integrated")
        
    def _get_system_error_rate(self) -> float:
        """Get current system-wide error rate."""
        stats = self.get_system_statistics()
        return stats.get("error_rate", 0.0)
    
    def _get_current_pattern(self) -> Optional[int]:
        """Get current system pattern."""
        if self._pattern_detector:
            return self._pattern_detector.current_pattern
        return None
    
    def _apply_current_parameters(self) -> None:
        """Apply current adaptive parameters to system components."""
        params = PARAMETER_MANAGER.get_all_parameters()
        
        # Apply to interval manager
        if self._interval_manager:
            if "min_scan_interval" in params:
                self._interval_manager.min_interval = params["min_scan_interval"]
            if "max_scan_interval" in params:
                self._interval_manager.max_interval = params["max_scan_interval"]
            if "registers_per_second_limit" in params:
                self._interval_manager.max_registers_per_second = params["registers_per_second_limit"]
        
        # Apply to pattern detector
        if self._pattern_detector:
            if "pattern_detection_sensitivity" in params:
                # Store adaptive sensitivity - actual application depends on the pattern detector implementation
                setattr(self._pattern_detector, "_adaptive_sensitivity", 
                       params["pattern_detection_sensitivity"])
            if "pattern_stability_threshold" in params:
                # Store adaptive threshold - actual application depends on the pattern detector implementation
                setattr(self._pattern_detector, "_adaptive_threshold", 
                       params["pattern_stability_threshold"])
        
        # Apply to correlation manager
        if self._correlation_manager:
            if "correlation_threshold" in params:
                self._correlation_manager.threshold = params["correlation_threshold"]
                
        _LOGGER.debug("Applied adaptive parameters to system components")
        
    def register_entity_tracker(self, entity_id: str, tracker: StatisticsTracker, 
                              register_count: int = 1) -> None:
        """Register an existing tracker with the manager."""
        self._entity_trackers[entity_id] = tracker
        
        # Register with interval manager
        if hasattr(tracker, "current_scan_interval"):
            self._interval_manager.register_entity(
                entity_id,
                register_count=register_count,
                initial_interval=tracker.current_scan_interval
            )
    
    def process_entity_update(self, entity_id: str, value: Any, 
                             value_changed: bool, timestamp: float = None) -> None:
        """Process an entity update for correlation analysis."""
        if timestamp is None:
            timestamp = time.time()
            
        # Add to correlation manager for analysis
        self._correlation_manager.add_entity_value(entity_id, value, timestamp)
        
        # Update pattern detector
        self._pattern_detector.update_entity_value(entity_id, value, timestamp)
        
        # Update both interval managers (legacy and advanced)
        self._interval_manager.record_poll(entity_id, timestamp, value_changed, value)
        self._advanced_interval_manager.record_poll(
            entity_id, timestamp, value_changed, value)
        
        # Run periodic analysis
        self._periodic_analysis(timestamp)
    
    def _periodic_analysis(self, current_time: float) -> None:
        """Run periodic analyses based on different schedules."""
        # Daily correlation analysis
        if current_time - self._last_daily_analysis > 86400:  # Once per day
            _LOGGER.info("Running scheduled daily statistical analysis")
            self._last_daily_analysis = current_time
            
            # Run correlation analysis
            clusters = self._correlation_manager.analyze_correlations()
            
            # Update interval manager with new clusters
            if clusters:
                self._interval_manager.update_clusters(clusters)
            
            # Clear cached statistics
            self._cached_entity_stats = {}
            
            # Run adaptive parameter adaptation
            self._run_parameter_adaptation()
            
            # Run pattern-correlation integration analysis
            if hasattr(PERSISTENT_STATISTICS_MANAGER, "perform_integration_analysis"):
                integration_results = PERSISTENT_STATISTICS_MANAGER.perform_integration_analysis(current_time)
                _LOGGER.info("Pattern-correlation integration analysis completed: %s", 
                           integration_results.get("total_relationships", 0))
    
    def _run_parameter_adaptation(self) -> None:
        """Run adaptive parameter adaptation if needed."""
        try:
            # Get system metrics for learning
            metrics = self._get_performance_metrics()
            
            # Run adaptation
            result = PARAMETER_MANAGER.adapt_parameters()
            
            if result.get("adapted", False):
                _LOGGER.info("Adaptive parameters: Activated profile '%s' (score: %.2f)", 
                           result.get("new_profile"), result.get("match_score", 0))
                
                # Apply new parameters
                self._apply_current_parameters()
            
            # Learn from performance metrics
            time_period = {
                "start": datetime.utcnow() - timedelta(days=1),
                "end": datetime.utcnow()
            }
            
            learn_result = PARAMETER_MANAGER.learn_from_performance(metrics, time_period)
            
            if "learned_profile" in learn_result:
                _LOGGER.info("Adaptive parameters: Created new learned profile '%s'", 
                           learn_result["learned_profile"])
                
        except Exception as e:
            _LOGGER.error("Error during adaptive parameter adaptation: %s", e)
    
    def _get_performance_metrics(self) -> Dict[str, float]:
        """Get current system performance metrics for parameter learning."""
        metrics = {}
        
        # Get system stats
        system_stats = self.get_system_statistics()
        
        # Extract relevant metrics
        metrics["error_rate"] = system_stats.get("error_rate", 0)
        metrics["polling_efficiency"] = system_stats.get("polling_efficiency", 100)
        
        if "avg_response_time_ms" in system_stats:
            metrics["response_time_ms"] = system_stats["avg_response_time_ms"]
            
        if "successful_polls" in system_stats:
            metrics["successful_polls"] = system_stats["successful_polls"]
            
        if "total_polls" in system_stats:
            metrics["total_polls"] = system_stats["total_polls"]
        
        # Could add more metrics from resource monitoring
        
        return metrics
    
    # Update get_entity_statistics method in ModbusStatisticsManager
    
    def get_entity_statistics(self, entity_id: str) -> Dict[str, Any]:
        """Get statistics for a specific entity."""
        if entity_id in self._entity_trackers:
            stats = self._entity_trackers[entity_id].get_statistics()
            
            # Add cluster information
            cluster = self._correlation_manager.get_cluster_for_entity(entity_id)
            if cluster:
                stats['cluster'] = cluster
                
            # Add current system pattern information
            pattern_info = self._pattern_detector.get_current_pattern_info()
            stats.update({
                'current_pattern': pattern_info.get('pattern_id'),
                'pattern_confidence': pattern_info.get('pattern_confidence'),
                'pattern_since': pattern_info.get('active_since'),
                'predicted_pattern_end': pattern_info.get('predicted_end'),
                'next_predicted_pattern': pattern_info.get('predicted_next_pattern'),
                'pattern_criteria': pattern_info.get('defining_criteria'),
            })
            
            # Add recommended scan intervals from both systems
            legacy_interval = self._interval_manager.get_recommended_interval(
                entity_id, time.time()
            )
            advanced_interval = self._advanced_interval_manager.get_recommended_interval(
                entity_id, time.time()
            )
            
            stats['legacy_scan_interval'] = legacy_interval
            stats['advanced_scan_interval'] = advanced_interval
            stats['dynamic_scan_interval'] = advanced_interval  # Use advanced as primary
            
            # Add detailed interval metrics
            advanced_stats = self._advanced_interval_manager.get_entity_statistics(entity_id)
            if advanced_stats:
                stats['interval_metrics'] = advanced_stats
            
            # Add integrated insights if available
            if hasattr(PERSISTENT_STATISTICS_MANAGER, "get_integrated_insights"):
                integrated_insights = PERSISTENT_STATISTICS_MANAGER.get_integrated_insights(entity_id)
                if "integrated_insights" in integrated_insights:
                    stats["integrated_insights"] = integrated_insights["integrated_insights"]
            
            # Save entity stats to persistent storage
            PERSISTENT_STATISTICS_MANAGER.save_entity_stats(entity_id, stats)
            
            return stats
            
        # Try to get from storage if not in active trackers
        return self._storage.get_entity_stats(entity_id) or {}
    
    def set_scan_interval(self, entity_id: str, interval: int) -> bool:
        """Set the scan interval for an entity."""
        # Update base scan interval in tracker
        if entity_id in self._entity_trackers:
            tracker = self._entity_trackers[entity_id]
            if hasattr(tracker, "current_scan_interval"):
                tracker.current_scan_interval = interval
                
            # Register updated interval with interval manager
            if hasattr(self, "_interval_manager"):
                self._interval_manager.register_entity(
                    entity_id,
                    initial_interval=interval
                )
                return True
                
        return False
        
    def get_correlation(self, entity_a: str, entity_b: str) -> Optional[float]:
        """Get correlation between two entities."""
        return self._correlation_manager.get_correlation(entity_a, entity_b)
        
    def get_cluster_for_entity(self, entity_id: str) -> Optional[str]:
        """Get the cluster containing an entity."""
        return self._correlation_manager.get_cluster_for_entity(entity_id)
        
    def get_clusters(self) -> Dict[str, List[str]]:
        """Get all entity clusters."""
        return self._correlation_manager.get_clusters()
        
    def analyze_correlations(self, force: bool = True) -> Dict[str, List[str]]:
        """Force correlation analysis."""
        clusters = self._correlation_manager.analyze_correlations(force=force)
        
        # Update interval manager with new clusters
        if clusters:
            self._interval_manager.update_clusters(clusters)
            
        return clusters
        
    def get_pattern_information(self) -> Dict[str, Any]:
        """Get current pattern detection information."""
        pattern_info = self._pattern_detector.get_current_pattern_info()
        
        # Add statistics about detected patterns
        pattern_stats = self._pattern_detector.get_statistics()
        pattern_info.update(pattern_stats)
        
        return pattern_info
        
    def set_entity_importance(self, entity_id: str, importance: float = 5.0) -> None:
        """Set the importance factor for an entity (how much it influences pattern detection)."""
        self._pattern_detector.set_entity_weight(entity_id, importance)
        
    def get_pattern_definition(self, pattern_id: int) -> Dict[str, Any]:
        """Get detailed definition of what characterizes a pattern."""
        return self._pattern_detector.get_pattern_definition(pattern_id)
        
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system-wide statistics about polling optimization."""
        return self._interval_manager.get_statistics()
        
    def optimize_all_intervals(self) -> Dict[str, int]:
        """Force optimization of all scan intervals."""
        current_time = time.time()
        optimized_intervals = {}
        
        for entity_id in self._entity_trackers:
            # Calculate optimal interval
            interval = self._interval_manager.get_recommended_interval(
                entity_id, current_time, force_recalculate=True
            )
            optimized_intervals[entity_id] = interval
            
        return optimized_intervals
        
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the storage system."""
        return PERSISTENT_STATISTICS_MANAGER.get_storage_info()
    
    def export_data(self) -> Dict[str, Any]:
        """Export all statistics data."""
        return PERSISTENT_STATISTICS_MANAGER.export_data()
    
    # Integrated Pattern-Correlation Methods
    
    def run_integration_analysis(self) -> Dict[str, Any]:
        """Run integration analysis between pattern and correlation systems.
        
        Returns:
            Dictionary with analysis results
        """
        if hasattr(PERSISTENT_STATISTICS_MANAGER, "perform_integration_analysis"):
            return PERSISTENT_STATISTICS_MANAGER.perform_integration_analysis()
        return {"error": "Integration analysis not available"}
    
    def get_integrated_insights(self, entity_id: str) -> Dict[str, Any]:
        """Get integrated insights for an entity.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Dictionary with integrated insights
        """
        if hasattr(PERSISTENT_STATISTICS_MANAGER, "get_integrated_insights"):
            return PERSISTENT_STATISTICS_MANAGER.get_integrated_insights(entity_id)
        return {"error": "Integrated insights not available"}
    
    def get_pattern_correlation_insights(self, pattern_id: Optional[int] = None) -> Dict[str, Any]:
        """Get correlation insights for patterns.
        
        Args:
            pattern_id: Optional pattern ID to filter results
            
        Returns:
            Dictionary with pattern correlation insights
        """
        if hasattr(PERSISTENT_STATISTICS_MANAGER, "get_pattern_correlation_insights"):
            return PERSISTENT_STATISTICS_MANAGER.get_pattern_correlation_insights(pattern_id)
        return {"error": "Pattern correlation insights not available"}
    
    # Adaptive Parameters System Methods
    
    def get_parameter_profiles(self, profile_name: Optional[str] = None) -> Dict[str, Any]:
        """Get parameter profiles.
        
        Args:
            profile_name: Optional profile name to get specific profile
            
        Returns:
            Parameter profiles data
        """
        if profile_name:
            return PARAMETER_MANAGER.get_profile(profile_name) or {"error": "Profile not found"}
        else:
            return PARAMETER_MANAGER.get_all_profiles()
    
    def create_parameter_profile(self, name: str, description: str, 
                              parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new parameter profile.
        
        Args:
            name: Profile name
            description: Profile description
            parameters: Optional parameters for the profile
            
        Returns:
            Result dictionary
        """
        if not name:
            return {"success": False, "error": "Profile name is required"}
            
        try:
            profile = PARAMETER_MANAGER.create_profile(name, description, "user")
            
            # Apply parameters if provided
            if parameters:
                for param, value in parameters.items():
                    profile.update_parameter(param, value)
            
            # Save changes
            PARAMETER_MANAGER._save_data()
            
            return {
                "success": True, 
                "profile": profile.to_dict(),
                "message": f"Created profile '{name}'"
            }
        except Exception as e:
            _LOGGER.error("Error creating parameter profile: %s", e)
            return {"success": False, "error": str(e)}
    
    def update_parameter_profile(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a parameter profile.
        
        Args:
            name: Profile name
            data: Update data (description, parameters)
            
        Returns:
            Result dictionary
        """
        if not name:
            return {"success": False, "error": "Profile name is required"}
            
        try:
            profile_dict = PARAMETER_MANAGER.get_profile(name)
            if not profile_dict:
                return {"success": False, "error": f"Profile '{name}' not found"}
                
            profile = PARAMETER_MANAGER.profiles.get(name)
            
            # Update description if provided
            if "description" in data:
                profile.description = data["description"]
                
            # Update parameters if provided
            if "parameters" in data and isinstance(data["parameters"], dict):
                for param, value in data["parameters"].items():
                    profile.update_parameter(param, value)
            
            # Update context match if provided
            if "context_match" in data and isinstance(data["context_match"], dict):
                for context_type_str, match_data in data["context_match"].items():
                    try:
                        context_type = ContextType(context_type_str)
                        profile.update_context_match(context_type, match_data)
                    except ValueError:
                        _LOGGER.warning("Invalid context type: %s", context_type_str)
            
            # Save changes
            PARAMETER_MANAGER._save_data()
            
            return {
                "success": True, 
                "profile": profile.to_dict(),
                "message": f"Updated profile '{name}'"
            }
        except Exception as e:
            _LOGGER.error("Error updating parameter profile: %s", e)
            return {"success": False, "error": str(e)}
    
    def delete_parameter_profile(self, name: str) -> Dict[str, Any]:
        """Delete a parameter profile.
        
        Args:
            name: Profile name
            
        Returns:
            Result dictionary
        """
        if not name:
            return {"success": False, "error": "Profile name is required"}
            
        try:
            if name == PARAMETER_MANAGER.active_profile_name:
                return {
                    "success": False, 
                    "error": "Cannot delete active profile"
                }
                
            if name not in PARAMETER_MANAGER.profiles:
                return {
                    "success": False, 
                    "error": f"Profile '{name}' not found"
                }
                
            success = PARAMETER_MANAGER.delete_profile(name)
            
            if success:
                return {
                    "success": True,
                    "message": f"Deleted profile '{name}'"
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to delete profile '{name}'"
                }
        except Exception as e:
            _LOGGER.error("Error deleting parameter profile: %s", e)
            return {"success": False, "error": str(e)}
    
    def activate_parameter_profile(self, name: str) -> Dict[str, Any]:
        """Activate a parameter profile.
        
        Args:
            name: Profile name
            
        Returns:
            Result dictionary
        """
        if not name:
            return {"success": False, "error": "Profile name is required"}
            
        try:
            if name not in PARAMETER_MANAGER.profiles:
                return {
                    "success": False, 
                    "error": f"Profile '{name}' not found"
                }
                
            success = PARAMETER_MANAGER.activate_profile(name)
            
            if success:
                # Apply updated parameters to components
                self._apply_current_parameters()
                
                return {
                    "success": True,
                    "message": f"Activated profile '{name}'",
                    "parameters": PARAMETER_MANAGER.get_all_parameters()
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to activate profile '{name}'"
                }
        except Exception as e:
            _LOGGER.error("Error activating parameter profile: %s", e)
            return {"success": False, "error": str(e)}
    
    def adapt_parameters(self, force: bool = False) -> Dict[str, Any]:
        """Adapt parameters to current context.
        
        Args:
            force: Force adaptation even if not needed
            
        Returns:
            Adaptation results
        """
        try:
            result = PARAMETER_MANAGER.adapt_parameters(force=force)
            
            if result.get("adapted", False):
                # Apply updated parameters to components
                self._apply_current_parameters()
            
            return result
        except Exception as e:
            _LOGGER.error("Error adapting parameters: %s", e)
            return {"success": False, "error": str(e)}
    
    def get_parameter_status(self) -> Dict[str, Any]:
        """Get parameter system status.
        
        Returns:
            Status information
        """
        try:
            status = PARAMETER_MANAGER.get_status()
            
            # Add current parameter values
            status["current_parameters"] = PARAMETER_MANAGER.get_all_parameters()
            
            return status
        except Exception as e:
            _LOGGER.error("Error getting parameter status: %s", e)
            return {"error": str(e)}
    
    def set_parameter(self, name: str, value: Any) -> Dict[str, Any]:
        """Set a specific parameter value.
        
        Args:
            name: Parameter name
            value: Parameter value
            
        Returns:
            Result dictionary
        """
        try:
            if not name:
                return {"success": False, "error": "Parameter name is required"}
                
            success = PARAMETER_MANAGER.set_parameter(name, value)
            
            if success:
                # Apply updated parameter to components
                self._apply_current_parameters()
                
                return {
                    "success": True,
                    "message": f"Updated parameter '{name}' to {value}",
                    "parameter": name,
                    "value": PARAMETER_MANAGER.get_parameter(name)
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to update parameter '{name}'"
                }
        except Exception as e:
            _LOGGER.error("Error setting parameter: %s", e)
            return {"success": False, "error": str(e)}
          

# Create singleton instance
STATISTICS_MANAGER = ModbusStatisticsManager.get_instance()