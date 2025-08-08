"""
Health dashboard for Modbus optimization system.

This module provides a comprehensive health dashboard that can be displayed
in Home Assistant to monitor system health and performance.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

from .health_check import HealthCheckManager
from .validation import ValidationManager
from .persistent_statistics import PERSISTENT_STATISTICS_MANAGER

_LOGGER = logging.getLogger(__name__)

class HealthDashboard:
    """Health dashboard for Modbus optimization system."""
    
    _instance = None
    _initialized = False
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = HealthDashboard()
        return cls._instance
    
    def __init__(self):
        """Initialize health dashboard."""
        self.health_check = HealthCheckManager.get_instance()
        self.validation = ValidationManager.get_instance()
        self.storage = None  # Initialize to None, we'll access it on-demand
        self.last_update_time = 0
        self.update_interval = 300  # Update every 5 minutes
        self.dashboard_data = {
            "status": "initializing",
            "status_emoji": "⏳",
            "health_score": 0,
            "entity_metrics": {},
            "system_metrics": {},
            "alerts": [],
            "recommendations": [],
            "updated_at": datetime.utcnow().isoformat()
        }
        self.hass = None
    
    def set_hass(self, hass):
        """Set Home Assistant instance for async operations."""
        self.hass = hass
    
    def initialize(self):
        """Initialize the health dashboard after storage is ready."""
        if HealthDashboard._initialized:
            return True
            
        # Check if PERSISTENT_STATISTICS_MANAGER is initialized
        if not hasattr(PERSISTENT_STATISTICS_MANAGER, 'enabled') or not PERSISTENT_STATISTICS_MANAGER.enabled:
            _LOGGER.debug("Statistics manager not enabled yet, health dashboard initialization deferred")
            return False
            
        HealthDashboard._initialized = True
        _LOGGER.info("Health dashboard initialized with storage access")
        return True
    
    def is_initialized(self):
        """Check if the health dashboard is properly initialized."""
        # If we're already marked as initialized, return True
        if HealthDashboard._initialized:
            return True
            
        # Try to initialize if not already done
        return self.initialize()
    
    def get_dashboard_data(self, current_entity_ids: List[str] = None, force_update: bool = False) -> Dict[str, Any]:
        """Get comprehensive dashboard data.
        
        Args:
            current_entity_ids: List of current entity IDs in Home Assistant
            force_update: If True, force a refresh regardless of time
            
        Returns:
            Dictionary with dashboard data
        """
        # Check if we're initialized
        if not self.is_initialized():
            # Return initialization status if we're not ready
            return {
                "status": "initializing",
                "status_emoji": "⏳",
                "health_score": 0,
                "entity_metrics": {"total": 0},
                "system_metrics": {},
                "alerts": [],
                "recommendations": [],
                "updated_at": datetime.utcnow().isoformat()
            }
        
        current_time = time.time()
        
        # Check if we need to update
        if not force_update and current_time - self.last_update_time < self.update_interval:
            return self.dashboard_data
        
        self.last_update_time = current_time
        
        # Run health check
        health_result = self.health_check.check_health(current_entity_ids)
        
        # Get entity metrics and system metrics
        # Since we're in a synchronous context, we'll use the synchronous methods
        # but in the async version below, we'll use the async methods
        entity_metrics = self._get_entity_metrics()
        system_metrics = self._get_system_metrics()
        
        # Build dashboard data
        self.dashboard_data = {
            "status": health_result.status,
            "health_score": health_result.score,
            "entity_metrics": entity_metrics,
            "system_metrics": system_metrics,
            "alerts": health_result.issues,
            "recommendations": health_result.recommendations,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Add emoji for status
        if health_result.status == "healthy":
            self.dashboard_data["status_emoji"] = "✅"
        elif health_result.status == "warning":
            self.dashboard_data["status_emoji"] = "⚠️"
        elif health_result.status == "critical":
            self.dashboard_data["status_emoji"] = "🔴"
        else:
            self.dashboard_data["status_emoji"] = "❓"
        
        return self.dashboard_data
    
    async def async_get_dashboard_data(self, current_entity_ids: List[str] = None, force_update: bool = False) -> Dict[str, Any]:
        """Get comprehensive dashboard data asynchronously.
        
        Args:
            current_entity_ids: List of current entity IDs in Home Assistant
            force_update: If True, force a refresh regardless of time
            
        Returns:
            Dictionary with dashboard data
        """
        if not self.hass:
            # If no hass instance, fall back to synchronous method
            return self.get_dashboard_data(current_entity_ids, force_update)
            
        # Check if we're initialized
        if not self.is_initialized():
            # Return initialization status if we're not ready
            return {
                "status": "initializing",
                "status_emoji": "⏳",
                "health_score": 0,
                "entity_metrics": {"total": 0},
                "system_metrics": {},
                "alerts": [],
                "recommendations": [],
                "updated_at": datetime.utcnow().isoformat()
            }
        
        current_time = time.time()
        
        # Check if we need to update
        if not force_update and current_time - self.last_update_time < self.update_interval:
            return self.dashboard_data
        
        self.last_update_time = current_time
        
        # Run health check - this is synchronous so run in executor
        health_result = await self.hass.async_add_executor_job(
            self.health_check.check_health, current_entity_ids)
        
        # Get entity metrics and system metrics asynchronously
        entity_metrics = await self._async_get_entity_metrics()
        system_metrics = await self._async_get_system_metrics()
        
        # Build dashboard data
        self.dashboard_data = {
            "status": health_result.status,
            "health_score": health_result.score,
            "entity_metrics": entity_metrics,
            "system_metrics": system_metrics,
            "alerts": health_result.issues,
            "recommendations": health_result.recommendations,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Add emoji for status
        if health_result.status == "healthy":
            self.dashboard_data["status_emoji"] = "✅"
        elif health_result.status == "warning":
            self.dashboard_data["status_emoji"] = "⚠️"
        elif health_result.status == "critical":
            self.dashboard_data["status_emoji"] = "🔴"
        else:
            self.dashboard_data["status_emoji"] = "❓"
        
        return self.dashboard_data
    
    def _get_entity_metrics(self) -> Dict[str, Any]:
        """Get entity-related metrics synchronously."""
        metrics = {
            "total": 0,
            "clustered": 0,
            "unclustered": 0,
            "cluster_count": 0,
            "black_holes": [],
            "black_hole_count": 0,
            "optimized": 0
        }
        
        try:
            # Get entity statistics
            entity_stats = PERSISTENT_STATISTICS_MANAGER.get_entity_stats()
            if not entity_stats:
                return metrics
                
            total_entities = len(entity_stats)
            metrics["total"] = total_entities
            
            # Get cluster information
            clusters = PERSISTENT_STATISTICS_MANAGER.get_clusters() or {}
            
            # Count clustered entities
            clustered_entities = set()
            for members in clusters.values():
                if isinstance(members, list):
                    clustered_entities.update(members)
            
            metrics["clustered"] = len(clustered_entities)
            metrics["unclustered"] = total_entities - len(clustered_entities)
            metrics["cluster_count"] = len(clusters)
            
            # Count black hole entities
            black_holes = []
            for entity_id, stats in entity_stats.items():
                if isinstance(stats, dict) and stats.get("is_blackhole", False):
                    black_holes.append(entity_id)
            
            metrics["black_holes"] = black_holes
            metrics["black_hole_count"] = len(black_holes)
            
            # Count optimized entities
            optimized_count = 0
            for stats in entity_stats.values():
                if isinstance(stats, dict) and stats.get("recommended_scan_interval") != stats.get("original_scan_interval"):
                    optimized_count += 1
            
            metrics["optimized"] = optimized_count
            
        except Exception as e:
            _LOGGER.error("Error calculating entity metrics: %s", e)
            
        return metrics
    
    async def _async_get_entity_metrics(self) -> Dict[str, Any]:
        """Get entity-related metrics asynchronously."""
        metrics = {
            "total": 0,
            "clustered": 0,
            "unclustered": 0,
            "cluster_count": 0,
            "black_holes": [],
            "black_hole_count": 0,
            "optimized": 0
        }
        
        if not self.hass:
            return self._get_entity_metrics()  # Fall back to sync method
            
        try:
            # Get entity statistics asynchronously
            entity_stats = await PERSISTENT_STATISTICS_MANAGER.async_get_entity_stats(self.hass)
            if not entity_stats:
                return metrics
                
            total_entities = len(entity_stats)
            metrics["total"] = total_entities
            
            # Get cluster information asynchronously
            clusters = await PERSISTENT_STATISTICS_MANAGER.async_get_clusters(self.hass) or {}
            
            # Count clustered entities
            clustered_entities = set()
            for members in clusters.values():
                if isinstance(members, list):
                    clustered_entities.update(members)
            
            metrics["clustered"] = len(clustered_entities)
            metrics["unclustered"] = total_entities - len(clustered_entities)
            metrics["cluster_count"] = len(clusters)
            
            # Count black hole entities
            black_holes = []
            for entity_id, stats in entity_stats.items():
                if isinstance(stats, dict) and stats.get("is_blackhole", False):
                    black_holes.append(entity_id)
            
            metrics["black_holes"] = black_holes
            metrics["black_hole_count"] = len(black_holes)
            
            # Count optimized entities
            optimized_count = 0
            for stats in entity_stats.values():
                if isinstance(stats, dict) and stats.get("recommended_scan_interval") != stats.get("original_scan_interval"):
                    optimized_count += 1
            
            metrics["optimized"] = optimized_count
            
        except Exception as e:
            _LOGGER.error("Error calculating entity metrics asynchronously: %s", e)
            
        return metrics
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-related metrics synchronously."""
        metrics = {}
        
        try:
            # Get pattern information
            patterns_data = PERSISTENT_STATISTICS_MANAGER.get_patterns() or {}
                
            if patterns_data:
                metrics["current_pattern"] = patterns_data.get("current_pattern")
                
                pattern_count = len(patterns_data.get("patterns", {}))
                metrics["pattern_count"] = pattern_count
                
                # Get last analysis time
                meta = patterns_data.get("meta", {})
                last_detection = meta.get("last_detection")
                if last_detection:
                    metrics["last_pattern_detection"] = last_detection
                    
                    try:
                        detection_time = datetime.fromisoformat(last_detection)
                        now = datetime.utcnow()
                        hours_ago = (now - detection_time).total_seconds() / 3600
                        metrics["pattern_detection_hours_ago"] = round(hours_ago, 1)
                    except (ValueError, TypeError):
                        pass
            
            # Get storage metrics - this will generate the warning
            storage_info = PERSISTENT_STATISTICS_MANAGER.get_storage_info()
            if storage_info:
                metrics["storage_size_kb"] = sum(
                    f.get("size_bytes", 0) / 1024
                    for f in storage_info.get("files", {}).values()
                    if isinstance(f, dict)
                )
                metrics["storage_last_update"] = storage_info.get("timestamp")
            
            # Try to get prediction accuracy
            try:
                from .prediction_evaluation import PredictionEvaluator
                evaluator = PredictionEvaluator.get_instance()
                accuracy_metrics = evaluator.calculate_overall_metrics()
                metrics["prediction_accuracy"] = accuracy_metrics.get("average_accuracy", 0)
            except Exception as e:
                _LOGGER.debug("Error getting prediction accuracy: %s", e)
            
            # Try to get scheduler information
            try:
                from .analysis_scheduler import AnalysisScheduler
                scheduler = AnalysisScheduler.get_instance()
                next_task = scheduler.get_next_task()
                metrics["next_analysis_task"] = next_task
            except Exception as e:
                _LOGGER.debug("Error getting next analysis task: %s", e)
                
        except Exception as e:
            _LOGGER.error("Error calculating system metrics: %s", e)
            
        return metrics
    
    async def _async_get_system_metrics(self) -> Dict[str, Any]:
        """Get system-related metrics asynchronously."""
        metrics = {}
        
        if not self.hass:
            return self._get_system_metrics()  # Fall back to sync method
            
        try:
            # Get pattern information asynchronously
            patterns_data = await PERSISTENT_STATISTICS_MANAGER.async_get_patterns(self.hass) or {}
                
            if patterns_data:
                metrics["current_pattern"] = patterns_data.get("current_pattern")
                
                pattern_count = len(patterns_data.get("patterns", {}))
                metrics["pattern_count"] = pattern_count
                
                # Get last analysis time
                meta = patterns_data.get("meta", {})
                last_detection = meta.get("last_detection")
                if last_detection:
                    metrics["last_pattern_detection"] = last_detection
                    
                    try:
                        detection_time = datetime.fromisoformat(last_detection)
                        now = datetime.utcnow()
                        hours_ago = (now - detection_time).total_seconds() / 3600
                        metrics["pattern_detection_hours_ago"] = round(hours_ago, 1)
                    except (ValueError, TypeError):
                        pass
            
            # Get storage metrics asynchronously - this is the proper async approach
            storage_info = await PERSISTENT_STATISTICS_MANAGER.async_get_storage_info(self.hass)
            if storage_info:
                metrics["storage_size_kb"] = sum(
                    f.get("size_bytes", 0) / 1024
                    for f in storage_info.get("files", {}).values()
                    if isinstance(f, dict)
                )
                metrics["storage_last_update"] = storage_info.get("timestamp")
            
            # Try to get prediction accuracy - run in executor if needed
            try:
                from .prediction_evaluation import PredictionEvaluator
                evaluator = PredictionEvaluator.get_instance()
                accuracy_metrics = await self.hass.async_add_executor_job(evaluator.calculate_overall_metrics)
                metrics["prediction_accuracy"] = accuracy_metrics.get("average_accuracy", 0)
            except Exception as e:
                _LOGGER.debug("Error getting prediction accuracy asynchronously: %s", e)
            
            # Try to get scheduler information - run in executor if needed
            try:
                from .analysis_scheduler import AnalysisScheduler
                scheduler = AnalysisScheduler.get_instance()
                next_task = await self.hass.async_add_executor_job(scheduler.get_next_task)
                metrics["next_analysis_task"] = next_task
            except Exception as e:
                _LOGGER.debug("Error getting next analysis task asynchronously: %s", e)
                
        except Exception as e:
            _LOGGER.error("Error calculating system metrics asynchronously: %s", e)
            
        return metrics
    
    def get_sensor_attributes(self) -> Dict[str, Any]:
        """Get attributes for a Home Assistant sensor."""
        # If we're not initialized, return minimal attributes
        if not self.is_initialized():
            return {
                "status": "initializing",
                "status_emoji": "⏳",
                "initialization_status": "Storage system not yet ready"
            }
        
        if not self.dashboard_data or self.dashboard_data.get("status") == "initializing":
            try:
                # Force an update
                self.get_dashboard_data(force_update=True)
            except Exception as e:
                _LOGGER.error("Error getting dashboard data: %s", e)
                return {
                    "status": "error",
                    "status_emoji": "❌",
                    "error": str(e)
                }
        
        # Build sensor attributes
        attributes = {
            "health_score": self.dashboard_data.get("health_score", 0),
            "status": self.dashboard_data.get("status", "unknown"),
            "status_emoji": self.dashboard_data.get("status_emoji", "❓"),
            "total_entities": self.dashboard_data.get("entity_metrics", {}).get("total", 0),
            "unclustered_entities": self.dashboard_data.get("entity_metrics", {}).get("unclustered", 0),
            "black_hole_entities": len(self.dashboard_data.get("entity_metrics", {}).get("black_holes", [])),
            "cluster_count": self.dashboard_data.get("entity_metrics", {}).get("cluster_count", 0),
            "current_pattern": self.dashboard_data.get("system_metrics", {}).get("current_pattern", None),
            "storage_size_kb": self.dashboard_data.get("system_metrics", {}).get("storage_size_kb", 0),
            "last_pattern_detection": self.dashboard_data.get("system_metrics", {}).get("last_pattern_detection"),
            "next_analysis_task": self.dashboard_data.get("system_metrics", {}).get("next_analysis_task"),
            "last_updated": self.dashboard_data.get("updated_at"),
        }
        
        # Add top recommendations
        recommendations = self.dashboard_data.get("recommendations", [])
        if recommendations:
            # Sort by priority (high, medium, low)
            priority_order = {"high": 0, "medium": 1, "low": 2}
            sorted_recommendations = sorted(
                recommendations,
                key=lambda x: priority_order.get(x.get("priority"), 3)
            )
            
            # Add top 3 recommendations
            for i, recommendation in enumerate(sorted_recommendations[:3]):
                attributes[f"recommendation_{i+1}"] = recommendation.get("action")
                attributes[f"recommendation_{i+1}_priority"] = recommendation.get("priority")
        
        # Add top alerts
        alerts = self.dashboard_data.get("alerts", [])
        if alerts:
            # Sort by severity (error, warning)
            severity_order = {"error": 0, "warning": 1}
            sorted_alerts = sorted(
                alerts,
                key=lambda x: severity_order.get(x.get("severity"), 2)
            )
            
            # Add top 3 alerts
            for i, alert in enumerate(sorted_alerts[:3]):
                attributes[f"alert_{i+1}"] = alert.get("message")
                attributes[f"alert_{i+1}_severity"] = alert.get("severity")
        
        return attributes
    
    async def async_get_sensor_attributes(self) -> Dict[str, Any]:
        """Get attributes for a Home Assistant sensor asynchronously."""
        if not self.hass:
            # Fall back to synchronous method
            return self.get_sensor_attributes()
            
        # If we're not initialized, return minimal attributes
        if not self.is_initialized():
            return {
                "status": "initializing",
                "status_emoji": "⏳",
                "initialization_status": "Storage system not yet ready"
            }
        
        if not self.dashboard_data or self.dashboard_data.get("status") == "initializing":
            try:
                # Force an update asynchronously
                await self.async_get_dashboard_data(force_update=True)
            except Exception as e:
                _LOGGER.error("Error getting dashboard data asynchronously: %s", e)
                return {
                    "status": "error",
                    "status_emoji": "❌",
                    "error": str(e)
                }
        
        # Build sensor attributes - same as synchronous version from here
        attributes = {
            "health_score": self.dashboard_data.get("health_score", 0),
            "status": self.dashboard_data.get("status", "unknown"),
            "status_emoji": self.dashboard_data.get("status_emoji", "❓"),
            "total_entities": self.dashboard_data.get("entity_metrics", {}).get("total", 0),
            "unclustered_entities": self.dashboard_data.get("entity_metrics", {}).get("unclustered", 0),
            "black_hole_entities": len(self.dashboard_data.get("entity_metrics", {}).get("black_holes", [])),
            "cluster_count": self.dashboard_data.get("entity_metrics", {}).get("cluster_count", 0),
            "current_pattern": self.dashboard_data.get("system_metrics", {}).get("current_pattern", None),
            "storage_size_kb": self.dashboard_data.get("system_metrics", {}).get("storage_size_kb", 0),
            "last_pattern_detection": self.dashboard_data.get("system_metrics", {}).get("last_pattern_detection"),
            "next_analysis_task": self.dashboard_data.get("system_metrics", {}).get("next_analysis_task"),
            "last_updated": self.dashboard_data.get("updated_at"),
        }
        
        # Add top recommendations
        recommendations = self.dashboard_data.get("recommendations", [])
        if recommendations:
            # Sort by priority (high, medium, low)
            priority_order = {"high": 0, "medium": 1, "low": 2}
            sorted_recommendations = sorted(
                recommendations,
                key=lambda x: priority_order.get(x.get("priority"), 3)
            )
            
            # Add top 3 recommendations
            for i, recommendation in enumerate(sorted_recommendations[:3]):
                attributes[f"recommendation_{i+1}"] = recommendation.get("action")
                attributes[f"recommendation_{i+1}_priority"] = recommendation.get("priority")
        
        # Add top alerts
        alerts = self.dashboard_data.get("alerts", [])
        if alerts:
            # Sort by severity (error, warning)
            severity_order = {"error": 0, "warning": 1}
            sorted_alerts = sorted(
                alerts,
                key=lambda x: severity_order.get(x.get("severity"), 2)
            )
            
            # Add top 3 alerts
            for i, alert in enumerate(sorted_alerts[:3]):
                attributes[f"alert_{i+1}"] = alert.get("message")
                attributes[f"alert_{i+1}_severity"] = alert.get("severity")
        
        return attributes


# Global singleton instance
HEALTH_DASHBOARD = HealthDashboard.get_instance()