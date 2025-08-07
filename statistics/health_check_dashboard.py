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
        self.storage = PERSISTENT_STATISTICS_MANAGER.storage
        self.last_update_time = 0
        self.update_interval = 300  # Update every 5 minutes
        self.dashboard_data = {}
    
    def get_dashboard_data(self, current_entity_ids: List[str] = None, force_update: bool = False) -> Dict[str, Any]:
        """Get comprehensive dashboard data.
        
        Args:
            current_entity_ids: List of current entity IDs in Home Assistant
            force_update: If True, force a refresh regardless of time
            
        Returns:
            Dictionary with dashboard data
        """
        current_time = time.time()
        
        # Check if we need to update
        if not force_update and current_time - self.last_update_time < self.update_interval:
            return self.dashboard_data
        
        self.last_update_time = current_time
        
        # Run health check
        health_result = self.health_check.check_health(current_entity_ids)
        
        # Get entity metrics
        entity_metrics = self._get_entity_metrics()
        
        # Get system metrics
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
    
    def _get_entity_metrics(self) -> Dict[str, Any]:
        """Get entity-related metrics."""
        metrics = {}
        
        # Get entity statistics
        entity_stats = self.storage.get_entity_stats()
        total_entities = len(entity_stats)
        metrics["total"] = total_entities
        
        # Get cluster information
        clusters = self.storage.get_clusters()
        
        # Count clustered entities
        clustered_entities = set()
        for members in clusters.values():
            clustered_entities.update(members)
        
        metrics["clustered"] = len(clustered_entities)
        metrics["unclustered"] = total_entities - len(clustered_entities)
        metrics["cluster_count"] = len(clusters)
        
        # Count black hole entities
        black_holes = []
        for entity_id, stats in entity_stats.items():
            if stats.get("is_blackhole", False):
                black_holes.append(entity_id)
        
        metrics["black_holes"] = black_holes
        metrics["black_hole_count"] = len(black_holes)
        
        # Count optimized entities
        optimized_count = 0
        for stats in entity_stats.values():
            if stats.get("recommended_scan_interval") != stats.get("original_scan_interval"):
                optimized_count += 1
        
        metrics["optimized"] = optimized_count
        
        return metrics
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-related metrics."""
        metrics = {}
        
        # Get pattern information
        patterns_data = self.storage.get_patterns()
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
        
        # Get storage metrics
        storage_info = self.storage.get_storage_info()
        if storage_info:
            metrics["storage_size_kb"] = sum(
                f.get("size_kb", 0)
                for f in storage_info.get("files", {}).values()
                if isinstance(f, dict)
            )
            metrics["storage_last_update"] = storage_info.get("last_update")
        
        # Try to get prediction accuracy
        try:
            from .prediction_evaluation import PredictionEvaluator
            evaluator = PredictionEvaluator.get_instance()
            accuracy_metrics = evaluator.calculate_overall_metrics()
            metrics["prediction_accuracy"] = accuracy_metrics.get("average_accuracy", 0)
        except Exception:
            pass
        
        # Try to get scheduler information
        try:
            from .analysis_scheduler import AnalysisScheduler
            scheduler = AnalysisScheduler.get_instance()
            next_task = scheduler.get_next_task()
            metrics["next_analysis_task"] = next_task
        except Exception:
            pass
        
        return metrics
    
    def get_sensor_attributes(self) -> Dict[str, Any]:
        """Get attributes for a Home Assistant sensor."""
        if not self.dashboard_data:
            # Force an update
            self.get_dashboard_data(force_update=True)
        
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


# Global singleton instance
HEALTH_DASHBOARD = HealthDashboard.get_instance()