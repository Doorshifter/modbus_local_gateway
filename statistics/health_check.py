"""
Health check system for Modbus optimization.

This module provides health check capabilities and recommendations for
maintaining system performance and reliability.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set

from .validation import ValidationManager
from .storage import StatisticsStorageManager
from .analysis_scheduler import AnalysisScheduler

_LOGGER = logging.getLogger(__name__)

class HealthCheckResult:
    """Represents the result of a health check."""
    
    def __init__(self):
        """Initialize health check result."""
        self.status = "unknown"  # unknown, healthy, warning, critical
        self.score = 0  # 0-100
        self.issues = []
        self.recommendations = []
        self.metrics = {}
        self.timestamp = datetime.utcnow().isoformat()
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status,
            "score": self.score,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "metrics": self.metrics,
            "timestamp": self.timestamp
        }


class HealthCheckManager:
    """Manages health checks and recommendations."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = HealthCheckManager()
        return cls._instance
    
    def __init__(self):
        """Initialize health check manager."""
        self.validation_manager = ValidationManager.get_instance()
        self.storage = StatisticsStorageManager.get_instance()
        self.last_check_time = 0
        self.last_result = None
        self.hass = None
        
        # Initialize component weights for health score
        self.component_weights = {
            "file_structure": 25,  # Critical - without valid files nothing works
            "entity_consistency": 20,  # High importance - entities are the core
            "clusters": 15,  # Moderately high - clustering affects optimization
            "scan_intervals": 20,  # High importance - directly affects performance
            "data_quality": 20,  # High importance - affects analysis quality
        }
    
    def set_hass(self, hass):
        """Set Home Assistant instance for async operations.
        
        Args:
            hass: Home Assistant instance
        """
        self.hass = hass
    
    def check_health(self, current_entity_ids: List[str] = None) -> HealthCheckResult:
        """Run a comprehensive health check. WARNING: Contains blocking calls!
        
        This method should not be called directly from the event loop.
        Use async_check_health instead when in an async context.
        
        Args:
            current_entity_ids: List of current entity IDs in Home Assistant
            
        Returns:
            HealthCheckResult object
        """
        self.last_check_time = time.time()
        
        # Create result object
        result = HealthCheckResult()
        
        # Run validation
        validation_report = self.validation_manager.validate_all(current_entity_ids)
        
        # Calculate health score based on validation results
        self._calculate_health_score(result, validation_report)
        
        # Generate recommendations
        self._generate_recommendations(result, validation_report)
        
        # Add system metrics (blocking)
        self._blocking_add_system_metrics(result)
        
        self.last_result = result
        return result
    
    async def async_check_health(self, current_entity_ids: List[str] = None) -> HealthCheckResult:
        """Run a comprehensive health check asynchronously.
        
        Args:
            current_entity_ids: List of current entity IDs in Home Assistant
            
        Returns:
            HealthCheckResult object
        """
        if not self.hass:
            raise RuntimeError("Home Assistant instance not set. Call set_hass() first.")
            
        self.last_check_time = time.time()
        
        # Create result object
        result = HealthCheckResult()
        
        # Run validation (assuming it's not blocking; if it is, should be moved to executor)
        validation_report = self.validation_manager.validate_all(current_entity_ids)
        
        # Calculate health score based on validation results
        self._calculate_health_score(result, validation_report)
        
        # Generate recommendations
        self._generate_recommendations(result, validation_report)
        
        # Add system metrics (non-blocking)
        await self._async_add_system_metrics(result)
        
        self.last_result = result
        return result
    
    def _calculate_health_score(self, result: HealthCheckResult, validation_report: Dict[str, Any]) -> None:
        """Calculate health score based on validation results.
        
        Args:
            result: HealthCheckResult to update
            validation_report: Validation report from ValidationManager
        """
        # Initialize score components
        score_components = {}
        
        # Get overall status
        overall_status = validation_report.get("overall_status", "fail")
        
        # Process each component
        for component, weight in self.component_weights.items():
            component_result = validation_report.get("results", {}).get(component)
            if not component_result:
                # Missing component, assign zero score
                score_components[component] = 0
                continue
            
            # Calculate component score based on errors and warnings
            error_count = len(component_result.get("errors", []))
            warning_count = len(component_result.get("warnings", []))
            
            if error_count > 0:
                # Errors cause significant score reduction
                component_score = max(0, weight * (1 - (error_count * 0.25)))
            elif warning_count > 0:
                # Warnings cause moderate score reduction
                component_score = max(weight * 0.5, weight * (1 - (warning_count * 0.1)))
            else:
                # No issues, full score
                component_score = weight
            
            score_components[component] = component_score
        
        # Calculate total score (0-100)
        total_score = sum(score_components.values())
        result.score = round(total_score)
        
        # Determine status based on score
        if total_score >= 90:
            result.status = "healthy"
        elif total_score >= 70:
            result.status = "warning"
        else:
            result.status = "critical"
        
        # Add score breakdown to metrics
        result.metrics["score_components"] = {
            component: round(score) for component, score in score_components.items()
        }
        
        # Extract issues from validation
        for component, component_result in validation_report.get("results", {}).items():
            if component == "overall":
                continue
                
            for error in component_result.get("errors", []):
                result.issues.append({
                    "component": component,
                    "severity": "error",
                    "message": error
                })
            
            for warning in component_result.get("warnings", []):
                result.issues.append({
                    "component": component,
                    "severity": "warning",
                    "message": warning
                })
    
    def _generate_recommendations(self, result: HealthCheckResult, validation_report: Dict[str, Any]) -> None:
        """Generate recommendations based on validation results.
        
        Args:
            result: HealthCheckResult to update
            validation_report: Validation report from ValidationManager
        """
        # Check for specific issues and generate recommendations
        
        # File structure issues
        file_result = validation_report.get("results", {}).get("file_structure")
        if file_result and file_result.get("status") == "fail":
            result.recommendations.append({
                "priority": "high",
                "action": "Fix file structure issues",
                "description": "Create missing files or restore from backup",
                "automated_fix_available": True,
                "fix_action": "create_missing_files"
            })
        
        # Entity consistency issues
        entity_result = validation_report.get("results", {}).get("entity_consistency")
        if entity_result:
            entities_to_remove = entity_result.get("details", {}).get("entities_to_remove", [])
            if entities_to_remove:
                result.recommendations.append({
                    "priority": "medium",
                    "action": "Remove orphaned entities",
                    "description": f"Remove {len(entities_to_remove)} entities that no longer exist",
                    "automated_fix_available": True,
                    "fix_action": "fix_orphaned_entities"
                })
            
            new_entities = entity_result.get("details", {}).get("new_entities", [])
            if new_entities:
                result.recommendations.append({
                    "priority": "medium",
                    "action": "Run correlation analysis",
                    "description": f"Analyze {len(new_entities)} new entities",
                    "automated_fix_available": False,
                    "manual_action": "Run correlation analysis service"
                })
        
        # Cluster issues
        cluster_result = validation_report.get("results", {}).get("clusters")
        if cluster_result:
            empty_clusters = cluster_result.get("details", {}).get("empty_clusters", [])
            if empty_clusters:
                result.recommendations.append({
                    "priority": "low",
                    "action": "Remove empty clusters",
                    "description": f"Clean up {len(empty_clusters)} empty clusters",
                    "automated_fix_available": True,
                    "fix_action": "fix_empty_clusters"
                })
            
            undersized_clusters = cluster_result.get("details", {}).get("undersized_clusters", [])
            if undersized_clusters:
                result.recommendations.append({
                    "priority": "low",
                    "action": "Fix undersized clusters",
                    "description": f"Merge {len(undersized_clusters)} clusters with fewer than 2 members",
                    "automated_fix_available": True,
                    "fix_action": "fix_undersized_clusters"
                })
            
            unclustered_entities = cluster_result.get("details", {}).get("unclustered_entities", [])
            if unclustered_entities:
                result.recommendations.append({
                    "priority": "medium",
                    "action": "Cluster unclustered entities",
                    "description": f"Assign {len(unclustered_entities)} entities to clusters",
                    "automated_fix_available": False,
                    "manual_action": "Run correlation analysis service"
                })
        
        # Scan interval issues
        interval_result = validation_report.get("results", {}).get("scan_intervals")
        if interval_result:
            stale_optimizations = interval_result.get("details", {}).get("stale_optimizations", [])
            if stale_optimizations:
                result.recommendations.append({
                    "priority": "medium",
                    "action": "Run scan interval optimization",
                    "description": f"Update {len(stale_optimizations)} stale scan interval optimizations",
                    "automated_fix_available": False,
                    "manual_action": "Run scan interval optimization service"
                })
            
            invalid_intervals = interval_result.get("details", {}).get("invalid_intervals", [])
            if invalid_intervals:
                result.recommendations.append({
                    "priority": "high",
                    "action": "Fix invalid scan intervals",
                    "description": f"Correct {len(invalid_intervals)} invalid scan intervals",
                    "automated_fix_available": False,
                    "manual_action": "Run scan interval optimization service"
                })
        
        # Data quality issues
        quality_result = validation_report.get("results", {}).get("data_quality")
        if quality_result:
            black_holes = quality_result.get("details", {}).get("black_holes", [])
            if black_holes:
                result.recommendations.append({
                    "priority": "low",
                    "action": "Review black hole entities",
                    "description": f"Check {len(black_holes)} entities with very low activity",
                    "automated_fix_available": False,
                    "manual_action": "Review entity activity and consider excluding from analysis"
                })
    
    def _blocking_add_system_metrics(self, result: HealthCheckResult) -> None:
        """Add system metrics to health check result - BLOCKING VERSION.
        
        Warning: This method contains blocking calls and should not be used in the event loop.
        
        Args:
            result: HealthCheckResult to update
        """
        # Get storage info
        storage_info = self.storage.get_storage_info()
        
        # Get scheduler info if available
        try:
            scheduler = AnalysisScheduler.get_instance()
            scheduler_stats = scheduler.get_task_stats()
        except Exception:
            scheduler_stats = None
        
        # Add storage metrics
        if storage_info:
            result.metrics["storage"] = {
                "stats_count": storage_info.get("stats_count", 0),
                "patterns_count": storage_info.get("patterns_count", 0),
                "clusters_count": storage_info.get("clusters_count", 0),
                "total_size_kb": sum(
                    f.get("size_kb", 0) 
                    for f in storage_info.get("files", {}).values() 
                    if isinstance(f, dict)
                )
            }
        
        # Add scheduler metrics
        if scheduler_stats:
            result.metrics["scheduler"] = scheduler_stats.get("scheduler", {})
            
            # Add next task info
            tasks = scheduler_stats.get("tasks", {})
            if tasks:
                next_tasks = []
                now = time.time()
                
                for task_name, task_info in tasks.items():
                    next_run_time_str = task_info.get("next_run_time")
                    if next_run_time_str:
                        try:
                            next_run_time = datetime.fromisoformat(next_run_time_str)
                            seconds_until = (next_run_time - datetime.now()).total_seconds()
                            
                            if seconds_until > 0:
                                next_tasks.append({
                                    "name": task_name,
                                    "next_run_in_seconds": int(seconds_until),
                                    "priority": task_info.get("priority", 0)
                                })
                        except (ValueError, TypeError):
                            pass
                
                # Sort by soonest first
                next_tasks.sort(key=lambda x: x["next_run_in_seconds"])
                result.metrics["next_tasks"] = next_tasks[:3]  # Show next 3 tasks
        
        # Add runtime metrics
        import os
        import sys
        
        try:
            import psutil
            process = psutil.Process(os.getpid())
            
            # THIS IS THE BLOCKING CALL THAT CAUSES THE WARNING:
            # We keep it here in the blocking version, but create a non-blocking alternative
            result.metrics["runtime"] = {
                "memory_mb": round(process.memory_info().rss / (1024 * 1024), 1),
                "cpu_percent": process.cpu_percent(interval=0.1),  # This blocks for 0.1 seconds
                "uptime_seconds": int(time.time() - process.create_time())
            }
        except ImportError:
            # psutil not available
            result.metrics["runtime"] = {
                "python_version": sys.version.split()[0],
                "interpreter_path": sys.executable
            }
        
        # Add timestamp
        result.metrics["current_time"] = datetime.utcnow().isoformat()
    
    def _add_system_metrics(self, result: HealthCheckResult) -> None:
        """Add system metrics to health check result.
        
        Warning: Contains blocking calls, use async_add_system_metrics when in event loop.
        
        Args:
            result: HealthCheckResult to update
        """
        # This just calls the blocking implementation
        # WARNING: This is a blocking call
        self._blocking_add_system_metrics(result)
    
    async def _async_add_system_metrics(self, result: HealthCheckResult) -> None:
        """Add system metrics to health check result asynchronously.
        
        Args:
            result: HealthCheckResult to update
        """
        if not self.hass:
            raise RuntimeError("Home Assistant instance not set. Call set_hass() first.")
        
        # Run the blocking operation in the executor
        def _get_metrics():
            # Create a dummy result to collect metrics
            dummy_result = HealthCheckResult()
            self._blocking_add_system_metrics(dummy_result)
            return dummy_result.metrics
            
        # Execute the blocking operation in a thread
        metrics = await self.hass.async_add_executor_job(_get_metrics)
        
        # Copy the metrics to our actual result
        result.metrics.update(metrics)