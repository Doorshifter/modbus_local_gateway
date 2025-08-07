"""
Visualization tools for the advanced interval system.

This module provides tools to visualize and understand how the interval
optimization system is working.
"""

import time
import logging
from typing import Dict, List, Any, Optional
import json

from .advanced_interval_manager import ADVANCED_INTERVAL_MANAGER

_LOGGER = logging.getLogger(__name__)

class IntervalVisualizationTool:
    """Provides visualization data for interval optimization."""
    
    @staticmethod
    def generate_system_overview() -> Dict[str, Any]:
        """Generate system-wide overview visualization data.
        
        Returns:
            Dictionary with visualization data
        """
        # Get system statistics
        stats = ADVANCED_INTERVAL_MANAGER.get_statistics()
        
        # Create visualization data
        visualization = {
            "system_load": {
                "current": stats.get("current_load", 0),
                "target": stats.get("target_utilization", 70),
                "color": _get_load_color(stats.get("current_load", 0)),
            },
            "interval_distribution": stats.get("interval_distribution", {}),
            "efficiency": stats.get("efficiency", 0),
            "total_entities": stats.get("entity_count", 0),
            "total_polls": stats.get("total_polls", 0),
            "total_changes": stats.get("total_changes", 0),
        }
        
        # Add donut chart data for intervals
        interval_dist = stats.get("interval_distribution", {}).get("count_by_range", {})
        visualization["interval_donut_chart"] = {
            "labels": list(interval_dist.keys()),
            "data": list(interval_dist.values()),
            "colors": [
                "#28a745",  # 0-30s - Green
                "#17a2b8",  # 31-60s - Blue
                "#ffc107",  # 61-120s - Yellow
                "#fd7e14",  # 121-300s - Orange
                "#dc3545"   # 301s+ - Red
            ]
        }
        
        # Add time series placeholder
        # In a real implementation, this would use historical data
        visualization["load_history"] = {
            "labels": ["-60m", "-50m", "-40m", "-30m", "-20m", "-10m", "now"],
            "data": [65, 60, 70, 75, 65, 70, stats.get("current_load", 0)]
        }
        
        return visualization
    
    @staticmethod
    def generate_entity_visualization(entity_id: str) -> Dict[str, Any]:
        """Generate visualization data for a specific entity.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Dictionary with visualization data
        """
        # Get entity statistics
        stats = ADVANCED_INTERVAL_MANAGER.get_entity_statistics(entity_id)
        if not stats:
            return {"error": "Entity not found"}
            
        # Calculate interval change trend
        current = stats.get("current_interval", 0)
        recommended = stats.get("recommended_interval", 0)
        dynamic = stats.get("dynamic_interval", 0)
        
        if recommended > current * 1.2:
            trend = "increasing"
            trend_color = "#dc3545"  # Red
        elif recommended < current * 0.8:
            trend = "decreasing"
            trend_color = "#28a745"  # Green
        else:
            trend = "stable"
            trend_color = "#17a2b8"  # Blue
            
        # Create visualization data
        visualization = {
            "intervals": {
                "current": current,
                "recommended": recommended,
                "dynamic": dynamic,
                "trend": trend,
                "trend_color": trend_color
            },
            "polling": {
                "total_polls": stats.get("poll_count", 0),
                "change_count": stats.get("change_count", 0),
                "change_frequency": stats.get("change_frequency", 0),
                "efficiency": stats.get("change_frequency", 0),  # Same as change frequency
                "next_poll_in": stats.get("next_poll_in", 0),
            },
            "importance": stats.get("importance", 5.0),
            "register_count": stats.get("register_count", 1),
            "cluster": stats.get("cluster_id"),
        }
        
        # Add pattern information if available
        if stats.get("pattern_interval"):
            visualization["pattern"] = {
                "interval": stats.get("pattern_interval"),
                "influence": _calculate_pattern_influence(current, stats.get("pattern_interval", 0))
            }
            
        return visualization
    
    @staticmethod
    def generate_cluster_visualization(cluster_id: str) -> Dict[str, Any]:
        """Generate visualization data for a cluster.
        
        Args:
            cluster_id: Cluster identifier
            
        Returns:
            Dictionary with visualization data
        """
        # Get all entities in this cluster
        entities_in_cluster = []
        cluster_entities = {}
        
        for entity_id, entity_data in ADVANCED_INTERVAL_MANAGER._entities.items():
            if entity_data.cluster_id == cluster_id:
                entities_in_cluster.append(entity_id)
                cluster_entities[entity_id] = ADVANCED_INTERVAL_MANAGER.get_entity_statistics(entity_id)
        
        if not entities_in_cluster:
            return {"error": "Cluster not found or empty"}
            
        # Calculate average interval and variance
        intervals = [e.get("dynamic_interval", 0) for e in cluster_entities.values()]
        avg_interval = sum(intervals) / len(intervals) if intervals else 0
        
        # Calculate variance to see how well coordinated the polling is
        variance = sum((i - avg_interval)**2 for i in intervals) / len(intervals) if intervals else 0
        
        # Determine if polling is well-coordinated
        if variance < (avg_interval * 0.1)**2:  # Within 10% of average
            coordination = "high"
        elif variance < (avg_interval * 0.3)**2:  # Within 30% of average
            coordination = "medium"
        else:
            coordination = "low"
            
        # Get poll schedule
        poll_schedule = []
        now = time.time()
        for entity_id, stats in cluster_entities.items():
            next_poll = stats.get("next_poll_in", 0)
            poll_schedule.append({
                "entity_id": entity_id,
                "next_poll_in": next_poll,
                "next_poll_at": now + next_poll
            })
            
        # Sort by next poll time
        poll_schedule.sort(key=lambda x: x["next_poll_in"])
        
        return {
            "cluster_id": cluster_id,
            "entity_count": len(entities_in_cluster),
            "entities": entities_in_cluster,
            "avg_interval": round(avg_interval, 1),
            "interval_variance": round(variance, 1),
            "coordination_level": coordination,
            "poll_schedule": poll_schedule[:5],  # First 5 scheduled polls
            "poll_distribution": {
                "0-30s": len([i for i in intervals if i <= 30]),
                "31-60s": len([i for i in intervals if 30 < i <= 60]),
                "61-120s": len([i for i in intervals if 60 < i <= 120]),
                "121-300s": len([i for i in intervals if 120 < i <= 300]),
                "301s+": len([i for i in intervals if i > 300])
            }
        }
    
    @staticmethod
    def export_visualization_data() -> str:
        """Export all visualization data as JSON.
        
        Returns:
            JSON string with visualization data
        """
        # Get system overview
        system_data = IntervalVisualizationTool.generate_system_overview()
        
        # Get data for each entity
        entity_data = {}
        for entity_id in ADVANCED_INTERVAL_MANAGER._entities:
            entity_data[entity_id] = IntervalVisualizationTool.generate_entity_visualization(entity_id)
            
        # Get cluster data
        cluster_data = {}
        cluster_ids = set()
        for entity_data in ADVANCED_INTERVAL_MANAGER._entities.values():
            if entity_data.cluster_id:
                cluster_ids.add(entity_data.cluster_id)
                
        for cluster_id in cluster_ids:
            cluster_data[cluster_id] = IntervalVisualizationTool.generate_cluster_visualization(cluster_id)
            
        # Compile all data
        export_data = {
            "system": system_data,
            "entities": entity_data,
            "clusters": cluster_data,
            "timestamp": time.time(),
            "generated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        
        # Convert to JSON
        return json.dumps(export_data, indent=2)


def _get_load_color(load: float) -> str:
    """Get color based on system load.
    
    Args:
        load: System load percentage
        
    Returns:
        Color hex code
    """
    if load < 50:
        return "#28a745"  # Green
    elif load < 70:
        return "#17a2b8"  # Blue
    elif load < 85:
        return "#ffc107"  # Yellow
    elif load < 95:
        return "#fd7e14"  # Orange
    else:
        return "#dc3545"  # Red
        
def _calculate_pattern_influence(current: int, pattern: int) -> float:
    """Calculate pattern influence on interval.
    
    Args:
        current: Current interval
        pattern: Pattern-based interval
        
    Returns:
        Influence percentage
    """
    if current == 0:
        return 0.0
        
    difference = abs(pattern - current)
    
    if difference < 0.1 * current:
        return 0.0  # Less than 10% difference
    
    # Calculate influence as percentage
    return min(100.0, (difference / current) * 100.0)