"""
Advanced adaptive interval management system.

This module provides an enhanced system for determining optimal scan intervals
that leverages patterns, correlations, and system-wide optimizations.
"""

import time
import logging
import heapq
import math
from typing import Dict, List, Set, Tuple, Any, Optional
from datetime import datetime

from .interval_manager import INTERVAL_MANAGER
from .pattern_detection import PatternDetector
from .correlation import EntityCorrelationManager
from .storage_aware import StorageAware

_LOGGER = logging.getLogger(__name__)

# Constants for interval management
MIN_SCAN_INTERVAL = 5  # Minimum scan interval in seconds
MAX_SCAN_INTERVAL = 600  # Maximum scan interval in seconds (10 minutes)
DEFAULT_SCAN_INTERVAL = 30  # Default scan interval

# Load management constants
MAX_REGISTERS_PER_SECOND = 100  # Maximum register polls per second
LOAD_CALCULATION_WINDOW = 60  # Window for calculating system load (seconds)
TARGET_SYSTEM_UTILIZATION = 0.7  # Target system utilization (70%)

# Pattern and prediction constants
PATTERN_CONFIDENCE_THRESHOLD = 70  # Minimum confidence to use pattern-based intervals
TRANSITION_WINDOW_BUFFER = 60  # Buffer window (seconds) around predicted transitions
HIGH_ACTIVITY_INTERVAL = 10  # Interval during high activity periods

class EntityScanData:
    """Stores scan data for a single entity."""
    
    def __init__(self, entity_id: str, initial_interval: int = DEFAULT_SCAN_INTERVAL,
                register_count: int = 1):
        """Initialize entity scan data.
        
        Args:
            entity_id: Entity identifier
            initial_interval: Initial scan interval in seconds
            register_count: Number of registers this entity reads
        """
        self.entity_id = entity_id
        self.current_interval = initial_interval
        self.recommended_interval = initial_interval
        self.dynamic_interval = initial_interval  # What's actually being used
        self.register_count = max(1, register_count)
        self.last_change_time = 0.0
        self.last_poll_time = 0.0
        self.change_count = 0
        self.poll_count = 0
        self.importance = 5.0  # Default importance on scale 1-10
        self.next_poll_due = 0.0
        self.cluster_id = None
        self.pattern_interval = None  # Pattern-based recommended interval
        self.correlation_factor = 1.0  # Multiplier based on correlation insights
        self.value_history = []  # Recent values
        
    # update method to store values:
    
    def update(self, timestamp: float, value_changed: bool, value: Any = None) -> None:
        """Update entity scan data with a new poll result."""
        self.poll_count += 1
        self.last_poll_time = timestamp
        
        if value_changed:
            self.change_count += 1
            self.last_change_time = timestamp
            
        if value is not None:
            self.value_history.append((timestamp, value))
            # Keep only the most recent values
            if len(self.value_history) > 10:
                self.value_history.pop(0)

    # Correlation adjustment method to AdvancedIntervalManager:
    
    def _adjust_for_correlations(self, base_interval: int, entity_id: str, timestamp: float) -> int:
        """Adjust interval based on correlation insights.
        
        Args:
            base_interval: Base interval in seconds
            entity_id: Entity identifier
            timestamp: Current timestamp
            
        Returns:
            Correlation-adjusted interval in seconds
        """
        if not self._correlation_manager:
            return base_interval
            
        # Get cluster for this entity
        cluster_id = self._correlation_manager.get_cluster_for_entity(entity_id)
        if not cluster_id:
            return base_interval
            
        # Store cluster ID for future reference
        if entity_id in self._entities:
            self._entities[entity_id].cluster_id = cluster_id
            
        # Get all entities in this cluster
        cluster = self._correlation_manager.get_clusters().get(cluster_id, [])
        if len(cluster) <= 1:
            return base_interval
            
        # Calculate average interval of recently changed entities in the cluster
        recently_changed = []
        for other_id in cluster:
            if other_id == entity_id:
                continue
                
            if other_id in self._entities:
                other_data = self._entities[other_id]
                
                # Check if recently changed (within 5 minutes)
                if timestamp - other_data.last_change_time < 300:
                    recently_changed.append(other_data.dynamic_interval)
        
        if recently_changed:
            # If cluster members are changing, sync intervals
            avg_interval = sum(recently_changed) / len(recently_changed)
            
            # Blend with base interval - 70% base, 30% cluster influence
            return int(0.7 * base_interval + 0.3 * avg_interval)
            
        # No recent changes in cluster, stick with base interval
        return base_interval
    
    def get_change_frequency(self) -> float:
        """Get the frequency of changes (changes per poll)."""
        if self.poll_count == 0:
            return 0.0
        return self.change_count / self.poll_count
    
    def __lt__(self, other):
        """Comparison for priority queue based on next poll time."""
        # This is used by the priority queue to determine poll order
        return self.next_poll_due < other.next_poll_due


class AdvancedIntervalManager(StorageAware):
    """Advanced interval management system with pattern and correlation awareness."""
    
    def __init__(self):
        """Initialize advanced interval manager."""
        super().__init__("advanced_interval_manager")
        
        # Core entity data
        self._entities: Dict[str, EntityScanData] = {}
        self._poll_queue = []  # Priority queue of entities to poll
        self._last_maintenance = 0.0
        
        # System-wide settings and limits
        self.min_interval = MIN_SCAN_INTERVAL
        self.max_interval = MAX_SCAN_INTERVAL
        self.max_registers_per_second = MAX_REGISTERS_PER_SECOND
        self.target_utilization = TARGET_SYSTEM_UTILIZATION
        
        # Components for enhanced interval management
        self._pattern_detector = None
        self._correlation_manager = None
        
        # Performance metrics
        self._total_polls = 0
        self._total_changes = 0
        self._poll_timestamps = []  # Recent poll timestamps for load calculation
        self._register_count_history = []  # Recent register counts for load calculation
        
        # Load data from storage
        self._load_from_storage()
    
    def set_components(self, pattern_detector: PatternDetector, 
                      correlation_manager: EntityCorrelationManager) -> None:
        """Set components for enhanced interval management."""
        self._pattern_detector = pattern_detector
        self._correlation_manager = correlation_manager
        _LOGGER.info("Advanced interval manager connected with pattern and correlation components")
    
    def register_entity(self, entity_id: str, initial_interval: int = DEFAULT_SCAN_INTERVAL,
                       register_count: int = 1, importance: float = 5.0) -> None:
        """Register an entity for interval management."""
        # Cap initial interval to min/max range
        initial_interval = max(self.min_interval, min(self.max_interval, initial_interval))
        
        # Create or update entity data
        if entity_id not in self._entities:
            entity_data = EntityScanData(entity_id, initial_interval, register_count)
            entity_data.importance = max(1.0, min(10.0, importance))
            entity_data.next_poll_due = time.time()  # Due immediately
            self._entities[entity_id] = entity_data
            
            # Add to poll queue
            heapq.heappush(self._poll_queue, entity_data)
        else:
            # Update existing entity
            entity_data = self._entities[entity_id]
            entity_data.current_interval = initial_interval
            entity_data.register_count = register_count
            entity_data.importance = max(1.0, min(10.0, importance))
            
        _LOGGER.debug("Registered entity %s with interval %d and %d registers",
                    entity_id, initial_interval, register_count)
    
    def record_poll(self, entity_id: str, timestamp: float, value_changed: bool,
                   value: Any = None, error: bool = False, response_time: float = None) -> None:
        """Record a polling event for an entity."""
        if entity_id not in self._entities:
            # Auto-register with default settings
            self.register_entity(entity_id)
            
        entity_data = self._entities[entity_id]
        
        if not error:
            # Update entity data
            entity_data.update(timestamp, value_changed, value)
            
            # Schedule next poll
            entity_data.next_poll_due = timestamp + entity_data.dynamic_interval
            
            # Remove and re-insert with updated priority
            self._rebuild_poll_queue()
            
            # Update system metrics
            self._total_polls += 1
            if value_changed:
                self._total_changes += 1
                
            # Record poll timestamp and register count for load calculation
            self._poll_timestamps.append(timestamp)
            self._register_count_history.append(entity_data.register_count)
            
            # Keep history within a reasonable size
            if len(self._poll_timestamps) > 1000:
                self._poll_timestamps = self._poll_timestamps[-1000:]
                self._register_count_history = self._register_count_history[-1000:]
            
            # Run periodic maintenance
            if timestamp - self._last_maintenance >= 300:  # Every 5 minutes
                self._perform_maintenance(timestamp)
    
    def get_next_entity_to_poll(self, current_time: float = None) -> Optional[str]:
        """Get the next entity that should be polled."""
        if current_time is None:
            current_time = time.time()
            
        if not self._poll_queue:
            # Rebuild queue if empty
            self._rebuild_poll_queue()
            if not self._poll_queue:
                return None
                
        # Check if top entity is due
        top_entity = self._poll_queue[0]
        if top_entity.next_poll_due <= current_time:
            # Mark as polled
            entity_id = top_entity.entity_id
            top_entity.next_poll_due = current_time + top_entity.dynamic_interval
            
            # Re-prioritize
            self._rebuild_poll_queue()
            return entity_id
        
        return None
    
    def _rebuild_poll_queue(self) -> None:
        """Rebuild the priority queue of entities to poll."""
        self._poll_queue = list(self._entities.values())
        heapq.heapify(self._poll_queue)

    # Include correlation adjustments:
    
    def get_recommended_interval(self, entity_id: str, timestamp: float = None,
                               force_recalculate: bool = False) -> int:
        """Calculate optimal scan interval for an entity.
        
        This enhanced version considers:
        1. Entity change frequency
        2. System load
        3. Pattern detection insights
        4. Correlation insights
        5. Entity importance
        
        Args:
            entity_id: Entity identifier
            timestamp: Current timestamp (defaults to now)
            force_recalculate: Force recalculation even if recently calculated
            
        Returns:
            Recommended scan interval in seconds
        """
        if timestamp is None:
            timestamp = time.time()
            
        if entity_id not in self._entities:
            return DEFAULT_SCAN_INTERVAL
            
        entity_data = self._entities[entity_id]
        
        # Start with base interval based on change frequency
        base_interval = self._calculate_base_interval(entity_data, timestamp)
        
        # Adjust based on system load
        load_adjusted_interval = self._adjust_for_system_load(base_interval, entity_data)
        
        # Adjust based on patterns if available
        pattern_adjusted_interval = load_adjusted_interval
        if self._pattern_detector:
            pattern_adjusted_interval = self._adjust_for_patterns(
                load_adjusted_interval, entity_id, timestamp)
            
            # Store pattern-recommended interval
            entity_data.pattern_interval = pattern_adjusted_interval
            
        # Adjust based on correlations if available
        correlation_adjusted_interval = pattern_adjusted_interval
        if self._correlation_manager:
            correlation_adjusted_interval = self._adjust_for_correlations(
                pattern_adjusted_interval, entity_id, timestamp)
        
        # Ensure within limits
        final_interval = max(self.min_interval, min(self.max_interval, correlation_adjusted_interval))
        
        # Update recommended interval
        entity_data.recommended_interval = final_interval
        
        # Apply to dynamic interval with smoothing to avoid rapid changes
        entity_data.dynamic_interval = self._smooth_interval_change(
            entity_data.dynamic_interval, final_interval)
            
        return final_interval
        
    # Add the pattern adjustment method:
    
    def _adjust_for_patterns(self, base_interval: int, entity_id: str, timestamp: float) -> int:
        """Adjust interval based on pattern detection insights.
        
        Args:
            base_interval: Base interval in seconds
            entity_id: Entity identifier
            timestamp: Current timestamp
            
        Returns:
            Pattern-adjusted interval in seconds
        """
        if not self._pattern_detector:
            return base_interval
            
        # Get current pattern info
        pattern_info = self._pattern_detector.get_current_pattern_info()
        current_pattern = pattern_info.get("pattern_id")
        pattern_confidence = pattern_info.get("pattern_confidence", 0)
        
        # If no pattern or low confidence, use base interval
        if current_pattern is None or pattern_confidence < PATTERN_CONFIDENCE_THRESHOLD:
            return base_interval
            
        # Get optimal interval for this entity in this pattern
        pattern_interval = self._pattern_detector.get_optimal_scan_interval(entity_id)
        if pattern_interval is None:
            return base_interval
            
        # Check if we're in a transition period
        in_transition = self._pattern_detector.is_transition_period()
        
        if in_transition:
            # Use more frequent polling during transitions
            return min(base_interval, pattern_interval)
        
        # Check if near predicted pattern change
        predicted_end = pattern_info.get("predicted_end")
        if predicted_end:
            try:
                # Convert ISO timestamp to epoch seconds
                predicted_end_time = datetime.fromisoformat(predicted_end).timestamp()
                
                # If we're approaching predicted end, decrease interval
                time_to_end = predicted_end_time - timestamp
                if 0 < time_to_end < TRANSITION_WINDOW_BUFFER:
                    # Near transition, poll more frequently
                    return min(base_interval, pattern_interval, HIGH_ACTIVITY_INTERVAL)
            except (ValueError, TypeError):
                pass
                
        # Blend base interval with pattern-based interval
        # Higher confidence gives more weight to pattern-based interval
        confidence_factor = pattern_confidence / 100.0
        blended_interval = int(
            (base_interval * (1 - confidence_factor)) +
            (pattern_interval * confidence_factor)
        )
        
        return blended_interval
    
    def _calculate_base_interval(self, entity_data: EntityScanData, timestamp: float) -> int:
        """Calculate base interval based on change frequency.
        
        Args:
            entity_data: Entity scan data
            timestamp: Current timestamp
            
        Returns:
            Base interval in seconds
        """
        if entity_data.poll_count < 10:
            # Not enough data yet, use current interval
            return entity_data.current_interval
            
        change_frequency = entity_data.get_change_frequency()
        
        if change_frequency == 0:
            # No changes detected yet
            time_since_last_poll = timestamp - entity_data.last_poll_time
            
            # Gradually increase interval for entities that don't change
            if time_since_last_poll > 3600:  # More than an hour
                return min(self.max_interval, entity_data.current_interval * 2)
            elif time_since_last_poll > 1800:  # More than 30 minutes
                return min(self.max_interval, int(entity_data.current_interval * 1.5))
            else:
                return min(self.max_interval, entity_data.current_interval + 30)
        
        # Calculate interval based on change frequency
        if change_frequency > 0.5:  # Changes on more than 50% of polls
            # Changes very frequently, poll more often
            return max(self.min_interval, min(30, entity_data.current_interval // 2))
        elif change_frequency > 0.2:  # Changes on 20-50% of polls
            # Changes somewhat frequently
            return max(self.min_interval, min(60, entity_data.current_interval))
        elif change_frequency > 0.1:  # Changes on 10-20% of polls
            # Changes occasionally
            return max(30, min(120, entity_data.current_interval))
        elif change_frequency > 0.05:  # Changes on 5-10% of polls
            # Changes rarely
            return max(60, min(300, int(entity_data.current_interval * 1.5)))
        else:  # Changes on less than 5% of polls
            # Changes very rarely
            return max(120, min(self.max_interval, entity_data.current_interval * 2))
    
    def _adjust_for_system_load(self, base_interval: int, entity_data: EntityScanData) -> int:
        """Adjust interval based on system load and entity importance.
        
        Args:
            base_interval: Base interval in seconds
            entity_data: Entity scan data
            
        Returns:
            Load-adjusted interval in seconds
        """
        # Get current system load
        current_load = self._calculate_system_load()
        
        # If system load is below target, no adjustment needed
        if current_load <= self.target_utilization:
            return base_interval
            
        # System is overloaded, adjust intervals based on entity importance
        load_factor = current_load / self.target_utilization
        
        # More important entities get less adjustment
        importance_factor = entity_data.importance / 10.0  # Scale to 0.1-1.0
        
        # Calculate adjustment factor (higher importance = lower adjustment)
        adjustment_factor = load_factor * (2.0 - importance_factor)
        
        # Apply adjustment (increase interval to reduce load)
        adjusted_interval = int(base_interval * adjustment_factor)
        
        # Ensure within limits
        return min(self.max_interval, adjusted_interval)
    
    def _smooth_interval_change(self, current_interval: int, target_interval: int) -> int:
        """Smooth interval changes to avoid rapid fluctuations.
        
        Args:
            current_interval: Current interval in seconds
            target_interval: Target interval in seconds
            
        Returns:
            Smoothed interval in seconds
        """
        # Limit change to max 50% increase or 30% decrease at a time
        if target_interval > current_interval:
            # Increasing interval
            return min(target_interval, int(current_interval * 1.5))
        else:
            # Decreasing interval
            return max(target_interval, int(current_interval * 0.7))
    
    def _perform_maintenance(self, timestamp: float) -> None:
        """Perform periodic maintenance tasks.
        
        Args:
            timestamp: Current timestamp
        """
        self._last_maintenance = timestamp
        
        # Recalculate all entity intervals
        for entity_id in self._entities:
            self.get_recommended_interval(entity_id, timestamp, True)
            
        # Update group-based scan optimization
        self._optimize_cluster_scanning()
            
        # Clean up stale poll timestamps
        self._poll_timestamps = [t for t in self._poll_timestamps 
                               if timestamp - t < 3600]  # Keep last hour
        
        # Save data to storage
        self._save_to_storage()
        
        _LOGGER.debug("Completed interval maintenance with %d entities", 
                    len(self._entities))
    
    def _calculate_system_load(self) -> float:
        """Calculate current system load as a ratio of capacity."""
        # Calculate recent polls per second
        now = time.time()
        recent_timestamps = [t for t in self._poll_timestamps 
                            if now - t < LOAD_CALCULATION_WINDOW]
                            
        if not recent_timestamps or len(recent_timestamps) < 2:
            return 0.0
            
        # Calculate time window
        window_size = max(1.0, now - min(recent_timestamps))
        
        # Calculate registers per second
        registers_in_window = sum(self._register_count_history[-len(recent_timestamps):])
        registers_per_second = registers_in_window / window_size
        
        # Calculate load as a ratio of capacity
        return registers_per_second / self.max_registers_per_second
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about interval management."""
        now = time.time()
        
        # Calculate system load
        current_load = self._calculate_system_load()
        
        # Calculate interval distribution
        intervals = [e.dynamic_interval for e in self._entities.values()]
        avg_interval = sum(intervals) / len(intervals) if intervals else 0
        
        interval_distribution = {
            "min": min(intervals) if intervals else 0,
            "max": max(intervals) if intervals else 0,
            "avg": avg_interval,
            "count_by_range": {
                "0-30s": len([i for i in intervals if i <= 30]),
                "31-60s": len([i for i in intervals if 30 < i <= 60]),
                "61-120s": len([i for i in intervals if 60 < i <= 120]),
                "121-300s": len([i for i in intervals if 120 < i <= 300]),
                "301s+": len([i for i in intervals if i > 300])
            }
        }
        
        # Calculate efficiency
        efficiency = 0
        if self._total_polls > 0:
            efficiency = (self._total_changes / self._total_polls) * 100
            
        # Calculate cluster statistics
        clusters_with_entities = set()
        for entity_data in self._entities.values():
            if entity_data.cluster_id:
                clusters_with_entities.add(entity_data.cluster_id)
                
        return {
            "entity_count": len(self._entities),
            "total_polls": self._total_polls,
            "total_changes": self._total_changes,
            "efficiency": round(efficiency, 1),
            "current_load": round(current_load * 100, 1),
            "target_utilization": round(self.target_utilization * 100, 1),
            "registers_per_second_limit": self.max_registers_per_second,
            "interval_distribution": interval_distribution,
            "active_clusters": len(clusters_with_entities),
            "pattern_integration": self._pattern_detector is not None,
            "correlation_integration": self._correlation_manager is not None,
            "last_maintenance": int(now - self._last_maintenance)
        }
    
    # Update the get_entity_statistics method to include pattern information:
    
    def get_entity_statistics(self, entity_id: str) -> Dict[str, Any]:
        """Get statistics for a specific entity."""
        if entity_id not in self._entities:
            return {}
            
        entity_data = self._entities[entity_id]
        
        # Calculate time to next poll
        now = time.time()
        time_to_next_poll = max(0, entity_data.next_poll_due - now)
        
        return {
            "current_interval": entity_data.current_interval,
            "recommended_interval": entity_data.recommended_interval,
            "dynamic_interval": entity_data.dynamic_interval,
            "register_count": entity_data.register_count,
            "importance": entity_data.importance,
            "poll_count": entity_data.poll_count,
            "change_count": entity_data.change_count,
            "change_frequency": round(entity_data.get_change_frequency() * 100, 1),
            "last_poll_age": round(now - entity_data.last_poll_time) if entity_data.last_poll_time else None,
            "last_change_age": round(now - entity_data.last_change_time) if entity_data.last_change_time else None,
            "next_poll_in": round(time_to_next_poll),
            "cluster_id": entity_data.cluster_id,
            "pattern_interval": entity_data.pattern_interval
        }
    
    def _load_from_storage(self) -> None:
        """Load interval management data from storage."""
        if not self._storage_manager:
            return
            
        try:
            data = self.get_metadata_file("advanced_interval_data")
            if not data:
                return
                
            # Load system settings
            if "settings" in data:
                settings = data["settings"]
                if "min_interval" in settings:
                    self.min_interval = settings["min_interval"]
                if "max_interval" in settings:
                    self.max_interval = settings["max_interval"]
                if "max_registers_per_second" in settings:
                    self.max_registers_per_second = settings["max_registers_per_second"]
                if "target_utilization" in settings:
                    self.target_utilization = settings["target_utilization"]
                    
            # Load entity data
            if "entities" in data:
                for entity_id, entity_info in data["entities"].items():
                    if "current_interval" in entity_info:
                        initial_interval = entity_info["current_interval"]
                        register_count = entity_info.get("register_count", 1)
                        importance = entity_info.get("importance", 5.0)
                        
                        self.register_entity(
                            entity_id, 
                            initial_interval=initial_interval,
                            register_count=register_count,
                            importance=importance
                        )
                        
                        # Set additional fields if available
                        if entity_id in self._entities:
                            entity_data = self._entities[entity_id]
                            entity_data.cluster_id = entity_info.get("cluster_id")
                            entity_data.dynamic_interval = entity_info.get(
                                "dynamic_interval", initial_interval)
                
            _LOGGER.info("Loaded interval management data with %d entities", 
                       len(self._entities))
        except Exception as e:
            _LOGGER.error("Failed to load interval management data: %s", e)
    
    def _save_to_storage(self) -> bool:
        """Save interval management data to storage."""
        if not self._storage_manager:
            return False
            
        try:
            # Prepare entity data
            entity_data = {}
            for entity_id, entity in self._entities.items():
                entity_data[entity_id] = {
                    "current_interval": entity.current_interval,
                    "dynamic_interval": entity.dynamic_interval,
                    "recommended_interval": entity.recommended_interval,
                    "register_count": entity.register_count,
                    "importance": entity.importance,
                    "cluster_id": entity.cluster_id,
                }
                
            # Prepare settings
            settings = {
                "min_interval": self.min_interval,
                "max_interval": self.max_interval,
                "max_registers_per_second": self.max_registers_per_second,
                "target_utilization": self.target_utilization,
            }
                
            # Prepare full data structure
            data = {
                "settings": settings,
                "entities": entity_data,
                "meta": {
                    "version": "1.0",
                    "timestamp": datetime.utcnow().isoformat(),
                    "entity_count": len(self._entities)
                }
            }
            
            success = self.save_metadata_file("advanced_interval_data", data)
            return success
        except Exception as e:
            _LOGGER.error("Failed to save interval management data: %s", e)
            return False

    # Add the update_clusters method to AdvancedIntervalManager:
    
    def update_clusters(self, clusters: Dict[str, List[str]]) -> None:
        """Update cluster information for interval optimization.
        
        Args:
            clusters: Dictionary mapping cluster IDs to lists of entity IDs
        """
        # Update cluster IDs for all entities
        for cluster_id, entity_ids in clusters.items():
            for entity_id in entity_ids:
                if entity_id in self._entities:
                    self._entities[entity_id].cluster_id = cluster_id
                    
        # Run optimization for the new clusters
        self._optimize_cluster_scanning()
    
    # Add cluster optimization methods to AdvancedIntervalManager:
    
    def _optimize_cluster_scanning(self) -> None:
        """Optimize scanning within clusters to improve efficiency."""
        if not self._correlation_manager:
            return
            
        # Group entities by cluster
        entities_by_cluster = {}
        for entity_id, entity_data in self._entities.items():
            cluster_id = entity_data.cluster_id
            if not cluster_id:
                cluster_id = self._correlation_manager.get_cluster_for_entity(entity_id)
                entity_data.cluster_id = cluster_id
                
            if cluster_id:
                if cluster_id not in entities_by_cluster:
                    entities_by_cluster[cluster_id] = []
                entities_by_cluster[cluster_id].append(entity_data)
        
        # Optimize intervals within each cluster
        for cluster_id, entities in entities_by_cluster.items():
            if len(entities) <= 1:
                continue
                
            # Calculate optimal staggering for entities in the same cluster
            self._optimize_cluster_intervals(cluster_id, entities)
    
    def _optimize_cluster_intervals(self, cluster_id: str, entities: List[EntityScanData]) -> None:
        """Optimize intervals for entities within a cluster.
        
        Args:
            cluster_id: Cluster identifier
            entities: List of entity data objects in the cluster
        """
        # Sort entities by importance (higher importance first)
        entities.sort(key=lambda e: e.importance, reverse=True)
        
        # Get average interval
        avg_interval = sum(e.recommended_interval for e in entities) / len(entities)
        
        # For entities with similar intervals, stagger their polling times
        # to avoid polling all related entities at once
        similar_entities = [e for e in entities 
                           if 0.7 * avg_interval <= e.recommended_interval <= 1.3 * avg_interval]
        
        if len(similar_entities) > 1:
            # Calculate staggered offsets
            now = time.time()
            stagger_step = avg_interval / len(similar_entities)
            
            for i, entity in enumerate(similar_entities):
                # Set next poll time with staggered offset
                entity.next_poll_due = now + (i * stagger_step)
                
            _LOGGER.debug("Optimized polling for cluster %s with %d entities",
                        cluster_id, len(similar_entities))

# Initialize the advanced interval manager
ADVANCED_INTERVAL_MANAGER = AdvancedIntervalManager()