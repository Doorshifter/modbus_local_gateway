"""
Unified interval management system for Modbus polling optimization.

This module provides intelligent, adaptive scaling of polling intervals based on:
1. Entity behavior patterns and change frequency
2. Pattern-awareness and operational modes
3. System-wide coordination and resource limitations
4. Statistical prediction of future changes
"""

import logging
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from collections import defaultdict
import statistics as stats_lib
import threading

_LOGGER = logging.getLogger(__name__)

# Constants for interval management
DEFAULT_MIN_INTERVAL = 10         # Default minimum scan interval in seconds
DEFAULT_MAX_INTERVAL = 3600       # Default maximum scan interval in seconds (1 hour)
DEFAULT_BASE_INTERVAL = 60        # Default scan interval when no other information
DEFAULT_BACKOFF_FACTOR = 1.5      # Factor for progressive backoff
DEFAULT_MAX_BACKOFF_STEPS = 5     # Maximum number of backoff steps
DEFAULT_STABILITY_THRESHOLD = 3   # Number of consecutive unchanged polls to consider stable
DEFAULT_PATTERN_WEIGHT = 0.7      # How much pattern detection influences intervals
DEFAULT_MAX_REGISTERS_PER_SEC = 100  # Default maximum registers per second


class EntityState:
    """Tracks and manages polling state for an individual entity."""
    
    def __init__(
        self, 
        entity_id: str, 
        initial_interval: int = DEFAULT_BASE_INTERVAL,
        register_count: int = 1
    ):
        """Initialize entity polling state.
        
        Args:
            entity_id: Unique entity identifier
            initial_interval: Starting scan interval in seconds
            register_count: Number of registers this entity represents
        """
        self.entity_id = entity_id
        self.base_interval = initial_interval
        self.current_interval = initial_interval
        self.recommended_interval = initial_interval
        self.register_count = register_count
        
        # Change tracking
        self.last_change_timestamp: float = 0
        self.last_poll_timestamp: float = 0
        self.last_value: Any = None
        self.change_intervals: List[float] = []  # Time between changes
        self.consecutive_no_changes: int = 0
        self.change_count: int = 0
        self.poll_count: int = 0
        self.error_count: int = 0
        
        # Adaptive state
        self.backoff_step: int = 0
        self.pattern_specific_intervals: Dict[int, int] = {}  # pattern_id -> interval
        self.adaptive_metrics: Dict[str, float] = {
            "stability_score": 0.5,    # 0-1, higher = more stable
            "priority_score": 0.5,     # 0-1, higher = more important
            "variance": 0.0,           # Variance in values
            "trend_direction": 0.0,    # Positive = increasing, negative = decreasing
            "error_rate": 0.0,         # Error rate (0-1)
            "change_rate": 0.0,        # Changes per poll (0-1)
            "prediction_quality": 0.5, # Quality of predictions (0-1)
        }
        
        # Value history and predictions
        self.last_values: List[float] = []  # Recent numeric values
        self.predicted_next_change: Optional[float] = None
        self.predicted_change_confidence: float = 0.0
        
        # Status tracking
        self.last_optimization_time: float = 0
    
    def record_poll(
        self, 
        timestamp: float, 
        value_changed: bool, 
        value: Any = None, 
        error: bool = False
    ) -> None:
        """Record a poll and update state.
        
        Args:
            timestamp: Time when poll occurred
            value_changed: Whether the value changed
            value: The polled value (if available)
            error: Whether an error occurred during polling
        """
        self.last_poll_timestamp = timestamp
        self.poll_count += 1
        
        if error:
            self.error_count += 1
            self.adaptive_metrics["error_rate"] = self.error_count / self.poll_count
            return
        
        # Track numeric values for trend analysis
        if value is not None and self.last_value != value:
            try:
                numeric_value = float(value)
                if len(self.last_values) >= 10:
                    self.last_values.pop(0)
                self.last_values.append(numeric_value)
                
                # Update variance and trend if we have enough values
                if len(self.last_values) >= 3:
                    self._update_value_metrics()
                    
                self.last_value = value
                    
            except (ValueError, TypeError):
                # Non-numeric values are tracked but not analyzed
                self.last_value = value
                pass
        
        if value_changed:
            self.change_count += 1
            
            # Calculate time since last change
            if self.last_change_timestamp > 0:
                change_interval = timestamp - self.last_change_timestamp
                self.change_intervals.append(change_interval)
                
                # Keep history bounded
                if len(self.change_intervals) > 50:  # Keep last 50 intervals
                    self.change_intervals.pop(0)
            
            self.last_change_timestamp = timestamp
            self.consecutive_no_changes = 0
            self.backoff_step = 0  # Reset backoff on change
        else:
            self.consecutive_no_changes += 1
        
        # Update adaptive metrics
        self.adaptive_metrics["change_rate"] = self.change_count / max(1, self.poll_count)
        self._update_stability_score()
    
    def _update_value_metrics(self) -> None:
        """Update metrics based on recent values."""
        if len(self.last_values) < 2:
            return
        
        try:
            # Calculate variance
            mean = sum(self.last_values) / len(self.last_values)
            variance = sum((x - mean) ** 2 for x in self.last_values) / len(self.last_values)
            self.adaptive_metrics["variance"] = variance
            
            # Calculate trend direction (positive = increasing, negative = decreasing)
            diffs = [self.last_values[i] - self.last_values[i-1] 
                    for i in range(1, len(self.last_values))]
            avg_diff = sum(diffs) / len(diffs)
            self.adaptive_metrics["trend_direction"] = avg_diff
            
        except (ValueError, ZeroDivisionError):
            pass
    
    def _update_stability_score(self) -> None:
        """Update the stability score based on current metrics."""
        # Combine multiple factors into stability score:
        # 1. Change rate (lower = more stable)
        # 2. Consecutive unchanged polls (higher = more stable)
        # 3. Variance in values (lower = more stable)
        
        # Convert consecutive unchanged to a 0-1 scale
        unchanged_factor = min(1.0, self.consecutive_no_changes / 10)
        
        # Normalize variance to 0-1 scale (inversed, so lower variance = higher score)
        variance_norm = 1.0 / (1.0 + self.adaptive_metrics["variance"])
        
        # Calculate stability as weighted combination
        stability = (
            (1.0 - self.adaptive_metrics["change_rate"]) * 0.5 +
            unchanged_factor * 0.3 +
            variance_norm * 0.2
        )
        
        self.adaptive_metrics["stability_score"] = max(0.0, min(1.0, stability))
        
        # Priority is roughly inverse of stability
        self.adaptive_metrics["priority_score"] = max(0.0, min(1.0, 
            (1.0 - self.adaptive_metrics["stability_score"]) * 0.8 +
            self.adaptive_metrics["error_rate"] * 0.2
        ))
    
    def update_prediction(
        self, 
        predicted_timestamp: Optional[float], 
        confidence: float
    ) -> None:
        """Update prediction information.
        
        Args:
            predicted_timestamp: Predicted time of next change
            confidence: Confidence level (0-1) in the prediction
        """
        self.predicted_next_change = predicted_timestamp
        self.predicted_change_confidence = confidence
        self.adaptive_metrics["prediction_quality"] = confidence
    
    def get_recommended_interval(
        self, 
        current_time: float,
        min_interval: int,
        max_interval: int,
        backoff_factor: float,
        pattern_id: Optional[int] = None,
        force_recalculate: bool = False
    ) -> int:
        """Calculate recommended polling interval for this entity.
        
        Args:
            current_time: Current timestamp
            min_interval: Minimum allowed interval
            max_interval: Maximum allowed interval
            backoff_factor: Factor for interval backoff
            pattern_id: Current system pattern ID (if applicable)
            force_recalculate: Whether to force recalculation
            
        Returns:
            Recommended polling interval in seconds
        """
        # Skip recalculation if recent and not forced
        if not force_recalculate and current_time - self.last_optimization_time < 300:  # 5 minutes
            return self.recommended_interval
            
        self.last_optimization_time = current_time
        
        # Start with base interval
        interval = self.base_interval
        
        # Apply pattern-specific interval if available
        if pattern_id is not None and pattern_id in self.pattern_specific_intervals:
            interval = self.pattern_specific_intervals[pattern_id]
        
        # Apply adaptive scaling based on stability
        stability = self.adaptive_metrics["stability_score"]
        
        if self.consecutive_no_changes > DEFAULT_STABILITY_THRESHOLD:
            # Apply progressive backoff for stable entities
            backoff_step = min(self.backoff_step, DEFAULT_MAX_BACKOFF_STEPS)
            backoff_multiplier = backoff_factor ** backoff_step
            interval = min(max_interval, int(interval * backoff_multiplier))
            
            # Increment backoff step for next time if stability continues
            if self.consecutive_no_changes % 3 == 0:  # Every 3 unchanged polls
                self.backoff_step = backoff_step + 1
        
        # Apply prediction-based adjustment if available and confidence is sufficient
        if (self.predicted_next_change and 
            self.predicted_change_confidence > 0.3 and 
            self.predicted_next_change > current_time):
            
            # Calculate time until predicted change
            time_to_change = self.predicted_next_change - current_time
            
            # Aim to poll before the predicted change (80% of time to change)
            predicted_interval = max(min_interval, min(max_interval, 
                                                     int(time_to_change * 0.8)))
            
            # Blend with stability-based interval according to prediction confidence
            confidence_factor = self.predicted_change_confidence
            interval = int(interval * (1 - confidence_factor) + 
                         predicted_interval * confidence_factor)
        
        # Adjust for optimal pattern detection if we have significant variance
        if self.adaptive_metrics["variance"] > 0.1:
            # If we have variance but not many detected changes,
            # we might need more frequent polling to catch transitions
            if self.adaptive_metrics["change_rate"] < 0.2:
                interval = max(min_interval, int(interval * 0.8))
        
        # Ensure interval stays within bounds
        interval = max(min_interval, min(max_interval, interval))
        
        self.recommended_interval = interval
        return interval
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this entity's polling behavior.
        
        Returns:
            Dictionary of statistics
        """
        avg_change_interval = (
            sum(self.change_intervals) / len(self.change_intervals) 
            if self.change_intervals else 0
        )
        
        return {
            "entity_id": self.entity_id,
            "current_interval": self.current_interval,
            "recommended_interval": self.recommended_interval,
            "base_interval": self.base_interval,
            "poll_count": self.poll_count,
            "change_count": self.change_count,
            "change_rate": round(self.adaptive_metrics["change_rate"] * 100, 1),
            "avg_change_interval": round(avg_change_interval, 1),
            "consecutive_no_changes": self.consecutive_no_changes,
            "stability_score": round(self.adaptive_metrics["stability_score"] * 100, 1),
            "priority_score": round(self.adaptive_metrics["priority_score"] * 100, 1),
            "registers": self.register_count,
            "error_rate": round(self.adaptive_metrics["error_rate"] * 100, 1),
            "last_change_timestamp": self.last_change_timestamp,
            "last_poll_timestamp": self.last_poll_timestamp,
            "pattern_specific": bool(self.pattern_specific_intervals)
        }


class IntervalManager:
    """Manages dynamic scaling of entity polling intervals."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance of the interval manager."""
        if cls._instance is None:
            cls._instance = IntervalManager()
        return cls._instance
    
    def __init__(
        self, 
        min_interval: int = DEFAULT_MIN_INTERVAL,
        max_interval: int = DEFAULT_MAX_INTERVAL,
        base_interval: int = DEFAULT_BASE_INTERVAL,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        max_registers_per_second: Optional[int] = DEFAULT_MAX_REGISTERS_PER_SEC
    ):
        """Initialize interval manager.
        
        Args:
            min_interval: Minimum allowed scan interval
            max_interval: Maximum allowed scan interval
            base_interval: Default scan interval when no other information
            backoff_factor: Factor for progressive backoff
            max_registers_per_second: Maximum registers per second limit
        """
        # Basic configuration
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.base_interval = base_interval
        self.backoff_factor = backoff_factor
        self.max_registers_per_second = max_registers_per_second
        
        # Entity tracking
        self.entity_states: Dict[str, EntityState] = {}
        
        # Clustering and grouping
        self.entity_clusters: Dict[str, str] = {}  # entity_id -> cluster_id
        self.cluster_members: Dict[str, Set[str]] = defaultdict(set)  # cluster_id -> entity_ids
        
        # Pattern awareness
        self.current_pattern: Optional[int] = None
        self.pattern_transition_time: Optional[float] = None
        self.pattern_weight = DEFAULT_PATTERN_WEIGHT
        
        # System statistics
        self.last_optimization_time: float = 0
        self.optimization_count: int = 0
        self.optimization_interval: int = 300  # 5 minutes between optimizations
        self.total_polls: int = 0
        self.successful_polls: int = 0
        self.total_changes: int = 0
        self.total_errors: int = 0
        self.register_polls_per_second: float = 0
        self.last_response_times: List[float] = []
        
        # Storage integration
        self._interval_history: Dict[str, List[Dict[str, Any]]] = {}
        self._history_max_size: int = 50
        self._storage = None
        
        # Thread safety
        self._lock = threading.RLock()
    
    def register_entity(
        self,
        entity_id: str,
        register_count: int = 1,
        initial_interval: Optional[int] = None
    ) -> None:
        """Register an entity for interval management.
        
        Args:
            entity_id: Entity ID to register
            register_count: Number of registers this entity represents
            initial_interval: Starting scan interval (uses base_interval if None)
        """
        if initial_interval is None:
            initial_interval = self.base_interval
            
        with self._lock:
            if entity_id in self.entity_states:
                # Update existing entity
                state = self.entity_states[entity_id]
                state.register_count = register_count
                state.base_interval = initial_interval
            else:
                # Create new entity state
                self.entity_states[entity_id] = EntityState(
                    entity_id,
                    initial_interval,
                    register_count
                )
    
    def record_poll(
        self,
        entity_id: str,
        timestamp: float,
        value_changed: bool,
        value: Any = None,
        error: bool = False,
        response_time: Optional[float] = None
    ) -> None:
        """Record a poll event for an entity.
        
        Args:
            entity_id: Entity ID
            timestamp: Time of the poll
            value_changed: Whether the entity's value changed
            value: The polled value (if available)
            error: Whether an error occurred
            response_time: Poll response time in milliseconds (if available)
        """
        if timestamp is None:
            timestamp = time.time()
            
        with self._lock:
            # Create entity state if it doesn't exist
            if entity_id not in self.entity_states:
                self.register_entity(entity_id)
                
            # Update entity state
            state = self.entity_states[entity_id]
            state.record_poll(timestamp, value_changed, value, error)
            
            # Update system statistics
            self.total_polls += 1
            if not error:
                self.successful_polls += 1
            else:
                self.total_errors += 1
                
            if value_changed:
                self.total_changes += 1
                
            # Track response times
            if response_time is not None:
                self.last_response_times.append(response_time)
                if len(self.last_response_times) > 100:  # Keep last 100 times
                    self.last_response_times.pop(0)
    
    def update_clusters(self, clusters: Dict[str, List[str]]) -> None:
        """Update entity clusters for coordinated polling.
        
        Args:
            clusters: Dictionary of cluster_id to list of entity_ids
        """
        with self._lock:
            # Reset cluster assignments
            self.entity_clusters = {}
            self.cluster_members = defaultdict(set)
            
            # Apply new clusters
            for cluster_id, members in clusters.items():
                for entity_id in members:
                    self.entity_clusters[entity_id] = cluster_id
                    self.cluster_members[cluster_id].add(entity_id)
    
    def set_current_pattern(self, pattern_id: Optional[int], transition_time: float) -> None:
        """Set the current system pattern.
        
        Args:
            pattern_id: Current pattern ID
            transition_time: When the transition occurred
        """
        with self._lock:
            # Only record if pattern actually changed
            if pattern_id != self.current_pattern:
                self.current_pattern = pattern_id
                self.pattern_transition_time = transition_time
                
                # Force optimization on pattern change
                self._optimize_system_intervals(transition_time, force=True)
    
    def record_pattern_interval(
        self,
        entity_id: str,
        pattern_id: int,
        interval: int
    ) -> None:
        """Record an optimal interval for an entity in a specific pattern.
        
        Args:
            entity_id: Entity ID
            pattern_id: Pattern ID
            interval: Optimal interval for this entity in this pattern
        """
        with self._lock:
            if entity_id not in self.entity_states:
                return
                
            state = self.entity_states[entity_id]
            state.pattern_specific_intervals[pattern_id] = interval
    
    def update_prediction(
        self,
        entity_id: str,
        predicted_timestamp: Optional[float],
        confidence: float
    ) -> None:
        """Update prediction for when an entity will next change.
        
        Args:
            entity_id: Entity ID
            predicted_timestamp: Predicted time of next change
            confidence: Confidence level (0-1) in this prediction
        """
        with self._lock:
            if entity_id not in self.entity_states:
                return
                
            state = self.entity_states[entity_id]
            state.update_prediction(predicted_timestamp, confidence)
    
    def get_recommended_interval(
        self,
        entity_id: str,
        current_time: Optional[float] = None,
        force_recalculate: bool = False
    ) -> int:
        """Get the recommended polling interval for an entity.
        
        Args:
            entity_id: Entity ID
            current_time: Current time (uses current time if None)
            force_recalculate: Whether to force recalculation
            
        Returns:
            Recommended interval in seconds
        """
        if current_time is None:
            current_time = time.time()
            
        with self._lock:
            # Run system-wide optimization if due
            if (force_recalculate or 
                current_time - self.last_optimization_time > self.optimization_interval):
                self._optimize_system_intervals(current_time)
            
            # If entity doesn't exist, use default
            if entity_id not in self.entity_states:
                return self.base_interval
            
            # Get entity's recommended interval
            state = self.entity_states[entity_id]
            interval = state.get_recommended_interval(
                current_time,
                self.min_interval,
                self.max_interval,
                self.backoff_factor,
                self.current_pattern,
                force_recalculate
            )
            
            # Apply cluster coordination
            interval = self._coordinate_with_cluster(entity_id, interval)
            
            # Apply system-wide rate limiting
            interval = self._apply_rate_limiting(entity_id, interval)
            
            # Update and return interval
            state.current_interval = interval
            
            # Record to history
            self._record_interval_history(entity_id, interval, current_time)
            
            return interval
    
    def _record_interval_history(
        self,
        entity_id: str,
        interval: int,
        timestamp: float
    ) -> None:
        """Record interval history for an entity.
        
        Args:
            entity_id: Entity ID
            interval: Interval value
            timestamp: Current timestamp
        """
        # Initialize entity history if needed
        if entity_id not in self._interval_history:
            self._interval_history[entity_id] = []
        
        # Add entry (only if different from previous)
        entity_history = self._interval_history[entity_id]
        
        if not entity_history or entity_history[-1]["interval"] != interval:
            entry = {
                "timestamp": timestamp,
                "interval": interval,
                "pattern": self.current_pattern
            }
            
            # Add entity metrics
            if entity_id in self.entity_states:
                state = self.entity_states[entity_id]
                entry["stability"] = state.adaptive_metrics["stability_score"]
                
            entity_history.append(entry)
            
            # Trim history if needed
            if len(entity_history) > self._history_max_size:
                entity_history = entity_history[-self._history_max_size:]
                self._interval_history[entity_id] = entity_history
            
            # Save interval history periodically
            # Only save every 10 changes to avoid excessive storage operations
            if len(entity_history) % 10 == 0 and self._storage:
                self._storage.save_interval_history(self._interval_history)
    
    def _coordinate_with_cluster(self, entity_id: str, interval: int) -> int:
        """Coordinate polling intervals within a cluster.
        
        Args:
            entity_id: Entity ID
            interval: Proposed interval
            
        Returns:
            Adjusted interval
        """
        # If entity isn't in a cluster, return unchanged
        if entity_id not in self.entity_clusters:
            return interval
            
        cluster_id = self.entity_clusters[entity_id]
        cluster_members = self.cluster_members.get(cluster_id, set())
        
        if not cluster_members or len(cluster_members) <= 1:
            return interval
            
        # Find relevant intervals within cluster
        active_intervals = []
        
        for member_id in cluster_members:
            if member_id == entity_id:
                continue
                
            if member_id in self.entity_states:
                member_state = self.entity_states[member_id]
                
                # Only consider entities with significant activity
                if member_state.change_count > 0:
                    # Add to relevant intervals if priority is significant
                    if member_state.adaptive_metrics["priority_score"] > 0.3:
                        active_intervals.append(member_state.current_interval)
        
        # If no relevant intervals, return unchanged
        if not active_intervals:
            return interval
            
        # Find the minimum active interval
        min_active = min(active_intervals)
        
        # If we're close to the minimum active interval, align with it
        if interval < min_active * 1.5:
            return min_active
            
        # Otherwise keep our interval
        return interval
    
    def _apply_rate_limiting(self, entity_id: str, interval: int) -> int:
        """Apply rate limiting to respect max_registers_per_second.
        
        Args:
            entity_id: Entity ID
            interval: Proposed interval
            
        Returns:
            Adjusted interval
        """
        # If no rate limiting is enabled, return unchanged
        if not self.max_registers_per_second:
            return interval
            
        # Calculate current registers per second
        registers_per_second = self._calculate_current_rps()
        
        # If we're under budget, return unchanged
        if registers_per_second <= self.max_registers_per_second:
            return interval
            
        # We're over budget, increase interval
        # Calculate entity's registers per second contribution
        entity_state = self.entity_states.get(entity_id)
        if not entity_state:
            return interval
            
        entity_rps = entity_state.register_count / interval
        
        # Calculate how overbudget we are
        excess_factor = registers_per_second / self.max_registers_per_second
        
        # Increase interval based on excess factor
        # The more we're over budget, the more we increase
        if excess_factor > 1.5:  # Significantly over budget
            return min(self.max_interval, int(interval * 1.5))
        elif excess_factor > 1.2:  # Moderately over budget
            return min(self.max_interval, int(interval * 1.3))
        else:  # Slightly over budget
            return min(self.max_interval, int(interval * 1.1))
    
    def _calculate_current_rps(self) -> float:
        """Calculate current registers per second rate.
        
        Returns:
            Current registers per second rate
        """
        total_rps = 0.0
        
        for entity_id, state in self.entity_states.items():
            # Skip entities with no interval
            if state.current_interval <= 0:
                continue
                
            # Add entity's contribution
            entity_rps = state.register_count / state.current_interval
            total_rps += entity_rps
            
        self.register_polls_per_second = total_rps
        return total_rps
    
    def _optimize_system_intervals(
        self,
        current_time: float,
        force: bool = False
    ) -> None:
        """Perform system-wide interval optimization.
        
        Args:
            current_time: Current timestamp
            force: Whether to force optimization
        """
        # Skip if recently optimized and not forced
        if not force and current_time - self.last_optimization_time < self.optimization_interval:
            return
            
        with self._lock:
            self.last_optimization_time = current_time
            self.optimization_count += 1
            
            _LOGGER.debug("Performing system-wide interval optimization (%d entities)",
                       len(self.entity_states))
            
            # First, calculate recommended intervals for all entities
            # Start with high priority entities
            entities_by_priority = sorted(
                self.entity_states.items(),
                key=lambda x: x[1].adaptive_metrics["priority_score"],
                reverse=True
            )
            
            # First pass: calculate recommended intervals
            for entity_id, state in entities_by_priority:
                interval = state.get_recommended_interval(
                    current_time,
                    self.min_interval,
                    self.max_interval,
                    self.backoff_factor,
                    self.current_pattern,
                    True  # Force recalculation
                )
                state.recommended_interval = interval
            
            # Second pass: apply cluster coordination
            for entity_id, state in self.entity_states.items():
                state.current_interval = self._coordinate_with_cluster(
                    entity_id, state.recommended_interval
                )
            
            # Third pass: apply system-wide rate limiting if needed
            current_rps = self._calculate_current_rps()
            
            if self.max_registers_per_second and current_rps > self.max_registers_per_second:
                # We're over budget, adjust starting with lowest priority entities
                for entity_id, state in reversed(entities_by_priority):
                    state.current_interval = self._apply_rate_limiting(
                        entity_id, state.current_interval
                    )
                    
                    # Recalculate RPS after each adjustment
                    current_rps = self._calculate_current_rps()
                    
                    # Stop if we're under budget
                    if current_rps <= self.max_registers_per_second:
                        break
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about interval management.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            # Calculate register statistics
            total_registers = sum(state.register_count for state in self.entity_states.values())
            
            # Calculate intervals statistics
            intervals = [state.current_interval for state in self.entity_states.values()]
            avg_interval = sum(intervals) / len(intervals) if intervals else 0
            
            # Calculate error rate
            error_rate = self.total_errors / max(1, self.total_polls)
            
            # Calculate polls per hour
            total_polls_per_hour = sum(
                3600 / state.current_interval for state in self.entity_states.values()
            )
            
            # Calculate average response time
            avg_response_time = (
                sum(self.last_response_times) / len(self.last_response_times)
                if self.last_response_times else 0
            )
            
            # Calculate polling efficiency (changes / polls)
            polling_efficiency = (
                (self.total_changes / max(1, self.successful_polls)) * 100
                if self.successful_polls else 0
            )
            
            # Count entities in different change rate categories
            high_change_entities = sum(
                1 for state in self.entity_states.values()
                if state.adaptive_metrics["change_rate"] > 0.5
            )
            
            medium_change_entities = sum(
                1 for state in self.entity_states.values()
                if 0.1 <= state.adaptive_metrics["change_rate"] <= 0.5
            )
            
            low_change_entities = sum(
                1 for state in self.entity_states.values()
                if state.adaptive_metrics["change_rate"] < 0.1
            )
            
            # Create statistics dictionary
            return {
                "total_entities": len(self.entity_states),
                "total_registers": total_registers,
                "total_polls": self.total_polls,
                "successful_polls": self.successful_polls,
                "total_changes": self.total_changes,
                "total_errors": self.total_errors,
                "error_rate": round(error_rate, 3),
                "polls_per_hour": round(total_polls_per_hour),
                "avg_scan_interval": round(avg_interval, 1),
                "min_scan_interval": min(intervals) if intervals else 0,
                "max_scan_interval": max(intervals) if intervals else 0,
                "high_change_entities": high_change_entities,
                "medium_change_entities": medium_change_entities,
                "low_change_entities": low_change_entities,
                "polling_efficiency": round(polling_efficiency, 1),
                "current_pattern": self.current_pattern,
                "registers_per_second": round(self.register_polls_per_second, 2),
                "avg_response_time_ms": round(avg_response_time, 2),
                "optimization_count": self.optimization_count,
                "last_optimization": datetime.fromtimestamp(self.last_optimization_time).isoformat()
                if self.last_optimization_time else None,
            }
    
    def get_entity_statistics(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific entity.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Entity statistics or None if not found
        """
        with self._lock:
            if entity_id not in self.entity_states:
                return None
                
            state = self.entity_states[entity_id]
            return state.get_stats()
    
    def get_interval_history(self, entity_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get interval history for entities.
        
        Args:
            entity_id: Specific entity ID or None for all entities
            
        Returns:
            Dictionary of entity ID to interval history
        """
        with self._lock:
            if entity_id:
                # Return history for specific entity
                if entity_id in self._interval_history:
                    return {entity_id: self._interval_history[entity_id]}
                else:
                    return {entity_id: []}
            else:
                # Return all histories
                return self._interval_history.copy()
    
    def optimize_all_intervals(self) -> Dict[str, int]:
        """Force optimization of all scan intervals.
        
        Returns:
            Dictionary of entity ID to optimized interval
        """
        current_time = time.time()
        optimized_intervals = {}
        
        with self._lock:
            # Force system-wide optimization
            self._optimize_system_intervals(current_time, force=True)
            
            # Collect all optimized intervals
            for entity_id, state in self.entity_states.items():
                optimized_intervals[entity_id] = state.current_interval
                
        return optimized_intervals
    
    def set_storage(self, storage_manager) -> None:
        """Set the storage manager for persistence.
        
        Args:
            storage_manager: Storage manager instance
        """
        self._storage = storage_manager
        
        # Load existing data if available
        self._load_from_storage()
    
    def _load_from_storage(self) -> None:
        """Load interval data from storage."""
        if not self._storage:
            return
            
        try:
            # Load interval history
            interval_history = self._storage.get_interval_history()
            if interval_history:
                self._interval_history = interval_history
                
            # Load entity data from stats
            entity_stats = self._storage.get_entity_stats()
            entities_restored = 0
            
            for entity_id, data in entity_stats.items():
                # Skip if entity doesn't have recommended interval data
                if "recommended_scan_interval" not in data:
                    continue
                    
                # Create or update entity state
                if entity_id not in self.entity_states:
                    # Create new entity state with stored interval
                    interval = data.get("recommended_scan_interval", self.base_interval)
                    register_count = data.get("register_count", 1)
                    self.register_entity(entity_id, register_count, interval)
                    
                    state = self.entity_states[entity_id]
                else:
                    # Update existing state
                    state = self.entity_states[entity_id]
                    state.base_interval = data.get("recommended_scan_interval", self.base_interval)
                    state.recommended_interval = state.base_interval
                    state.current_interval = state.base_interval
                
                # Add pattern-specific intervals if available
                if "current_pattern" in data and "pattern_specific_scan_interval" in data:
                    pattern_id = data["current_pattern"]
                    if pattern_id is not None:
                        interval = data["pattern_specific_scan_interval"]
                        state.pattern_specific_intervals[pattern_id] = interval
                
                # Load metrics if available
                if "stability_score" in data:
                    state.adaptive_metrics["stability_score"] = data["stability_score"] / 100
                if "change_rate" in data:
                    state.adaptive_metrics["change_rate"] = data["change_percentage"] / 100
                    
                entities_restored += 1
                
            if entities_restored > 0:
                _LOGGER.info("Restored interval data for %d entities from storage", entities_restored)
            if interval_history:
                _LOGGER.info("Loaded interval history for %d entities", len(interval_history))
                
        except Exception as e:
            _LOGGER.error("Error loading data from storage: %s", e)


# Global singleton instance
INTERVAL_MANAGER = IntervalManager.get_instance()