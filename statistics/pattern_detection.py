"""
Pattern detection for Modbus entities.

This module provides device-agnostic statistical pattern detection to identify
consistent operating patterns in entity value changes. Patterns are assigned
neutral identifiers and characterized by their statistical properties.
"""

import time
import logging
from datetime import datetime
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import deque, defaultdict
import statistics
import math

_LOGGER = logging.getLogger(__name__)

# Constants for pattern detection
PATTERN_HISTORY_SIZE = 100  # Number of pattern transitions to store
PATTERN_STABILITY_THRESHOLD = 300  # Seconds a pattern should be stable before considering it established
MIN_PATTERN_DURATION = 120  # Minimum duration (seconds) to consider a stable pattern
MAX_PATTERNS = 8  # Maximum number of patterns to detect to avoid over-fragmentation
MIN_SAMPLES_FOR_PATTERN = 10  # Minimum samples needed to establish a pattern
PATTERN_TRANSITION_METRIC_THRESHOLD = 0.5  # Threshold for detecting pattern transitions

class PatternState:
    """Represents a specific statistical pattern with its characteristics."""
    
    def __init__(self, pattern_id: int, characteristic_values: Dict[str, float] = None):
        """Initialize a pattern state."""
        self.pattern_id = pattern_id
        self.characteristic_values = characteristic_values or {}
        self.occurrences = 0
        self.total_duration = 0
        self.average_duration = 0
        self.last_start_time = None
        self.last_end_time = None
        self.typical_next_pattern = None  # The most common pattern that follows this one
        self.entity_scan_intervals = {}  # Entity-specific scan intervals for this pattern
        self.defining_characteristics = {}  # Key statistical properties defining this pattern
    
    def set_active(self, timestamp: float = None) -> None:
        """Mark this pattern as becoming active."""
        self.last_start_time = timestamp or time.time()
    
    def set_inactive(self, timestamp: float = None) -> None:
        """Mark this pattern as becoming inactive and update statistics."""
        end_time = timestamp or time.time()
        if self.last_start_time:
            duration = end_time - self.last_start_time
            if duration >= MIN_PATTERN_DURATION:
                self.occurrences += 1
                self.total_duration += duration
                self.average_duration = self.total_duration / self.occurrences
            self.last_end_time = end_time
            self.last_start_time = None
    
    def add_value_point(self, entity_id: str, value: float) -> None:
        """Add a characteristic value for an entity in this pattern."""
        if entity_id not in self.characteristic_values:
            self.characteristic_values[entity_id] = []
        
        self.characteristic_values[entity_id].append(value)
        
        # Keep only the most recent values (max 20 per entity)
        if len(self.characteristic_values[entity_id]) > 20:
            self.characteristic_values[entity_id] = self.characteristic_values[entity_id][-20:]
        
        # Update defining characteristics whenever we add values
        self._update_defining_characteristics()
    
    def _update_defining_characteristics(self) -> None:
        """Extract the statistical properties that define this pattern."""
        characteristics = {}
        
        for entity_id, values in self.characteristic_values.items():
            if len(values) < 3:
                continue
                
            try:
                # Calculate basic statistics
                mean_val = statistics.mean(values)
                try:
                    std_dev = statistics.stdev(values)
                except statistics.StatisticsError:
                    std_dev = 0
                
                # Calculate change rate (first derivative)
                changes = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
                mean_change = statistics.mean(changes) if changes else 0
                
                # Calculate stability (coefficient of variation)
                stability = (std_dev / mean_val) if mean_val != 0 else float('inf')
                
                # Store characteristics
                characteristics[entity_id] = {
                    "mean": mean_val,
                    "std_dev": std_dev,
                    "stability": min(1.0, 1.0 / (1.0 + stability)) if stability != float('inf') else 0,
                    "change_rate": mean_change,
                    "range": max(values) - min(values) if len(values) > 1 else 0,
                }
            except (ValueError, TypeError, ZeroDivisionError):
                # Skip problematic calculations
                pass
        
        self.defining_characteristics = characteristics
    
    def get_center_values(self) -> Dict[str, float]:
        """Get the average values for each entity in this pattern."""
        centers = {}
        for entity_id, values in self.characteristic_values.items():
            if values:
                try:
                    centers[entity_id] = statistics.mean(values)
                except statistics.StatisticsError:
                    # Handle case with only one value
                    centers[entity_id] = values[0] if values else 0
        return centers
    
    def calculate_distance(self, values: Dict[str, float], entity_weights: Dict[str, float] = None) -> float:
        """
        Calculate distance between current values and this pattern's characteristic values.
        
        Args:
            values: Current entity values
            entity_weights: Optional importance weights for entities (higher = more influential)
        """
        if not self.characteristic_values or not values:
            return float('inf')
        
        centers = self.get_center_values()
        common_entities = set(centers.keys()) & set(values.keys())
        
        if not common_entities:
            return float('inf')
        
        # Calculate normalized Euclidean distance with optional weighting
        distance_sum = 0
        total_weight = 0
        
        for entity_id in common_entities:
            try:
                # Get all historical values for this entity in this pattern
                historical_values = self.characteristic_values[entity_id]
                if not historical_values:
                    continue
                
                # Calculate standard deviation if we have enough data points
                if len(historical_values) > 1:
                    std_dev = statistics.stdev(historical_values)
                else:
                    # Estimate std_dev as 10% of the value if only one data point
                    std_dev = abs(historical_values[0] * 0.1) if historical_values[0] else 1.0
                
                # Avoid division by zero
                if std_dev == 0:
                    std_dev = 1.0
                
                # Calculate normalized squared difference
                center = centers[entity_id]
                current = values[entity_id]
                normalized_diff = (current - center) / std_dev
                
                # Apply weight if provided
                weight = entity_weights.get(entity_id, 1.0) if entity_weights else 1.0
                distance_sum += (normalized_diff ** 2) * weight
                total_weight += weight
                
            except (ValueError, TypeError, ZeroDivisionError):
                # Skip problematic calculations
                continue
        
        # Return weighted root mean square distance
        if total_weight == 0:
            return float('inf')
        
        return math.sqrt(distance_sum / total_weight)
    
    def estimate_next_transition(self, current_time: float) -> Optional[float]:
        """Estimate when this pattern will likely transition."""
        if not self.last_start_time or self.average_duration == 0:
            return None
        
        # Simple prediction: start time + average duration
        return self.last_start_time + self.average_duration
    
    def get_defining_criteria(self) -> List[str]:
        """Extract human-readable criteria that define this pattern."""
        criteria = []
        
        # Find entities with stable values (low variability)
        stable_entities = []
        for entity_id, stats in self.defining_characteristics.items():
            if stats.get("stability", 0) > 0.8:  # High stability
                stable_entities.append(entity_id)
                
        if stable_entities:
            criteria.append(f"Stable values for {len(stable_entities)} entities")
        
        # Find entities with high variability
        variable_entities = []
        for entity_id, stats in self.defining_characteristics.items():
            if stats.get("stability", 0) < 0.3 and stats.get("change_rate", 0) > 0:
                variable_entities.append(entity_id)
                
        if variable_entities:
            criteria.append(f"Variable values for {len(variable_entities)} entities")
        
        # Find correlations between entities
        if len(self.characteristic_values) >= 2:
            criteria.append("Correlated value changes between multiple entities")
        
        # Add average duration if we have enough occurrences
        if self.occurrences >= 3:
            duration_mins = self.average_duration / 60
            if duration_mins < 60:
                criteria.append(f"Typically lasts {duration_mins:.1f} minutes")
            else:
                criteria.append(f"Typically lasts {duration_mins/60:.1f} hours")
        
        # If no criteria found, add a generic one
        if not criteria:
            criteria.append("Statistical pattern with insufficient data to characterize")
            
        return criteria


class PatternSequence:
    """Tracks sequences of pattern transitions."""
    
    def __init__(self, max_history: int = 20):
        """Initialize pattern sequence tracker."""
        self.transitions = deque(maxlen=max_history)
        self.transition_counts = defaultdict(int)  # (from_pattern, to_pattern) -> count
    
    def add_transition(self, from_pattern: int, to_pattern: int) -> None:
        """Record a pattern transition."""
        self.transitions.append((from_pattern, to_pattern))
        self.transition_counts[(from_pattern, to_pattern)] += 1
    
    def get_most_likely_next(self, current_pattern: int) -> Optional[int]:
        """Get the most likely next pattern based on historical transitions."""
        transitions_from_current = [(key[1], count) for key, count in self.transition_counts.items() 
                                   if key[0] == current_pattern]
        
        if not transitions_from_current:
            return None
            
        # Find the pattern with highest transition count
        return max(transitions_from_current, key=lambda x: x[1])[0]


class PatternDetector:
    """Detects and tracks patterns from entity value behaviors."""
    
    def __init__(self, max_patterns: int = MAX_PATTERNS):
        """Initialize the pattern detector."""
        self.patterns: Dict[int, PatternState] = {}  # pattern_id -> PatternState
        self.current_pattern: Optional[int] = None
        self.previous_pattern: Optional[int] = None
        self.last_pattern_change: float = 0
        self.max_patterns = max_patterns
        self.entity_values: Dict[str, float] = {}  # Latest values by entity
        self.entity_weights: Dict[str, float] = {}  # Importance weights by entity
        self.pattern_sequence = PatternSequence()
        self.pattern_durations: Dict[int, List[float]] = {}  # pattern_id -> list of durations
        self.pattern_change_timestamps: List[float] = []  # When pattern changes occurred
        
        # Enhanced tracking of entity changes
        self._last_entity_values: Dict[str, Any] = {}  # Last known value per entity
        self._entity_change_times: Dict[str, float] = {}  # Last change time per entity
        self._recent_change_window = 300  # Consider changes within 5 minutes as "recent"
        
        # Correlation integration
        self._correlation_manager = None  # Will be set via set_correlation_manager
        self._cluster_weight_multiplier = 1.5  # Entities in same cluster get higher weight
        self._transition_threshold_adjustment = 0.8  # Adjustment factor for threshold when using correlation
        
    def set_correlation_manager(self, correlation_manager) -> None:
        """Set the correlation manager for enhanced pattern detection.
        
        Args:
            correlation_manager: Correlation manager to integrate with
        """
        self._correlation_manager = correlation_manager
        
    def set_entity_weight(self, entity_id: str, weight: float = 5.0) -> None:
        """Set the importance weight for an entity (how much it influences pattern detection)."""
        self.entity_weights[entity_id] = max(1.0, min(10.0, weight))
        
    def update_entity_value(self, entity_id: str, value: Any, timestamp: float = None) -> None:
        """Update an entity's value and check for pattern changes."""
        if timestamp is None:
            timestamp = time.time()
            
        try:
            # Only process numeric values
            numeric_value = float(value)
            
            # Track if this is a value change
            value_changed = False
            if entity_id in self._last_entity_values:
                old_value = self._last_entity_values[entity_id]
                if abs(numeric_value - old_value) > 0:
                    value_changed = True
                    # Record change time
                    self._entity_change_times[entity_id] = timestamp
                    
            # Store the current value
            self._last_entity_values[entity_id] = numeric_value
            old_value = self.entity_values.get(entity_id)
            self.entity_values[entity_id] = numeric_value
            
            # Apply correlation-enhanced weights if possible
            if value_changed and self._correlation_manager:
                self._apply_correlation_weights(entity_id)
            
            # If this is the first value or we don't have an active pattern, initialize
            if self.current_pattern is None and len(self.entity_values) >= MIN_SAMPLES_FOR_PATTERN:
                self._initialize_patterns()
                return
                
            # Update the current pattern's characteristic values if we have a pattern
            if self.current_pattern is not None:
                current_pattern = self.patterns.get(self.current_pattern)
                if current_pattern:
                    current_pattern.add_value_point(entity_id, numeric_value)
            
            # Check if this update indicates a pattern change
            if old_value is not None and abs(numeric_value - old_value) > 0:
                # Value changed - check if it's a significant change for pattern detection
                self.check_for_pattern_change_with_correlation(timestamp)
        
        except (ValueError, TypeError):
            # Skip non-numeric values
            pass
    
    def _initialize_patterns(self) -> None:
        """Initialize the first pattern based on current values."""
        if not self.entity_values:
            return
        
        # Create initial pattern
        pattern_id = 0
        initial_pattern = PatternState(pattern_id, {k: [v] for k, v in self.entity_values.items()})
        initial_pattern.set_active()
        self.patterns[pattern_id] = initial_pattern
        self.current_pattern = pattern_id
    
    def _check_for_pattern_change(self, timestamp: float) -> None:
        """Determine if current values constitute a pattern change."""
        # Don't consider pattern changes too frequently
        if timestamp - self.last_pattern_change < PATTERN_STABILITY_THRESHOLD:
            return
            
        # Find closest pattern to current values
        closest_pattern_id = None
        closest_distance = float('inf')
        
        for pattern_id, pattern in self.patterns.items():
            distance = pattern.calculate_distance(self.entity_values, self.entity_weights)
            if distance < closest_distance:
                closest_distance = distance
                closest_pattern_id = pattern_id
        
        # If we're too far from any existing pattern, create a new one
        if closest_distance > PATTERN_TRANSITION_METRIC_THRESHOLD and len(self.patterns) < self.max_patterns:
            new_pattern_id = max(self.patterns.keys()) + 1 if self.patterns else 0
            new_pattern = PatternState(new_pattern_id, {k: [v] for k, v in self.entity_values.items()})
            new_pattern.set_active(timestamp)
            self.patterns[new_pattern_id] = new_pattern
            
            # Record transition
            if self.current_pattern is not None:
                self._record_pattern_transition(self.current_pattern, new_pattern_id, timestamp)
                
            self.current_pattern = new_pattern_id
            self.last_pattern_change = timestamp
            self.pattern_change_timestamps.append(timestamp)
            
        # Or switch to the closest existing pattern
        elif closest_pattern_id is not None and closest_pattern_id != self.current_pattern:
            # Record transition
            if self.current_pattern is not None:
                self._record_pattern_transition(self.current_pattern, closest_pattern_id, timestamp)
                
            # Update pattern states
            if self.current_pattern is not None:
                current_pattern = self.patterns.get(self.current_pattern)
                if current_pattern:
                    current_pattern.set_inactive(timestamp)
                    
            closest_pattern = self.patterns.get(closest_pattern_id)
            if closest_pattern:
                closest_pattern.set_active(timestamp)
                
            self.previous_pattern = self.current_pattern
            self.current_pattern = closest_pattern_id
            self.last_pattern_change = timestamp
            self.pattern_change_timestamps.append(timestamp)
    
    def check_for_pattern_change_with_correlation(self, timestamp: float) -> None:
        """Check for pattern changes with correlation awareness.
        
        This enhances pattern detection by using correlation information
        to adjust sensitivity for pattern transitions.
        
        Args:
            timestamp: Current timestamp
        """
        # If we don't have a correlation manager, use default detection
        if not self._correlation_manager:
            self._check_for_pattern_change(timestamp)
            return
            
        # Enhanced detection logic
        # First, find entities that have recently changed
        recently_changed_entities = self._get_recently_changed_entities()
        
        # Group by cluster
        entities_by_cluster = {}
        for entity_id in recently_changed_entities:
            cluster = self._correlation_manager.get_cluster_for_entity(entity_id)
            if cluster:
                if cluster not in entities_by_cluster:
                    entities_by_cluster[cluster] = []
                entities_by_cluster[cluster].append(entity_id)
        
        # If we have multiple entities changing in the same cluster,
        # it's a stronger signal for a pattern change
        strong_cluster_changes = False
        for cluster, entities in entities_by_cluster.items():
            if len(entities) >= 2:
                strong_cluster_changes = True
                break
                
        if strong_cluster_changes:
            # Temporarily adjust the pattern transition threshold to be more sensitive
            global PATTERN_TRANSITION_METRIC_THRESHOLD
            old_threshold = PATTERN_TRANSITION_METRIC_THRESHOLD
            PATTERN_TRANSITION_METRIC_THRESHOLD = old_threshold * self._transition_threshold_adjustment
            
            # Call the normal pattern detection with the adjusted threshold
            self._check_for_pattern_change(timestamp)
            
            # Restore the original threshold
            PATTERN_TRANSITION_METRIC_THRESHOLD = old_threshold
        else:
            # Use the normal pattern detection
            self._check_for_pattern_change(timestamp)
    
    def _apply_correlation_weights(self, trigger_entity_id: str) -> None:
        """Apply correlation-based weights to improve pattern detection.
        
        Args:
            trigger_entity_id: The entity being updated
        """
        if not self._correlation_manager:
            return
            
        # Get the cluster for this entity
        cluster = self._correlation_manager.get_cluster_for_entity(trigger_entity_id)
        if not cluster:
            return
            
        # Get all entities in the same cluster
        cluster_entities = self._correlation_manager.get_clusters().get(cluster, [])
        if not cluster_entities:
            return
            
        # Apply weight multiplier to entities in the same cluster
        # This makes changes in correlated entities more influential in pattern detection
        for entity_id in cluster_entities:
            if entity_id != trigger_entity_id:
                # Get current weight
                current_weight = self.entity_weights.get(entity_id, 1.0)
                
                # Get correlation strength
                corr = self._correlation_manager.get_correlation(trigger_entity_id, entity_id)
                
                # Apply weight based on correlation strength
                if corr is not None and corr > 0.5:  # Only boost for strong correlations
                    # Scale multiplier based on correlation strength
                    multiplier = 1.0 + (corr - 0.5) * self._cluster_weight_multiplier
                    new_weight = current_weight * multiplier
                    
                    # Cap the weight to prevent one entity from dominating
                    self.entity_weights[entity_id] = min(10.0, new_weight)
    
    def _get_recently_changed_entities(self) -> List[str]:
        """Get list of entities that have recently changed values.
        
        Returns:
            List of entity IDs that have changed within the recent change window
        """
        current_time = time.time()
        recent_entities = []
        
        for entity_id, change_time in self._entity_change_times.items():
            # Check if change happened within the recent window
            if current_time - change_time <= self._recent_change_window:
                recent_entities.append(entity_id)
        
        return recent_entities
    
    def set_recent_change_window(self, seconds: int) -> None:
        """Set the time window for considering changes as 'recent'.
        
        Args:
            seconds: Number of seconds to consider changes as recent
        """
        self._recent_change_window = max(60, min(3600, seconds))  # Between 1 minute and 1 hour
    
    def get_change_metrics(self) -> Dict[str, Any]:
        """Get metrics about entity changes.
        
        Returns:
            Dictionary with change metrics
        """
        current_time = time.time()
        recent_count = 0
        all_changes = len(self._entity_change_times)
        
        # Count recent changes
        for change_time in self._entity_change_times.values():
            if current_time - change_time <= self._recent_change_window:
                recent_count += 1
        
        # Calculate change rates
        minutes_5 = 0
        minutes_60 = 0
        
        for change_time in self._entity_change_times.values():
            if current_time - change_time <= 300:  # 5 minutes
                minutes_5 += 1
            if current_time - change_time <= 3600:  # 60 minutes
                minutes_60 += 1
        
        return {
            "total_tracked_entities": len(self.entity_values),
            "entities_with_changes": all_changes,
            "recent_changes": recent_count,
            "changes_last_5min": minutes_5,
            "changes_last_60min": minutes_60,
            "recent_change_window": self._recent_change_window,
        }
    
    def _record_pattern_transition(self, from_pattern_id: int, to_pattern_id: int, timestamp: float) -> None:
        """Record a transition between patterns."""
        self.pattern_sequence.add_transition(from_pattern_id, to_pattern_id)
        
        # Update durations
        from_pattern = self.patterns.get(from_pattern_id)
        if from_pattern and from_pattern.last_start_time:
            duration = timestamp - from_pattern.last_start_time
            if duration >= MIN_PATTERN_DURATION:
                if from_pattern_id not in self.pattern_durations:
                    self.pattern_durations[from_pattern_id] = []
                self.pattern_durations[from_pattern_id].append(duration)
        
        # Update most likely next pattern
        from_pattern = self.patterns.get(from_pattern_id)
        if from_pattern:
            from_pattern.typical_next_pattern = to_pattern_id
    
    def get_current_pattern_info(self) -> Dict[str, Any]:
        """Get information about the current pattern."""
        if self.current_pattern is None:
            return {
                "pattern_id": None,
                "active_since": None,
                "predicted_duration": None,
                "predicted_end": None,
                "predicted_next_pattern": None,
                "stability": 0,
                "previous_pattern": None,
                "pattern_confidence": 0,
                "defining_criteria": [],
            }
        
        current_pattern = self.patterns.get(self.current_pattern)
        if not current_pattern:
            return {
                "pattern_id": self.current_pattern,
                "active_since": None,
                "predicted_duration": None,
                "predicted_end": None,
                "predicted_next_pattern": None,
                "stability": 0,
                "previous_pattern": self.previous_pattern,
                "pattern_confidence": 0,
                "defining_criteria": [],
            }
        
        # Get current time
        now = time.time()
        
        # Format active_since timestamp
        active_since = None
        if current_pattern.last_start_time:
            try:
                active_since = datetime.fromtimestamp(current_pattern.last_start_time).isoformat()
            except (ValueError, OverflowError):
                active_since = None
        
        # Calculate pattern stability - higher value means more stable
        time_in_current_pattern = now - current_pattern.last_start_time if current_pattern.last_start_time else 0
        stability = min(100, (time_in_current_pattern / PATTERN_STABILITY_THRESHOLD) * 100) \
                  if PATTERN_STABILITY_THRESHOLD > 0 else 0
        
        # Predict pattern end time
        predicted_end = None
        predicted_duration = current_pattern.average_duration if current_pattern.occurrences > 0 else None
        
        if current_pattern.last_start_time and predicted_duration:
            predicted_end_timestamp = current_pattern.last_start_time + predicted_duration
            try:
                predicted_end = datetime.fromtimestamp(predicted_end_timestamp).isoformat()
            except (ValueError, OverflowError):
                predicted_end = None
        
        # Predict next pattern
        predicted_next_pattern = None
        if current_pattern.typical_next_pattern is not None:
            predicted_next_pattern = current_pattern.typical_next_pattern
        
        # Determine pattern confidence
        if current_pattern.occurrences > 10:
            pattern_confidence = 90
        elif current_pattern.occurrences > 5:
            pattern_confidence = 75
        elif current_pattern.occurrences > 2:
            pattern_confidence = 50
        else:
            pattern_confidence = 25
            
        # Adjust for stability
        pattern_confidence = (pattern_confidence + stability) / 2
        
        # Get defining criteria
        defining_criteria = current_pattern.get_defining_criteria()
        
        # Add correlation information if available
        correlation_info = {}
        if self._correlation_manager:
            cluster_counts = {}
            entity_count = 0
            
            # Count entities by cluster
            for entity_id in current_pattern.characteristic_values:
                entity_count += 1
                cluster = self._correlation_manager.get_cluster_for_entity(entity_id)
                if cluster:
                    if cluster not in cluster_counts:
                        cluster_counts[cluster] = 0
                    cluster_counts[cluster] += 1
                    
            # Find primary cluster
            primary_cluster = None
            max_count = 0
            for cluster, count in cluster_counts.items():
                if count > max_count:
                    max_count = count
                    primary_cluster = cluster
                    
            if primary_cluster:
                correlation_info = {
                    "primary_cluster": primary_cluster,
                    "cluster_entity_count": max_count,
                    "total_entity_count": entity_count,
                    "cluster_coverage": round(max_count / entity_count * 100, 1) if entity_count > 0 else 0
                }
        
        result = {
            "pattern_id": self.current_pattern,
            "active_since": active_since,
            "predicted_duration": predicted_duration,
            "predicted_end": predicted_end,
            "predicted_next_pattern": predicted_next_pattern,
            "stability": int(stability),
            "previous_pattern": self.previous_pattern,
            "pattern_confidence": int(pattern_confidence),
            "defining_criteria": defining_criteria,
        }
        
        # Add correlation info if available
        if correlation_info:
            result["correlation"] = correlation_info
            
        return result
    
    def get_optimal_scan_interval(self, entity_id: str) -> Optional[int]:
        """Get optimal scan interval for an entity based on current pattern."""
        if self.current_pattern is None:
            return None
            
        current_pattern = self.patterns.get(self.current_pattern)
        if not current_pattern:
            return None
            
        # If we have a specific recommendation for this entity in this pattern, use it
        if entity_id in current_pattern.entity_scan_intervals:
            return current_pattern.entity_scan_intervals[entity_id]
            
        # Check if we're in a transition between patterns
        if self.is_transition_period():
            # More frequent scanning during transitions
            return 60  # 1 minute
            
        # Check entity characteristics in this pattern
        entity_stats = current_pattern.defining_characteristics.get(entity_id, {})
        
        # Calculate optimal scan interval based on statistical behavior
        stability = entity_stats.get("stability", 0.5)
        change_rate = entity_stats.get("change_rate", 0)
        
        if change_rate > 0:
            # Higher change rate = more frequent scanning
            # Lower stability = more frequent scanning
            optimal_interval = int(300 * stability / (1 + change_rate))
            
            # Keep within reasonable bounds
            return max(60, min(600, optimal_interval))
        
        # Fall back to entity weight
        weight = self.entity_weights.get(entity_id, 5.0)
        
        # Higher weight = more frequent scanning
        if weight > 8:
            return 60  # 1 minute
        elif weight > 5:
            return 120  # 2 minutes
        else:
            return 300  # 5 minutes
    
    def is_transition_period(self) -> bool:
        """Determine if we're currently in a pattern transition period."""
        if self.current_pattern is None:
            return False
            
        current_pattern = self.patterns.get(self.current_pattern)
        if not current_pattern or not current_pattern.last_start_time:
            return False
            
        # Consider first 2 minutes of a pattern as transition period
        now = time.time()
        time_in_pattern = now - current_pattern.last_start_time
        return time_in_pattern < 120  # 2 minutes
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about pattern detection."""
        patterns_info = {}
        for pattern_id, pattern in self.patterns.items():
            patterns_info[f"pattern_{pattern_id}"] = {
                "occurrences": pattern.occurrences,
                "average_duration": pattern.average_duration,
                "total_duration": pattern.total_duration,
                "defining_criteria": pattern.get_defining_criteria(),
            }
            
        # Include change metrics
        change_metrics = self.get_change_metrics()
        
        stats = {
            "detected_patterns": len(self.patterns),
            "pattern_changes": len(self.pattern_change_timestamps),
            "current_pattern": self.current_pattern,
            "patterns": patterns_info,
            "change_metrics": change_metrics,
        }
        
        # Add correlation integration info if available
        if self._correlation_manager:
            stats["correlation_integrated"] = True
            stats["correlation_weight_multiplier"] = self._cluster_weight_multiplier
        else:
            stats["correlation_integrated"] = False
            
        return stats
    
    def get_pattern_definition(self, pattern_id: int) -> Dict[str, Any]:
        """Get detailed information about what defines a specific pattern."""
        if pattern_id not in self.patterns:
            return {"error": "Pattern not found"}
            
        pattern = self.patterns[pattern_id]
        
        # Calculate primary statistical characteristics
        most_stable_entities = []
        most_variable_entities = []
        
        for entity_id, stats in pattern.defining_characteristics.items():
            if stats.get("stability", 0) > 0.8:
                most_stable_entities.append((entity_id, stats))
            if stats.get("stability", 0) < 0.3:
                most_variable_entities.append((entity_id, stats))
                
        # Sort by stability
        most_stable_entities.sort(key=lambda x: x[1].get("stability", 0), reverse=True)
        most_variable_entities.sort(key=lambda x: x[1].get("stability", 0))
        
        # Get next pattern
        next_pattern_id = pattern.typical_next_pattern
        next_pattern_probability = 0
        
        if next_pattern_id is not None:
            # Calculate probability
            transitions_from = sum(count for (from_id, _), count in 
                                 self.pattern_sequence.transition_counts.items()
                                 if from_id == pattern_id)
                                 
            if transitions_from > 0:
                next_pattern_probability = self.pattern_sequence.transition_counts.get(
                    (pattern_id, next_pattern_id), 0) / transitions_from
        
        result = {
            "pattern_id": pattern_id,
            "occurrences": pattern.occurrences,
            "average_duration": pattern.average_duration,
            "total_duration": pattern.total_duration,
            "defining_criteria": pattern.get_defining_criteria(),
            "most_stable_entities": [e[0] for e in most_stable_entities[:5]],
            "most_variable_entities": [e[0] for e in most_variable_entities[:5]],
            "typical_next_pattern": next_pattern_id,
            "next_pattern_probability": int(next_pattern_probability * 100),
        }
        
        # Add correlation information if available
        if self._correlation_manager:
            cluster_distribution = {}
            
            # Group entities by cluster
            for entity_id in pattern.characteristic_values:
                cluster = self._correlation_manager.get_cluster_for_entity(entity_id)
                if cluster:
                    if cluster not in cluster_distribution:
                        cluster_distribution[cluster] = []
                    cluster_distribution[cluster].append(entity_id)
            
            if cluster_distribution:
                result["cluster_distribution"] = {
                    k: len(v) for k, v in cluster_distribution.items()
                }
                
                # Find primary cluster
                primary_cluster = max(cluster_distribution.items(), key=lambda x: len(x[1]))[0]
                result["primary_cluster"] = primary_cluster
                
        return result