"""Mode detection for Modbus devices like HVAC equipment, pumps, and industrial systems."""

import time
import logging
from datetime import datetime
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import deque, defaultdict
import statistics

_LOGGER = logging.getLogger(__name__)

# Constants for mode detection
MODE_HISTORY_SIZE = 100  # Number of mode transitions to store
MODE_STABILITY_THRESHOLD = 300  # Seconds a mode should be stable before considering it "active"
MIN_MODE_DURATION = 120  # Minimum duration (seconds) to consider a stable mode
MAX_MODES = 8  # Maximum number of modes to detect to avoid over-fragmentation
MIN_SAMPLES_FOR_MODE = 10  # Minimum samples needed to establish a mode
MODE_TRANSITION_METRIC_THRESHOLD = 0.5  # Threshold for detecting mode transitions

class ModeState:
    """Represents a specific operational mode with its characteristics."""
    
    def __init__(self, mode_id: int, characteristic_values: Dict[str, float] = None):
        """Initialize a mode state."""
        self.mode_id = mode_id
        self.characteristic_values = characteristic_values or {}
        self.occurrences = 0
        self.total_duration = 0
        self.average_duration = 0
        self.last_start_time = None
        self.last_end_time = None
        self.typical_next_mode = None  # The most common mode that follows this one
        self.name = f"mode_{mode_id}"  # Default name
        self.entity_scan_intervals = {}  # Entity-specific scan intervals for this mode
    
    def set_active(self, timestamp: float = None) -> None:
        """Mark this mode as becoming active."""
        self.last_start_time = timestamp or time.time()
    
    def set_inactive(self, timestamp: float = None) -> None:
        """Mark this mode as becoming inactive and update statistics."""
        end_time = timestamp or time.time()
        if self.last_start_time:
            duration = end_time - self.last_start_time
            if duration >= MIN_MODE_DURATION:
                self.occurrences += 1
                self.total_duration += duration
                self.average_duration = self.total_duration / self.occurrences
            self.last_end_time = end_time
            self.last_start_time = None
    
    def add_value_point(self, entity_id: str, value: float) -> None:
        """Add a characteristic value for an entity in this mode."""
        if entity_id not in self.characteristic_values:
            self.characteristic_values[entity_id] = []
        
        self.characteristic_values[entity_id].append(value)
        
        # Keep only the most recent values (max 20 per entity)
        if len(self.characteristic_values[entity_id]) > 20:
            self.characteristic_values[entity_id] = self.characteristic_values[entity_id][-20:]
    
    def get_center_values(self) -> Dict[str, float]:
        """Get the average values for each entity in this mode."""
        centers = {}
        for entity_id, values in self.characteristic_values.items():
            if values:
                try:
                    centers[entity_id] = statistics.mean(values)
                except statistics.StatisticsError:
                    # Handle case with only one value
                    centers[entity_id] = values[0] if values else 0
        return centers
    
    def calculate_distance(self, values: Dict[str, float]) -> float:
        """Calculate distance between current values and this mode's characteristic values."""
        if not self.characteristic_values or not values:
            return float('inf')
        
        centers = self.get_center_values()
        common_entities = set(centers.keys()) & set(values.keys())
        
        if not common_entities:
            return float('inf')
        
        # Calculate normalized Euclidean distance
        distance_sum = 0
        for entity_id in common_entities:
            try:
                # Get all historical values for this entity in this mode
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
                distance_sum += normalized_diff ** 2
            except (ValueError, TypeError, ZeroDivisionError):
                # Skip problematic calculations
                continue
        
        # Return root mean square distance
        return (distance_sum / len(common_entities)) ** 0.5 if common_entities else float('inf')
    
    def estimate_next_transition(self, current_time: float) -> Optional[float]:
        """Estimate when this mode will likely transition."""
        if not self.last_start_time or self.average_duration == 0:
            return None
        
        # Simple prediction: start time + average duration
        return self.last_start_time + self.average_duration


class ModeSequence:
    """Tracks sequences of mode transitions."""
    
    def __init__(self, max_history: int = 20):
        """Initialize mode sequence tracker."""
        self.transitions = deque(maxlen=max_history)
        self.transition_counts = defaultdict(int)  # (from_mode, to_mode) -> count
    
    def add_transition(self, from_mode: int, to_mode: int) -> None:
        """Record a mode transition."""
        self.transitions.append((from_mode, to_mode))
        self.transition_counts[(from_mode, to_mode)] += 1
    
    def get_most_likely_next(self, current_mode: int) -> Optional[int]:
        """Get the most likely next mode based on historical transitions."""
        transitions_from_current = [(key[1], count) for key, count in self.transition_counts.items() 
                                   if key[0] == current_mode]
        
        if not transitions_from_current:
            return None
            
        # Find the mode with highest transition count
        return max(transitions_from_current, key=lambda x: x[1])[0]


class ModeDetector:
    """Detects and tracks operational modes from entity value patterns."""
    
    def __init__(self, max_modes: int = MAX_MODES):
        """Initialize the mode detector."""
        self.modes: Dict[int, ModeState] = {}  # mode_id -> ModeState
        self.current_mode: Optional[int] = None
        self.previous_mode: Optional[int] = None
        self.last_mode_change: float = 0
        self.max_modes = max_modes
        self.entity_values: Dict[str, float] = {}  # Latest values by entity
        self.critical_entities: Set[str] = set()  # Entities that strongly influence mode
        self.mode_sequence = ModeSequence()
        self.mode_durations: Dict[int, List[float]] = {}  # mode_id -> list of durations
        self.mode_change_timestamps: List[float] = []  # When mode changes occurred
        
        # Tracking for heating/cooling detection
        self.heating_indicators: Set[str] = set()
        self.cooling_indicators: Set[str] = set()
        
    def update_entity_value(self, entity_id: str, value: Any, timestamp: float = None) -> None:
        """Update an entity's value and check for mode changes."""
        if timestamp is None:
            timestamp = time.time()
            
        try:
            # Only process numeric values
            numeric_value = float(value)
            
            # Store the value
            old_value = self.entity_values.get(entity_id)
            self.entity_values[entity_id] = numeric_value
            
            # If this is the first value or we don't have an active mode, initialize
            if self.current_mode is None and len(self.entity_values) >= MIN_SAMPLES_FOR_MODE:
                self._initialize_modes()
                return
                
            # Update the current mode's characteristic values if we have a mode
            if self.current_mode is not None:
                current_mode = self.modes.get(self.current_mode)
                if current_mode:
                    current_mode.add_value_point(entity_id, numeric_value)
            
            # Check if this update indicates a mode change
            if old_value is not None and abs(numeric_value - old_value) > 0:
                # Value changed - check if it's a significant change for mode detection
                self._check_for_mode_change(timestamp)
        
        except (ValueError, TypeError):
            # Skip non-numeric values
            pass
    
    def _initialize_modes(self) -> None:
        """Initialize the first mode based on current values."""
        if not self.entity_values:
            return
        
        # Create initial mode
        mode_id = 0
        initial_mode = ModeState(mode_id, {k: v for k, v in self.entity_values.items()})
        initial_mode.set_active()
        self.modes[mode_id] = initial_mode
        self.current_mode = mode_id
        
        # Try to auto-detect heating/cooling indicators
        self._auto_detect_mode_indicators()
        
    def _auto_detect_mode_indicators(self) -> None:
        """Try to automatically identify heating/cooling mode indicators."""
        # Look for temperature-related entities
        for entity_id in self.entity_values:
            name = entity_id.lower()
            
            # Heating indicators
            if any(keyword in name for keyword in ['heat', 'warm', 'hot', 'temp']):
                self.heating_indicators.add(entity_id)
                
            # Cooling indicators
            if any(keyword in name for keyword in ['cool', 'cold', 'chill']):
                self.cooling_indicators.add(entity_id)
                
            # Generic temperature sensors might indicate both
            if 'temperature' in name and not any(
                keyword in name for keyword in 
                ['setpoint', 'target', 'desired', 'requested']):
                self.critical_entities.add(entity_id)
    
    def _check_for_mode_change(self, timestamp: float) -> None:
        """Determine if current values constitute a mode change."""
        # Don't consider mode changes too frequently
        if timestamp - self.last_mode_change < MODE_STABILITY_THRESHOLD:
            return
            
        # Find closest mode to current values
        closest_mode_id = None
        closest_distance = float('inf')
        
        for mode_id, mode in self.modes.items():
            distance = mode.calculate_distance(self.entity_values)
            if distance < closest_distance:
                closest_distance = distance
                closest_mode_id = mode_id
        
        # If we're too far from any existing mode, create a new one
        if closest_distance > MODE_TRANSITION_METRIC_THRESHOLD and len(self.modes) < self.max_modes:
            new_mode_id = max(self.modes.keys()) + 1 if self.modes else 0
            new_mode = ModeState(new_mode_id, {k: v for k, v in self.entity_values.items()})
            new_mode.set_active(timestamp)
            self.modes[new_mode_id] = new_mode
            
            # Record transition
            if self.current_mode is not None:
                self._record_mode_transition(self.current_mode, new_mode_id, timestamp)
                
            self.current_mode = new_mode_id
            self.last_mode_change = timestamp
            self.mode_change_timestamps.append(timestamp)
            
            # Try to identify the mode name
            self._identify_mode_name(new_mode_id)
            
        # Or switch to the closest existing mode
        elif closest_mode_id is not None and closest_mode_id != self.current_mode:
            # Record transition
            if self.current_mode is not None:
                self._record_mode_transition(self.current_mode, closest_mode_id, timestamp)
                
            # Update mode states
            if self.current_mode is not None:
                current_mode = self.modes.get(self.current_mode)
                if current_mode:
                    current_mode.set_inactive(timestamp)
                    
            closest_mode = self.modes.get(closest_mode_id)
            if closest_mode:
                closest_mode.set_active(timestamp)
                
            self.previous_mode = self.current_mode
            self.current_mode = closest_mode_id
            self.last_mode_change = timestamp
            self.mode_change_timestamps.append(timestamp)
    
    def _record_mode_transition(self, from_mode_id: int, to_mode_id: int, timestamp: float) -> None:
        """Record a transition between modes."""
        self.mode_sequence.add_transition(from_mode_id, to_mode_id)
        
        # Update durations
        from_mode = self.modes.get(from_mode_id)
        if from_mode and from_mode.last_start_time:
            duration = timestamp - from_mode.last_start_time
            if duration >= MIN_MODE_DURATION:
                if from_mode_id not in self.mode_durations:
                    self.mode_durations[from_mode_id] = []
                self.mode_durations[from_mode_id].append(duration)
        
        # Update most likely next mode
        from_mode = self.modes.get(from_mode_id)
        if from_mode:
            from_mode.typical_next_mode = to_mode_id
    
    def _identify_mode_name(self, mode_id: int) -> None:
        """Try to identify a human-readable name for a mode."""
        mode = self.modes.get(mode_id)
        if not mode:
            return
            
        # Start with default name
        mode_name = f"mode_{mode_id}"
        
        # Check for heating mode
        heating_indicators_active = False
        for indicator in self.heating_indicators:
            if indicator in mode.characteristic_values:
                # If the heating indicator is significantly positive, it might be heating mode
                if statistics.mean(mode.characteristic_values[indicator]) > 0:
                    heating_indicators_active = True
        
        # Check for cooling mode
        cooling_indicators_active = False
        for indicator in self.cooling_indicators:
            if indicator in mode.characteristic_values:
                # If the cooling indicator is significantly negative, it might be cooling mode
                if statistics.mean(mode.characteristic_values[indicator]) < 0:
                    cooling_indicators_active = True
        
        # Determine the mode name
        if heating_indicators_active and not cooling_indicators_active:
            mode_name = "heating"
        elif cooling_indicators_active and not heating_indicators_active:
            mode_name = "cooling"
        elif not heating_indicators_active and not cooling_indicators_active:
            mode_name = "standby"
        else:
            # Both indicators active could be defrost or transition
            mode_name = "transition"
            
        mode.name = mode_name
    
    def get_current_mode_info(self) -> Dict[str, Any]:
        """Get information about the current mode."""
        if self.current_mode is None:
            return {
                "mode_id": None,
                "mode_name": "unknown",
                "active_since": None,
                "predicted_duration": None,
                "predicted_end": None,
                "predicted_next_mode": None,
                "stability": 0,
                "previous_mode": None,
                "mode_confidence": 0,
            }
        
        current_mode = self.modes.get(self.current_mode)
        if not current_mode:
            return {
                "mode_id": self.current_mode,
                "mode_name": "unknown",
                "active_since": None,
                "predicted_duration": None,
                "predicted_end": None,
                "predicted_next_mode": None,
                "stability": 0,
                "previous_mode": self.previous_mode,
                "mode_confidence": 0,
            }
        
        # Get current time
        now = time.time()
        
        # Format active_since timestamp
        active_since = None
        if current_mode.last_start_time:
            try:
                active_since = datetime.fromtimestamp(current_mode.last_start_time).isoformat()
            except (ValueError, OverflowError):
                active_since = None
        
        # Calculate mode stability - higher value means more stable
        time_in_current_mode = now - current_mode.last_start_time if current_mode.last_start_time else 0
        stability = min(100, (time_in_current_mode / MODE_STABILITY_THRESHOLD) * 100) if MODE_STABILITY_THRESHOLD > 0 else 0
        
        # Predict mode end time
        predicted_end = None
        predicted_duration = current_mode.average_duration if current_mode.occurrences > 0 else None
        
        if current_mode.last_start_time and predicted_duration:
            predicted_end_timestamp = current_mode.last_start_time + predicted_duration
            try:
                predicted_end = datetime.fromtimestamp(predicted_end_timestamp).isoformat()
            except (ValueError, OverflowError):
                predicted_end = None
        
        # Predict next mode
        predicted_next_mode = None
        if current_mode.typical_next_mode is not None:
            next_mode = self.modes.get(current_mode.typical_next_mode)
            if next_mode:
                predicted_next_mode = next_mode.name
        
        # Determine mode confidence
        if current_mode.occurrences > 10:
            mode_confidence = 90
        elif current_mode.occurrences > 5:
            mode_confidence = 75
        elif current_mode.occurrences > 2:
            mode_confidence = 50
        else:
            mode_confidence = 25
            
        # Adjust for stability
        mode_confidence = (mode_confidence + stability) / 2
        
        return {
            "mode_id": self.current_mode,
            "mode_name": current_mode.name,
            "active_since": active_since,
            "predicted_duration": predicted_duration,
            "predicted_end": predicted_end,
            "predicted_next_mode": predicted_next_mode,
            "stability": int(stability),
            "previous_mode": self.previous_mode,
            "mode_confidence": int(mode_confidence),
        }
    
    def get_optimal_scan_interval(self, entity_id: str) -> Optional[int]:
        """Get optimal scan interval for an entity based on current mode."""
        if self.current_mode is None:
            return None
            
        current_mode = self.modes.get(self.current_mode)
        if not current_mode:
            return None
            
        # If we have a specific recommendation for this entity in this mode, use it
        if entity_id in current_mode.entity_scan_intervals:
            return current_mode.entity_scan_intervals[entity_id]
            
        # Otherwise use mode-specific logic
        if current_mode.name == "standby":
            # In standby mode, we can poll less frequently
            return 300  # 5 minutes
        elif current_mode.name == "transition":
            # In transition mode, poll more frequently
            return 60  # 1 minute
        elif entity_id in self.critical_entities:
            # Critical entities should be polled more frequently
            return 120  # 2 minutes
            
        # Default case
        return 180  # 3 minutes
    
    def is_transition_period(self) -> bool:
        """Determine if we're currently in a mode transition period."""
        if self.current_mode is None:
            return False
            
        current_mode = self.modes.get(self.current_mode)
        if not current_mode or not current_mode.last_start_time:
            return False
            
        # Consider first 2 minutes of a mode as transition period
        now = time.time()
        time_in_mode = now - current_mode.last_start_time
        return time_in_mode < 120  # 2 minutes
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about mode detection."""
        modes_info = {}
        for mode_id, mode in self.modes.items():
            modes_info[f"mode_{mode_id}"] = {
                "name": mode.name,
                "occurrences": mode.occurrences,
                "average_duration": mode.average_duration,
                "total_duration": mode.total_duration,
            }
            
        return {
            "detected_modes": len(self.modes),
            "mode_changes": len(self.mode_change_timestamps),
            "current_mode": self.current_mode,
            "current_mode_name": self.modes[self.current_mode].name if self.current_mode in self.modes else "unknown",
            "modes": modes_info,
        }