"""
Unified statistics tracking module for Modbus entities.

This module provides comprehensive statistics tracking, pattern detection,
and optimization capabilities for Modbus entities.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Deque
from collections import deque
import statistics as stats_lib

from .storage_aware import StorageAware
from .interval_manager import INTERVAL_MANAGER
from .pattern_detection import PatternDetector

_LOGGER = logging.getLogger(__name__)

# Constants for statistics settings
STATS_UPDATE_INTERVAL = 300  # 5 minutes between stats calculations
STATS_MAX_HISTORY = 100      # Maximum change intervals to track
STATS_LOW_CHANGE_THRESHOLD = 5.0  # Percentage threshold to consider inefficient polling
MIN_POLLS_REQUIRED = 5       # Minimum polls required for comprehensive statistics

# Default values
DEFAULT_INTERVAL = 30  # Default recommended interval if no data
MIN_RECOMMENDED_INTERVAL = 5  # Minimum recommended scan interval
MAX_RECOMMENDED_INTERVAL = 300  # Maximum recommended scan interval
CHANGE_HISTORY_SIZE = 100    # Maximum number of changes to track
BLACKHOLE_THRESHOLD_HOURS = 48  # Hours without changes to consider entity a "black hole"

# Global pattern detector instance
_GLOBAL_PATTERN_DETECTOR = PatternDetector()


class StatisticMetric:
    """Statistics metric name constants."""
    
    CHANGE_PERCENTAGE = "change_percentage"
    AVG_TIME_BETWEEN_CHANGES = "avg_time_between_changes" 
    POLLS_PER_HOUR = "polls_per_hour"
    WASTED_POLLS = "wasted_polls"
    LAST_CHANGE_AGO = "last_change_ago"
    RECOMMENDED_SCAN_INTERVAL = "recommended_scan_interval"
    POLLING_EFFICIENCY = "polling_efficiency"
    LAST_STATS_UPDATE = "last_stats_update"
    INSUFFICIENT_DATA = "insufficient_data"


class StatisticsTracker(StorageAware):
    """Unified statistics tracker with comprehensive features."""
    
    def __init__(self, entity_id: str = None, current_scan_interval: int = 30, register_count: int = 1):
        """Initialize statistics tracker.
        
        Args:
            entity_id: Entity identifier
            current_scan_interval: Current scan interval in seconds
            register_count: Number of registers this entity represents
        """
        # Initialize storage capabilities
        super().__init__("entity_tracker")
        
        self.entity_id = entity_id
        self.current_scan_interval = current_scan_interval
        self.register_count = register_count
        
        # Basic polling statistics
        self.poll_count = 0
        self.error_count = 0
        self.change_count = 0
        self.last_change_timestamp: Optional[float] = None
        self.last_success_time: Optional[str] = None
        self.last_error_time: Optional[str] = None
        self.change_intervals: List[float] = []
        self.recent_values: List[Any] = []
        self.response_times: List[float] = []
        self._last_stats_calc = 0
        
        # Enhanced tracking for pattern analysis
        self._value_history: Deque[Tuple[float, Any]] = deque(maxlen=CHANGE_HISTORY_SIZE)
        self._change_deltas: Deque[float] = deque(maxlen=CHANGE_HISTORY_SIZE)
        
        # Enhanced metrics
        self._last_change_delta: Optional[float] = None
        self._delta_total_24h: float = 0.0
        self._is_blackhole: bool = False
        
        # Pattern detection
        self._last_enhanced_analysis: float = 0
        self._predicted_next_change: Optional[float] = None
        self._prediction_accuracy: float = 0.0
        
        # Interval optimization
        if entity_id:  # Only register if we have an entity ID
            INTERVAL_MANAGER.register_entity(
                entity_id, 
                register_count=register_count,
                initial_interval=current_scan_interval
            )
        
        # Initialize statistics dictionary with placeholder values
        current_time = time.time()
        self.stats: Dict[str, Any] = {
            StatisticMetric.INSUFFICIENT_DATA: True,
            StatisticMetric.POLLS_PER_HOUR: round(3600 / current_scan_interval, 1),
            "poll_count": 0,
            "error_count": 0,
            "success_rate": 0.0,
            "last_success_time": None,
            "last_error_time": None,
            "value_changes": 0,
            "recent_values": [],
            "avg_response_time": 0.0,
            "more_polls_needed": MIN_POLLS_REQUIRED,
            StatisticMetric.RECOMMENDED_SCAN_INTERVAL: current_scan_interval,
            StatisticMetric.LAST_STATS_UPDATE: current_time,
            StatisticMetric.POLLING_EFFICIENCY: 0,
            StatisticMetric.CHANGE_PERCENTAGE: 0,
            StatisticMetric.WASTED_POLLS: 0,
            # Enhanced stats
            "last_delta": None,
            "change_count_24h": 0,
            "delta_total_24h": 0.0,
            "is_blackhole": False,
            "predicted_next_change": None,
            "prediction_accuracy": "0.0%",
            "current_pattern": None,
            "dynamic_scan_interval": current_scan_interval,
            "register_count": register_count,
        }
    
    def record_poll(self, value_changed: bool = False, timestamp: float = None,
                   error: bool = False, response_time: float = None, value: Any = None) -> None:
        """Record a polling event with comprehensive information.
        
        Args:
            value_changed: True if the entity value changed in this poll
            timestamp: Current timestamp when the poll occurred
            error: True if this poll resulted in an error
            response_time: Time taken for the poll in milliseconds
            value: The actual value retrieved (for recent values tracking)
        """
        if timestamp is None:
            timestamp = time.time()
            
        self.poll_count += 1
        
        # Track success/error
        if error:
            self.error_count += 1
            self.last_error_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        else:
            self.last_success_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            
            # Only track changes and values on successful polls
            if value_changed:
                self.change_count += 1
                if self.last_change_timestamp:
                    interval = timestamp - self.last_change_timestamp
                    self.change_intervals.append(interval)
                    # Keep a reasonable history size
                    if len(self.change_intervals) > STATS_MAX_HISTORY:
                        self.change_intervals.pop(0)
                self.last_change_timestamp = timestamp
            
            # Track recent values (successful polls only)
            if value is not None:
                self.recent_values.append(value)
                if len(self.recent_values) > 20:  # Keep last 20 values
                    self.recent_values.pop(0)
        
        # Track response times
        if response_time is not None:
            self.response_times.append(response_time)
            if len(self.response_times) > 50:  # Keep last 50 response times
                self.response_times.pop(0)
        
        # Always record value history for trend analysis
        self._value_history.append((timestamp, value))
        
        # Process value change with enhanced metrics
        if value_changed and not error and value is not None:
            # Calculate delta if possible
            prev_value = None
            for t, v in reversed(list(self._value_history)[:-1]):
                if v is not None:  # Find last non-None value
                    prev_value = v
                    break
                    
            try:
                # Try to calculate numeric delta
                if prev_value is not None:
                    delta = float(value) - float(prev_value)
                    self._change_deltas.append(delta)
                    self._last_change_delta = delta
                    
                # Reset blackhole flag on any change
                self._is_blackhole = False
                
            except (ValueError, TypeError):
                # Non-numeric values can't have delta
                pass
                
        # Check for "black hole" status (long time without changes)
        elif self.last_change_timestamp and timestamp - self.last_change_timestamp > BLACKHOLE_THRESHOLD_HOURS * 3600:
            self._is_blackhole = True
        
        # Update interval manager with poll result
        if self.entity_id:
            INTERVAL_MANAGER.record_poll(
                self.entity_id,
                timestamp,
                value_changed,
                value,
                error,
                response_time
            )
            
            # Update global pattern detector with this entity's value
            if not error and value is not None:
                _GLOBAL_PATTERN_DETECTOR.update_entity_value(self.entity_id, value, timestamp)
            
        # Run enhanced analysis periodically
        if timestamp - self._last_enhanced_analysis >= STATS_UPDATE_INTERVAL:
            self._enhanced_analysis(timestamp)
        
        # Only calculate basic statistics on milestone polls or periodically
        should_update = (
            # First few polls to establish basic stats
            self.poll_count in (1, 5, 10) or
            # Periodic updates based on time
            (timestamp - self._last_stats_calc) >= STATS_UPDATE_INTERVAL
        )
        
        if should_update:
            self.calculate_statistics(timestamp)
    
    def calculate_statistics(self, current_time: float, force: bool = False) -> Dict[str, Any]:
        """Calculate statistics about entity polling efficiency.
        
        Args:
            current_time: Current timestamp
            force: Force recalculation even if the update interval hasn't passed
            
        Returns:
            Dictionary containing statistical metrics
        """
        # Only recalculate stats periodically to reduce CPU usage
        if not force and current_time - self._last_stats_calc < STATS_UPDATE_INTERVAL:
            return self.stats
            
        # Calculate success rate
        success_rate = ((self.poll_count - self.error_count) / self.poll_count * 100) if self.poll_count > 0 else 0
        
        # Calculate average response time
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
            
        if self.poll_count < MIN_POLLS_REQUIRED:
            # Provide basic statistics even with insufficient data
            self.stats.update({
                StatisticMetric.INSUFFICIENT_DATA: True,
                StatisticMetric.POLLS_PER_HOUR: round(3600 / self.current_scan_interval, 1),
                "poll_count": self.poll_count,
                "error_count": self.error_count,
                "success_rate": round(success_rate, 1),
                "last_success_time": self.last_success_time,
                "last_error_time": self.last_error_time,
                "value_changes": self.change_count,
                "recent_values": self.recent_values.copy(),
                "avg_response_time": round(avg_response_time, 2),
                "more_polls_needed": MIN_POLLS_REQUIRED - self.poll_count,
                StatisticMetric.LAST_STATS_UPDATE: current_time
            })
            self._last_stats_calc = current_time
            return self.stats
            
        change_percentage = (self.change_count / self.poll_count) * 100 if self.poll_count > 0 else 0
        avg_interval = sum(self.change_intervals) / len(self.change_intervals) if self.change_intervals else 0
        
        # Recommend a new interval based on change patterns
        recommended_interval = DEFAULT_INTERVAL
        if avg_interval > 0:
            # Set polling to fraction of average change interval, with limits
            recommended_interval = max(
                MIN_RECOMMENDED_INTERVAL, 
                min(MAX_RECOMMENDED_INTERVAL, int(avg_interval / 3))  # Divide by 3 for safety margin
            )
        
        # Calculate polling efficiency
        efficiency = 100
        if change_percentage < STATS_LOW_CHANGE_THRESHOLD:
            efficiency = change_percentage * 20  # Scale to 0-100
        
        time_since_last_change = current_time - self.last_change_timestamp if self.last_change_timestamp else 0
        
        self.stats.update({
            StatisticMetric.INSUFFICIENT_DATA: False,  # Now we have enough data
            StatisticMetric.CHANGE_PERCENTAGE: round(change_percentage, 1),
            StatisticMetric.AVG_TIME_BETWEEN_CHANGES: round(avg_interval, 1),
            StatisticMetric.POLLS_PER_HOUR: round(3600 / self.current_scan_interval, 1),
            StatisticMetric.WASTED_POLLS: self.poll_count - self.change_count,
            StatisticMetric.LAST_CHANGE_AGO: round(time_since_last_change, 1),
            StatisticMetric.RECOMMENDED_SCAN_INTERVAL: recommended_interval,
            StatisticMetric.POLLING_EFFICIENCY: round(efficiency, 1),
            StatisticMetric.LAST_STATS_UPDATE: current_time,
            "poll_count": self.poll_count,
            "error_count": self.error_count,
            "success_rate": round(success_rate, 1),
            "last_success_time": self.last_success_time,
            "last_error_time": self.last_error_time,
            "value_changes": self.change_count,
            "recent_values": self.recent_values.copy(),
            "avg_response_time": round(avg_response_time, 2),
        })
        self._last_stats_calc = current_time
        return self.stats
    
    def _enhanced_analysis(self, current_time: float) -> None:
        """Perform enhanced pattern analysis."""
        self._last_enhanced_analysis = current_time
        
        # Count changes and deltas in the last 24 hours
        day_ago = current_time - 86400
        
        # Filter recent changes and calculate 24h metrics
        change_count_24h = 0
        delta_total = 0.0
        
        # Count changes in last 24h
        for t in self.change_intervals:
            if self.last_change_timestamp and self.last_change_timestamp - t > day_ago:
                change_count_24h += 1
                
        # Sum deltas in last 24h (if they exist)
        for delta in self._change_deltas:
            delta_total += delta
            
        # Update enhanced metrics
        self.stats.update({
            "last_delta": self._last_change_delta,
            "change_count_24h": change_count_24h,
            "delta_total_24h": delta_total,
            "is_blackhole": self._is_blackhole,
        })
        
        # Get pattern information
        if self.entity_id:
            pattern_info = _GLOBAL_PATTERN_DETECTOR.get_current_pattern_info()
            self.stats.update({
                "current_pattern": pattern_info.get("pattern_id"),
                "pattern_confidence": pattern_info.get("pattern_confidence"),
                "pattern_since": pattern_info.get("active_since"),
                "predicted_pattern_end": pattern_info.get("predicted_end"),
                "next_predicted_pattern": pattern_info.get("predicted_next_pattern"),
            })
            
            # Get optimal scan interval from pattern detector
            pattern_interval = _GLOBAL_PATTERN_DETECTOR.get_optimal_scan_interval(self.entity_id)
            if pattern_interval:
                self.stats["pattern_based_interval"] = pattern_interval
        
        # Predict next change time
        self._predict_next_change(current_time)
        
        # Update recommended scan interval from interval manager
        if self.entity_id:
            recommended_interval = INTERVAL_MANAGER.get_recommended_interval(
                self.entity_id, current_time
            )
            self.stats["recommended_scan_interval"] = recommended_interval
            self.stats["dynamic_scan_interval"] = recommended_interval
            
        # Save entity stats to storage if available
        self._save_to_storage()
    
    def _predict_next_change(self, current_time: float) -> None:
        """Predict next change time based on historical patterns."""
        if not self.change_intervals or len(self.change_intervals) < 2:
            # Not enough data for prediction
            return
            
        # Calculate average interval between changes
        avg_interval = sum(self.change_intervals) / len(self.change_intervals)
        
        # Calculate standard deviation if we have enough samples
        if len(self.change_intervals) > 2:
            try:
                stdev_interval = stats_lib.stdev(self.change_intervals)
            except stats_lib.StatisticsError:
                stdev_interval = avg_interval * 0.25  # Fallback
        else:
            stdev_interval = avg_interval * 0.25  # Estimate
        
        # Make prediction
        if self.last_change_timestamp:
            self._predicted_next_change = self.last_change_timestamp + avg_interval
            
            # Format for display
            try:
                predicted_time = datetime.fromtimestamp(self._predicted_next_change).isoformat()
            except (ValueError, OverflowError):
                predicted_time = None
                
            self.stats["predicted_next_change"] = predicted_time
            
            # Check prediction accuracy if we have a previous prediction to compare
            if current_time > self._predicted_next_change:
                # We can evaluate our prediction
                accuracy = self._evaluate_prediction(current_time)
        
    def _evaluate_prediction(self, current_time: float) -> Optional[float]:
        """Evaluate accuracy of previous prediction."""
        if not self._predicted_next_change or not self.last_change_timestamp:
            return None
            
        # Check if a change happened after our prediction
        if self.last_change_timestamp > self._predicted_next_change:
            # A change did occur - how close was our prediction?
            error = abs(self.last_change_timestamp - self._predicted_next_change)
            avg_interval = sum(self.change_intervals) / len(self.change_intervals) if self.change_intervals else 3600
            
            # Normalize error by average interval
            accuracy = max(0.0, 1.0 - (error / avg_interval))
            
            # Update running accuracy (weighted average)
            self._prediction_accuracy = 0.7 * self._prediction_accuracy + 0.3 * accuracy
            
            # Format for display
            self.stats["prediction_accuracy"] = f"{self._prediction_accuracy:.1%}"
            
            return accuracy
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about entity polling.
        
        Returns:
            Dictionary with statistics
        """
        # Add interval manager data for this entity if available
        if self.entity_id and hasattr(INTERVAL_MANAGER, "get_entity_statistics"):
            entity_stats = INTERVAL_MANAGER.get_entity_statistics(self.entity_id)
            if entity_stats:
                self.stats["interval_stats"] = entity_stats
        
        return self.stats
        
    def reset(self) -> None:
        """Reset all statistics."""
        self.poll_count = 0
        self.error_count = 0
        self.change_count = 0
        self.last_change_timestamp = None
        self.last_success_time = None
        self.last_error_time = None
        self.change_intervals = []
        self.recent_values = []
        self.response_times = []
        self._last_stats_calc = 0
        self._value_history.clear()
        self._change_deltas.clear()
        self._last_change_delta = None
        self._delta_total_24h = 0.0
        self._is_blackhole = False
        self._predicted_next_change = None
        self._prediction_accuracy = 0.0
        
        # Reset statistics dictionary
        current_time = time.time()
        self.stats = {
            StatisticMetric.INSUFFICIENT_DATA: True,
            StatisticMetric.POLLS_PER_HOUR: round(3600 / self.current_scan_interval, 1),
            "poll_count": 0,
            "error_count": 0,
            "success_rate": 0.0,
            "last_success_time": None,
            "last_error_time": None,
            "value_changes": 0,
            "recent_values": [],
            "avg_response_time": 0.0,
            "more_polls_needed": MIN_POLLS_REQUIRED,
            StatisticMetric.RECOMMENDED_SCAN_INTERVAL: self.current_scan_interval,
            StatisticMetric.LAST_STATS_UPDATE: current_time,
            StatisticMetric.POLLING_EFFICIENCY: 0,
            StatisticMetric.CHANGE_PERCENTAGE: 0,
            StatisticMetric.WASTED_POLLS: 0,
            "last_delta": None,
            "change_count_24h": 0,
            "delta_total_24h": 0.0,
            "is_blackhole": False,
            "predicted_next_change": None,
            "prediction_accuracy": "0.0%",
            "current_pattern": None,
            "dynamic_scan_interval": self.current_scan_interval,
            "register_count": self.register_count,
        }
    
    def _load_from_storage(self) -> None:
        """Load entity statistics from storage."""
        if not self._storage_manager or not self.entity_id:
            return
            
        entity_stats = self._storage_manager.get_entity_stats().get(self.entity_id)
        if not entity_stats:
            return
            
        # Restore only metrics that make sense to persist
        # Don't restore dynamic counters like poll_count
        if "recommended_scan_interval" in entity_stats:
            self.stats[StatisticMetric.RECOMMENDED_SCAN_INTERVAL] = entity_stats["recommended_scan_interval"]
            
        if "is_blackhole" in entity_stats:
            self._is_blackhole = entity_stats["is_blackhole"]
            self.stats["is_blackhole"] = entity_stats["is_blackhole"]
    
    def _save_to_storage(self) -> bool:
        """Save entity statistics to storage."""
        if not self._storage_manager or not self.entity_id:
            return False
            
        # Save to storage
        try:
            self._storage_manager.save_entity_stats(self.entity_id, self.stats)
            return True
        except Exception as e:
            _LOGGER.error("Error saving entity stats: %s", e)
            return False