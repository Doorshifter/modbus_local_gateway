"""
Statistics tracking for Modbus Local Gateway entities.

This module provides statistical analysis of entity polling behavior to help
users optimize scan intervals based on actual data change patterns.
"""

import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Any

_LOGGER = logging.getLogger(__name__)

# Constants for statistics settings
STATS_UPDATE_INTERVAL = 300  # 5 minutes between stats calculations
STATS_MAX_HISTORY = 100      # Maximum change intervals to track
STATS_LOW_CHANGE_THRESHOLD = 5.0  # Percentage threshold to consider inefficient polling
MIN_POLLS_REQUIRED = 5       # Reduced from 10 for faster feedback

# Default values
DEFAULT_INTERVAL = 30  # Default recommended interval if no data
MIN_RECOMMENDED_INTERVAL = 5  # Minimum recommended scan interval
MAX_RECOMMENDED_INTERVAL = 300  # Maximum recommended scan interval
CHANGE_INTERVAL_FACTOR = 3  # Divisor for determining recommended interval (1/3 of avg change interval)


class StatisticMetric(Enum):
    """Enum for statistics metric names."""
    
    CHANGE_PERCENTAGE = "change_percentage"
    AVG_TIME_BETWEEN_CHANGES = "avg_time_between_changes" 
    POLLS_PER_HOUR = "polls_per_hour"
    WASTED_POLLS = "wasted_polls"
    LAST_CHANGE_AGO = "last_change_ago"
    RECOMMENDED_SCAN_INTERVAL = "recommended_scan_interval"
    POLLING_EFFICIENCY = "polling_efficiency"
    LAST_STATS_UPDATE = "last_stats_update"
    INSUFFICIENT_DATA = "insufficient_data"


class EntityStatisticsTracker:
    """Track statistics for a Modbus entity to optimize polling frequency."""
    
    def __init__(self, current_scan_interval: int = 30):
        """Initialize the statistics tracker with the current scan interval."""
        self.current_scan_interval = current_scan_interval
        self.poll_count = 0
        self.error_count = 0
        self.change_count = 0
        self.last_change_timestamp: Optional[float] = None
        self.last_success_time: Optional[str] = None
        self.last_error_time: Optional[str] = None
        self.change_intervals: List[float] = []
        self.recent_values: List[Any] = []
        self.value_changes = 0
        self.response_times: List[float] = []
        self._last_stats_calc = 0
        
        # Initialize with placeholder values so attributes appear immediately
        current_time = time.time()
        self.stats: Dict[str, Any] = {
            StatisticMetric.INSUFFICIENT_DATA.value: True,
            StatisticMetric.POLLS_PER_HOUR.value: round(3600 / current_scan_interval, 1),
            "poll_count": 0,
            "error_count": 0,
            "success_rate": 0.0,
            "last_success_time": None,
            "last_error_time": None,
            "value_changes": 0,
            "recent_values": [],
            "avg_response_time": 0.0,
            "more_polls_needed": MIN_POLLS_REQUIRED,
            StatisticMetric.RECOMMENDED_SCAN_INTERVAL.value: current_scan_interval,
            StatisticMetric.LAST_STATS_UPDATE.value: current_time,
            StatisticMetric.POLLING_EFFICIENCY.value: 0,
            StatisticMetric.CHANGE_PERCENTAGE.value: 0,
            StatisticMetric.WASTED_POLLS.value: 0,
        }
        
    def record_poll(self, value_changed: bool = False, timestamp: float = None, 
                   error: bool = False, response_time: float = None, value: Any = None) -> None:
        """
        Record a polling event with comprehensive information.
        
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
                self.value_changes += 1
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
        
        # Only calculate statistics on milestone polls or periodically
        should_update = (
            # First few polls to establish basic stats
            self.poll_count in (1, 5, 10) or
            # Periodic updates based on time
            (timestamp - self._last_stats_calc) >= STATS_UPDATE_INTERVAL
        )
        
        if should_update:
            self.calculate_statistics(timestamp)
    
    def calculate_statistics(self, current_time: float, force: bool = False) -> Dict[str, Any]:
        """
        Calculate statistics about entity polling efficiency.
        
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
                StatisticMetric.INSUFFICIENT_DATA.value: True,
                StatisticMetric.POLLS_PER_HOUR.value: round(3600 / self.current_scan_interval, 1),
                "poll_count": self.poll_count,
                "error_count": self.error_count,
                "success_rate": round(success_rate, 1),
                "last_success_time": self.last_success_time,
                "last_error_time": self.last_error_time,
                "value_changes": self.value_changes,
                "recent_values": self.recent_values.copy(),
                "avg_response_time": round(avg_response_time, 2),
                "more_polls_needed": MIN_POLLS_REQUIRED - self.poll_count,
                StatisticMetric.LAST_STATS_UPDATE.value: current_time
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
                min(MAX_RECOMMENDED_INTERVAL, int(avg_interval / CHANGE_INTERVAL_FACTOR))
            )
        
        # Calculate polling efficiency
        efficiency = 100
        if change_percentage < STATS_LOW_CHANGE_THRESHOLD:
            efficiency = change_percentage * 20  # Scale to 0-100
        
        time_since_last_change = current_time - self.last_change_timestamp if self.last_change_timestamp else 0
        
        self.stats.update({
            StatisticMetric.INSUFFICIENT_DATA.value: False,  # Now we have enough data
            StatisticMetric.CHANGE_PERCENTAGE.value: round(change_percentage, 1),
            StatisticMetric.AVG_TIME_BETWEEN_CHANGES.value: round(avg_interval, 1),
            StatisticMetric.POLLS_PER_HOUR.value: round(3600 / self.current_scan_interval, 1),
            StatisticMetric.WASTED_POLLS.value: self.poll_count - self.change_count,
            StatisticMetric.LAST_CHANGE_AGO.value: round(time_since_last_change, 1),
            StatisticMetric.RECOMMENDED_SCAN_INTERVAL.value: recommended_interval,
            StatisticMetric.POLLING_EFFICIENCY.value: round(efficiency, 1),
            StatisticMetric.LAST_STATS_UPDATE.value: current_time,
            "poll_count": self.poll_count,
            "error_count": self.error_count,
            "success_rate": round(success_rate, 1),
            "last_success_time": self.last_success_time,
            "last_error_time": self.last_error_time,
            "value_changes": self.value_changes,
            "recent_values": self.recent_values.copy(),
            "avg_response_time": round(avg_response_time, 2),
        })
        self._last_stats_calc = current_time
        return self.stats
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get the current statistics."""
        return self.stats
        
    def reset(self) -> None:
        """Reset all statistics."""
        self.poll_count = 0
        self.error_count = 0
        self.change_count = 0
        self.value_changes = 0
        self.last_change_timestamp = None
        self.last_success_time = None
        self.last_error_time = None
        self.change_intervals = []
        self.recent_values = []
        self.response_times = []
        self._last_stats_calc = 0
        current_time = time.time()
        # Reset to initial placeholder values
        self.stats = {
            StatisticMetric.INSUFFICIENT_DATA.value: True,
            StatisticMetric.POLLS_PER_HOUR.value: round(3600 / self.current_scan_interval, 1),
            "poll_count": 0,
            "error_count": 0,
            "success_rate": 0.0,
            "last_success_time": None,
            "last_error_time": None,
            "value_changes": 0,
            "recent_values": [],
            "avg_response_time": 0.0,
            "more_polls_needed": MIN_POLLS_REQUIRED,
            StatisticMetric.RECOMMENDED_SCAN_INTERVAL.value: self.current_scan_interval,
            StatisticMetric.LAST_STATS_UPDATE.value: current_time,
            StatisticMetric.POLLING_EFFICIENCY.value: 0,
            StatisticMetric.CHANGE_PERCENTAGE.value: 0,
            StatisticMetric.WASTED_POLLS.value: 0,
        }