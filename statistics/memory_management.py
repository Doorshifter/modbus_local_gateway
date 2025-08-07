"""
Memory management for Modbus optimization data.

This module provides utilities for managing historical data and
optimizing memory usage for statistical analysis.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, TypeVar, Generic, Callable
import statistics
from collections import deque

_LOGGER = logging.getLogger(__name__)

T = TypeVar('T')

class RollingBuffer(Generic[T]):
    """
    A rolling buffer that efficiently stores a fixed number of values.
    Automatically calculates statistics over the stored data.
    """
    
    def __init__(self, max_size: int = 100, stat_calc_interval: int = 60):
        """Initialize rolling buffer.
        
        Args:
            max_size: Maximum number of items to store
            stat_calc_interval: How often (in seconds) to recalculate stats
        """
        self._buffer = deque(maxlen=max_size)
        self._max_size = max_size
        self._stat_calc_interval = stat_calc_interval
        self._last_stat_calc = 0
        self._stats = {}
    
    def add(self, value: T) -> None:
        """Add a value to the buffer."""
        self._buffer.append(value)
        
        # Check if we should recalculate stats
        now = time.time()
        if now - self._last_stat_calc >= self._stat_calc_interval:
            self._calculate_stats()
    
    def _calculate_stats(self) -> None:
        """Calculate statistics on the buffer data.
        Only applicable for numeric data.
        """
        self._last_stat_calc = time.time()
        
        try:
            # Check if we can calculate numeric statistics
            if self._buffer and all(isinstance(x, (int, float)) for x in self._buffer):
                values = list(self._buffer)
                
                self._stats = {
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "count": len(values),
                    "last_value": values[-1] if values else None,
                    "updated_at": datetime.now().isoformat()
                }
                
                # Calculate standard deviation if we have enough values
                if len(values) >= 2:
                    try:
                        self._stats["stdev"] = statistics.stdev(values)
                    except Exception:
                        self._stats["stdev"] = 0
                else:
                    self._stats["stdev"] = 0
            else:
                self._stats = {
                    "count": len(self._buffer),
                    "updated_at": datetime.now().isoformat()
                }
        except Exception as e:
            _LOGGER.warning("Error calculating stats: %s", e)
            self._stats = {
                "count": len(self._buffer),
                "error": str(e),
                "updated_at": datetime.now().isoformat()
            }
    
    def clear(self) -> None:
        """Clear all data in the buffer."""
        self._buffer.clear()
        self._stats = {}
    
    def get_values(self) -> List[T]:
        """Get all values in the buffer as a list."""
        return list(self._buffer)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the data in the buffer."""
        # Recalculate if not done recently
        now = time.time()
        if now - self._last_stat_calc >= self._stat_calc_interval:
            self._calculate_stats()
            
        return self._stats
    
    @property
    def count(self) -> int:
        """Get the number of items in the buffer."""
        return len(self._buffer)
    
    @property
    def full(self) -> bool:
        """Check if the buffer is full."""
        return len(self._buffer) == self._max_size


class TimestampedBuffer(RollingBuffer[Dict[str, Any]]):
    """A rolling buffer for timestamped values with downsampling capability."""
    
    def __init__(
        self, 
        max_size: int = 100, 
        stat_calc_interval: int = 60,
        max_age_hours: Optional[int] = None,
        value_key: str = "value"
    ):
        """Initialize timestamped buffer.
        
        Args:
            max_size: Maximum number of items to store
            stat_calc_interval: How often (in seconds) to recalculate stats
            max_age_hours: Maximum age of data to keep (None = keep all up to max_size)
            value_key: Key in dict where the numeric value is stored
        """
        super().__init__(max_size, stat_calc_interval)
        self._max_age_hours = max_age_hours
        self._value_key = value_key
        self._downsampling_enabled = False
        self._original_max_size = max_size
        
        # Downsampling settings
        self._downsampling_threshold = max_size * 0.9  # Start downsampling at 90% capacity
        self._downsampled_resolution = 300  # 5 minutes (in seconds)
    
    def add(self, timestamp: float, value: Any, metadata: Dict[str, Any] = None) -> None:
        """Add a timestamped value to the buffer.
        
        Args:
            timestamp: Unix timestamp
            value: Value to store
            metadata: Additional metadata to store with the value
        """
        item = {
            "timestamp": timestamp,
            self._value_key: value
        }
        
        if metadata:
            item.update(metadata)
            
        super().add(item)
        
        # Check for old data to remove if max_age_hours is set
        self._purge_old_data()
        
        # Check if we should enable downsampling
        if not self._downsampling_enabled and len(self._buffer) > self._downsampling_threshold:
            self._enable_downsampling()
    
    def _purge_old_data(self) -> None:
        """Remove data older than max_age_hours."""
        if self._max_age_hours is None:
            return
            
        now = time.time()
        cutoff = now - (self._max_age_hours * 3600)
        
        while self._buffer and self._buffer[0]["timestamp"] < cutoff:
            self._buffer.popleft()
    
    def _enable_downsampling(self) -> None:
        """Enable downsampling to prevent buffer overflow."""
        if self._downsampling_enabled:
            return
            
        _LOGGER.debug("Enabling downsampling for buffer")
        self._downsampling_enabled = True
        
        # Group values by time buckets
        buckets = {}
        for item in self._buffer:
            # Round timestamp to bucket boundary
            bucket_time = int(item["timestamp"] // self._downsampled_resolution) * self._downsampled_resolution
            if bucket_time not in buckets:
                buckets[bucket_time] = []
            buckets[bucket_time].append(item)
        
        # Keep only one value per bucket (the latest one in each bucket)
        new_buffer = deque(maxlen=self._max_size)
        for bucket_time in sorted(buckets.keys()):
            items = buckets[bucket_time]
            latest_item = max(items, key=lambda x: x["timestamp"])
            if len(items) > 1:
                # If we're combining multiple items, add a count
                latest_item["downsampled_count"] = len(items)
            new_buffer.append(latest_item)
        
        self._buffer = new_buffer
        _LOGGER.debug("Downsampling reduced buffer size from %d to %d", 
                    sum(len(items) for items in buckets.values()), len(new_buffer))
    
    def get_values_in_range(self, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """Get values within a time range.
        
        Args:
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)
            
        Returns:
            List of values within the time range
        """
        return [item for item in self._buffer 
                if start_time <= item["timestamp"] <= end_time]
    
    def get_value_at_time(self, target_time: float, interpolate: bool = False) -> Optional[Dict[str, Any]]:
        """Get the value at a specific time.
        
        Args:
            target_time: Target timestamp
            interpolate: If True, interpolate between values if exact time not found
            
        Returns:
            Value at the specified time, or None if not found
        """
        # Fast path: check if buffer is empty
        if not self._buffer:
            return None
            
        # Check if target_time is outside buffer range
        earliest = self._buffer[0]["timestamp"]
        latest = self._buffer[-1]["timestamp"]
        
        if target_time < earliest or target_time > latest:
            return None
            
        # Try to find exact match
        for item in reversed(self._buffer):
            if item["timestamp"] == target_time:
                return item
        
        if not interpolate:
            # Find closest item
            closest = min(self._buffer, key=lambda x: abs(x["timestamp"] - target_time))
            return closest
            
        # Interpolate between values
        before = None
        after = None
        
        for item in self._buffer: