"""
Representation of Modbus Gateway Context.

This context object encapsulates all necessary information about a Modbus entity,
including its slave address, entity description, and statistical tracking.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Any, Dict

from .entity_management.base import ModbusEntityDescription

_LOGGER = logging.getLogger(__name__)

@dataclass
class ModbusContext:
    """
    Context object for use with the Modbus coordinator and batching logic.
    """
    slave_id: int
    desc: ModbusEntityDescription
    scan_interval: int = 30
    last_update: float = 0.0
    statistics: Optional[Any] = None  # Type as Any to avoid circular imports
    last_value: Any = None

    def __post_init__(self):
        """Initialize derived attributes after main initialization."""
        if self.scan_interval <= 0:
            raise ValueError("scan_interval must be positive")
        
        # Initialize statistics tracker with current scan interval
        if self.statistics is None:
            self._initialize_statistics()

    def _initialize_statistics(self):
        """Initialize statistics tracker lazily to avoid circular imports."""
        try:
            # Import here to avoid circular imports at module level
            from .statistics.statistics_tracker import StatisticsTracker as EntityStatisticsTracker
            
            self.statistics = EntityStatisticsTracker(self.scan_interval)
            # Immediately populate with placeholder values so dropdown appears
            self.statistics.stats = {
                "insufficient_data": True,
                "poll_count": 0, 
                "change_count": 0,
                "polls_per_hour": round(3600 / self.scan_interval, 1),
                "recommended_scan_interval": self.scan_interval,
                "last_stats_update": time.time()
            }
        except ImportError as e:
            _LOGGER.warning("Could not import StatisticsTracker: %s", e)
            
            # Create a minimal placeholder object
            class DummyStatistics:
                """Dummy statistics tracker that stores data but does no processing."""
                def __init__(self):
                    self.stats = {
                        "insufficient_data": True,
                        "poll_count": 0,
                        "polls_per_hour": round(3600 / self.scan_interval, 1),
                        "recommended_scan_interval": self.scan_interval,
                        "last_stats_update": time.time()
                    }
                
                def __getattr__(self, name):
                    return lambda *args, **kwargs: None
            
            self.statistics = DummyStatistics()

    def __repr__(self):
        return (f"ModbusContext(slave_id={self.slave_id}, "
                f"desc.key={getattr(self.desc, 'key', None)}, "
                f"scan_interval={self.scan_interval}, "
                f"last_update={self.last_update})")