"""
Representation of Modbus Gateway Context.

This context object encapsulates all necessary information about a Modbus entity,
including its slave address, entity description, and statistical tracking.
"""

import time
from dataclasses import dataclass
from typing import Optional, Any
from .entity_management.base import ModbusEntityDescription
from .statistics import EntityStatisticsTracker

@dataclass
class ModbusContext:
    """
    Context object for use with the Modbus coordinator and batching logic.
    """
    slave_id: int
    desc: ModbusEntityDescription
    scan_interval: int = 30
    last_update: float = 0.0
    statistics: Optional[EntityStatisticsTracker] = None
    last_value: Any = None

    def __post_init__(self):
        """Initialize derived attributes after main initialization."""
        if self.scan_interval <= 0:
            raise ValueError("scan_interval must be positive")
        
        # Initialize statistics tracker with current scan interval
        if self.statistics is None:
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

    def __repr__(self):
        return (f"ModbusContext(slave_id={self.slave_id}, "
                f"desc.key={getattr(self.desc, 'key', None)}, "
                f"scan_interval={self.scan_interval}, "
                f"last_update={self.last_update})")