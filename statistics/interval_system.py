"""
Interval management system initialization.

This module initializes the unified interval management system
and integrates it with the statistics storage system.
"""

import logging
from typing import Optional

from .interval_manager import IntervalManager, INTERVAL_MANAGER
from .persistent_statistics import PERSISTENT_STATISTICS_MANAGER

_LOGGER = logging.getLogger(__name__)

def initialize_interval_system(min_interval: Optional[int] = None,
                             max_interval: Optional[int] = None,
                             max_registers_per_second: Optional[int] = None) -> IntervalManager:
    """Initialize the unified interval management system.
    
    Args:
        min_interval: Minimum scan interval
        max_interval: Maximum scan interval
        max_registers_per_second: Maximum registers per second limit
        
    Returns:
        Initialized interval manager instance
    """
    # Get instances
    manager = INTERVAL_MANAGER
    storage = PERSISTENT_STATISTICS_MANAGER.storage
    
    # Apply custom settings if provided
    if min_interval is not None:
        manager.min_interval = min_interval
    if max_interval is not None:
        manager.max_interval = max_interval
    if max_registers_per_second is not None:
        manager.max_registers_per_second = max_registers_per_second
    
    # Connect storage
    manager.set_storage(storage)
    
    _LOGGER.info("Interval management system initialized")
    
    return manager