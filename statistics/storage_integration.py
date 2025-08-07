"""
Storage integration for statistics components.

This module handles initializing and connecting storage-aware components
with the persistent storage system.
"""

import logging
from typing import Optional

from .storage_aware import StorageAware
from .persistent_statistics import PERSISTENT_STATISTICS_MANAGER

_LOGGER = logging.getLogger(__name__)

def initialize_statistics_storage(statistics_manager) -> None:
    """Initialize storage for statistics components through the manager.
    
    Args:
        statistics_manager: The ModbusStatisticsManager instance
    """
    storage = PERSISTENT_STATISTICS_MANAGER.storage
    
    # Get components from manager that need storage integration
    components = []
    
    # Add correlation manager if available
    if hasattr(statistics_manager, "_correlation_manager"):
        components.append(statistics_manager._correlation_manager)
        
    # Add pattern detector if available
    if hasattr(statistics_manager, "_pattern_detector"):
        components.append(statistics_manager._pattern_detector)
    
    # Initialize storage for each component
    for component in components:
        if isinstance(component, StorageAware):
            try:
                component.set_storage_manager(storage)
                _LOGGER.debug("Initialized storage for %s", component._component_name)
            except Exception as e:
                _LOGGER.error("Error initializing storage for component: %s", e)
        else:
            _LOGGER.debug("Component is not storage-aware, skipping")