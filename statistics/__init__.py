"""
Statistics package for Modbus Local Gateway.

Contains modules for tracking entity behavior, identifying patterns,
detecting correlations, and optimizing scan intervals.
"""

import logging

_LOGGER = logging.getLogger(__name__)

# Dictionary to store lazy-loaded components
_COMPONENTS = {}

def get_statistics_manager():
    """Get the statistics manager instance."""
    if "statistics_manager" not in _COMPONENTS:
        from .manager import ModbusStatisticsManager
        _COMPONENTS["statistics_manager"] = ModbusStatisticsManager.get_instance()
    return _COMPONENTS["statistics_manager"]

def get_self_healing_system():
    """Get the self healing system instance."""
    if "self_healing_system" not in _COMPONENTS:
        from .self_healing import SELF_HEALING_SYSTEM
        _COMPONENTS["self_healing_system"] = SELF_HEALING_SYSTEM
    return _COMPONENTS["self_healing_system"]

def get_resource_adapter():
    """Get the resource adapter instance."""
    if "resource_adapter" not in _COMPONENTS:
        from .resource_adaptation import RESOURCE_ADAPTER
        _COMPONENTS["resource_adapter"] = RESOURCE_ADAPTER
    return _COMPONENTS["resource_adapter"]

def get_persistent_statistics_manager():
    """Get the persistent statistics manager instance."""
    if "persistent_statistics_manager" not in _COMPONENTS:
        from .persistent_statistics import PERSISTENT_STATISTICS_MANAGER
        _COMPONENTS["persistent_statistics_manager"] = PERSISTENT_STATISTICS_MANAGER
    return _COMPONENTS["persistent_statistics_manager"]

def get_performance_optimizer():
    """Get the performance optimizer instance."""
    if "performance_optimizer" not in _COMPONENTS:
        from .performance_optimization import PERFORMANCE_OPTIMIZER
        _COMPONENTS["performance_optimizer"] = PERFORMANCE_OPTIMIZER
    return _COMPONENTS["performance_optimizer"]

def get_advanced_interval_manager():
    """Get the advanced interval manager instance."""
    if "advanced_interval_manager" not in _COMPONENTS:
        from .advanced_interval_manager import ADVANCED_INTERVAL_MANAGER
        _COMPONENTS["advanced_interval_manager"] = ADVANCED_INTERVAL_MANAGER
    return _COMPONENTS["advanced_interval_manager"]

def initialize_all():
    """Initialize all statistics systems in the correct order."""
    try:
        # Import systems
        from .interval_system import initialize_interval_system
        from .storage_integration import initialize_statistics_storage
        
        # First initialize interval system
        initialize_interval_system()
        
        # Get the statistics manager (creates it if needed)
        statistics_manager = get_statistics_manager()
        
        # Initialize storage for components
        initialize_statistics_storage(statistics_manager)
        
        # Initialize advanced interval manager with components
        persistent_manager = get_persistent_statistics_manager()
        persistent_manager.initialize_advanced_interval_manager()
        
        _LOGGER.info("All statistics components initialized")
        return True
    except Exception as e:
        _LOGGER.error("Failed to initialize statistics components: %s", e)
        return False

# Direct imports for types only - these don't cause circular import problems
from .statistics_tracker import StatisticMetric
try:
    from .statistics_tracker import StatisticsTracker
    from .pattern_detection import PatternDetector
    from .correlation import EntityCorrelationManager
    from .interval_visualization import IntervalVisualizationTool
except ImportError as e:
    _LOGGER.debug("Some statistics modules could not be imported directly: %s", e)

# Export commonly used components for backward compatibility
STATISTICS_MANAGER = None
try:
    from .manager import STATISTICS_MANAGER
except ImportError:
    pass

# Create a fallback with a warning if needed
if STATISTICS_MANAGER is None:
    _LOGGER.warning("STATISTICS_MANAGER not available at import time, use get_statistics_manager() instead")
    
    class DummyStatisticsManager:
        """Dummy statistics manager that logs operations."""
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    STATISTICS_MANAGER = DummyStatisticsManager()