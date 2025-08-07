"""
Statistics package for Modbus Local Gateway.

Contains modules for tracking entity behavior, identifying patterns,
detecting correlations, and optimizing scan intervals.
"""

from .statistics_tracker import StatisticsTracker, StatisticMetric
from .manager import ModbusStatisticsManager
from .self_healing import SELF_HEALING_SYSTEM
from .resource_adaptation import RESOURCE_ADAPTER
from .performance_optimization import PERFORMANCE_OPTIMIZER
from .interval_system import initialize_interval_system
from .storage_integration import initialize_statistics_storage
from .pattern_detection import PatternDetector
from .correlation import EntityCorrelationManager
from .persistent_statistics import PERSISTENT_STATISTICS_MANAGER
from .advanced_interval_manager import ADVANCED_INTERVAL_MANAGER
from .interval_visualization import IntervalVisualizationTool

# Initialize systems
initialize_interval_system()

# Create singleton instance of statistics manager
STATISTICS_MANAGER = ModbusStatisticsManager()

# Initialize storage for components
initialize_statistics_storage(STATISTICS_MANAGER)

# Initialize advanced interval manager with components
PERSISTENT_STATISTICS_MANAGER.initialize_advanced_interval_manager()

__all__ = [
    "StatisticsTracker", "StatisticMetric", 
    "STATISTICS_MANAGER", "PERSISTENT_STATISTICS_MANAGER",
    "SELF_HEALING_SYSTEM", "RESOURCE_ADAPTER", 
    "PERFORMANCE_OPTIMIZER", "PatternDetector", 
    "EntityCorrelationManager", "ADVANCED_INTERVAL_MANAGER",
    "IntervalVisualizationTool"
]