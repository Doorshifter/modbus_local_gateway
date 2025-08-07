"""
Performance optimization loop for Modbus polling.

This module provides a continuous improvement cycle that:
1. Monitors system performance metrics
2. Applies and evaluates optimization strategies
3. Learns from previous optimizations
4. Automatically adjusts parameters for optimal reliability and efficiency
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
import statistics
from collections import deque

from .analysis_scheduler import AnalysisScheduler
from .resource_adaptation import RESOURCE_ADAPTER
from .storage import StatisticsStorageManager
from .prediction_evaluation import PredictionEvaluator
from .self_healing import SELF_HEALING_SYSTEM

_LOGGER = logging.getLogger(__name__)

# Constants for optimization
DEFAULT_LEARNING_RATE = 0.2  # How quickly to adapt based on feedback (0-1)
MIN_SAMPLES_FOR_LEARNING = 5  # Minimum number of optimization cycles before learning
MAX_HISTORY_SIZE = 50  # Maximum number of optimization results to store
OPTIMIZATION_COOLDOWN = 12 * 3600  # 12 hours between major optimizations
QUICK_OPTIMIZATION_INTERVAL = 4 * 3600  # 4 hours between quick optimizations

class OptimizationStrategy:
    """Represents a specific optimization strategy."""
    
    def __init__(self, name: str, priority: int = 5, impact_level: str = "medium"):
        """Initialize an optimization strategy.
        
        Args:
            name: Strategy name
            priority: Priority level (1-10, higher = more important)
            impact_level: Impact on system (low, medium, high)
        """
        self.name = name
        self.priority = priority
        self.impact_level = impact_level
        self.success_rate = 0.5  # Start with neutral success rate
        self.application_count = 0
        self.last_application = None
        self.last_improvement = 0.0  # Percentage improvement
        self.cooldown_seconds = 0  # Additional cooldown beyond global cooldown
        
        # Strategy-specific parameters
        self.parameters: Dict[str, Any] = {}
    
    def apply(self, optimization_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply this strategy to the system.
        
        Args:
            optimization_context: Context information for optimization
            
        Returns:
            Dictionary with optimization results
        """
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement apply()")
    
    def evaluate_result(self, before_metrics: Dict[str, Any], 
                       after_metrics: Dict[str, Any]) -> float:
        """Evaluate how effective this strategy was.
        
        Args:
            before_metrics: System metrics before optimization
            after_metrics: System metrics after optimization
            
        Returns:
            Improvement score (-1.0 to 1.0, higher is better)
        """
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement evaluate_result()")
    
    def update_success_rate(self, improvement_score: float) -> None:
        """Update success rate based on improvement score.
        
        Args:
            improvement_score: Improvement score from evaluate_result()
        """
        # Use exponential moving average to update success rate
        weight = min(0.8, self.application_count / 10.0)  # Weight increases with experience
        self.success_rate = (self.success_rate * weight) + (improvement_score > 0) * (1 - weight)
        
        # Record improvement
        self.last_improvement = improvement_score
        self.application_count += 1
        self.last_application = datetime.utcnow().isoformat()


class IntervalOptimizationStrategy(OptimizationStrategy):
    """Strategy for optimizing polling intervals."""
    
    def __init__(self):
        """Initialize interval optimization strategy."""
        super().__init__(
            name="interval_optimization", 
            priority=8,
            impact_level="medium"
        )
        self.parameters = {
            "min_interval": 5,
            "max_interval": 300,
            "aggressiveness": 0.5,  # 0-1, higher = more aggressive changes
            "max_entities_per_run": 100
        }
    
    def apply(self, optimization_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply interval optimization strategy."""
        start_time = time.time()
        entity_stats = optimization_context.get("entity_stats", {})
        interval_manager = optimization_context.get("interval_manager")
        
        if not entity_stats or not interval_manager:
            return {
                "success": False,
                "error": "Missing required context data",
                "entities_optimized": 0
            }
            
        # Sort entities by optimization potential
        entities_by_potential = self._sort_entities_by_potential(entity_stats)
        
        # Limit number of entities to optimize
        entity_limit = self.parameters["max_entities_per_run"]
        entities_to_optimize = entities_by_potential[:entity_limit]
        
        optimization_results = {}
        optimized_count = 0
        
        # Apply optimizations
        for entity_id in entities_to_optimize:
            # Get current interval
            current_interval = entity_stats.get(entity_id, {}).get("current_scan_interval")
            if not current_interval:
                continue
                
            # Calculate optimal interval
            optimal_interval = self._calculate_optimal_interval(entity_id, entity_stats)
            if optimal_interval == current_interval:
                continue
            
            # Apply new interval
            try:
                interval_manager.get_recommended_interval(
                    entity_id, time.time(), force_recalculate=True
                )
                optimization_results[entity_id] = {
                    "before": current_interval,
                    "after": optimal_interval
                }
                optimized_count += 1
            except Exception as e:
                _LOGGER.warning("Failed to optimize interval for %s: %s", entity_id, e)
                
        return {
            "success": True,
            "entities_optimized": optimized_count,
            "duration_seconds": time.time() - start_time,
            "details": optimization_results
        }
    
    def evaluate_result(self, before_metrics: Dict[str, Any], 
                       after_metrics: Dict[str, Any]) -> float:
        """Evaluate effectiveness of interval optimization."""
        # Get polling efficiency before and after
        before_efficiency = before_metrics.get("polling_efficiency", 0)
        after_efficiency = after_metrics.get("polling_efficiency", 0)
        
        # Get polling load before and after
        before_load = before_metrics.get("polls_per_second", 0)
        after_load = after_metrics.get("polls_per_second", 0)
        
        # Calculate efficiency improvement
        efficiency_change = after_efficiency - before_efficiency
        
        # Calculate load reduction (positive = good)
        load_change = 0
        if before_load > 0:
            load_change = (before_load - after_load) / before_load
        
        # Combined score (weighted more toward efficiency)
        improvement_score = (efficiency_change * 0.7) + (load_change * 0.3)
        
        # Scale to -1 to 1 range
        return max(-1.0, min(1.0, improvement_score))
    
    def _sort_entities_by_potential(self, entity_stats: Dict[str, Dict[str, Any]]) -> List[str]:
        """Sort entities by optimization potential."""
        # Calculate optimization score for each entity
        entity_scores = []
        
        for entity_id, stats in entity_stats.items():
            # Skip entities with insufficient data
            if stats.get("insufficient_data", True):
                continue
                
            # Calculate optimization potential score
            polling_efficiency = stats.get("polling_efficiency", 0)
            change_percentage = stats.get("change_percentage", 0)
            recommended_interval = stats.get("recommended_scan_interval")
            current_interval = stats.get("current_scan_interval")
            
            if not recommended_interval or not current_interval:
                continue
                
            # Higher score = higher optimization potential
            interval_diff = abs(recommended_interval - current_interval) / max(current_interval, 1)
            efficiency_factor = (100 - polling_efficiency) / 100  # Lower efficiency = higher potential
            
            score = interval_diff * efficiency_factor
            
            entity_scores.append((entity_id, score))
        
        # Sort by score (highest potential first)
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the entity IDs
        return [entity_id for entity_id, _ in entity_scores]
    
    def _calculate_optimal_interval(self, entity_id: str, entity_stats: Dict[str, Dict[str, Any]]) -> int:
        """Calculate optimal polling interval for an entity."""
        stats = entity_stats.get(entity_id, {})
        
        # Get recommended interval from statistics
        recommended_interval = stats.get("recommended_scan_interval")
        current_interval = stats.get("current_scan_interval")
        
        if not recommended_interval or not current_interval:
            return current_interval or 30  # Default to current or 30 seconds
            
        # Apply aggressiveness factor to control how quickly we adapt
        aggressiveness = self.parameters["aggressiveness"]
        
        # Calculate new interval (weighted average)
        new_interval = int((current_interval * (1 - aggressiveness)) + 
                         (recommended_interval * aggressiveness))
        
        # Ensure within bounds
        min_interval = self.parameters["min_interval"]
        max_interval = self.parameters["max_interval"]
        
        return max(min_interval, min(max_interval, new_interval))


class ClusterOptimizationStrategy(OptimizationStrategy):
    """Strategy for optimizing entity clustering."""
    
    def __init__(self):
        """Initialize cluster optimization strategy."""
        super().__init__(
            name="cluster_optimization",
            priority=6,
            impact_level="high"
        )
        self.parameters = {
            "correlation_threshold": 0.6,
            "min_cluster_size": 2,
            "max_cluster_size": 20
        }
        self.cooldown_seconds = 24 * 3600  # 24 hour cooldown (expensive operation)
    
    def apply(self, optimization_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply cluster optimization strategy."""
        start_time = time.time()
        correlation_manager = optimization_context.get("correlation_manager")
        
        if not correlation_manager:
            return {
                "success": False,
                "error": "Missing correlation manager",
                "clusters_optimized": 0
            }
            
        # Update correlation threshold
        if hasattr(correlation_manager, "correlation_threshold"):
            correlation_manager.correlation_threshold = self.parameters["correlation_threshold"]
        
        # Run correlation analysis
        try:
            original_clusters = correlation_manager.get_clusters()
            clusters = correlation_manager.analyze_correlations(force=True)
            
            # Count changes
            added_clusters = 0
            removed_clusters = 0
            modified_clusters = 0
            
            # Compare with original clusters
            if original_clusters:
                original_set = set(original_clusters.keys())
                new_set = set(clusters.keys())
                
                added_clusters = len(new_set - original_set)
                removed_clusters = len(original_set - new_set)
                
                # Check for modified clusters
                for cluster_id in original_set & new_set:
                    original_members = set(original_clusters[cluster_id])
                    new_members = set(clusters[cluster_id])
                    
                    if original_members != new_members:
                        modified_clusters += 1
            else:
                # All clusters are new
                added_clusters = len(clusters)
            
            return {
                "success": True,
                "clusters_created": added_clusters,
                "clusters_removed": removed_clusters,
                "clusters_modified": modified_clusters,
                "total_clusters": len(clusters),
                "duration_seconds": time.time() - start_time
            }
        except Exception as e:
            _LOGGER.error("Error in cluster optimization: %s", e)
            return {
                "success": False,
                "error": str(e),
                "clusters_optimized": 0,
                "duration_seconds": time.time() - start_time
            }
    
    def evaluate_result(self, before_metrics: Dict[str, Any], 
                       after_metrics: Dict[str, Any]) -> float:
        """Evaluate effectiveness of cluster optimization."""
        # Check cluster coverage
        before_coverage = before_metrics.get("cluster_coverage", 0)
        after_coverage = after_metrics.get("cluster_coverage", 0)
        
        # Check cluster quality
        before_quality = before_metrics.get("cluster_quality", 0)
        after_quality = after_metrics.get("cluster_quality", 0)
        
        # Combined score
        coverage_change = after_coverage - before_coverage
        quality_change = after_quality - before_quality
        
        improvement_score = (coverage_change * 0.5) + (quality_change * 0.5)
        
        # Scale to -1 to 1 range
        return max(-1.0, min(1.0, improvement_score))


class PatternOptimizationStrategy(OptimizationStrategy):
    """Strategy for optimizing pattern detection."""
    
    def __init__(self):
        """Initialize pattern optimization strategy."""
        super().__init__(
            name="pattern_optimization",
            priority=7,
            impact_level="medium"
        )
        self.parameters = {
            "max_patterns": 8,
            "pattern_stability_threshold": 0.6
        }
        self.cooldown_seconds = 12 * 3600  # 12 hour cooldown
    
    def apply(self, optimization_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply pattern optimization strategy."""
        start_time = time.time()
        pattern_detector = optimization_context.get("pattern_detector")
        
        if not pattern_detector:
            return {
                "success": False,
                "error": "Missing pattern detector",
                "patterns_optimized": 0
            }
            
        # Update parameters
        if hasattr(pattern_detector, "max_patterns"):
            pattern_detector.max_patterns = self.parameters["max_patterns"]
        
        # Run pattern detection
        try:
            original_patterns = pattern_detector.get_statistics()
            original_count = original_patterns.get("detected_patterns", 0)
            
            # Detect patterns (if there's a detect_patterns method)
            if hasattr(pattern_detector, "detect_patterns"):
                pattern_detector.detect_patterns()
            
            # Get updated statistics
            new_patterns = pattern_detector.get_statistics()
            new_count = new_patterns.get("detected_patterns", 0)
            
            return {
                "success": True,
                "original_pattern_count": original_count,
                "new_pattern_count": new_count,
                "patterns_changed": original_count != new_count,
                "current_pattern": pattern_detector.current_pattern,
                "duration_seconds": time.time() - start_time
            }
        except Exception as e:
            _LOGGER.error("Error in pattern optimization: %s", e)
            return {
                "success": False,
                "error": str(e),
                "patterns_optimized": 0,
                "duration_seconds": time.time() - start_time
            }
    
    def evaluate_result(self, before_metrics: Dict[str, Any], 
                       after_metrics: Dict[str, Any]) -> float:
        """Evaluate effectiveness of pattern optimization."""
        # Check pattern confidence
        before_confidence = before_metrics.get("pattern_confidence", 0)
        after_confidence = after_metrics.get("pattern_confidence", 0)
        
        # Check prediction accuracy
        before_accuracy = before_metrics.get("prediction_accuracy", 0)
        after_accuracy = after_metrics.get("prediction_accuracy", 0)
        
        # Combined score
        confidence_change = after_confidence - before_confidence
        accuracy_change = after_accuracy - before_accuracy
        
        improvement_score = (confidence_change * 0.4) + (accuracy_change * 0.6)
        
        # Scale to -1 to 1 range
        return max(-1.0, min(1.0, improvement_score))


class PerformanceOptimizationLoop:
    """Manages the continuous performance optimization loop."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = PerformanceOptimizationLoop()
        return cls._instance
    
    def __init__(self):
        """Initialize the performance optimization loop."""
        self.scheduler = AnalysisScheduler.get_instance()
        self.storage = StatisticsStorageManager.get_instance()
        self.prediction_evaluator = PredictionEvaluator.get_instance()
        
        # Register optimization strategies
        self.strategies = {
            "interval": IntervalOptimizationStrategy(),
            "cluster": ClusterOptimizationStrategy(),
            "pattern": PatternOptimizationStrategy(),
        }
        
        # Optimization history
        self.optimization_history = deque(maxlen=MAX_HISTORY_SIZE)
        
        # Learning rate
        self.learning_rate = DEFAULT_LEARNING_RATE
        
        # Last optimization times
        self.last_major_optimization = 0
        self.last_quick_optimization = 0
        
        # Register optimization tasks with scheduler
        self._register_tasks()
        
        # Load optimization history from storage
        self._load_history()
        
        _LOGGER.info("Performance optimization loop initialized with %d strategies", 
                   len(self.strategies))
    
    def _register_tasks(self) -> None:
        """Register optimization tasks with scheduler."""
        try:
            # Register quick optimization task (runs more frequently)
            self.scheduler.register_task(
                name="quick_optimization",
                callback=self.run_quick_optimization,
                interval_seconds=QUICK_OPTIMIZATION_INTERVAL,
                priority=6,
                max_runtime_seconds=300,  # 5 minutes max runtime
                condition=self._should_run_quick_optimization
            )
            
            # Register full optimization task (runs less frequently)
            self.scheduler.register_task(
                name="full_optimization",
                callback=self.run_full_optimization,
                interval_seconds=OPTIMIZATION_COOLDOWN,
                priority=4,
                max_runtime_seconds=600,  # 10 minutes max runtime
                condition=self._should_run_full_optimization
            )
            
            # Register evaluation task (runs after optimizations)
            self.scheduler.register_task(
                name="optimization_evaluation",
                callback=self.evaluate_optimizations,
                interval_seconds=3600,  # Every hour
                priority=3,
                max_runtime_seconds=180,  # 3 minutes max runtime
            )
        except Exception as e:
            _LOGGER.error("Failed to register optimization tasks: %s", e)
    
    def _should_run_quick_optimization(self) -> bool:
        """Determine if quick optimization should run."""
        current_time = time.time()
        
        # Don't run if we did it recently
        if current_time - self.last_quick_optimization < QUICK_OPTIMIZATION_INTERVAL:
            return False
            
        # Check if system has enough resources
        if hasattr(RESOURCE_ADAPTER, "should_run_expensive_operation"):
            return RESOURCE_ADAPTER.should_run_expensive_operation("quick_optimization")
            
        return True
    
    def _should_run_full_optimization(self) -> bool:
        """Determine if full optimization should run."""
        current_time = time.time()
        
        # Don't run if we did it recently
        if current_time - self.last_major_optimization < OPTIMIZATION_COOLDOWN:
            return False
            
        # Check if system has enough resources
        if hasattr(RESOURCE_ADAPTER, "should_run_expensive_operation"):
            return RESOURCE_ADAPTER.should_run_expensive_operation("full_optimization")
            
        return True
    
    def run_quick_optimization(self) -> Dict[str, Any]:
        """Run a quick optimization cycle (just interval optimization)."""
        _LOGGER.info("Running quick optimization cycle")
        start_time = time.time()
        self.last_quick_optimization = start_time
        
        # Get optimization context
        context = self._get_optimization_context()
        
        # Run only the interval optimization strategy
        interval_strategy = self.strategies.get("interval")
        
        if not interval_strategy:
            return {
                "success": False,
                "error": "Interval optimization strategy not found",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        # Collect before metrics
        before_metrics = self._collect_system_metrics()
        
        # Run the strategy
        result = interval_strategy.apply(context)
        
        # Collect after metrics (we'll evaluate later in the dedicated task)
        after_metrics = self._collect_system_metrics()
        
        # Record optimization attempt
        optimization_record = {
            "type": "quick",
            "strategies": ["interval"],
            "timestamp": datetime.utcnow().isoformat(),
            "duration_seconds": time.time() - start_time,
            "results": {
                "interval": result
            },
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "evaluated": False
        }
        
        self.optimization_history.append(optimization_record)
        self._save_history()
        
        return {
            "success": True,
            "optimization_type": "quick",
            "strategies_run": ["interval"],
            "duration_seconds": time.time() - start_time,
            "entities_optimized": result.get("entities_optimized", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def run_full_optimization(self) -> Dict[str, Any]:
        """Run a full optimization cycle (all strategies)."""
        _LOGGER.info("Running full optimization cycle")
        start_time = time.time()
        self.last_major_optimization = start_time
        
        # Get optimization context
        context = self._get_optimization_context()
        
        # Collect before metrics
        before_metrics = self._collect_system_metrics()
        
        # Run strategies in priority order
        results = {}
        strategies_run = []
        
        # Get strategies sorted by priority (highest first)
        sorted_strategies = sorted(
            self.strategies.values(), 
            key=lambda s: s.priority, 
            reverse=True
        )
        
        for strategy in sorted_strategies:
            # Skip if on cooldown
            if strategy.last_application:
                try:
                    last_run = datetime.fromisoformat(strategy.last_application)
                    now = datetime.utcnow()
                    cooldown = timedelta(seconds=strategy.cooldown_seconds)
                    
                    if now - last_run < cooldown:
                        _LOGGER.debug(
                            "Skipping strategy %s (on cooldown)", 
                            strategy.name
                        )
                        continue
                except (ValueError, TypeError):
                    # Invalid date format, don't skip
                    pass
            
            # Run the strategy
            _LOGGER.debug("Running optimization strategy: %s", strategy.name)
            result = strategy.apply(context)
            
            # Store result
            results[strategy.name] = result
            strategies_run.append(strategy.name)
            
            # Short pause between strategies to let system stabilize
            time.sleep(2)
        
        # Collect after metrics (we'll evaluate later in the dedicated task)
        after_metrics = self._collect_system_metrics()
        
        # Record optimization attempt
        optimization_record = {
            "type": "full",
            "strategies": strategies_run,
            "timestamp": datetime.utcnow().isoformat(),
            "duration_seconds": time.time() - start_time,
            "results": results,
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "evaluated": False
        }
        
        self.optimization_history.append(optimization_record)
        self._save_history()
        
        return {
            "success": True,
            "optimization_type": "full",
            "strategies_run": strategies_run,
            "duration_seconds": time.time() - start_time,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def evaluate_optimizations(self) -> Dict[str, Any]:
        """Evaluate recent optimization attempts."""
        evaluations = []
        
        # Find unevaluated optimization records
        for record in self.optimization_history:
            if record.get("evaluated", True):
                continue
                
            # Get metrics
            before_metrics = record.get("before_metrics", {})
            after_metrics = record.get("after_metrics", {})
            
            if not before_metrics or not after_metrics:
                # Skip if missing metrics
                record["evaluated"] = True
                record["evaluation"] = {
                    "error": "Missing metrics data"
                }
                continue
            
            # Evaluate each strategy
            strategy_evaluations = {}
            for strategy_name in record.get("strategies", []):
                strategy = self.strategies.get(strategy_name)
                if not strategy:
                    continue
                
                # Evaluate strategy
                improvement_score = strategy.evaluate_result(before_metrics, after_metrics)
                
                # Update strategy success rate
                strategy.update_success_rate(improvement_score)
                
                # Record evaluation
                strategy_evaluations[strategy_name] = {
                    "improvement_score": round(improvement_score, 3),
                    "success_rate": round(strategy.success_rate, 3)
                }
            
            # Calculate overall improvement
            overall_improvement = 0
            if strategy_evaluations:
                improvement_scores = [
                    eval_data["improvement_score"] 
                    for eval_data in strategy_evaluations.values()
                ]
                overall_improvement = sum(improvement_scores) / len(improvement_scores)
            
            # Mark as evaluated
            record["evaluated"] = True
            record["evaluation"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "strategies": strategy_evaluations,
                "overall_improvement": round(overall_improvement, 3)
            }
            
            # Add to evaluations list
            evaluations.append({
                "optimization_type": record.get("type"),
                "timestamp": record.get("timestamp"),
                "strategies": record.get("strategies"),
                "overall_improvement": round(overall_improvement, 3)
            })
            
        # Update storage
        self._save_history()
        
        # Return evaluation results
        return {
            "evaluations_performed": len(evaluations),
            "evaluations": evaluations
        }
    
    def _get_optimization_context(self) -> Dict[str, Any]:
        """Get context data for optimization strategies."""
        # Get managers and data from your existing components
        context = {}
        
        # Try to get entity stats
        try:
            from .manager import ModbusStatisticsManager
            stats_manager = ModbusStatisticsManager.get_instance()
            
            # Get entity statistics
            entity_stats = {}
            for entity_id, tracker in getattr(stats_manager, "_entity_trackers", {}).items():
                entity_stats[entity_id] = tracker.get_statistics()
            
            context["entity_stats"] = entity_stats
            context["statistics_manager"] = stats_manager
            
            # Try to get interval manager
            if hasattr(stats_manager, "_interval_manager"):
                context["interval_manager"] = stats_manager._interval_manager
            
            # Try to get correlation manager
            if hasattr(stats_manager, "_correlation_manager"):
                context["correlation_manager"] = stats_manager._correlation_manager
            
            # Try to get pattern detector
            if hasattr(stats_manager, "_pattern_detector"):
                context["pattern_detector"] = stats_manager._pattern_detector
                
        except (ImportError, AttributeError) as e:
            _LOGGER.debug("Could not get all statistics managers: %s", e)
        
        return context
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-wide metrics for optimization evaluation."""
        metrics = {}
        
        # Try to get system metrics from various components
        try:
            # Get interval manager metrics
            from .manager import ModbusStatisticsManager
            stats_manager = ModbusStatisticsManager.get_instance()
            
            if hasattr(stats_manager, "get_system_statistics"):
                metrics.update(stats_manager.get_system_statistics())
            
            # Get pattern metrics
            if hasattr(stats_manager, "get_pattern_information"):
                pattern_info = stats_manager.get_pattern_information()
                metrics["pattern_confidence"] = pattern_info.get("pattern_confidence", 0)
                metrics["current_pattern"] = pattern_info.get("pattern_id")
            
            # Get prediction accuracy
            prediction_metrics = self.prediction_evaluator.calculate_overall_metrics()
            metrics["prediction_accuracy"] = prediction_metrics.get("average_accuracy", 0) / 100.0
            
            # Get cluster metrics
            if hasattr(stats_manager, "get_clusters"):
                clusters = stats_manager.get_clusters()
                
                # Calculate cluster coverage
                entity_count = len(getattr(stats_manager, "_entity_trackers", {}))
                if entity_count > 0:
                    clustered_entities = set()
                    for cluster in clusters.values():
                        clustered_entities.update(cluster)
                    
                    metrics["cluster_coverage"] = len(clustered_entities) / entity_count
                    
                # Estimate cluster quality
                if hasattr(stats_manager, "_correlation_manager"):
                    corr_manager = stats_manager._correlation_manager
                    metrics["cluster_quality"] = getattr(corr_manager, "correlation_threshold", 0.6)
            
            # Calculate polling efficiency
            entity_stats = {}
            for entity_id, tracker in getattr(stats_manager, "_entity_trackers", {}).items():
                entity_stats[entity_id] = tracker.get_statistics()
                
            if entity_stats:
                efficiency_values = [
                    stats.get("polling_efficiency", 0) 
                    for stats in entity_stats.values()
                ]
                metrics["polling_efficiency"] = statistics.mean(efficiency_values) if efficiency_values else 0
                
                # Calculate polling load
                polls_per_second = sum(
                    3600 / stats.get("current_scan_interval", 30) / 3600
                    for stats in entity_stats.values()
                )
                metrics["polls_per_second"] = polls_per_second
            
        except (ImportError, AttributeError) as e:
            _LOGGER.debug("Could not collect all system metrics: %s", e)
        
        # Add timestamp
        metrics["timestamp"] = datetime.utcnow().isoformat()
        
        return metrics
    
    def _load_history(self) -> None:
        """Load optimization history from storage."""
        try:
            history_data = self.storage.get_storage_info()
            optimization_history = history_data.get("optimization_history", [])
            
            if optimization_history:
                # Convert list to deque with max length
                self.optimization_history = deque(
                    optimization_history[-MAX_HISTORY_SIZE:],
                    maxlen=MAX_HISTORY_SIZE
                )
                _LOGGER.debug("Loaded %d optimization records from storage", 
                            len(self.optimization_history))
                
                # Update strategy success rates
                for record in self.optimization_history:
                    if not record.get("evaluated", False):
                        continue
                        
                    evaluation = record.get("evaluation", {})
                    strategy_evals = evaluation.get("strategies", {})
                    
                    for strategy_name, eval_data in strategy_evals.items():
                        strategy = self.strategies.get(strategy_name)
                        if strategy and "improvement_score" in eval_data:
                            improvement = eval_data["improvement_score"]
                            strategy.update_success_rate(improvement)
        except Exception as e:
            _LOGGER.warning("Failed to load optimization history: %s", e)
    
    def _save_history(self) -> None:
        """Save optimization history to storage."""
        try:
            # Convert deque to list for serialization
            history_list = list(self.optimization_history)
            
            # Add to storage info
            storage_info = self.storage.get_storage_info()
            storage_info["optimization_history"] = history_list
            
            # Save optimization_strategies state
            strategy_state = {}
            for name, strategy in self.strategies.items():
                strategy_state[name] = {
                    "success_rate": strategy.success_rate,
                    "application_count": strategy.application_count,
                    "last_application": strategy.last_application,
                    "last_improvement": strategy.last_improvement
                }
            
            storage_info["optimization_strategies"] = strategy_state
            
            # No direct method to save this, but we can add it next time the storage is saved
            # This would be handled better with dedicated storage method
        except Exception as e:
            _LOGGER.warning("Failed to save optimization history: %s", e)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the performance optimization loop."""
        stats = {
            "last_quick_optimization": datetime.fromtimestamp(
                self.last_quick_optimization
            ).isoformat() if self.last_quick_optimization else None,
            "last_major_optimization": datetime.fromtimestamp(
                self.last_major_optimization
            ).isoformat() if self.last_major_optimization else None,
            "optimization_history_length": len(self.optimization_history),
            "learning_rate": self.learning_rate,
            "strategies": {}
        }
        
        # Add strategy statistics
        for name, strategy in self.strategies.items():
            stats["strategies"][name] = {
                "success_rate": round(strategy.success_rate, 3),
                "application_count": strategy.application_count,
                "last_application": strategy.last_application,
                "last_improvement": round(strategy.last_improvement, 3) if strategy.last_improvement else None,
                "priority": strategy.priority,
                "impact_level": strategy.impact_level
            }
        
        # Get recent optimizations
        recent_optimizations = []
        for record in list(self.optimization_history)[-5:]:  # Last 5 optimizations
            recent_optimizations.append({
                "type": record.get("type"),
                "timestamp": record.get("timestamp"),
                "strategies": record.get("strategies"),
                "evaluated": record.get("evaluated", False),
                "improvement": record.get("evaluation", {}).get("overall_improvement") 
                             if record.get("evaluated") else None
            })
        
        stats["recent_optimizations"] = recent_optimizations
        
        return stats
    
    def update_strategy_parameters(self, strategy_name: str, 
                                  parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Update parameters for a strategy.
        
        Args:
            strategy_name: Name of strategy to update
            parameters: New parameters
            
        Returns:
            Dictionary with update results
        """
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            return {
                "success": False,
                "error": f"Strategy {strategy_name} not found"
            }
            
        # Update parameters
        original_params = strategy.parameters.copy()
        for key, value in parameters.items():
            if key in strategy.parameters:
                strategy.parameters[key] = value
                
        return {
            "success": True,
            "strategy": strategy_name,
            "original_parameters": original_params,
            "updated_parameters": strategy.parameters.copy()
        }
    
    def run_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """Run a specific optimization strategy.
        
        Args:
            strategy_name: Name of strategy to run
            
        Returns:
            Dictionary with strategy results
        """
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            return {
                "success": False,
                "error": f"Strategy {strategy_name} not found"
            }
            
        # Get optimization context
        context = self._get_optimization_context()
        
        # Run strategy
        start_time = time.time()
        result = strategy.apply(context)
        
        return {
            "success": True,
            "strategy": strategy_name,
            "duration_seconds": time.time() - start_time,
            "result": result
        }


# Global instance
PERFORMANCE_OPTIMIZER = PerformanceOptimizationLoop.get_instance()