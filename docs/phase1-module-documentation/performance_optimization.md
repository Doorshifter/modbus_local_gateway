# Module: performance_optimization.py

## Functional Purpose
Implements a continuous improvement/performance optimization loop for Modbus polling and system efficiency.
Monitors system metrics, applies and evaluates optimization strategies (interval, clustering, pattern), and learns from results to automatically tune parameters for reliability and efficiency.

## Core Responsibilities
- Define multiple optimization strategies (interval, cluster, pattern) and manage their application and learning.
- Use a scheduler to periodically trigger quick and full optimization cycles, as well as post-optimization evaluation.
- Collect, record, and learn from pre- and post-optimization system metrics for ongoing improvement.
- Persist and restore optimization history and strategy learning state.
- Expose APIs for running optimizations, updating strategy parameters, and retrieving statistics.

## Key Public Classes and Methods

| Class/Method                       | Purpose                                                        | Log Level When Called     | Success Indicator                        |
|------------------------------------|----------------------------------------------------------------|--------------------------|------------------------------------------|
| `OptimizationStrategy`             | Abstract base class for optimization strategies                | (none)                   | Subclassed for each strategy             |
| `IntervalOptimizationStrategy`     | Optimizes entity polling intervals                             | WARNING/INFO             | Dict with results                        |
| `ClusterOptimizationStrategy`      | Optimizes entity clustering and correlations                   | ERROR/INFO               | Dict with results                        |
| `PatternOptimizationStrategy`      | Optimizes pattern detection and configuration                  | ERROR/INFO               | Dict with results                        |
| `PerformanceOptimizationLoop`      | Main class for managing scheduling, learning, and orchestration| INFO/ERROR/DEBUG         | Singleton pattern                        |
| `PerformanceOptimizationLoop.run_quick_optimization()` | Run only interval optimization (quick)            | INFO                     | Dict with results                        |
| `PerformanceOptimizationLoop.run_full_optimization()`  | Run all strategies (full)                         | INFO/DEBUG               | Dict with results                        |
| `PerformanceOptimizationLoop.evaluate_optimizations()` | Evaluate all non-evaluated optimizations          | (none)                   | Dict with evaluation summaries           |
| `PerformanceOptimizationLoop.update_strategy_parameters()` | Update params for a strategy                | (none)                   | Dict with before/after param values      |
| `PerformanceOptimizationLoop.run_strategy()`           | Manually run a specific strategy                 | (none)                   | Dict with results                        |
| `PerformanceOptimizationLoop.get_statistics()`         | Return stats on optimization history/learning    | (none)                   | Dict with stats                          |

## Current Logging Philosophy

### INFO Level
- On initialization of performance loop:
  - `"Performance optimization loop initialized with %d strategies"`
- On running quick/full optimization cycles:
  - `"Running quick optimization cycle"`
  - `"Running full optimization cycle"`

### WARNING Level
- When interval optimization fails for entity:
  - `"Failed to optimize interval for %s: %s"`
- When failing to load/save optimization history:
  - `"Failed to load optimization history: %s"`
  - `"Failed to save optimization history: %s"`

### ERROR Level
- On failed registration of optimization tasks:
  - `"Failed to register optimization tasks: %s"`
- On error in cluster or pattern optimization:
  - `"Error in cluster optimization: %s"`
  - `"Error in pattern optimization: %s"`

### DEBUG Level
- When skipping strategy due to cooldown:
  - `"Skipping strategy %s (on cooldown)"`
- When unable to collect all system metrics:
  - `"Could not collect all system metrics: %s"`
- When unable to get all statistics managers:
  - `"Could not get all statistics managers: %s"`
- On loading optimization records from storage:
  - `"Loaded %d optimization records from storage"`

## Dependencies (Imports From)
- `analysis_scheduler.AnalysisScheduler`
- `resource_adaptation.RESOURCE_ADAPTER`
- `storage.StatisticsStorageManager`
- `prediction_evaluation.PredictionEvaluator`
- `self_healing.SELF_HEALING_SYSTEM`
- `collections.deque`, `statistics`, `datetime`, `time`, `logging`
- Typing: For type hints

## Used By (Imported By)
- System orchestrator, analysis scheduler, or admin tools for automated or on-demand performance optimization.
- Any UI or service needing optimization history, learning, or manual trigger capabilities.

## Key Data Flows

1. **Input:**
   - System and entity statistics, pattern/correlation/interval managers, resource adaptation context.
2. **Processing:**
   - Periodically collects system metrics and applies optimization strategies according to schedule and cooldowns.
   - Each strategy applies its own logic to propose/implement improvements.
   - Pre- and post-optimization metrics are collected for learning.
   - Evaluates optimization effectiveness and updates strategy success rates.
   - Optimization history and learning state are persisted for future runs.
3. **Output:**
   - System and strategy statistics, optimization/evaluation records, and learning outcomes.
   - Return values from direct and scheduled optimization runs.

## Integration Points
- **Scheduler:**  
  Registers and runs as scheduled analysis tasks.
- **Resource Adaptation:**  
  Checks with resource adapter before running heavy optimizations.
- **System/Entity Stats Managers:**  
  Consumes statistics for optimization context and metrics collection.
- **Pattern/Cluster/Interval Managers:**  
  Strategies interact directly to update system configuration.
- **Storage:**  
  Persists history and learning state with system statistics.

## Current Issues/Tech Debt
- Storage of optimization history/learning is somewhat ad hoc, likely not robust on restart or in multi-instance scenarios.
- Strategy application is synchronous and blocking; no async or parallel execution.
- No persistence for strategy parameter changes unless storage is explicitly saved.
- Not all error states are logged at the appropriate level; some are silent.
- "Self-healing" and prediction evaluation hooks are placeholders/not detailed here.
- No access control/auditing for manual or automated strategy changes.