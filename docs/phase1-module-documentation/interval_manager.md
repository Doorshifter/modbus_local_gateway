# Module: interval_manager.py

## Functional Purpose
Provides unified, intelligent, and adaptive management of polling intervals for Modbus entities.  
It dynamically scales scan intervals for each entity based on observed change frequency, detected patterns, system constraints, and predictive analytics — optimizing both responsiveness and resource usage.

## Core Responsibilities
- Track per-entity polling state, change frequency, and adaptive metrics.
- Calculate and recommend optimal polling intervals for each entity, considering stability, prediction, clustering, and rate limits.
- Coordinate system-wide optimizations and interval adjustments across all registered entities.
- Integrate pattern detection, clustering/grouping, and predictive change timing into interval decisions.
- Provide statistics, history, and persistence support for interval management.

## Key Public Methods
| Method                          | Purpose                                                            | Log Level When Called     | Success Indicator                               |
|----------------------------------|--------------------------------------------------------------------|--------------------------|-------------------------------------------------|
| `register_entity()`              | Register a new entity or update parameters for interval management | (none)                   | Entity state is present in manager              |
| `record_poll()`                  | Record polling results for an entity (value, change, error, etc.)  | (none)                   | Updates stats and metrics                       |
| `get_recommended_interval()`     | Calculate and return recommended scan interval for an entity       | (none)                   | Returns integer interval (seconds)              |
| `update_clusters()`              | Update clusters of entities for coordinated polling                | (none)                   | Cluster assignments updated                     |
| `set_current_pattern()`          | Set system pattern and trigger optimization on change              | DEBUG                    | "Performing system-wide interval optimization"  |
| `record_pattern_interval()`      | Record optimal interval for entity in a specific pattern           | (none)                   | Pattern-specific interval updated               |
| `update_prediction()`            | Update prediction for an entity's next value change                | (none)                   | Prediction and confidence updated               |
| `get_statistics()`               | Retrieve overall interval management statistics                    | (none)                   | Returns dict of statistics                      |
| `get_entity_statistics()`        | Retrieve polling statistics for a specific entity                  | (none)                   | Returns dict of stats or None                   |
| `get_interval_history()`         | Retrieve interval history for one/all entities                     | (none)                   | Returns dict of interval histories              |
| `optimize_all_intervals()`       | Force system-wide scan interval optimization                       | DEBUG                    | "Performing system-wide interval optimization"  |
| `set_storage()`                  | Attach a storage manager for persistence                           | INFO/ERROR               | "Restored interval data..." or error message    |

## Current Logging Philosophy

### INFO Level
- Storage/data restoration from persistence
  - Example: `"Restored interval data for %d entities from storage"`
  - Example: `"Loaded interval history for %d entities"`

### WARNING Level  
- (No explicit WARNING logs in current code, but rate/excess conditions could be candidates.)

### ERROR Level
- Storage/data loading failures
  - Example: `"Error loading data from storage: %s"`

### DEBUG Level
- System-wide optimization triggers
  - Example: `"Performing system-wide interval optimization (%d entities)"`

## Dependencies (Imports From)
- `logging`: For diagnostic and event logging
- `collections`, `defaultdict`: For cluster and history management
- `datetime`, `time`, `math`, `statistics`: For time and statistical calculations
- `threading`: For thread safety (RLock)
- `storage_manager` (external): For persistence integration

## Used By (Imported By)
- `manager.py`: Used as a singleton (`INTERVAL_MANAGER`)
- Any module needing adaptive Modbus scan intervals or statistics

## Key Data Flows
1. **Input:**  
   - Entity registration (entity_id, register_count, interval)
   - Polling results (timestamp, value_changed, value, error, response_time)
   - Pattern/cluster changes and predictions
2. **Processing:**  
   - Adaptive metrics are updated (change rate, stability, error rate, etc.)
   - Intervals are recalculated using stability, prediction, clustering, and system rate limits
   - System-wide optimizations and persistence updates
3. **Output:**  
   - Recommended intervals per entity
   - Statistics and interval history for diagnostics and analytics
   - Cluster-coordinated and rate-limited polling schedules

## Integration Points
- **Coordinator/Manager Integration:**  
  Receives registration and polling events, provides interval recommendations.
- **Pattern and Cluster Integration:**  
  Adjusts intervals based on detected patterns and entity clusters.
- **Storage Integration:**  
  Periodically persists interval history and statistics; restores at startup.
- **Prediction Integration:**  
  Incorporates next-change predictions into interval calculation.

## Current Issues/Tech Debt
- **No explicit WARNING logs for suboptimal but not erroneous conditions** (could improve observability).
- **Complexity in coordination:**  
  Multiple passes (priority, clustering, rate limiting) could be refactored for clarity.
- **Heavy use of internal state and locking:**  
  Concurrency handled via RLock but could be further validated under heavy load.
- **Persistence logic is basic:**  
  Only saves every 10th interval change; could be made more robust or configurable.
- **Cluster and pattern logic tightly coupled:**  
  Might be candidates for further modularization.