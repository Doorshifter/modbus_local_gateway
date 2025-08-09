# Module: advanced_interval_manager.py

## Functional Purpose
Provides an advanced, adaptive system for determining optimal scan intervals for Modbus entities, leveraging pattern detection, correlation analysis, and dynamic system-wide optimizations.  
Aims to maximize efficiency and responsiveness while maintaining system constraints and cluster coordination.

## Core Responsibilities
- Track, update, and manage scan interval settings per entity, including importance and register count.
- Integrate pattern and correlation managers to refine interval recommendations based on behavioral patterns and group activity.
- Dynamically adjust intervals in response to system load, entity change frequency, pattern transitions, and cluster states.
- Schedule entity polling using a priority queue, optimizing for system utilization and staggered cluster polling.
- Persist and restore state, settings, and statistics from storage.
- Provide detailed statistics and diagnostics at both system and entity level.

## Key Public Methods
| Method                               | Purpose                                                     | Log Level When Called | Success Indicator                              |
|---------------------------------------|-------------------------------------------------------------|----------------------|-----------------------------------------------|
| `set_components()`                    | Integrate pattern and correlation managers                  | INFO                 | Logs connection, enables advanced features     |
| `register_entity()`                   | Register or update entity for interval management           | DEBUG                | Entity data present in manager                 |
| `record_poll()`                       | Record poll event for entity, schedule next, update stats   | (none)               | Poll count, changes, and queue updated         |
| `get_next_entity_to_poll()`           | Get the next entity due for polling, respecting scheduling  | (none)               | Returns entity_id or None                      |
| `get_recommended_interval()`          | Compute optimal scan interval for an entity                 | (none)               | Integer interval, updated in entity data       |
| `update_clusters()`                   | Update cluster assignments and trigger optimization         | DEBUG                | Clusters set, intervals staggered/stabilized   |
| `get_statistics()`                    | Return interval management and system-level diagnostics     | (none)               | Dict of stats                                  |
| `get_entity_statistics()`             | Get stats for a specific entity (intervals, cluster, etc.)  | (none)               | Dict of stats                                  |

## Current Logging Philosophy

### INFO Level
- On successful storage load:
  - `"Loaded interval management data with %d entities"`
- On connecting to pattern/correlation managers:
  - `"Advanced interval manager connected with pattern and correlation components"`

### WARNING Level  
- None in this module.

### ERROR Level
- On failed storage operations:
  - `"Failed to load interval management data: %s"`
  - `"Failed to save interval management data: %s"`

### DEBUG Level
- On entity registration:
  - `"Registered entity %s with interval %d and %d registers"`
- On cluster polling optimization:
  - `"Optimized polling for cluster %s with %d entities"`
- On maintenance:
  - `"Completed interval maintenance with %d entities"`

## Dependencies (Imports From)
- `interval_manager`, `pattern_detection`, `correlation`: For integration with other optimization systems
- `storage_aware`: For persistence of interval and entity data
- `logging`, `heapq`, `math`, `datetime`, `time`: Standard library for diagnostics, scheduling, and calculations

## Used By (Imported By)
- System orchestrator, manager, or statistics controller
- Any component requiring advanced scan interval recommendations, cluster-aware polling, or diagnostics

## Key Data Flows
1. **Input:**
   - Entity registration and poll events
   - Pattern and correlation insights via component integration
   - Cluster updates (from correlation or external logic)
   - System load and performance metrics
2. **Processing:**
   - Schedules polling using a heap-based priority queue, rescheduling as needed
   - Calculates recommended intervals using:
      - Change frequency and history
      - System-wide load and utilization targets
      - Pattern-based insights (including transitions and confidence)
      - Correlation/cluster-based adjustments
      - Entity importance scoring
      - Smoothing to avoid rapid changes
   - Performs regular maintenance: recalculation, cluster optimization, persistence
   - Persists/restores state and settings via abstracted storage
3. **Output:**
   - Recommended and dynamic intervals per entity
   - Next entity to poll
   - System and entity-level statistics and interval diagnostics
   - Logs for registration, optimization, and errors

## Integration Points
- **Pattern Detector:**  
  Refines intervals based on detected operating patterns and transitions.
- **Correlation Manager:**  
  Enables cluster-based interval adjustment and staggered polling for correlated entities.
- **Persistent Storage:**  
  Persists system and entity interval data for recovery and audit.
- **Orchestrator/Coordinator:**  
  Consumes interval recommendations and polling schedules for actual device operations.

## Current Issues/Tech Debt
- Some methods are large and multi-responsibility (e.g., maintenance, interval calculation).
- Lacks per-entity or per-cluster warning/error reporting for outlier or failing behaviors.
- Priority queue is rebuilt frequently, which may be inefficient for large numbers of entities.
- Cluster optimization and smoothing logic could be further modularized for clarity.
- Thread safety is not explicitly managed; assumes single-threaded or externally coordinated access.
- Error handling for storage is minimal—could expose error state to callers.