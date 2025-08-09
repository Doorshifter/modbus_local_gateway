# Module: statistics_tracker.py

## Functional Purpose
Provides unified, comprehensive statistics tracking, pattern detection, and optimization for Modbus entities.  
Enables efficient polling by measuring, analyzing, and recommending polling intervals and strategies based on observed entity behavior, trends, and errors.

## Core Responsibilities
- Track per-entity polling events, value changes, errors, and response times.
- Calculate and update efficiency, change rates, and optimal polling intervals for each entity.
- Integrate pattern detection and prediction to inform interval recommendations.
- Persist statistics for reliability and diagnostics across restarts.
- Provide blackhole (stale/no-change) detection and reporting.

## Key Public Methods
| Method                      | Purpose                                               | Log Level When Called | Success Indicator                             |
|-----------------------------|-------------------------------------------------------|----------------------|-----------------------------------------------|
| `record_poll()`             | Record and process a polling event                    | ERROR (for save)     | Stats updated, value/history tracked          |
| `calculate_statistics()`    | Compute and update core statistics                    | (none)               | Stats dict returned/updated                   |
| `get_statistics()`          | Return current statistics, including interval manager | (none)               | Dict with current statistics                  |
| `reset()`                   | Reset all statistics and metrics                      | (none)               | Stats zeroed, history cleared                 |
| `_load_from_storage()`      | Restore statistics from persistent storage            | (none)               | Stats restored from storage                   |
| `_save_to_storage()`        | Save statistics to persistent storage                 | ERROR                | True if no error, logs error if failure       |
| `_enhanced_analysis()`      | Perform advanced trend & pattern analysis             | (none)               | Stats updated with pattern/prediction fields  |
| `_predict_next_change()`    | Predict timestamp of next change                      | (none)               | Prediction fields updated in stats            |
| `_evaluate_prediction()`    | Evaluate prediction accuracy                          | (none)               | Prediction accuracy field updated             |

## Current Logging Philosophy

### INFO Level
- None in this module.

### WARNING Level  
- None in this module.

### ERROR Level
- Storage/persistence failures:
  - `"Error saving entity stats: %s"`

### DEBUG Level
- None in this module.

## Dependencies (Imports From)
- `interval_manager`: For interval recommendations and poll recording (`INTERVAL_MANAGER`)
- `pattern_detection`: For pattern-aware analysis and predictions (`PatternDetector`)
- `storage_aware`: For persisting statistics to disk/storage (`StorageAware`)
- `logging`, `datetime`, `collections`, `statistics`, `time`: Standard library utilities

## Used By (Imported By)
- `manager.py`: For tracking, collecting, and reporting entity statistics
- Other system modules that require detailed per-entity stats and diagnostics

## Key Data Flows
1. **Input:**
   - Polling events: value, change status, error, timestamp, response time
   - Registration: entity_id, scan interval, register count
2. **Processing:**
   - Update counters, change intervals, and efficiency metrics
   - Calculate/recommend scan intervals based on statistics and patterns
   - Run advanced analysis (24h trends, blackhole detection, predictions)
   - Persist stats to storage if available
3. **Output:**
   - Dict of statistics, including efficiency, change rates, recommended intervals, prediction info, blackhole status, and pattern data

## Integration Points
- **Interval Manager:**  
  - Registers entity at initialization
  - Records each poll with interval manager
  - Retrieves recommended intervals from interval manager
- **Pattern Detector:**  
  - Updates and queries pattern detector for value/pattern info
  - Uses pattern-based scan interval suggestions
- **Storage Layer:**  
  - Loads and saves statistics using provided storage manager
- **Manager/Coordinator:**  
  - Used as statistics collection/reporting backend for system diagnostics and optimization

## Current Issues/Tech Debt
- No INFO/DEBUG logs for normal operation or milestone stat updates (harder to audit in production).
- Enhanced analysis and prediction are passive, not actively scheduled—could miss trends if polling is sparse.
- Blackhole logic is simple (time-based), could be made more robust/context-aware.
- Pattern and prediction logic could be further modularized for clarity.
- Thread safety not explicitly handled (relies on single-threaded use or external coordination).