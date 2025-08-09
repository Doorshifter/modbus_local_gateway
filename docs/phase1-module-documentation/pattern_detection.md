# Module: pattern_detection.py

## Functional Purpose
Detects and tracks statistical operating patterns in Modbus entity value changes in a device-agnostic manner.  
Assigns neutral pattern identifiers, characterizes their statistical properties, and provides pattern-aware recommendations for scan intervals and diagnostics.

## Core Responsibilities
- Identify, characterize, and track recurring patterns in entity behaviors based on statistical analysis.
- Detect transitions between patterns, including sensitivity adjustments using entity correlations.
- Maintain pattern life cycles (activation, deactivation, duration, transitions).
- Provide pattern-based scan interval recommendations and pattern metadata.
- Integrate with correlation manager to enhance pattern detection and weighting.
- Expose metrics and statistics about patterns, transitions, stability, and change rates.

## Key Public Methods
| Method                                | Purpose                                                      | Log Level When Called | Success Indicator                     |
|----------------------------------------|--------------------------------------------------------------|----------------------|---------------------------------------|
| `update_entity_value()`                | Feed new entity value for pattern analysis and transition     | (none)               | Pattern state/metrics updated         |
| `set_correlation_manager()`            | Integrate correlation manager for enhanced detection          | (none)               | Correlation-aware detection enabled   |
| `get_current_pattern_info()`           | Returns info about active pattern and its metrics             | (none)               | Dict with pattern/stability/confidence|
| `get_optimal_scan_interval()`          | Recommend scan interval for entity based on current pattern   | (none)               | Integer interval or None              |
| `is_transition_period()`               | Indicates if currently in a pattern transition                | (none)               | Boolean                               |
| `get_statistics()`                     | Get summary of patterns, transitions, and change metrics      | (none)               | Dict of statistics                    |
| `get_pattern_definition()`             | Detailed info about what defines a specific pattern           | (none)               | Dict with pattern/entity criteria     |
| `set_entity_weight()`                  | Set entity importance for pattern detection                   | (none)               | Weighting updated                     |
| `set_recent_change_window()`           | Set time window for "recent" change sensitivity               | (none)               | Window applied                        |

## Current Logging Philosophy

### INFO Level
- None in this module.

### WARNING Level  
- None in this module.

### ERROR Level
- None in this module.

### DEBUG Level
- None in this module.

## Dependencies (Imports From)
- `collections`: For deques, defaultdicts (pattern transitions, histories)
- `datetime`, `time`, `statistics`, `math`: For time and statistical calculations
- `logging`: For possible logging (minimal used)
- External: Correlation manager for cluster- and correlation-aware detection (optional, set at runtime)

## Used By (Imported By)
- `statistics_tracker.py`: For global and per-entity pattern detection and scan interval recommendations
- `manager.py`: For system-wide pattern integration, interval, and diagnostics coordination

## Key Data Flows
1. **Input:**
   - Entity values (numeric, per-poll or on change)
   - Entity weights (for importance in pattern detection)
   - Correlation manager and cluster assignments (optional)
2. **Processing:**
   - Track recent values, detect and create new patterns as needed
   - Assign and update pattern states, transitions, and durations
   - Adjust pattern sensitivity using correlation/cluster context
   - Characterize each pattern statistically (mean, variance, change rate, stability, etc.)
   - Predict pattern transitions, durations, and next likely patterns
3. **Output:**
   - Current pattern info and confidence
   - Optimal scan interval recommendations for each entity
   - Pattern-level statistics and definitions
   - Change metrics and transition history

## Integration Points
- **Correlation Manager Integration:**  
  - Sensitivity adjustments for transitions and entity weighting
  - Cluster-aware detection for stronger pattern signals
- **Statistics Tracker/Manager:**  
  - Receives value updates, provides pattern info and interval advice
- **System Diagnostics:**  
  - Exposes pattern and transition metrics for analytics and optimization

## Current Issues/Tech Debt
- No INFO/WARNING/DEBUG logs present; pattern transitions and new pattern creations are silent.
- Some global constants (e.g., transition thresholds) are modified at runtime (could be encapsulated).
- Direct use of mutable globals (e.g., `PATTERN_TRANSITION_METRIC_THRESHOLD`) may lead to hard-to-debug side effects.
- Correlation manager is optional and set at runtime, but not type-checked.
- Some methods are large/complex (e.g., pattern change checks, pattern definition) and could be refactored for clarity.
- No explicit thread safety; assumes single-threaded or externally managed concurrency.