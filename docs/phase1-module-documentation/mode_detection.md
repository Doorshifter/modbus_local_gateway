# Module: mode_detection.py

## Functional Purpose
Detects, tracks, and characterizes operational modes (such as heating, cooling, standby, or custom) for Modbus devices (e.g., HVAC, pumps, industrial systems) based on real-time and historical entity value patterns.  
Provides insight into mode transitions, stability, and predicts optimal scan intervals and future mode changes.

## Core Responsibilities
- Identify and manage a limited number of operational modes using entity value characteristics and transitions.
- Track and update statistical properties of each detected mode (occurrences, durations, characteristic values, etc.).
- Detect and record mode transitions, maintain transition history, and predict likely next modes.
- Assign human-readable names to modes (e.g., heating, cooling, standby, transition) based on value patterns and entity indicators.
- Provide mode stability metrics, confidence levels, and transition/interval recommendations.
- Integrate with scan interval optimization by exposing mode-aware scan interval suggestions.

## Key Public Classes and Methods

| Class/Method                    | Purpose                                                               | Log Level When Called | Success Indicator                  |
|---------------------------------|-----------------------------------------------------------------------|----------------------|------------------------------------|
| `ModeDetector`                  | Main class for detecting, tracking, and analyzing modes               | (none)               | Modes detected, transitions tracked|
| `ModeDetector.update_entity_value()` | Update entity value, trigger mode checks, update pattern           | (none)               | Value recorded, modes updated      |
| `ModeDetector.get_current_mode_info()` | Returns dict with current mode, stability, prediction           | (none)               | Dict with mode info                |
| `ModeDetector.get_optimal_scan_interval()` | Suggests scan interval for an entity given current mode        | (none)               | Integer or None                    |
| `ModeDetector.is_transition_period()`     | Returns True if system is in a transition/unstable period      | (none)               | Boolean                            |
| `ModeDetector.get_statistics()`          | Returns stats about all detected modes and transitions         | (none)               | Dict of mode information           |

### Supporting Classes
| Class/Method                    | Purpose                                         |
|---------------------------------|-------------------------------------------------|
| `ModeState`                     | Represents a single mode, stores char. values, durations, etc. |
| `ModeState.calculate_distance()`| Computes normalized distance to a value vector   |
| `ModeState.add_value_point()`    | Adds a value to mode's historical profile        |
| `ModeSequence`                  | Tracks transition history and frequency          |
| `ModeSequence.get_most_likely_next()` | Predicts likely next mode                   |

## Current Logging Philosophy

- No INFO/WARNING/ERROR/DEBUG logs, except for module-level logger declaration.
- All exception handling is silent (value errors in numeric conversions, statistics errors, etc.).
- No runtime warnings, errors, or info for user-facing diagnostic events.

## Dependencies (Imports From)
- `time`, `datetime`, `statistics`, `collections.deque`, `collections.defaultdict`
- `logging`
- Typing: For type hints

## Used By (Imported By)
- Pattern detection, interval optimization, or analytics subsystems needing to react to operational mode changes.
- Any system needing to query current mode, transition predictions, or mode-specific scan interval recommendations.

## Key Data Flows

1. **Input:**
   - Entity value updates (numerics, per entity, with timestamp).
2. **Processing:**
   - Maintains rolling history of values and transitions for each mode.
   - Assigns and updates modes using normalized distance from current values to each mode's "center".
   - Tracks mode durations, occurrences, and transition frequencies.
   - Infers names and types (heating, cooling, standby, transition) based on detected value patterns and entity name heuristics.
   - Predicts next likely mode and end time of current mode.
3. **Output:**
   - Current mode info: id, name, stability, predicted end and next mode, confidence.
   - Mode-specific scan interval recommendations for entities.
   - Mode and transition statistics for diagnostics and optimization.

## Integration Points

- **Interval Optimization:**  
  Used to determine scan intervals based on operational mode and transition stability.
- **Pattern/Anomaly Detection:**  
  Provides mode transition history and statistics for deeper pattern analysis.
- **Dashboards/UIs:**  
  Expose mode info, stability, and predictions for user interfaces and status panels.

## Current Issues/Tech Debt

- No persistent storage of mode history; all mode detection is in-memory.
- No external callbacks or event hooks on mode change.
- No logging for mode change, errors, or significant events.
- Mode naming and detection heuristics are basic and could be improved with domain knowledge or ML.
- Confidence and stability are simple calculations, not probabilistic or learned.
- No explicit thread safety; not safe for concurrent updates.