# Module: adaptive_parameters.py

## Functional Purpose
Provides an adaptive parameter management system for Modbus integration that dynamically tunes operational parameters (such as scan intervals, batch size, thresholds, etc.) based on current context, historical performance, detected patterns, and learned effectiveness—enabling self-optimizing, context-aware operation.

## Core Responsibilities
- Define, constrain, and manage system parameters and their defaults.
- Maintain multiple parameter profiles optimized for different system contexts (e.g., normal, high load, recovery, night).
- Match current system context (load, errors, time, etc.) to the best-fitting profile and activate it.
- Support learning and parameter adaptation from historical performance data.
- Persist, restore, and track changes/history of parameter sets.
- Provide a singleton interface (`PARAMETER_MANAGER`) for global parameter access and tuning.

## Key Public Methods
| Method                              | Purpose                                                                        | Log Level When Called  | Success Indicator                                 |
|--------------------------------------|--------------------------------------------------------------------------------|-----------------------|---------------------------------------------------|
| `get_instance()`                     | Get the global singleton instance                                              | INFO                  | Instance returned                                 |
| `get_parameter()`                    | Retrieve current value for a parameter                                         | (none)                | Value returned                                    |
| `set_parameter()`                    | Set a parameter value (with constraints/history/persistence)                   | (none)                | Returns True, value stored and saved if possible  |
| `get_all_parameters()`               | Get all current parameter values                                               | (none)                | Dict returned                                     |
| `set_parameter_constraints()`        | Set constraints (min/max/default) for a parameter                              | (none)                | Constraints updated, value forced into bounds      |
| `create_profile()`                   | Create a parameter profile (optionally with description/source)                | (none)                | Profile created                                   |
| `delete_profile()`                   | Delete a parameter profile (if not active)                                     | (none)                | Returns True if deleted                           |
| `activate_profile()`                 | Activate a named profile and apply its parameters                             | INFO                  | True if activated, applies values                 |
| `get_best_profile_for_context()`     | Find and return the best-fitting profile for current context                   | (none)                | Tuple (profile_name, score)                       |
| `adapt_parameters()`                 | Adapt parameters to context (if needed/forced)                                 | INFO                  | Returns adaptation result dict                    |
| `learn_from_performance()`           | Learn from performance metrics, suggest/record profile improvements            | (none)                | Returns learning result dict                      |
| `get_status()`                       | Report status including active profile, context, history, etc.                 | (none)                | Dict returned                                     |
| `get_profile()`, `get_all_profiles()`| Retrieve profile(s) as dict(s)                                                 | (none)                | Profile dict(s)                                   |
| `reset_to_defaults()`                | Reset all parameters and profiles to defaults                                  | INFO                  | True if reset                                     |
| `initialize_storage()`               | Initialize/verify persistent storage before loading/saving parameters          | INFO/WARN/ERROR       | True if storage available, logs error otherwise   |

## Current Logging Philosophy

### INFO Level
- Initialization of the adaptive parameter manager and storage connections:
  - `"Adaptive parameter manager initialized with default profiles"`
  - `"Successfully connected to persistent storage"`
  - `"Loaded %d profiles from storage"`
  - `"Connected to already initialized storage"`
  - `"Initialized storage with base path: %s"`
  - `"Activated parameter profile: %s"`
  - `"Reset parameter manager to default settings"`

### WARNING Level  
- Storage not ready or used with defaults:
  - `"Storage not yet initialized - using defaults"`
  - `"Cannot save parameters - storage not ready"`

### ERROR Level
- Storage or profile load/save errors:
  - `"Error checking storage readiness: %s"`
  - `"Failed to initialize storage: %s"`
  - `"Error loading parameter data: %s"`
  - `"Error saving parameter data: %s"`
  - `"Error loading profile %s: %s"`

### DEBUG Level
- Loaded profiles or parameter data:
  - `"Loaded profile: %s"`
  - `"No adaptive parameter data found in storage"`
  - `"Saved parameter data to storage"`

## Dependencies (Imports From)
- `persistent_statistics`: For persistent storage of profiles and parameters
- `statistics`, `math`, `time`, `threading`, `datetime`, `copy`, `enum`, `pathlib`: Standard library
- Logging: For diagnostics and error handling

## Used By (Imported By)
- Any Modbus statistics or optimization subsystem requiring context-aware, tunable parameters 
- Pattern detection, correlation management, polling system, and self-healing subsystems

## Key Data Flows
1. **Input:**
   - Requests to get/set parameters, define constraints, create/activate/delete profiles
   - Context data: load, error rates, time, system state (static or via callbacks)
   - Performance metrics for learning (error rates, efficiency, usage stats)
2. **Processing:**
   - Match current context to best profile using criteria and scoring
   - Apply constraints to parameter values before setting
   - Learn from historical performance and create improved profiles
   - Save/load all parameter and profile data to persistent storage if available
   - Track all parameter changes in history (for audit/learning)
3. **Output:**
   - Current parameters, constraints, and history
   - Profile data and recommendations
   - Logs for initialization, activation, adaptation, and errors

## Integration Points
- **Persistent Statistics Manager:**  
  For loading and saving adaptive parameter data (profiles, constraints, history)
- **Pattern/correlation/optimization systems:**  
  Provide and adapt parameters based on contexts such as load, error, pattern, time
- **Self-healing/Recovery:**  
  Adjust aggressiveness and scan intervals based on health/recovery state
- **Context Providers:**  
  Dynamic callbacks for context (load, time, errors, etc.) and static context

## Current Issues/Tech Debt
- Storage readiness is checked in many public methods (may cause repeated log noise)
- Context and profile matching logic could be further abstracted for extensibility/testing
- Profiles and constraints are managed in-memory; storage loss means reset to defaults
- Thread safety is via RLock but not all internal components are protected (especially context providers)
- Logging is thorough for errors and info, but lacks DEBUG/TRACE for adaptation logic details