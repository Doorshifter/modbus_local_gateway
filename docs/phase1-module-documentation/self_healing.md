# Module: self_healing.py

## Functional Purpose
Provides self-healing functionality for the Modbus optimization system.  
Automatically detects, attempts to fix, and records common system issues (file structure, data integrity, basic entity consistency) to maximize reliability and minimize manual intervention.

## Core Responsibilities
- Check and automatically repair file structure issues (missing directories/files, meta.json, required stats files).
- Detect and attempt to fix JSON corruption or missing data in core statistics files.
- (Planned) Detect and heal entity-level consistency/data issues (integration stub provided).
- Maintain a bounded history of healing operations for diagnostics and review.
- Expose APIs for triggering healing checks (full or partial), entity-specific healing, and retrieving healing history.

## Key Public Classes and Methods

| Class/Method                              | Purpose                                                        | Log Level When Called   | Success Indicator                    |
|-------------------------------------------|----------------------------------------------------------------|------------------------|--------------------------------------|
| `SelfHealingSystem`                       | Main class for self-healing checks and recovery                | EXCEPTION              | Singleton instance available         |
| `.check_and_heal_system(full_check)`      | Run healing checks (file/data/entity) and perform fixes        | EXCEPTION              | Dict with status/results             |
| `.heal_entity_data(entity_id, issue_type)`| (Stub) Attempt to heal entity-specific issue                   | EXCEPTION              | Dict with outcome (integration stub) |
| `.get_healing_history()`                  | Return list of recent healing operations                       | (none)                 | List of dicts                        |
| `.initialize(storage_path)`               | Set storage path for all healing operations                    | (none)                 | Path set                             |

## Current Logging Philosophy

### EXCEPTION Level
- On error during self-healing or entity healing:
  - `"Error during self-healing: %s"`
  - `"Error healing entity %s: %s"`

### INFO/WARNING/DEBUG Level
- None in this module (file/data creation errors are only added to results, not logged).

## Dependencies (Imports From)
- `logging`, `os`, `json`, `datetime`, `time`, `threading`
- `pathlib.Path`
- Typing: For type hints

## Used By (Imported By)
- System orchestrator, diagnostics, or admin tools needing automatic recovery from file/data issues.
- Performance optimization or health check systems seeking to trigger self-healing as a remediation.

## Key Data Flows

1. **Input:**
   - Optional: storage path to operate on, full_check flag, entity_id and issue_type for entity healing.
2. **Processing:**
   - Checks for missing directories/files and creates them as needed.
   - Checks for corrupt JSON in known files and repairs/replaces with empty defaults.
   - (Planned) Can check for and fix entity-level data issues (stubbed).
   - Records all operations in a bounded in-memory history.
3. **Output:**
   - Dicts with status, issues detected/fixed, operations performed, and messages.
   - List of recent healing operations.

## Integration Points

- **Statistics Manager:**  
  Expected to provide storage path and in the future coordinate entity-level healing.
- **Health Check/Optimization:**  
  May invoke healing as an automatic remediation for detected issues.

## Current Issues/Tech Debt

- Entity-level healing is only stubbed, not integrated with statistics/coordinator managers.
- All operations are performed synchronously; no async/HA support.
- No logging for file/data creation/repair except on exceptions.
- No concurrency or locking for file I/O, only for history.
- No notification or alerting on healing beyond internal history.
- No persistence of healing history across restarts.