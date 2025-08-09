# Module: validation.py

## Functional Purpose
Implements a comprehensive validation framework for the Modbus optimization system, integrating with Home Assistant (HA) lifecycle.  
Ensures persistent statistics and configuration data remain consistent, complete, and within integrity constraints; provides system health checks, auto-fixes, and reporting.

## Core Responsibilities
- Validate presence, integrity, and schema of core persistent files (meta, entity_stats, patterns, clusters, etc.).
- Detect and report issues in file structure, entity consistency, cluster assignments, scan intervals, and data quality.
- Integrate with HA startup/shutdown, deferring or scheduling validation as appropriate.
- Provide both async and sync validation/fix methods for use in event loop or background workers.
- Offer auto-remediation for common issues (e.g., missing files, orphaned entities, empty/undersized clusters).
- Maintain a cache of validation results and prepare summary reports for diagnostics or UI.

## Key Public Classes and Methods

| Class/Method                                 | Purpose                                                                   | Log Level When Called   | Success Indicator                |
|----------------------------------------------|---------------------------------------------------------------------------|------------------------|----------------------------------|
| `ValidationManager`                          | Main class for orchestrating all system validation                        | INFO/WARNING/ERROR     | Singleton instance, caches ready |
| `.set_hass(hass)`                            | Attach HA instance, register lifecycle, and schedule init/validation      | INFO                   | HA integration active            |
| `.async_validate_all(current_entity_ids)`     | Run all validations asynchronously (HA safe)                              | INFO                   | Dict with results/report         |
| `.validate_all(current_entity_ids)`           | Synchronous validation (for executor/background use)                      | INFO/WARNING/ERROR     | Dict with results/report         |
| `.async_fix_issues(issues_to_fix)`           | Auto-fix selected validation issues async                                 | (none)                 | Dict with fix results            |
| `.fix_issues(issues_to_fix)`                 | Synchronous fix for selected validation issues                            | (none)                 | Dict with fix results            |
| `.async_create_missing_files()` / `.create_missing_files()` | Ensure all required files exist (async/sync)                   | INFO/ERROR             | Dict with created/failed files   |
| `.async_validate_file_structure()` / `.validate_file_structure()` | Validate file presence, schema, version               | WARNING/ERROR           | ValidationResult                 |
| `.async_validate_entity_consistency()` / `.validate_entity_consistency()` | Validate entity stats vs. HA, clusters         | (none)                 | ValidationResult                 |
| `.async_validate_clusters()` / `.validate_clusters()` | Validate cluster assignments, structure                        | (none)                 | ValidationResult                 |
| `.async_validate_scan_intervals()` / `.validate_scan_intervals()` | Validate scan interval presence/range                 | (none)                 | ValidationResult                 |
| `.async_validate_data_quality()` / `.validate_data_quality()` | Check data sufficiency, black holes, staleness           | (none)                 | ValidationResult                 |
| `.async_read_json_file()` / `._blocking_read_json_file()` | Async/sync JSON file read                                | (none)                 | Tuple: (success, data, error)    |
| `.ValidationResult`                          | Tracks and reports validation status for a component                       | (none)                 | Includes errors, warnings, info  |
| `.ValidationStage`                           | Enum for lifecycle stage (INITIALIZING, READY, SHUTDOWN)                   | (none)                 | Used in report/status            |
| `VALIDATION_MANAGER`                         | Global singleton instance                                                 | (none)                 | Available everywhere             |

## Current Logging Philosophy

### INFO Level
- On HA startup, validation scheduling, file creation, and successful validation completion.
- Examples: `"Home Assistant started - switching to READY validation stage"`, `"First-time setup: Creating required storage files"`

### WARNING Level
- On file structure validation failures and when skipping further checks due to critical errors:
  - `"Skipping further validation due to file structure errors: ..."`

### ERROR Level
- On failure to create required files, or on exceptions during validation or auto-fix:
  - `"Failed to create some files: ..."`
  - `"Failed to initialize storage files: ..."`
  - `"Failed to fix file structure issues: ..."`

## Dependencies (Imports From)
- Python stdlib: `os`, `json`, `time`, `asyncio`, `datetime`, `enum`, `pathlib`
- Home Assistant: `core.HomeAssistant`, `helpers.event.async_call_later`
- Local: `storage.StatisticsStorageManager`, `data_validation.StorageValidator`
- Typing: For type hints

## Used By (Imported By)
- System orchestrator, admin UI, automation, or diagnostics tools needing validation or system health info.
- Performance optimization and self-healing systems for pre-flight or post-fix validation cycles.

## Key Data Flows

1. **Input:**
   - Current entity IDs (from HA), persistent stats/config files, and optionally explicit issues to fix.
2. **Processing:**
   - Checks for file existence, schema, and version integrity.
   - Detects orphaned/missing entities, empty/undersized clusters, invalid scan intervals, and data quality issues.
   - Caches results and prepares a comprehensive validation report.
   - Can perform auto-remediation of common issues (removing orphans, fixing clusters, etc.).
   - Listens to HA events to schedule or defer validation as needed (startup/shutdown).
3. **Output:**
   - Dict reports of validation status for each subsystem/component.
   - ValidationResult objects (as dict) with errors, warnings, info, and details.
   - Fix results for auto-remediation actions.

## Integration Points

- **Home Assistant:**  
  Registers for lifecycle events and uses async/scheduler integration for safe operation.
- **Persistent Storage/Statistics:**  
  Uses and validates all persistent data for system health.
- **Self-Healing/Optimization:**  
  Can be called to ensure system is valid before or after automated fixes/optimizations.

## Current Issues/Tech Debt

- Many methods are sync/blocking and require careful async wrapping for HA/event loop use.
- Only basic auto-remediation is provided for some issue types.
- No persistent logging or reporting of validation history.
- Some error/warning/skip logic is ad hoc and could be refactored for clarity.
- Validation is only as good as the schemas/checks defined in StorageValidator.
- Entity/cluster validation assumes canonical file and object structure.