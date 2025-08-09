# Module: data_validation.py

## Functional Purpose
Ensures integrity, validity, and consistency of persistent Modbus optimization data files.  
Provides routine and on-demand validation, integrity checks, and orphaned reference cleanup for all key storage files (meta, entity_stats, clusters, correlation, patterns, etc).

## Core Responsibilities
- Validate structure and content of all persistent data files against schemas.
- Detect corruption, missing files, invalid references, and schema mismatches.
- Provide both synchronous and asynchronous validation and fixing methods.
- Identify and remove orphaned entities and broken cluster references from storage.
- Facilitate integration with Home Assistant’s async executor for non-blocking validation.

## Key Public Methods
| Method                           | Purpose                                                      | Log Level When Called   | Success Indicator                             |
|-----------------------------------|--------------------------------------------------------------|------------------------|-----------------------------------------------|
| `validate_file()`                 | Validate a given data structure against its schema           | (none)                 | Returns (is_valid, list_of_errors)            |
| `validate_meta()`                 | Validate the meta file and all referenced files exist        | (none)                 | Returns (is_valid, list_of_errors)            |
| `validate_all_files()`            | Validate all storage files (structure, type, existence)      | (none)                 | Returns results dict with overall/file status |
| `validate_entity_references()`    | Detect orphaned/missing entities in stats/clusters           | (none)                 | Returns (is_valid, issues_dict)               |
| `fix_orphaned_entities()`         | Remove orphaned entities from stats/clusters files           | (none)                 | Dict of removals/errors                       |
| `async_validate_meta()`           | Async wrapper for meta validation (HA integration)           | (none)                 | As above                                      |
| `async_validate_all_files()`      | Async wrapper for all-file validation                        | (none)                 | As above                                      |
| `async_validate_entity_references()` | Async wrapper for entity reference validation            | (none)                 | As above                                      |
| `async_fix_orphaned_entities()`   | Async wrapper for fixing orphaned entity references          | (none)                 | As above                                      |
| `get_validator()`                 | Factory for StorageValidator                                 | (none)                 | Returns StorageValidator instance             |

## Current Logging Philosophy

### INFO Level
- None in this module.

### WARNING Level
- None in this module.

### ERROR Level
- None in this module.

### DEBUG Level
- When non-numeric values are encountered and ignored:
  - `"Non-numeric value for %s: %s"`

## Dependencies (Imports From)
- `json`, `os`, `pathlib.Path`: File access and parsing
- `logging`: For debug logs
- Typing: For type hints

## Used By (Imported By)
- Storage and statistics subsystems needing to validate or repair persistent Modbus optimization data
- Home Assistant and management scripts for preflight checks and maintenance

## Key Data Flows
1. **Input:**
   - Paths to storage base and current valid entity IDs
   - Data files to validate, as loaded from disk
2. **Processing:**
   - Loads, parses, and checks data files against internal schema definitions
   - Detects missing, corrupt, or invalid files and schema violations
   - Identifies orphaned entities (not present in current IDs), missing entities, and broken cluster references
   - Can automatically remove orphaned entries from files if needed
3. **Output:**
   - Validation status and error lists per file
   - Overall storage health status
   - Cleaned or fixed storage files (when fix methods executed)
   - Debug logs for ignored values

## Integration Points
- **Home Assistant:**  
  Provides async wrappers for non-blocking validation/execution using HA’s async executor.
- **Persistent Storage/Management:**  
  Used to routinely or on-demand validate and clean persistent data files for integrity.

## Current Issues/Tech Debt
- Only validates structure and simple content, not deep semantic correctness.
- Schemas for most files are permissive/dynamic; may not catch misuse or subtle corruption.
- No INFO/WARNING/ERROR logs for validation/fix events—makes tracking harder in production.
- No explicit concurrency or file-locking—race conditions possible if used in parallel with writers.
- Does not handle all possible cross-file consistency checks (e.g., pattern/cluster/entity interlinks).