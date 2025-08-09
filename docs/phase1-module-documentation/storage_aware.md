# Module: storage_aware.py

## Functional Purpose
Provides a base class (`StorageAware`) for components that require persistent storage and retrieval of data, enabling a common interface to plug into a storage manager.  
Facilitates loading and saving component-specific metadata files using an attached storage manager, abstracting away the details of underlying storage operations.

## Core Responsibilities
- Allow components to set (and switch) their persistent storage manager at runtime.
- Offer base methods to load and save component-specific metadata files via the storage manager.
- Define overrideable hooks (`_load_from_storage`, `_save_to_storage`) for components to implement custom persistence logic.
- Provide safe, fallback behavior if no storage manager is attached.

## Key Public Methods

| Method                           | Purpose                                                      | Log Level When Called   | Success Indicator           |
|---------------------------------- |--------------------------------------------------------------|------------------------|-----------------------------|
| `set_storage_manager()`           | Attach a storage manager and trigger loading from storage     | (none)                 | Storage manager set         |
| `get_metadata_file()`             | Fetch a metadata file via the storage manager                | (none)                 | Dict (empty if not found)   |
| `save_metadata_file()`            | Save a metadata file via the storage manager                 | (none)                 | True if successful          |

### Methods to Override in Subclasses

| Method                | Purpose                                   |
|-----------------------|-------------------------------------------|
| `_load_from_storage()`| Load component data from storage          |
| `_save_to_storage()`  | Save component data to storage            |

## Current Logging Philosophy

- No INFO/WARNING/ERROR/DEBUG logs in normal operation.
- Only logger defined at module level; all operations are silent on failure or fallback.

## Dependencies (Imports From)
- `logging`
- Typing: For type hints

## Used By (Imported By)
- Any component or manager requiring persistent state across restarts (pattern detectors, correlation managers, entity trackers, etc.).
- Facilitates modular storage by plugging into any compatible storage manager with `get_metadata_file`/`save_metadata_file` methods.

## Key Data Flows

1. **Input:**
   - Storage manager instance (set at runtime).
   - File name and data dicts for metadata file operations.
2. **Processing:**
   - Loads or saves data via the attached storage manager.
   - Subclasses can implement custom load/save hooks for their own data.
3. **Output:**
   - Returns file contents (dict) or success status (bool) for metadata operations.
   - No errors raised—returns defaults if storage unavailable.

## Integration Points

- **Persistent Statistics/Storage Managers:**  
  Must provide `get_metadata_file` and `save_metadata_file` compatible APIs.
- **Component Architecture:**  
  Any component needing persistent, modular, or swappable storage can inherit this base class.

## Current Issues/Tech Debt

- No error logging or exceptions—failures are silent and may be hard to diagnose.
- No async support—purely synchronous, not suited for async event loops.
- No validation or schema enforcement on metadata files.
- No notification/callback mechanism on load/save.
- `set_storage_manager` triggers load but doesn't confirm success or propagate errors.