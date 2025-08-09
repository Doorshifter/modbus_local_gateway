# Module: storage.py

## Functional Purpose
Implements the main persistent storage manager for Modbus statistics and optimization data.  
Manages all disk-backed statistics files and advanced chunked storage for large-scale or long-running installations.  
Supports both synchronous and asynchronous (Home Assistant) operation.

## Core Responsibilities
- Provide CRUD and caching for:
  - Entity statistics
  - Pattern detection data
  - Cluster definitions
  - Correlation matrices
  - Interval optimization history
  - (Enhanced) Chunked and indexed data for scalable storage
- Support atomic, safe file writes with temp/replace strategy and directory management.
- Integrate with Home Assistant's async executor for all I/O operations in event loop.
- Centralize and expose storage metadata/status for diagnostics and admin tools.
- Facilitate migration to chunked/extensible storage as system scales.

## Key Public Classes and Methods

| Class/Method                                       | Purpose                                                               | Log Level When Called | Success Indicator                  |
|----------------------------------------------------|-----------------------------------------------------------------------|----------------------|------------------------------------|
| `StatisticsStorageManager`                         | Main class for all statistics storage and metadata                    | INFO/ERROR/WARNING/DEBUG | Singleton instance, caches loaded  |
| `get_instance()`                                   | Singleton getter for global access                                    | (none)               | Instance returned                  |
| `.async_initialize(force)`                         | Async setup: dirs, chunk manager, load caches                         | INFO/ERROR           | True if initialized                |
| `.set_hass(hass)`                                  | Attach Home Assistant instance for async                              | (none)               | HA context ready                   |
| `.async_get_entity_stats()` / `.get_entity_stats()`| Get entity stats (async/sync, per entity or all)                      | ERROR                | Dict/None                          |
| `.async_save_entity_stats()` / `.save_entity_stats()`| Save entity stats (async/sync)                                       | ERROR                | True if saved                      |
| `.async_get_patterns()` / `.get_patterns()`         | Get pattern data (async/sync)                                         | ERROR                | Dict                               |
| `.async_save_patterns()` / `.save_patterns()`       | Save pattern data (async/sync)                                        | ERROR                | True if saved                      |
| `.async_get_clusters()` / `.get_clusters()`         | Get cluster data (async/sync)                                         | ERROR                | Dict                               |
| `.async_save_clusters()` / `.save_clusters()`       | Save cluster data (async/sync)                                        | ERROR                | True if saved                      |
| `.async_get_correlation_matrix()` / `.get_correlation_matrix()` | Get correlation matrix (async/sync)                      | INFO/ERROR                | Dict                               |
| `.async_save_correlation_matrix()` / `.save_correlation_matrix()`| Save correlation matrix (async/sync)                         | ERROR                | True if saved                      |
| `.async_get_interval_history()` / `.get_interval_history()` | Get interval optimization history (with filtering, async/sync)         | ERROR                | Dict                               |
| `.async_save_interval_history()` / `.save_interval_history()` | Save interval optimization history (async/sync)                 | ERROR                | True if saved                      |
| `.async_get_storage_info()` / `.get_storage_info()` | Return storage system info, file stats, version (async/sync)           | DEBUG                | Dict                               |

## Current Logging Philosophy

### INFO Level
- On initialization:
  - `"Starting async initialization of statistics storage"`
  - `"Statistics storage initialized at %s"`

### ERROR Level
- On failed file/directory operations or cache loads:
  - `"Failed to create directory %s: %s"`
  - `"Error loading metadata: %s"`
  - `"Error loading entity stats: %s"`
  - `"Error loading patterns: %s"`
  - `"Error loading clusters: %s"`
  - `"Error loading interval history: %s"`
  - `"Failed to save %s: %s"`
  - `"Error loading correlation matrix: %s"`

### WARNING Level
- On JSON decode or IO errors:
  - `"Failed to load %s: %s"`

### DEBUG Level
- On file info/stat gathering errors:
  - `"Error getting file info: %s"`

## Dependencies (Imports From)
- `extensible_storage`: For chunk/index management and migration.
- `json`, `os`, `shutil`, `time`, `asyncio`
- `logging`
- `datetime`, `pathlib.Path`
- Typing: For type hints

## Used By (Imported By)
- All statistics, analytics, and optimization modules that need persistent storage.
- Home Assistant integrations for async-safe statistics management.
- System and admin diagnostics needing storage status info.

## Key Data Flows

1. **Input:**
   - Stats, patterns, clusters, correlation matrices, and interval histories to persist/load as JSON.
   - Home Assistant instance for async context.
   - Enhanced storage: chunked/statistical data (via extensible_storage).
2. **Processing:**
   - Caches all data in memory, flushes to disk periodically or on forced save.
   - Handles both sync and async file I/O, with atomic save via temp file/replace.
   - Initializes directories and storage metadata as needed.
   - Provides chunked/indexed storage for scalable operation.
3. **Output:**
   - Returns or persists all relevant statistics, metadata, and diagnostic info.
   - Returns storage info (file sizes, last modified, version, etc.).
   - Logs all errors, warnings, and key events.

## Integration Points

- **Async/HA Context:**  
  All async I/O runs in executor for event loop safety.
- **Chunked/Extensible Storage:**  
  Supports migration and operation at scale.
- **Other Analytics Modules:**  
  Exposes full CRUD for all statistics and stateful data.

## Current Issues/Tech Debt

- Some sync/async branching is verbose and error-prone.
- Cache invalidation and sync between memory/disk is simplistic.
- Enhanced storage (chunks/indexes) is present but not fully utilized by all consumers.
- Not all operations are atomic in failure scenarios.
- No explicit lock or concurrency control for cache/disk sync.
- No recovery/migration from legacy data formats beyond what is in extensible_storage.