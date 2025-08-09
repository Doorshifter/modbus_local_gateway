# Module: persistent_statistics.py

## Functional Purpose
Provides persistent, disk-based storage and management for statistical data, patterns, clusters, and metadata relevant to Modbus optimization analysis.  
Enables both synchronous (blocking) and asynchronous (Home Assistant-friendly) access to all major statistics files and supports modular initialization and component wiring.

## Core Responsibilities
- Handle file I/O for all persistent statistics: entity stats, patterns, clusters, interval history, and metadata.
- Provide both blocking (direct) and async (HA executor) methods for reading/writing all data.
- Support caching for storage info to minimize disk I/O.
- Enable integration and initialization with Home Assistant and modular subcomponents (pattern/correlation/interval managers).
- Serve as the central manager for all persistent data across restarts and for system diagnostics.

## Key Public Classes and Methods

| Class/Method                               | Purpose                                                          | Log Level When Called   | Success Indicator                      |
|--------------------------------------------|------------------------------------------------------------------|------------------------|----------------------------------------|
| `StatisticsStorageManager`                 | Core class for low-level file I/O and JSON serialization         | DEBUG/ERROR/WARNING    | Files read/written, cache updated      |
| `.get_patterns()` / `.async_get_patterns()`| Retrieve pattern data (blocking/async)                           | DEBUG                  | Dict returned                          |
| `.save_patterns()` / `.async_save_patterns()`| Save pattern data                                              | DEBUG                  | True if successful                     |
| `.get_clusters()` / `.async_get_clusters()`| Retrieve cluster data (blocking/async)                           | DEBUG                  | Dict returned                          |
| `.save_clusters()` / `.async_save_clusters()`| Save cluster data                                              | DEBUG                  | True if successful                     |
| `.get_metadata_file()` / `.async_get_metadata_file()`| Retrieve arbitrary metadata JSON files                | DEBUG                  | Dict                                   |
| `.save_metadata_file()` / `.async_save_metadata_file()`| Save arbitrary metadata JSON files                    | DEBUG                  | True if successful                     |
| `.get_storage_info()` / `.async_get_storage_info()`  | Get complete storage status, file sizes, etc           | WARNING/DEBUG          | Dict with summary info                 |
| `.get_interval_history()` / `.async_get_interval_history()` | Retrieve interval history for all entities        | DEBUG                  | Dict                                   |
| `.save_interval_history()` / `.async_save_interval_history()` | Save interval history for all entities           | DEBUG                  | True if successful                     |

### PersistentStatisticsManager (high-level)
| Method                                    | Purpose                                                          | Log Level When Called   | Success Indicator                      |
|--------------------------------------------|------------------------------------------------------------------|------------------------|----------------------------------------|
| `.async_initialize()` / `.initialize()`    | Set up manager and storage, wiring subcomponents                 | INFO/WARNING/DEBUG     | Manager ready, components initialized  |
| `.set_hass()`                             | Wire up Home Assistant for async                                 | (none)                 | Async methods enabled                  |
| `.async_get_storage_info()` / `.get_storage_info()` | High-level storage info                                   | WARNING                | Dict                                   |
| `.async_save_entity_stats()` / `.save_entity_stats()` | Save stats for a specific entity                        | (none)                 | True if successful                     |
| `.async_get_entity_stats()` / `.get_entity_stats()` | Retrieve stats for one/all entities                      | DEBUG                  | Dict/None                              |
| `.async_get_patterns()` / `.get_patterns()`| Retrieve patterns                                             | (none)                 | Dict                                   |
| `.async_get_clusters()` / `.get_clusters()`| Retrieve clusters                                             | (none)                 | Dict                                   |

## Current Logging Philosophy

### DEBUG Level
- On directory creation, reading/writing, and I/O errors:
  - `"Statistics directory exists at %s"`
  - `"Error getting stats for %s: %s"`
  - Various for blocking calls in async context

### INFO Level
- On initialization:
  - `"Persistent statistics initialized with storage path: %s"`
  - `"Advanced interval manager initialized"`

### WARNING Level
- On blocking usage in async context:
  - `"Called blocking get_storage_info() - use async_get_storage_info instead"`
- On missing Home Assistant instance:
  - `"Home Assistant instance not set, using default path"`
- On initialization fallback or errors:
  - `"Could not create statistics directory: %s"`
  - `"Could not initialize advanced interval manager: %s"`

### ERROR Level
- On failed reads/writes:
  - `"Error reading from %s: %s"`
  - `"Error writing to %s: %s"`
  - `"Error reading meta file: %s"`

## Dependencies (Imports From)
- `json`, `os`, `time`, `asyncio`, `hashlib`
- `logging`
- `pathlib.Path`
- Typing: For type hints

## Used By (Imported By)
- Statistics, interval, pattern, and correlation managers needing persistent storage and retrieval.
- Home Assistant integrations and system orchestrators needing disk-backed, async-safe statistics.

## Key Data Flows

1. **Input:**
   - Stats, patterns, clusters, interval history, and metadata to be saved/loaded as JSON.
   - Home Assistant instance for async/HA context.
2. **Processing:**
   - Reads/writes JSON files, manages directory creation.
   - Caches storage info for quick repeated access.
   - Wires up subcomponents (pattern/correlation/interval) as available.
   - Initializes with correct base/config path, supporting both sync and async.
3. **Output:**
   - Returns or persists all relevant stats, pattern, cluster, and metadata info.
   - Returns storage info (size, file counts, status).
   - Logs relevant events, warnings, and errors.

## Integration Points

- **Home Assistant:**  
  Async methods offload I/O to executor for safe use in event loop.
- **Pattern/Correlation/Interval Managers:**  
  Optionally initialized and exposed as subcomponents for advanced analytics.
- **System Orchestrator:**  
  Used as the persistent data backend for all analytics and optimization modules.

## Current Issues/Tech Debt

- Blocking/sync methods are still available and not always clearly separated from async in usage.
- No explicit schema or validation on file contents—assumes all files are well-formed.
- No locking or concurrency protection on file I/O.
- Initialization of subcomponents is best-effort and may fail silently.
- Storage info cache is basic and may become stale.
- Error handling for file I/O is mostly logging; no robust recovery or retries.