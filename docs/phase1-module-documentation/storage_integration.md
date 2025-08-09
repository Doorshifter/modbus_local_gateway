# Module: storage_integration.py

## Functional Purpose
Handles the automatic wiring of storage-aware statistics components (such as correlation managers and pattern detectors) to the persistent storage system.  
Ensures that all relevant components in the statistics manager are connected to persistent storage at initialization time.

## Core Responsibilities
- Discover storage-aware components on the statistics manager (e.g., correlation manager, pattern detector).
- Attach the persistent storage manager to each component that inherits from `StorageAware`.
- Log the integration steps and any issues encountered during initialization.

## Key Public Methods

| Method                                     | Purpose                                                      | Log Level When Called        | Success Indicator                  |
|--------------------------------------------|--------------------------------------------------------------|-----------------------------|------------------------------------|
| `initialize_statistics_storage(statistics_manager)` | Connects all storage-aware components to persistent storage | DEBUG/ERROR                 | Components wired to storage        |

## Current Logging Philosophy

### DEBUG Level
- On successfully initializing storage for a component:
  - `"Initialized storage for %s"`

- When a component is not storage-aware:
  - `"Component is not storage-aware, skipping"`

### ERROR Level
- If storage initialization for a component fails:
  - `"Error initializing storage for component: %s"`

## Dependencies (Imports From)
- `storage_aware.StorageAware`
- `persistent_statistics.PERSISTENT_STATISTICS_MANAGER`
- `logging`
- Typing: For type hints

## Used By (Imported By)
- System/module initialization code or orchestrators setting up the statistics and storage system.

## Key Data Flows

1. **Input:**
   - The statistics manager instance, which may have private attributes for subcomponents.
2. **Processing:**
   - Collects all known subcomponents that might support storage integration.
   - Checks if each component is an instance of `StorageAware`.
   - Sets the storage manager for each, triggering their internal load-from-storage hooks.
   - Logs each step for diagnostics.
3. **Output:**
   - All storage-aware components are connected to persistent storage and ready to load/save state.

## Integration Points

- **Persistent Statistics Manager:**  
  Provides the underlying storage system used for all statistics components.
- **Correlation/Pattern/Other Managers:**  
  Any component inheriting from `StorageAware` can be automatically integrated.

## Current Issues/Tech Debt

- Only looks for `_correlation_manager` and `_pattern_detector`; new component types must be manually added.
- Relies on private attributes of the statistics manager; not robust to API changes.
- No async support; assumes synchronous storage operations.
- No error propagation—failures are logged but do not halt initialization.
- No notification of storage integration status to the calling code.