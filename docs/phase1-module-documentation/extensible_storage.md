# Module: extensible_storage.py

## Functional Purpose
Provides a scalable, extensible, and versioned storage system for Modbus optimization data.  
Supports chunked, indexed, and optionally compressed storage for large entity datasets, as well as version-aware migrations and storage compaction.

## Core Responsibilities
- Store large statistical/optimization data efficiently using chunked files (with optional compression).
- Maintain robust indexing for fast lookup and retrieval of entity data.
- Handle storage versioning and support migration between storage formats.
- Provide mechanisms for automatic chunk splitting, compaction, and retention of historical data.
- Enable scalable read/write access with minimal memory footprint and efficient eviction of unused data.
- Support metadata and backup management for reliability.

## Key Public Classes and Methods
| Class/Method                        | Purpose                                                         | Log Level When Called         | Success Indicator                                 |
|------------------------------------- |-----------------------------------------------------------------|------------------------------|---------------------------------------------------|
| `StorageMigrator`                   | Handles storage version migration and structure upgrades         | INFO/ERROR/EXCEPTION         | True if migration succeeds                        |
| `StorageMigrator.check_and_migrate`  | Checks if migration is needed and performs it                   | INFO/ERROR/EXCEPTION         | Returns (success, from_version, to_version)       |
| `StorageMigrator._migrate_1_0_0_to_2_0_0` | Migrates from version 1 to 2, chunking and indexing           | INFO/ERROR/EXCEPTION         | True if migration completed                       |
| `DataChunk`                         | Represents a single chunk of data, supports load/save/compress  | ERROR                        | True if load/save succeeds                        |
| `DataChunk.save`                    | Persists data to disk, optionally compressed                    | ERROR                        | True if write succeeds                            |
| `ChunkManager`                      | Manages in-memory and on-disk chunks, eviction, and caching     | (none)                       | Chunks loaded/evicted as needed                   |
| `ChunkManager.get_chunk`            | Loads chunk into memory cache, evicting if necessary            | (none)                       | Returns DataChunk                                 |
| `ChunkManager.save_all_chunks`       | Saves all dirty chunks to disk                                  | (none)                       | True if all saves succeed                         |
| `EntityIndex`                       | Maintains mapping from entity IDs to chunk files                | ERROR                        | Index is loaded, saved, or updated                |
| `EntityIndex.save`                  | Persists entity index to disk                                   | ERROR                        | True if write succeeds                            |
| `EntityIndex.get_chunk_id`          | Retrieves the chunk ID for a given entity                       | (none)                       | Returns chunk ID or None                          |

## Current Logging Philosophy

### INFO Level
- Migration start, completion, and per-file migration actions:
  - `"Migrating storage from %s to %s"`
  - `"Migrated entity stats: %d entities"`
  - `"Copied %s to metadata directory"`
  - `"Migration completed successfully"`

### WARNING Level
- None in this module.

### ERROR Level
- Migration path or function missing:
  - `"No migration path found from %s to %s"`
  - `"Missing migration function from %s to next version"`
- Error during migration, chunk load/save, or index load/save:
  - `"Error migrating entity stats: %s"`
  - `"Error copying %s: %s"`
  - `"Error loading chunk %s: %s"`
  - `"Error saving chunk %s: %s"`
  - `"Error loading entity index: %s"`
  - `"Error saving entity index: %s"`
  - `"Unexpected error during migration: %s"`

### EXCEPTION/DEBUG Level
- Full stacktraces for migration or storage exceptions.

## Dependencies (Imports From)
- `json`, `os`, `shutil`, `hashlib`, `gzip`, `threading`, `time`, `datetime`, `collections`
- `pathlib.Path`: For file and directory handling
- `logging`: For diagnostics

## Used By (Imported By)
- Any system or statistics module requiring scalable, chunked, indexed, and/or compressed persistent storage for large entity datasets
- Migration and backup utilities needing to manage or upgrade storage format

## Key Data Flows
1. **Input:**
   - Entity statistics and optimization data (potentially large sets)
   - Storage version metadata and migration requirements
2. **Processing:**
   - Splits large entity data files into manageable chunk files, compressing if needed
   - Maintains indices for mapping entity IDs to chunks
   - Supports migration from old storage formats to new ones (1.x → 2.x)
   - Loads/saves chunks and indices on demand, evicting least-recently-used chunks from memory
   - Tracks and manages chunk size, compaction, and retention
3. **Output:**
   - Chunked, indexed, and optionally compressed persistent data files
   - Migration progress and error logs
   - Updated metadata and backup files

## Integration Points
- **Migration/Upgrade:**  
  Provides robust migration logic for upgrading storage format/version, including backups.
- **Statistics/Entity Data:**  
  Used as backend storage for entity statistics, patterns, clusters, and correlation data (after migration).
- **Compaction and Retention:**  
  Supports future integration of auto-compaction and retention enforcement for efficient long-term storage.

## Current Issues/Tech Debt
- Migration path logic is very simple; only handles direct upgrade from 1.0.0 → 2.0.0.
- LRU eviction in `ChunkManager` is basic; could be improved for high concurrency or very large datasets.
- Error handling is present but does not always propagate failures upward.
- Chunk splitting/compaction logic is basic; more advanced strategies could be added.
- Thread safety is provided for chunk manager, but not necessarily for all file operations.
- No explicit schema validation for chunked or migrated data.