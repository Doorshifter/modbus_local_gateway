# Module: correlation_manager.py

## Functional Purpose
Manages detection and maintenance of correlations between entities in the system, clustering entities with similar behaviors and persisting correlation data. Integrates with a storage backend to persist correlation matrices and clusters, and uses NumPy for accurate correlation when available.

## Core Responsibilities
- Collect and store timestamped numeric values for each entity.
- Compute pairwise correlations between entities, using either NumPy or a fallback method.
- Cluster entities based on correlation strength and update mappings from entities to clusters.
- Persist and restore correlation matrices and clusters using the StorageAware interface.
- Provide methods for querying clusters and correlation values for specific entities.

## Key Public Methods
| Method                         | Purpose                                                           | Log Level When Called        | Success Indicator                          |
|---------------------------------|-------------------------------------------------------------------|------------------------------|--------------------------------------------|
| `add_entity_value()`            | Store a new value for an entity (for correlation analysis)         | DEBUG (non-numeric value)    | Value appended to entity history           |
| `analyze_correlations()`        | Run correlation analysis and update clusters                       | INFO/DEBUG                   | Clusters and matrix updated, storage saved |
| `get_clusters()`                | Return current clusters as dict of cluster_id -> entity_ids        | (none)                       | Dict of clusters returned                  |
| `get_correlation()`             | Get correlation coefficient between two entities                   | (none)                       | Float or None                              |
| `get_cluster_for_entity()`      | Get the cluster ID for a given entity                              | (none)                       | Cluster ID string or None                  |

## Current Logging Philosophy

### INFO Level
- When correlation analysis starts and completes:
  - `"Running entity correlation analysis"`
  - `"Correlation analysis complete. Created %d clusters."`
- When correlation matrix or clusters are loaded from storage:
  - `"Loaded correlation matrix from storage"`
  - `"Loaded %d clusters from storage"`

### WARNING Level
- On NumPy import failure (at module load time):
  - `"NumPy not available, correlation analysis will use simple methods"`
- On failure to load/save matrix or clusters:
  - `"Failed to load correlation matrix: %s"`
  - `"Failed to load clusters: %s"`
  - `"Failed to save correlation matrix: %s"`
  - `"Failed to save clusters: %s"`
- On exception during correlation calculation with NumPy:
  - `"Error calculating correlation: %s"`

### ERROR Level
- On failure to save correlation matrix or clusters:
  - `"Failed to save correlation matrix: %s"`
  - `"Failed to save clusters: %s"`

### DEBUG Level
- On receiving non-numeric values for correlation:
  - `"Non-numeric value for %s: %s"` (entity/value)
- On not enough entities for analysis:
  - `"Not enough entities with data for correlation analysis"`

## Dependencies (Imports From)
- `storage_aware`: For persistence of clusters/matrix
- `numpy` (optional): For advanced correlation
- `logging`, `time`, `typing`, `collections`: Standard library

## Used By (Imported By)
- Pattern detection, statistics, or advanced interval modules requiring up-to-date cluster and correlation data
- Any system component needing correlation or grouping context

## Key Data Flows
1. **Input:**
   - New values for entities, with timestamps
   - Requests to analyze correlations (periodic, forced, or as needed)
2. **Processing:**
   - Stores recent values (up to 1000 per entity)
   - Computes pairwise correlations for entities with enough data
   - Uses NumPy if available, else falls back to simple statistical method
   - Aligns values in time for fair correlation computation
   - Builds or updates clusters: correlated entities grouped, others as singletons
   - Persists updated matrix and clusters to storage
3. **Output:**
   - Dicts of clusters, per-entity cluster assignments, and correlation coefficients
   - Logs for analysis, errors, and storage actions

## Integration Points
- **Storage System:**  
  Save/load correlation matrix and clusters for persistence and recovery
- **Pattern/Interval/Statistics Modules:**  
  Provide cluster and correlation lookups for optimization and pattern analysis

## Current Issues/Tech Debt
- Only analyzes correlations once per day by default (unless forced).
- NumPy is optional; fallback is less robust for large/high-frequency data.
- Not thread-safe; assumes single-threaded or externally coordinated use.
- No pruning of very old entity values except for the most recent 1000.
- Cluster logic is simple and may not find optimal groupings in all cases.
- Error/warning logs are present but could be more granular for recovery scenarios.