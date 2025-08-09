# Module: correlation.py

## Functional Purpose
Detects statistical correlations between entity value changes, groups related entities into clusters, and provides this relational information to optimize polling, pattern detection, and system analysis in the Modbus optimization system.

## Core Responsibilities
- Track and store timestamped numeric values for each entity.
- Calculate pairwise correlation coefficients between entities.
- Identify and manage clusters of highly correlated entities, including dynamic merging and cluster size limits.
- Provide correlation data and clustering information to other modules (e.g., for pattern detection, adaptive polling).
- Expose system-wide and per-entity statistics on correlation and clustering status.

## Key Public Methods
| Method                         | Purpose                                                         | Log Level When Called | Success Indicator                                       |
|---------------------------------|-----------------------------------------------------------------|----------------------|---------------------------------------------------------|
| `add_entity_value()`            | Record a new value (with timestamp) for an entity               | (none)               | Value appended to entity’s history                      |
| `get_correlation()`             | Get correlation coefficient between two entities                | (none)               | Float correlation returned or calculated                |
| `analyze_correlations()`        | Analyze all entity correlations and build clusters              | INFO                 | `"Correlation analysis complete: found %d clusters"`    |
| `get_clusters()`                | Get current clusters of correlated entities                     | (none)               | Dict of cluster_id to list of entity_ids                |
| `get_cluster_for_entity()`      | Get the cluster assignment for a specific entity                | (none)               | Cluster ID string or None                               |
| `get_correlation_matrix()`      | Return the cached correlation matrix                            | (none)               | Dict of entity-pair: correlation                        |
| `get_statistics()`              | Provide system-wide correlation/clustering statistics           | (none)               | Dict of stats                                           |
| `get_instance()` (classmethod)  | Get singleton instance of EntityCorrelationManager              | (none)               | Singleton returned                                      |

## Current Logging Philosophy

### INFO Level
- System-level summary after analysis:
  - `"Correlation analysis complete: found %d clusters"`

### WARNING Level  
- None in this module.

### ERROR Level
- None in this module.

### DEBUG Level
- None in this module.

## Dependencies (Imports From)
- `math`: For Pearson correlation calculations
- `logging`: For system event logging
- `time`: For timestamps and analysis throttling
- `collections`: For defaultdicts (internal data structures)

## Used By (Imported By)
- `pattern_detection.py`: For correlation-aware pattern detection and clustering
- `statistics_tracker.py`: For providing cluster context for statistics/adaptive behaviors
- Any module needing entity clusters or correlation values

## Key Data Flows
1. **Input:**
   - Entity value updates: entity_id, value, optional timestamp
2. **Processing:**
   - Stores recent value histories (up to 100 per entity)
   - Calculates pairwise Pearson correlations for entities with sufficient data and timestamp alignment
   - Builds clusters by merging highly correlated entities, respecting size limits
   - Updates correlation matrix cache
3. **Output:**
   - Clusters of correlated entities (for grouping, pattern, or polling optimization)
   - Correlation coefficients (for weighting, diagnostics, or pattern sensitivity)
   - System statistics (counts, strengths, thresholds)

## Integration Points
- **Pattern Detection:**  
  Supplies correlation and cluster info to pattern detection module for more accurate and robust pattern transitions and entity weighting.
- **Statistics Tracker:**  
  Entity clusters and correlation values can inform adaptive polling, group analysis, and efficiency metrics.
- **Manager/Coordinator:**  
  Used for system diagnostics, optimization, and reporting of correlated entity groups.

## Current Issues/Tech Debt
- No WARNING/ERROR/DEBUG logs for exceptional or suboptimal conditions (e.g., too-small sample sizes, failed correlations).
- Clustering is greedy and may not always find optimal groups; could be enhanced with more advanced algorithms.
- Does not prune old data beyond last 100 values; could have time-based data expiration.
- Assumes single-threaded access or external concurrency management (no explicit thread safety).
- No persistence; correlation matrix and clusters are rebuilt in-memory each run.