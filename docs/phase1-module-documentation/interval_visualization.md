# Module: interval_visualization.py

## Functional Purpose
Provides visualization tools and data exports for the advanced interval optimization system.  
Enables real-time and historical understanding of how scan intervals, polling efficiency, and clustering are functioning and evolving in the system.

## Core Responsibilities
- Generate system-wide overview data for load, interval distribution, efficiency, and entity/poll counts.
- Visualize per-entity interval trends, polling characteristics, and pattern influence.
- Visualize per-cluster coordination, interval variance, and next poll scheduling.
- Export full visualization datasets (system, entities, clusters) as JSON for dashboards or external tools.
- Provide color-coded and trend information for easy understanding by UIs and dashboards.

## Key Public Methods
| Method                                     | Purpose                                                           | Log Level When Called | Success Indicator             |
|---------------------------------------------|-------------------------------------------------------------------|----------------------|-------------------------------|
| `generate_system_overview()`                | Returns dict of overall system stats and visualization elements   | (none)               | Dict returned                 |
| `generate_entity_visualization(entity_id)`  | Returns dict for one entity’s polling, interval, and trend info   | (none)               | Dict returned                 |
| `generate_cluster_visualization(cluster_id)`| Returns dict for cluster-level interval/coordination data         | (none)               | Dict returned                 |
| `export_visualization_data()`               | Returns JSON string with all visualization data                   | (none)               | JSON string                   |

## Current Logging Philosophy

### INFO Level
- None in this module

### WARNING Level
- None in this module

### ERROR Level
- None in this module

### DEBUG Level
- None in this module

## Dependencies (Imports From)
- `advanced_interval_manager.ADVANCED_INTERVAL_MANAGER`: Source of all stats and entity/cluster data
- `logging`, `time`, `json`: For logging, time, and data export

## Used By (Imported By)
- Frontend dashboards, admin UIs, or REST APIs needing interval, load, or cluster visualizations
- System diagnostics and data export scripts

## Key Data Flows
1. **Input:**
   - Entity or cluster IDs for targeted visualization
   - No direct input for system-wide overview/export (reads from manager)
2. **Processing:**
   - Aggregates stats from the interval manager for system, entities, and clusters
   - Computes trends, variances, coordination levels, and pattern influence
   - Formats output for ready consumption by dashboards
   - Exports complete snapshot as JSON for external tools
3. **Output:**
   - Dicts or JSONs with visualization data for system, entity, or cluster
   - Color codes, trend directions, and coordination levels for UI use

## Integration Points
- **Advanced Interval Manager:**  
  All statistics and state for visualization are sourced from ADVANCED_INTERVAL_MANAGER
- **Dashboards/UIs:**  
  Data is formatted for easy use in donut charts, time series, entity/cluster panels, etc.
- **Export:**  
  Supports batch exports for diagnostics, external analytics, or offline review.

## Current Issues/Tech Debt
- Uses only current snapshot stats; no true historical time series (load_history is a placeholder).
- Cluster and entity stats access private fields in `ADVANCED_INTERVAL_MANAGER` (violates encapsulation).
- No error logging if stats are missing, and cluster/entity not found errors are shallow.
- Color and trend rules are hardcoded; could be made more flexible/configurable.
- No thread safety; assumes single-threaded or read-only usage.