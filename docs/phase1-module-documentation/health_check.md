# Module: health_check.py

## Functional Purpose
Performs comprehensive health checks and generates recommendations for the Modbus optimization system, ensuring ongoing performance, reliability, and data integrity.  
Aggregates validation, consistency, storage, and scheduling diagnostics into a unified health score and actionable recommendations.

## Core Responsibilities
- Run blocking or async health checks across all system components, integrating validation, storage, and analysis scheduling.
- Calculate a composite health score (0-100) based on weighted status of file structure, entity consistency, clusters, scan intervals, and data quality.
- Detect and report errors, warnings, and issues from all subsystems.
- Generate actionable recommendations for fixing or improving system health, including auto-fixable and manual actions.
- Collect and attach system/storage/scheduler/runtime metrics to health check results.
- Provide async and sync interfaces for integration with Home Assistant and other orchestrators.

## Key Public Methods
| Method                          | Purpose                                                         | Log Level When Called   | Success Indicator                              |
|----------------------------------|-----------------------------------------------------------------|------------------------|------------------------------------------------|
| `check_health()`                 | Run a full (blocking) health check                              | (none)                 | Returns HealthCheckResult                      |
| `async_check_health()`           | Run a full health check asynchronously (HA integration)         | (none)                 | Returns HealthCheckResult                      |
| `set_hass()`                     | Set Home Assistant instance for async execution                 | (none)                 | Enables async health checks                    |

### Internal Methods
| Method                                 | Purpose                                                    | Log Level When Called   | Success Indicator         |
|-----------------------------------------|------------------------------------------------------------|------------------------|--------------------------|
| `_calculate_health_score()`             | Compute score from validation results                      | (none)                 | Updates HealthCheckResult |
| `_generate_recommendations()`           | Generate fix/improve actions from validation issues         | (none)                 | Updates HealthCheckResult |
| `_blocking_add_system_metrics()`        | Gathers metrics (storage, scheduler, runtime) (blocking)    | (none)                 | Updates HealthCheckResult |
| `_async_add_system_metrics()`           | Gathers metrics asynchronously (HA executor)               | (none)                 | Updates HealthCheckResult |

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
- `validation.ValidationManager`: For validation and integrity checks
- `storage.StatisticsStorageManager`: For storage status and metrics
- `analysis_scheduler.AnalysisScheduler`: For scheduling metrics
- `logging`, `datetime`, `time`, `os`, `sys`, `psutil` (optional): For logging/metrics

## Used By (Imported By)
- System orchestrator, Home Assistant integration, or diagnostics modules
- Any user interface or monitoring tools needing a comprehensive health dashboard

## Key Data Flows
1. **Input:**
   - List of current entity IDs (from runtime or HA)
   - Storage and validation reports from other modules
2. **Processing:**
   - Runs validation on all components and collects results
   - Calculates health score by weighted aggregation of component statuses
   - Extracts issues/errors/warnings for reporting
   - Generates actionable recommendations (auto/manual)
   - Gathers metrics: storage size, pattern/cluster counts, scheduler status, runtime stats
3. **Output:**
   - HealthCheckResult object (as dict) containing:
     - Health status and score
     - List of issues and recommendations
     - Detailed metrics from storage, scheduler, and runtime
     - Timestamp

## Integration Points
- **ValidationManager:**  
  Supplies validation and consistency status for all persistent and runtime data.
- **StatisticsStorageManager:**  
  Provides current storage metrics and diagnostics.
- **AnalysisScheduler:**  
  Supplies scheduled analysis status, next tasks, and execution stats.
- **Home Assistant:**  
  Supports async health checks via HA’s executor for non-blocking operation.

## Current Issues/Tech Debt
- Error/warning/info logs are almost entirely absent—issues only tracked in result, not in logs.
- Health scoring and recommendation logic is tightly coupled to validation schema and weighting.
- Blocking and async versions both must be maintained for HA and non-HA environments.
- Metrics gathering is tightly bound to `psutil` and may miss some info if unavailable.
- Recommendations/actions are static; no dynamic or learning-based health advice yet.