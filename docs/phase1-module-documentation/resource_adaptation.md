# Module: resource_adaptation.py

## Functional Purpose
Analyzes system resources (CPU, memory) and dynamically adapts Modbus polling and request behavior to optimize performance and reliability on the Local Gateway.  
Provides recommendations for polling rate, concurrency, and timeout settings based on real-time and recent system conditions.

## Core Responsibilities
- Measure and cache system resource capabilities (CPU count/usage, memory, platform, etc.).
- Calculate throughput and concurrency recommendations based on current and historical resource usage.
- Adapt polling and request strategies based on system load and recent performance trends.
- Expose recommendations for use by interval/optimization modules and system diagnostics.
- Support both blocking and async (executor-compatible) measurement methods for Home Assistant integration.

## Key Public Classes and Methods

| Class/Method                           | Purpose                                                               | Log Level When Called   | Success Indicator                  |
|----------------------------------------|-----------------------------------------------------------------------|------------------------|------------------------------------|
| `ResourceAdapter`                      | Main class for system resource analysis and adaptation                | ERROR/DEBUG            | Recommendations/system info updated|
| `.measure_system_capabilities_blocking()` | Blocking, measures and caches system resource data                  | ERROR                  | System capabilities dict returned  |
| `.async_measure_system_capabilities()` | Async (executor) wrapper for safe use in event loop                   | (none)                 | System capabilities dict returned  |
| `.get_throughput_recommendation()`     | Returns current polling/concurrency/timeout recommendations           | (none)                 | Dict returned                      |
| `.analyze_performance_trend()`         | Adjust recommendations based on recent performance stats              | ERROR/DEBUG            | New recommendations dict returned  |
| `.get_memory_recommendation()`         | Returns memory-based optimization hints                               | (none)                 | Dict returned                      |

## Current Logging Philosophy

### ERROR Level
- On failure to measure system capabilities:
  - `"Error measuring system capabilities: %s"`
- On failure to calculate recommendations:
  - `"Error calculating resource recommendations: %s"`
- On failure to analyze performance trend:
  - `"Error analyzing performance trend: %s"`

### DEBUG Level
- On successful calculation of recommendations:
  - `"Resource recommendations calculated: %s"`
  - `"Performance-based recommendations: %s"`

### INFO/WARNING Level
- None in this module.

## Dependencies (Imports From)
- `psutil`, `sys`
- Typing: For type hints

## Used By (Imported By)
- Performance optimization loop, interval manager, system orchestrator, or diagnostics needing to adapt polling to available resources.

## Key Data Flows

1. **Input:**
   - System resource measurements (CPU usage, memory, core count, etc.)
   - Recent performance data: response times, error counts, request totals.
2. **Processing:**
   - Measures and caches system resource info.
   - Calculates polling/concurrency/timeouts based on CPU/memory and load.
   - Adjusts recommendations in response to recent performance trends and error rates.
   - Provides memory-specific recommendations for safe operation.
3. **Output:**
   - Dicts of recommended settings for polling, concurrency, timeouts, and memory pressure mitigation.

## Integration Points

- **Performance Optimization Loop:**  
  Used to decide if expensive optimizations are safe and to adapt polling behavior dynamically.
- **Diagnostics and Health Checks:**  
  Reports system capabilities and recommendations for dashboards or troubleshooting.
- **Interval/Cluster Manager:**  
  May use recommendations to throttle or adapt scan rates based on resource pressure.

## Current Issues/Tech Debt

- Recommendations are heuristic and may need tuning for real-world workloads.
- No persistent storage of past recommendations or performance—recommendations reset on restart.
- All resource measurements are synchronous (except for async executor wrapper).
- No explicit thread safety on shared state.
- No support for distributed/multi-instance resource coordination.