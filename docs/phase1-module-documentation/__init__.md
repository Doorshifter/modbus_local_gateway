# Module: \_\_init\_\_.py

## Functional Purpose
Provides the initialization and public API for the statistics package in the Modbus Local Gateway.  
Manages lazy loading, instantiation, and cross-module orchestration of core statistics subsystems (tracking, healing, adaptation, persistence, optimization, interval management).

## Core Responsibilities
- Centralized entry point for accessing singleton/stateless instances of key statistics modules.
- Lazy-loads and caches subsystem components to avoid circular imports and optimize resource usage.
- Provides `initialize_all()` to initialize the entire statistics stack in a safe, ordered manner.
- Handles fallback and warnings for unavailable components.
- Exposes type imports and backward compatibility aliases for legacy integration.

## Key Public Methods
| Method                           | Purpose                                                      | Log Level When Called | Success Indicator                                |
|-----------------------------------|--------------------------------------------------------------|----------------------|--------------------------------------------------|
| `get_statistics_manager()`        | Returns (lazy-loads) global statistics manager instance      | (none)               | Instance returned                                |
| `get_self_healing_system()`       | Returns (lazy-loads) self-healing system instance            | (none)               | Instance returned                                |
| `get_resource_adapter()`          | Returns (lazy-loads) resource adapter instance               | (none)               | Instance returned                                |
| `get_persistent_statistics_manager()` | Returns (lazy-loads) persistent statistics manager        | (none)               | Instance returned                                |
| `get_performance_optimizer()`     | Returns (lazy-loads) performance optimizer                   | (none)               | Instance returned                                |
| `get_advanced_interval_manager()` | Returns (lazy-loads) advanced interval manager               | (none)               | Instance returned                                |
| `initialize_all()`                | Initializes all statistics systems in correct order           | INFO/ERROR           | True if all initialized, error log on failure     |

## Current Logging Philosophy

### INFO Level
- On successful global initialization:
  - `"All statistics components initialized"`

### WARNING Level  
- If backward-compatibility alias (`STATISTICS_MANAGER`) is not available at import time:
  - `"STATISTICS_MANAGER not available at import time, use get_statistics_manager() instead"`

### ERROR Level
- If initialization fails:
  - `"Failed to initialize statistics components: %s"`

### DEBUG Level
- If some optional/statistics modules can't be imported directly:
  - `"Some statistics modules could not be imported directly: %s"`

## Dependencies (Imports From)
- `manager`, `self_healing`, `resource_adaptation`, `persistent_statistics`, `performance_optimization`, `advanced_interval_manager`: Core statistics components
- `interval_system`, `storage_integration`: For ordered initialization
- `statistics_tracker`, `pattern_detection`, `correlation`, `interval_visualization`: Types and public symbols
- `logging`: For diagnostics

## Used By (Imported By)
- Any consumer of the statistics package (main gateway, diagnostics, tests)
- Other system modules needing a statistics manager or related subsystem

## Key Data Flows
1. **Input:**
   - Requests for subsystem instances (via getter functions)
   - Initialization call (`initialize_all`)
2. **Processing:**
   - Lazy-loads and instantiates components as needed, caching them
   - Initializes interval system and storage, wires up dependencies in correct order
   - Handles import errors and missing dependencies gracefully
3. **Output:**
   - Returns requested singleton instances for use throughout system
   - Logs initialization, warnings, and errors

## Integration Points
- **Subsystem Access:**  
  Provides getter functions for statistics manager, self-healing, resource adapter, persistent statistics manager, performance optimizer, and advanced interval manager.
- **System Initialization:**  
  `initialize_all()` sets up all statistics-related systems in the correct order, ensuring dependencies are met.
- **Type Exports:**  
  Exports common types and classes for use by other modules and for backward compatibility.

## Current Issues/Tech Debt
- Fallback to dummy statistics manager with a warning if import fails; may mask deeper issues.
- Direct imports inside functions to avoid circular dependencies; can obscure dependency structure.
- Some modules imported only for types, others for singleton access; could document more clearly.
- No explicit thread safety on the global `_COMPONENTS` dict.
- Some error handling is generic (logs errors but does not raise).