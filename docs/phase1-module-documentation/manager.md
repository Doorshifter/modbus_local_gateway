# Module: statistics/manager.py

## Functional Purpose
Central orchestrator for the entire Modbus statistics and optimization system. Acts as a singleton coordinator that integrates multiple specialized subsystems (correlation, pattern detection, interval optimization, storage) into a unified interface for the coordinator and Home Assistant services.

## Core Responsibilities
- **Subsystem Integration**: Coordinates 8+ specialized statistics modules through a single interface
- **Entity Lifecycle Management**: Registers/unregisters entity trackers and coordinators 
- **Periodic Analysis Orchestration**: Runs daily correlation analysis and adaptive parameter optimization
- **Unified Statistics API**: Provides consolidated statistics combining data from all subsystems
- **Parameter Management**: Applies adaptive parameters across all integrated components

## Key Public Methods
| Method | Purpose | Log Level When Called | Success Indicator |
|--------|---------|---------------------|-------------------|
| `get_instance()` | Singleton access pattern | DEBUG (implicit) | Instance created/returned |
| `register_coordinator()` | Register gateway coordinator | INFO | "Registered coordinator for gateway {key}" |
| `register_entity_tracker()` | Register entity for tracking | DEBUG (implicit) | Entity added to internal trackers |
| `process_entity_update()` | Process value changes for analysis | DEBUG (implicit) | Updates propagated to all subsystems |
| `get_entity_statistics()` | Get comprehensive entity stats | DEBUG (implicit) | Statistics object returned with all subsystem data |
| `optimize_all_intervals()` | Force interval optimization | INFO (in interval manager) | Dictionary of optimized intervals returned |
| `analyze_correlations()` | Force correlation analysis | INFO | "Running scheduled daily statistical analysis" |
| `adapt_parameters()` | Trigger parameter adaptation | INFO | "Adaptive parameters: Activated profile..." |

## Current Logging Philosophy

### INFO Level
- **Operational milestones**: Successful coordinator registration, daily analysis completion
- **Adaptive parameter changes**: Profile activations and system adaptations
- **Examples from code**:
  - `"Registered coordinator for gateway {gateway_key}"`
  - `"Running scheduled daily statistical analysis"`
  - `"Adaptive parameters: Activated profile '%s' (score: %.2f)"`
  - `"Pattern-correlation integration analysis completed: %s"`

### WARNING Level  
- **Initialization failures**: When components can't be initialized
- **Examples from code**:
  - `"Cannot register coordinator - statistics manager not initialized"`

### ERROR Level
- **Component failures**: Subsystem initialization errors, parameter adaptation failures
- **Examples from code**:
  - `"Error initializing statistics manager: %s"`
  - `"Error during adaptive parameter adaptation: %s"`
  - `"Error creating parameter profile: %s"`

### DEBUG Level
- **Parameter applications**: When adaptive parameters are applied to subsystems
- **Internal coordination**: Subsystem communication details
- **Examples from code**:
  - `"Applied adaptive parameters to system components"`

## Dependencies (Imports From)
- `statistics_tracker.py`: `StatisticsTracker`, `StatisticMetric` - Entity-level tracking
- `correlation.py`: `EntityCorrelationManager` - Cross-entity correlation analysis  
- `pattern_detection.py`: `PatternDetector` - Behavioral pattern recognition
- `interval_manager.py`: `INTERVAL_MANAGER` - Legacy interval optimization
- `persistent_statistics.py`: `PERSISTENT_STATISTICS_MANAGER` - Data persistence layer
- `adaptive_parameters.py`: `PARAMETER_MANAGER`, `ContextType` - Adaptive system tuning
- `advanced_interval_manager.py`: `ADVANCED_INTERVAL_MANAGER` - Enhanced interval optimization
- `interval_visualization.py`: `IntervalVisualizationTool` - Statistics visualization

## Used By (Imported By)
- `coordinator.py`: Uses manager for statistics coordination during polling cycles
- `__init__.py`: Imports `STATISTICS_MANAGER` singleton for Home Assistant integration  
- Home Assistant services: Accesses statistics through the singleton interface

## Key Data Flows

### 1. Entity Registration Flow
**Input**: Coordinator registers with gateway key and entity trackers
**Processing**: Creates internal tracking structures, initializes subsystem connections
**Output**: Registered entities ready for statistics collection

### 2. Real-time Update Flow  
**Input**: Entity value changes from coordinator polling
**Processing**: Propagates updates to correlation, pattern detection, and interval managers
**Output**: Updated statistics available through unified API

### 3. Periodic Analysis Flow
**Input**: Time-based triggers (daily analysis schedule)
**Processing**: Orchestrates correlation analysis, parameter adaptation, pattern-correlation integration
**Output**: Optimized system parameters and updated entity clusters

### 4. Statistics Retrieval Flow
**Input**: Requests for entity or system statistics
**Processing**: Aggregates data from all subsystems (tracking, correlation, patterns, intervals)
**Output**: Comprehensive statistics with recommendations and insights

## Integration Points
- **Coordinator Integration**: Receives entity updates during polling cycles, provides statistics back to entities
- **Storage Integration**: Persists statistics through `PERSISTENT_STATISTICS_MANAGER` 
- **Parameter Integration**: Applies adaptive parameters to all managed subsystems
- **Home Assistant Integration**: Exposes singleton interface for HA services and diagnostics

## Current Issues/Tech Debt
- **Circular Dependencies**: Heavy reliance on global singletons (`INTERVAL_MANAGER`, `PERSISTENT_STATISTICS_MANAGER`)
- **Initialization Complexity**: Complex initialization order dependencies between subsystems
- **Method Proliferation**: 40+ public methods handling diverse responsibilities (statistics, parameters, visualization)
- **Error Handling Inconsistency**: Some methods return `Dict[str, Any]` with error keys, others raise exceptions
- **Dual Interval Managers**: Maintains both legacy and advanced interval managers for compatibility
- **Large Method Size**: Some methods (like `get_entity_statistics`) handle multiple concerns in 40+ lines

## Potential Consolidation Opportunities
- **Parameter Management Methods**: 8 parameter-related methods could be grouped into a parameter management interface
- **Statistics Retrieval Methods**: Multiple `get_*` methods could be consolidated with a unified statistics API
- **Integration Analysis Methods**: Pattern-correlation integration methods are candidates for extraction to specialized module