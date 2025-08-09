# Module: analysis_scheduler.py

## Functional Purpose
Schedules and manages the execution of various analysis operations (such as pattern detection, correlation analysis, and optimization tasks) within the Modbus optimization system. Ensures that analyses are performed at appropriate intervals, under suitable system conditions, and with error/backoff handling.

## Core Responsibilities
- Define and register recurrent analysis tasks with intervals, priorities, and optional execution conditions.
- Enforce execution timing, maximum runtime, and priority among multiple analysis tasks.
- Provide error handling (including exponential backoff) and runtime enforcement for scheduled tasks.
- Integrate system load checks before running tasks (if psutil is available).
- Track and report execution statistics for all registered tasks.

## Key Public Methods
| Method                    | Purpose                                                               | Log Level When Called      | Success Indicator                                |
|---------------------------|-----------------------------------------------------------------------|---------------------------|--------------------------------------------------|
| `register_task()`         | Register a new scheduled analysis task                                | INFO                      | Task registered, appears in `tasks` dict         |
| `unregister_task()`       | Remove a scheduled analysis task by name                              | INFO                      | Returns True if removed                          |
| `check_runnable_tasks()`  | Returns list of due tasks (by time/condition/priority/load)           | WARNING                   | List sorted by descending priority               |
| `execute_due_tasks()`     | Executes all tasks that are due to run                                | WARNING/ERROR/DEBUG       | Returns count of tasks executed                  |
| `get_task_stats()`        | Returns detailed statistics for scheduler and all tasks                | (none)                    | Dict of scheduler and task stats                 |
| `get_next_task()`         | Returns name and ETA of next scheduled task                           | (none)                    | String or None                                   |

### `AnalysisTask` Methods
| Method                  | Purpose                                                   | Log Level When Called     | Success Indicator          |
|-------------------------|-----------------------------------------------------------|--------------------------|----------------------------|
| `should_run()`          | Determines if a task should execute at current time       | (none)                   | Boolean                    |
| `execute()`             | Executes callback, tracks timing/errors/backoff           | WARNING/ERROR/DEBUG      | Returns True on success    |

## Current Logging Philosophy

### INFO Level
- Task registration/unregistration:
  - `"Registered analysis task: %s (interval: %ds, priority: %d)"`
  - `"Unregistered analysis task: %s"`

### WARNING Level
- If a task is already running when another execute is attempted:
  - `"Task %s is already running"`
- If a task runtime exceeds its max:
  - `"Analysis task %s took %.2fs (exceeds max allowed %ds)"`
- If system load (via psutil) is too high to run tasks:
  - `"System load too high (%.1f%%), deferring analysis tasks"`

### ERROR Level
- When a task's callback raises an exception:
  - `"Error in analysis task %s (attempt #%d): %s"`

### DEBUG Level
- When starting/completing an analysis task:
  - `"Starting analysis task: %s"`
  - `"Completed analysis task: %s (%.2fs)"`

## Dependencies (Imports From)
- `logging`, `time`, `datetime`, `timedelta`: For logging, time tracking, and scheduling
- `psutil`: Used (if available) to check system CPU load before running analysis
- Typing: For type annotations

## Used By (Imported By)
- System orchestrator, manager, or controller for scheduling regular background analysis tasks
- Pattern detection, correlation, and optimization modules that require scheduled execution

## Key Data Flows
1. **Input:**
   - Task definitions and callbacks from analysis modules
   - Optional execution conditions and priorities
2. **Processing:**
   - Tracks last/next run times, enforces min interval and max runtime
   - Handles failures with exponential backoff before retry
   - Checks system load (if psutil is available) before running
   - Prioritizes tasks by importance
   - Updates statistics for executions, errors, and scheduling
3. **Output:**
   - Executes registered task callbacks as appropriate
   - Provides statistics and next task scheduling information
   - Logs execution, errors, and scheduling events

## Integration Points
- **Analysis Modules:**  
  Modules register their analysis callbacks (pattern detection, correlation, optimization, etc.) for scheduled execution.
- **System Load Integration:**  
  Optionally checks system CPU load using `psutil` before running due tasks.
- **Diagnostics/Reporting:**  
  Exposes stats for monitoring and diagnostics (via `get_task_stats()`).

## Current Issues/Tech Debt
- No explicit thread safety; assumes single-threaded or externally-coordinated scheduling.
- Exponential backoff is simple and does not persist across restarts.
- System load check is only available if `psutil` is installed; else skipped silently.
- Tasks are executed synchronously; long-running tasks may block others or the main thread.
- No support for recurring tasks with variable intervals or cron-like scheduling.