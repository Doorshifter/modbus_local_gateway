"""
Analysis scheduler for Modbus optimization.

This module manages the scheduling of various analysis operations like
pattern detection, correlation analysis, and optimization processes.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Callable, Optional, List

_LOGGER = logging.getLogger(__name__)

class AnalysisTask:
    """Represents a scheduled analysis task."""
    
    def __init__(
        self, 
        name: str, 
        interval_seconds: int, 
        callback: Callable,
        condition: Optional[Callable[[], bool]] = None,
        priority: int = 5,
        max_runtime_seconds: int = 60,
    ):
        """Initialize an analysis task.
        
        Args:
            name: Task name
            interval_seconds: Minimum seconds between executions
            callback: Function to call when task executes
            condition: Optional function that returns True when task should run
            priority: Task priority (1-10, higher means more important)
            max_runtime_seconds: Maximum allowed runtime in seconds
        """
        self.name = name
        self.interval_seconds = interval_seconds
        self.callback = callback
        self.condition = condition
        self.priority = min(10, max(1, priority))
        self.max_runtime_seconds = max_runtime_seconds
        self.last_run_time = 0
        self.next_run_time = 0
        self.last_runtime_seconds = 0
        self.last_error: Optional[str] = None
        self.execution_count = 0
        self.success_count = 0
        self.consecutive_errors = 0
        self.running = False
    
    def should_run(self, current_time: float) -> bool:
        """Check if the task should be executed now."""
        if self.running:
            return False
            
        if current_time < self.next_run_time:
            return False
            
        # Check custom condition if provided
        if self.condition is not None and not self.condition():
            return False
            
        return True
    
    def execute(self) -> bool:
        """Execute the task and record metrics."""
        if self.running:
            _LOGGER.warning("Task %s is already running", self.name)
            return False
            
        self.running = True
        current_time = time.time()
        self.last_run_time = current_time
        self.next_run_time = current_time + self.interval_seconds
        self.execution_count += 1
        
        try:
            start_time = time.time()
            _LOGGER.debug("Starting analysis task: %s", self.name)
            
            result = self.callback()
            
            self.last_runtime_seconds = time.time() - start_time
            self.success_count += 1
            self.consecutive_errors = 0
            self.last_error = None
            
            # Check if task took too long
            if self.last_runtime_seconds > self.max_runtime_seconds:
                _LOGGER.warning(
                    "Analysis task %s took %.2fs (exceeds max allowed %ds)",
                    self.name, self.last_runtime_seconds, self.max_runtime_seconds
                )
            
            _LOGGER.debug(
                "Completed analysis task: %s (%.2fs)", 
                self.name, self.last_runtime_seconds
            )
            return True
            
        except Exception as e:
            self.last_error = str(e)
            self.consecutive_errors += 1
            self.last_runtime_seconds = time.time() - start_time
            
            # Exponential backoff for failing tasks
            backoff_factor = min(5, self.consecutive_errors)
            self.next_run_time = current_time + (self.interval_seconds * backoff_factor)
            
            _LOGGER.error(
                "Error in analysis task %s (attempt #%d): %s",
                self.name, self.consecutive_errors, e
            )
            return False
        finally:
            self.running = False


class AnalysisScheduler:
    """Manages scheduling of analysis tasks."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = AnalysisScheduler()
        return cls._instance
    
    def __init__(self):
        """Initialize the scheduler."""
        self.tasks: Dict[str, AnalysisTask] = {}
        self.last_check_time = 0
        self.check_interval = 15  # Check for runnable tasks every 15 seconds
        self.system_load_limit = 0.85  # Maximum allowed system load
        self.enabled = True
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "last_check_time": None,
        }
    
    def register_task(
        self,
        name: str,
        callback: Callable,
        interval_seconds: int = 3600,
        condition: Optional[Callable[[], bool]] = None,
        priority: int = 5,
        max_runtime_seconds: int = 60,
        immediate_first_run: bool = False
    ) -> AnalysisTask:
        """Register a new analysis task.
        
        Args:
            name: Task name
            callback: Function to call when task executes
            interval_seconds: Minimum seconds between executions
            condition: Optional function that returns True when task should run
            priority: Task priority (1-10, higher means more important)
            max_runtime_seconds: Maximum allowed runtime in seconds
            immediate_first_run: If True, run immediately on first check
            
        Returns:
            The created task object
        """
        task = AnalysisTask(
            name=name,
            interval_seconds=interval_seconds,
            callback=callback,
            condition=condition,
            priority=priority,
            max_runtime_seconds=max_runtime_seconds,
        )
        
        if immediate_first_run:
            task.next_run_time = 0
        else:
            current_time = time.time()
            task.next_run_time = current_time + interval_seconds
        
        self.tasks[name] = task
        _LOGGER.info(
            "Registered analysis task: %s (interval: %ds, priority: %d)",
            name, interval_seconds, priority
        )
        return task
    
    def unregister_task(self, name: str) -> bool:
        """Unregister a task by name."""
        if name in self.tasks:
            del self.tasks[name]
            _LOGGER.info("Unregistered analysis task: %s", name)
            return True
        return False
    
    def check_runnable_tasks(self) -> List[AnalysisTask]:
        """Get list of tasks that should run now, ordered by priority."""
        current_time = time.time()
        
        # Only check periodically to reduce overhead
        if current_time - self.last_check_time < self.check_interval:
            return []
        
        self.last_check_time = current_time
        self.stats["last_check_time"] = datetime.fromtimestamp(current_time).isoformat()
        
        if not self.enabled:
            return []
            
        # Check system load if we can
        try:
            import psutil
            if psutil.cpu_percent(interval=0.1) / 100 > self.system_load_limit:
                _LOGGER.warning(
                    "System load too high (%.1f%%), deferring analysis tasks",
                    psutil.cpu_percent()
                )
                return []
        except ImportError:
            # psutil not available, skip load check
            pass
            
        # Get runnable tasks
        runnable_tasks = [
            task for task in self.tasks.values()
            if task.should_run(current_time)
        ]
        
        # Sort by priority (highest first)
        runnable_tasks.sort(key=lambda t: t.priority, reverse=True)
        return runnable_tasks
    
    def execute_due_tasks(self) -> int:
        """Execute all tasks that are due to run."""
        runnable_tasks = self.check_runnable_tasks()
        
        if not runnable_tasks:
            return 0
            
        executed_count = 0
        for task in runnable_tasks:
            result = task.execute()
            self.stats["total_executions"] += 1
            if result:
                self.stats["successful_executions"] += 1
                executed_count += 1
            else:
                self.stats["failed_executions"] += 1
        
        return executed_count
    
    def get_task_stats(self) -> Dict[str, Any]:
        """Get statistics for all tasks."""
        task_stats = {}
        for name, task in self.tasks.items():
            next_run_time_str = datetime.fromtimestamp(task.next_run_time).isoformat() \
                if task.next_run_time else None
            last_run_time_str = datetime.fromtimestamp(task.last_run_time).isoformat() \
                if task.last_run_time else None
                
            task_stats[name] = {
                "executions": task.execution_count,
                "successes": task.success_count,
                "failures": task.execution_count - task.success_count,
                "last_runtime_seconds": task.last_runtime_seconds,
                "last_run_time": last_run_time_str,
                "next_run_time": next_run_time_str,
                "last_error": task.last_error,
                "consecutive_errors": task.consecutive_errors,
                "priority": task.priority,
                "interval_seconds": task.interval_seconds,
                "running": task.running,
            }
        
        return {
            "scheduler": {
                "enabled": self.enabled,
                "task_count": len(self.tasks),
                "check_interval": self.check_interval,
                "system_load_limit": self.system_load_limit,
                "last_check_time": self.stats["last_check_time"],
                "total_executions": self.stats["total_executions"],
                "successful_executions": self.stats["successful_executions"],
                "failed_executions": self.stats["failed_executions"],
            },
            "tasks": task_stats
        }
    
    def get_next_task(self) -> Optional[str]:
        """Get the name of the next task scheduled to run."""
        if not self.tasks:
            return None
            
        current_time = time.time()
        next_task = min(
            self.tasks.values(),
            key=lambda t: t.next_run_time if t.next_run_time > current_time else float("inf")
        )
        
        if next_task.next_run_time < float("inf"):
            seconds_until = max(0, next_task.next_run_time - current_time)
            return f"{next_task.name} (in {seconds_until:.0f}s)"
        return None