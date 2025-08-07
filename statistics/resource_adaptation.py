"""
Resource adaptation system for Modbus optimization.

This module dynamically adjusts processing intensity based on available
system resources to ensure reliable operation across different hardware.
"""

import logging
import time
from datetime import datetime
import threading
from typing import Dict, Any, Tuple, Optional

_LOGGER = logging.getLogger(__name__)

class ResourceAdaptiveScaling:
    """Adapts processing based on available system resources."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = ResourceAdaptiveScaling()
        return cls._instance
    
    def __init__(self):
        """Initialize resource adaptation system."""
        self.parallelism = 1
        self.batch_size = 50
        self.analysis_depth = "standard"
        self.last_measurement = 0
        self.measurement_interval = 300  # 5 minutes
        self.last_capabilities = None
        self.resource_history = []
        self.max_history = 24  # Keep 24 measurements
        self.capability_lock = threading.Lock()
        
        # Try to import psutil, but don't fail if not available
        try:
            import psutil
            self.psutil_available = True
        except ImportError:
            self.psutil_available = False
            _LOGGER.info("psutil not available - resource adaptation will use limited capability estimation")
    
    def measure_system_capabilities(self) -> Dict[str, Any]:
        """Measure current system capabilities."""
        current_time = time.time()
        
        # Use cached measurements if recent enough
        with self.capability_lock:
            if (current_time - self.last_measurement < self.measurement_interval and 
                self.last_capabilities is not None):
                return self.last_capabilities.copy()
        
        try:
            if self.psutil_available:
                import psutil
                
                # Get CPU and memory metrics
                cpu_count = psutil.cpu_count()
                available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
                cpu_usage_percent = psutil.cpu_percent(interval=0.5)
                
                # Calculate capacity score (0.0-1.0)
                # Higher means more available capacity
                cpu_capacity = 1.0 - (cpu_usage_percent / 100.0)
                memory_capacity = min(1.0, available_memory_mb / 1000.0)  # Cap at 1.0
                
                # Combined capacity score (weighted)
                capacity_score = (cpu_capacity * 0.7) + (memory_capacity * 0.3)
                
                capabilities = {
                    "cpu_count": cpu_count,
                    "cpu_usage_percent": round(cpu_usage_percent, 1),
                    "available_memory_mb": round(available_memory_mb),
                    "capacity_score": round(capacity_score, 2),
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                # Fall back to basic capability estimate
                capabilities = self._estimate_capabilities()
                
            # Store measurement
            with self.capability_lock:
                self.last_measurement = current_time
                self.last_capabilities = capabilities.copy()
                
                # Add to history
                self.resource_history.append(capabilities)
                
                # Keep history bounded
                while len(self.resource_history) > self.max_history:
                    self.resource_history.pop(0)
            
            return capabilities
            
        except Exception as e:
            _LOGGER.exception("Error measuring system capabilities: %s", e)
            return self._estimate_capabilities()
    
    def _estimate_capabilities(self) -> Dict[str, Any]:
        """Estimate system capabilities when psutil is not available."""
        import os
        import sys
        
        # Make conservative estimates
        try:
            # Try to get CPU count through os
            cpu_count = os.cpu_count() or 1
        except Exception:
            cpu_count = 1
            
        # Assume moderate capacity
        capacity_score = 0.5
        
        return {
            "cpu_count": cpu_count,
            "cpu_usage_percent": None,
            "available_memory_mb": None,
            "capacity_score": capacity_score,
            "estimated": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def adapt_processing_intensity(self) -> Tuple[int, int, str]:
        """Adapt processing intensity based on system capabilities."""
        capabilities = self.measure_system_capabilities()
        capacity_score = capabilities["capacity_score"]
        
        # Adjust processing parameters based on capacity
        if capacity_score > 0.7:
            # High capacity - use full capabilities
            self.parallelism = max(1, capabilities.get("cpu_count", 1) - 1)
            self.batch_size = 100
            self.analysis_depth = "full"
        elif capacity_score > 0.4:
            # Medium capacity - moderate resource usage
            self.parallelism = max(1, (capabilities.get("cpu_count", 1) // 2))
            self.batch_size = 50
            self.analysis_depth = "standard"
        else:
            # Low capacity - conservative resource usage
            self.parallelism = 1
            self.batch_size = 25
            self.analysis_depth = "basic"
            
        # Log adaptation
        _LOGGER.debug(
            "Adapted processing: capacity=%.2f, parallelism=%d, batch_size=%d, depth=%s",
            capacity_score, self.parallelism, self.batch_size, self.analysis_depth
        )
        
        return self.parallelism, self.batch_size, self.analysis_depth
    
    def get_resource_history(self) -> Dict[str, Any]:
        """Get resource usage history."""
        with self.capability_lock:
            history = self.resource_history.copy()
            
        # Calculate trend if we have enough history
        trend = None
        if len(history) >= 2:
            first = history[0]["capacity_score"]
            last = history[-1]["capacity_score"]
            trend = round(last - first, 2)
        
        # Get min/max/avg
        if history:
            capacity_scores = [entry["capacity_score"] for entry in history]
            capacity_min = min(capacity_scores)
            capacity_max = max(capacity_scores)
            capacity_avg = sum(capacity_scores) / len(capacity_scores)
        else:
            capacity_min = capacity_max = capacity_avg = 0
        
        return {
            "current": self.last_capabilities,
            "history_points": len(history),
            "capacity_trend": trend,
            "capacity_min": round(capacity_min, 2),
            "capacity_max": round(capacity_max, 2),
            "capacity_avg": round(capacity_avg, 2),
            "adaptation": {
                "parallelism": self.parallelism,
                "batch_size": self.batch_size,
                "analysis_depth": self.analysis_depth
            }
        }
    
    def get_throughput_recommendation(self) -> Dict[str, Any]:
        """Get recommended throughput parameters."""
        capabilities = self.measure_system_capabilities()
        capacity = capabilities["capacity_score"]
        
        # Calculate recommended polling throughput based on your existing batch management
        if capabilities.get("cpu_count"):
            base_throughput = capabilities["cpu_count"] * 5  # Base throughput per CPU core
        else:
            base_throughput = 10  # Conservative default
        
        # Adjust by capacity
        throughput = base_throughput * capacity
        
        # Safety factor (80% of theoretical maximum)
        safe_throughput = throughput * 0.8
        
        return {
            "polls_per_second": round(safe_throughput, 1),
            "capacity_score": capacity,
            "estimated_entity_capacity": {
                "1s_interval": int(safe_throughput),
                "5s_interval": int(safe_throughput * 5),
                "30s_interval": int(safe_throughput * 30),
                "60s_interval": int(safe_throughput * 60)
            }
        }

    def should_run_expensive_operation(self, operation_name: str) -> bool:
        """Determine if an expensive operation should run based on current resources."""
        capabilities = self.measure_system_capabilities()
        capacity = capabilities["capacity_score"]
        
        # Define capacity thresholds based on operation
        if operation_name == "correlation_analysis":
            required_capacity = 0.4
        elif operation_name == "pattern_detection":
            required_capacity = 0.3
        elif operation_name == "prediction_training":
            required_capacity = 0.5
        else:
            # Default for unknown operations
            required_capacity = 0.4
            
        should_run = capacity >= required_capacity
        
        if not should_run:
            _LOGGER.info(
                "Deferring %s operation due to insufficient capacity (%.2f < %.2f required)",
                operation_name, capacity, required_capacity
            )
            
        return should_run


# Global instance
RESOURCE_ADAPTER = ResourceAdaptiveScaling.get_instance()