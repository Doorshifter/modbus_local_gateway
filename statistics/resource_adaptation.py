"""
Resource adaptation system for Modbus Local Gateway.

This module analyzes system resources and adapts polling behavior
to optimize performance based on available CPU/memory.
"""

import logging
import sys
import psutil
from typing import Dict, Any, Optional

_LOGGER = logging.getLogger(__name__)

class ResourceAdapter:
    """Analyzes system resources and makes recommendations for optimal settings."""

    def __init__(self):
        """Initialize the resource adapter."""
        self._system_capabilities = {}
        self._recommendations = {
            "polls_per_second": 10,  # Default conservative value
            "concurrent_requests": 1,
            "read_timeout": 10,
            "write_timeout": 10,
        }

    def measure_system_capabilities_blocking(self):
        """Measure system capabilities - BLOCKING VERSION for executor_job."""
        # This is a blocking function and should only be called from executor_job
        try:
            # Get system information
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count(logical=True)
            cpu_usage_percent = psutil.cpu_percent(interval=0.5)  # This was the blocking call at line 65
            
            # Store results
            self._system_capabilities = {
                "memory_total": memory.total,
                "memory_available": memory.available,
                "cpu_count": cpu_count,
                "cpu_usage": cpu_usage_percent,
                "platform": sys.platform,
                "python_version": sys.version,
            }
            
            # Calculate recommendations based on system capabilities
            self._calculate_recommendations()
            
            return self._system_capabilities
        except Exception as e:
            _LOGGER.error("Error measuring system capabilities: %s", e)
            return {}

    def measure_system_capabilities(self):
        """Non-blocking wrapper that returns cached capabilities.
        
        For actual measurement, use async_measure_system_capabilities.
        """
        # Just return whatever we have (might be empty on first call)
        return self._system_capabilities

    async def async_measure_system_capabilities(self, hass):
        """Async wrapper for measuring system capabilities."""
        return await hass.async_add_executor_job(self.measure_system_capabilities_blocking)

    def _calculate_recommendations(self):
        """Calculate recommendations based on system capabilities."""
        try:
            # Only calculate if we have valid system data
            if not self._system_capabilities:
                return
                
            # Get system values
            memory_available_gb = self._system_capabilities.get("memory_available", 0) / (1024 * 1024 * 1024)
            cpu_count = self._system_capabilities.get("cpu_count", 1)
            cpu_usage = self._system_capabilities.get("cpu_usage", 50)
            
            # Calculate polls per second based on system resources
            # More conservative if CPU usage is high or memory is low
            if cpu_usage > 80:
                polls_per_second = max(5, min(10, cpu_count * 2))
            elif cpu_usage > 50:
                polls_per_second = max(10, min(20, cpu_count * 3))
            else:
                polls_per_second = max(15, min(30, cpu_count * 4))
                
            # Adjust for available memory
            if memory_available_gb < 0.5:
                polls_per_second = max(5, polls_per_second // 2)
            elif memory_available_gb > 2:
                polls_per_second = min(50, polls_per_second * 1.5)
                
            # Set concurrent requests based on CPU count and usage
            concurrent_requests = max(1, min(4, cpu_count // 2))
            if cpu_usage > 70:
                concurrent_requests = 1
                
            # Update recommendations
            self._recommendations = {
                "polls_per_second": int(polls_per_second),
                "concurrent_requests": concurrent_requests,
                "read_timeout": 10 if cpu_usage < 70 else 15,
                "write_timeout": 10 if cpu_usage < 70 else 15,
            }
            
            _LOGGER.debug("Resource recommendations calculated: %s", self._recommendations)
        except Exception as e:
            _LOGGER.error("Error calculating resource recommendations: %s", e)

    def get_throughput_recommendation(self) -> Dict[str, Any]:
        """Get recommended throughput settings based on system capabilities."""
        return self._recommendations

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for diagnostics."""
        return self._system_capabilities
    
    def analyze_performance_trend(self, recent_performances: list) -> Dict[str, Any]:
        """Analyze recent performance measurements and adjust recommendations."""
        if not recent_performances or len(recent_performances) < 5:
            return self._recommendations
            
        try:
            # Calculate average response times
            avg_response_time = sum(perf.get('response_time', 0) for perf in recent_performances) / len(recent_performances)
            
            # Calculate error rates
            total_requests = sum(perf.get('total_requests', 0) for perf in recent_performances)
            total_errors = sum(perf.get('errors', 0) for perf in recent_performances)
            error_rate = (total_errors / total_requests) if total_requests > 0 else 0
            
            # Adjust recommendations based on performance metrics
            new_recommendations = dict(self._recommendations)
            
            # Adjust polls per second based on response time and error rate
            if avg_response_time > 1.0 or error_rate > 0.05:
                new_recommendations["polls_per_second"] = max(5, int(new_recommendations["polls_per_second"] * 0.8))
            elif avg_response_time < 0.2 and error_rate < 0.01:
                new_recommendations["polls_per_second"] = min(50, int(new_recommendations["polls_per_second"] * 1.2))
                
            # Adjust timeouts based on response times
            if avg_response_time > 5.0:
                new_recommendations["read_timeout"] = max(15, new_recommendations["read_timeout"] + 5)
                new_recommendations["write_timeout"] = max(15, new_recommendations["write_timeout"] + 5)
            
            _LOGGER.debug("Performance-based recommendations: %s", new_recommendations)
            return new_recommendations
            
        except Exception as e:
            _LOGGER.error("Error analyzing performance trend: %s", e)
            return self._recommendations

    def get_memory_recommendation(self) -> Dict[str, Any]:
        """Get memory-specific optimization recommendations."""
        if not self._system_capabilities:
            return {"reduce_poll_rate": False, "reduce_entities": False}
            
        memory_available_gb = self._system_capabilities.get("memory_available", 4) / (1024 * 1024 * 1024)
        memory_total_gb = self._system_capabilities.get("memory_total", 8) / (1024 * 1024 * 1024)
        memory_percent_used = 100 - (memory_available_gb / memory_total_gb * 100) if memory_total_gb > 0 else 50
        
        return {
            "reduce_poll_rate": memory_percent_used > 80 or memory_available_gb < 0.5,
            "reduce_entities": memory_percent_used > 90 or memory_available_gb < 0.25,
            "memory_pressure": memory_percent_used,
            "recommended_max_entities": max(50, int(memory_available_gb * 200)),
        }

# Global singleton instance
RESOURCE_ADAPTER = ResourceAdapter()