"""
Adaptive parameters system for Modbus integration.

Provides dynamic tuning of system parameters based on:
- Current system conditions (context)
- Historical performance data
- Device operational patterns
- Machine learning from parameter effectiveness
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
import statistics
import math
import threading
import copy
from enum import Enum
from pathlib import Path

from .persistent_statistics import PERSISTENT_STATISTICS_MANAGER

_LOGGER = logging.getLogger(__name__)

# Default parameter constraints
DEFAULT_PARAMETER_CONSTRAINTS = {
    # Scan intervals
    "min_scan_interval": {
        "min": 5,
        "max": 30,
        "default": 10
    },
    "max_scan_interval": {
        "min": 300,
        "max": 3600,
        "default": 900
    },
    "default_scan_interval": {
        "min": 15,
        "max": 60,
        "default": 30
    },
    
    # Batch processing
    "max_batch_size": {
        "min": 5,
        "max": 50,
        "default": 20
    },
    
    # Correlation settings
    "correlation_threshold": {
        "min": 0.3,
        "max": 0.9,
        "default": 0.6
    },
    "correlation_min_samples": {
        "min": 10,
        "max": 100,
        "default": 20
    },
    
    # Pattern detection
    "pattern_detection_sensitivity": {
        "min": 0.1,
        "max": 0.9,
        "default": 0.5
    },
    "pattern_stability_threshold": {
        "min": 0.3,
        "max": 0.9,
        "default": 0.6
    },
    
    # Polling system
    "max_parallel_requests": {
        "min": 1,
        "max": 10,
        "default": 3
    },
    "registers_per_second_limit": {
        "min": 20,
        "max": 1000,
        "default": 100
    },
    
    # Memory management
    "memory_cache_size_mb": {
        "min": 10,
        "max": 200,
        "default": 50
    },
    
    # Self-healing
    "self_healing_aggressiveness": {
        "min": 0.1,
        "max": 1.0,
        "default": 0.5
    }
}


class ContextType(Enum):
    """Types of context that can affect parameter tuning."""
    
    SYSTEM_LOAD = "system_load"
    TIME_OF_DAY = "time_of_day"
    DAY_OF_WEEK = "day_of_week"
    CURRENT_PATTERN = "current_pattern"
    ERROR_RATE = "error_rate"
    DEVICE_TYPE = "device_type"
    NETWORK_QUALITY = "network_quality"
    ENTITY_COUNT = "entity_count"
    FEATURE_USAGE = "feature_usage"


class ParameterProfile:
    """Represents a set of parameters optimized for a specific context."""
    
    def __init__(self, name: str, description: str = ""):
        """Initialize a parameter profile.
        
        Args:
            name: Profile name
            description: Profile description
        """
        self.name = name
        self.description = description
        self.parameters: Dict[str, Any] = {}
        self.context_match: Dict[ContextType, Dict[str, float]] = {}
        self.created = datetime.utcnow().isoformat()
        self.last_updated = self.created
        self.usage_count = 0
        self.effectiveness_score = 0.5  # Default to neutral score
        self.source = "system"  # Where this profile came from (system, learned, user)
    
    def update_parameter(self, name: str, value: Any) -> None:
        """Update a parameter value.
        
        Args:
            name: Parameter name
            value: Parameter value
        """
        self.parameters[name] = value
        self.last_updated = datetime.utcnow().isoformat()
    
    def update_context_match(self, 
                           context_type: ContextType, 
                           match_criteria: Dict[str, float]) -> None:
        """Update context matching criteria.
        
        Args:
            context_type: Type of context
            match_criteria: Dictionary of match values and their weights
        """
        self.context_match[context_type] = match_criteria
        self.last_updated = datetime.utcnow().isoformat()
    
    def get_context_match_score(self, 
                              current_context: Dict[ContextType, Any]) -> float:
        """Calculate how well this profile matches the given context.
        
        Args:
            current_context: Current system context
            
        Returns:
            Match score from 0.0 to 1.0, higher is better match
        """
        if not self.context_match:
            return 0.0  # No context matching criteria defined
            
        total_score = 0.0
        total_weight = 0.0
        
        # Check each context type we have matching criteria for
        for context_type, match_criteria in self.context_match.items():
            if context_type not in current_context:
                continue  # Skip if this context type isn't available
                
            current_value = current_context[context_type]
            
            # Different handling based on context type
            if context_type == ContextType.TIME_OF_DAY:
                # Time of day is a special case - circular value
                time_score = self._calculate_time_match(current_value, match_criteria)
                weight = match_criteria.get("weight", 1.0)
                total_score += time_score * weight
                total_weight += weight
                
            elif context_type == ContextType.DAY_OF_WEEK:
                # Day of week is a simple match
                day_score = 0.0
                if str(current_value) in match_criteria:
                    day_score = match_criteria[str(current_value)]
                weight = match_criteria.get("weight", 1.0)
                total_score += day_score * weight
                total_weight += weight
                
            elif context_type == ContextType.CURRENT_PATTERN:
                # Pattern is a simple match
                pattern_score = 0.0
                if str(current_value) in match_criteria:
                    pattern_score = match_criteria[str(current_value)]
                weight = match_criteria.get("weight", 1.0)
                total_score += pattern_score * weight
                total_weight += weight
                
            elif context_type == ContextType.SYSTEM_LOAD:
                # System load is a range match
                load_score = self._calculate_range_match(
                    current_value,
                    match_criteria.get("min", 0.0),
                    match_criteria.get("max", 1.0),
                    match_criteria.get("optimal", 0.5)
                )
                weight = match_criteria.get("weight", 1.0)
                total_score += load_score * weight
                total_weight += weight
                
            elif context_type == ContextType.ERROR_RATE:
                # Error rate is a range match
                error_score = self._calculate_range_match(
                    current_value,
                    match_criteria.get("min", 0.0),
                    match_criteria.get("max", 0.5),
                    match_criteria.get("optimal", 0.0)
                )
                weight = match_criteria.get("weight", 1.0)
                total_score += error_score * weight
                total_weight += weight
                
            elif context_type == ContextType.ENTITY_COUNT:
                # Entity count is a range match
                count_score = self._calculate_range_match(
                    current_value,
                    match_criteria.get("min", 0),
                    match_criteria.get("max", 1000),
                    match_criteria.get("optimal", 100)
                )
                weight = match_criteria.get("weight", 1.0)
                total_score += count_score * weight
                total_weight += weight
                
            elif context_type == ContextType.NETWORK_QUALITY:
                # Network quality is a simple match or range
                if isinstance(current_value, str) and current_value in match_criteria:
                    quality_score = match_criteria[current_value]
                else:
                    # Try as numeric range
                    quality_score = self._calculate_range_match(
                        current_value,
                        match_criteria.get("min", 0.0),
                        match_criteria.get("max", 1.0),
                        match_criteria.get("optimal", 1.0)
                    )
                weight = match_criteria.get("weight", 1.0)
                total_score += quality_score * weight
                total_weight += weight
                
            elif context_type == ContextType.DEVICE_TYPE:
                # Device type is a simple match
                device_score = 0.0
                if str(current_value) in match_criteria:
                    device_score = match_criteria[str(current_value)]
                weight = match_criteria.get("weight", 1.0)
                total_score += device_score * weight
                total_weight += weight
                
        # Calculate final score - normalized to 0.0-1.0 range
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0
    
    def _calculate_time_match(self, 
                            current_time: datetime, 
                            time_criteria: Dict[str, Any]) -> float:
        """Calculate how well current time matches the time criteria.
        
        Args:
            current_time: Current time
            time_criteria: Time matching criteria
            
        Returns:
            Match score from 0.0 to 1.0
        """
        # Get current hour (0-23)
        if isinstance(current_time, datetime):
            current_hour = current_time.hour
        else:
            # Assume it's already an hour number
            current_hour = current_time
        
        # Get optimal hours
        optimal_hours = time_criteria.get("optimal_hours", [])
        if not optimal_hours:
            return 0.5  # Neutral if no optimal hours specified
            
        # Find closest optimal hour
        if current_hour in optimal_hours:
            return 1.0
            
        # Calculate distance to nearest optimal hour, accounting for circular nature
        min_distance = 24
        for hour in optimal_hours:
            # Regular distance
            distance = abs(current_hour - hour)
            # Circular distance (e.g., distance between 23 and 0 is 1, not 23)
            circular_distance = min(distance, 24 - distance)
            min_distance = min(min_distance, circular_distance)
            
        # Convert distance to score (0 distance = 1.0 score, 12 distance = 0.0 score)
        if min_distance >= 12:
            return 0.0
        else:
            return 1.0 - (min_distance / 12.0)
    
    def _calculate_range_match(self, 
                             current_value: float, 
                             min_value: float,
                             max_value: float, 
                             optimal_value: float) -> float:
        """Calculate how well current value matches the range criteria.
        
        Args:
            current_value: Current value
            min_value: Minimum acceptable value
            max_value: Maximum acceptable value
            optimal_value: Optimal value
            
        Returns:
            Match score from 0.0 to 1.0
        """
        # Ensure current_value is a number
        try:
            current_value = float(current_value)
        except (ValueError, TypeError):
            return 0.0
            
        # Outside acceptable range
        if current_value < min_value or current_value > max_value:
            return 0.0
            
        # At optimal value
        if current_value == optimal_value:
            return 1.0
            
        # Calculate distance from optimal as percentage of range
        if current_value < optimal_value:
            # Below optimal
            range_size = optimal_value - min_value
            if range_size == 0:
                return 0.0
            distance = optimal_value - current_value
            return 1.0 - (distance / range_size)
        else:
            # Above optimal
            range_size = max_value - optimal_value
            if range_size == 0:
                return 0.0
            distance = current_value - optimal_value
            return 1.0 - (distance / range_size)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for storage.
        
        Returns:
            Dictionary representation of profile
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "context_match": {
                context_type.value: criteria 
                for context_type, criteria in self.context_match.items()
            },
            "created": self.created,
            "last_updated": self.last_updated,
            "usage_count": self.usage_count,
            "effectiveness_score": self.effectiveness_score,
            "source": self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterProfile':
        """Create profile from dictionary.
        
        Args:
            data: Dictionary representation of profile
            
        Returns:
            ParameterProfile instance
        """
        profile = cls(data.get("name", "unnamed"), data.get("description", ""))
        profile.parameters = data.get("parameters", {})
        
        # Convert string context types back to enum
        for context_type_str, criteria in data.get("context_match", {}).items():
            try:
                context_type = ContextType(context_type_str)
                profile.context_match[context_type] = criteria
            except ValueError:
                # Unknown context type - skip
                pass
                
        profile.created = data.get("created", datetime.utcnow().isoformat())
        profile.last_updated = data.get("last_updated", profile.created)
        profile.usage_count = data.get("usage_count", 0)
        profile.effectiveness_score = data.get("effectiveness_score", 0.5)
        profile.source = data.get("source", "system")
        
        return profile


class ContextProvider:
    """Provides current context for parameter adaptation."""
    
    def __init__(self):
        """Initialize context provider."""
        self.static_context: Dict[ContextType, Any] = {}
        self.context_callbacks: Dict[ContextType, Callable[[], Any]] = {}
        self.context_cache: Dict[ContextType, Tuple[Any, datetime]] = {}
        self.cache_ttl = 60  # Cache TTL in seconds
    
    def set_static_context(self, context_type: ContextType, value: Any) -> None:
        """Set a static context value that doesn't change frequently.
        
        Args:
            context_type: Type of context
            value: Context value
        """
        self.static_context[context_type] = value
        
        # Also update cache
        self.context_cache[context_type] = (value, datetime.utcnow())
    
    def register_context_provider(self, 
                                context_type: ContextType, 
                                callback: Callable[[], Any]) -> None:
        """Register a callback function to provide dynamic context.
        
        Args:
            context_type: Type of context
            callback: Function that returns the current value
        """
        self.context_callbacks[context_type] = callback
    
    def get_context(self, 
                  context_type: ContextType, 
                  ignore_cache: bool = False) -> Any:
        """Get current value for a specific context type.
        
        Args:
            context_type: Type of context
            ignore_cache: If True, ignore cached values
            
        Returns:
            Current context value
        """
        # Check static context first
        if context_type in self.static_context:
            return self.static_context[context_type]
            
        # Check cache if not ignoring
        if not ignore_cache and context_type in self.context_cache:
            value, timestamp = self.context_cache[context_type]
            age = (datetime.utcnow() - timestamp).total_seconds()
            if age < self.cache_ttl:
                return value
        
        # Get from callback
        if context_type in self.context_callbacks:
            try:
                value = self.context_callbacks[context_type]()
                # Update cache
                self.context_cache[context_type] = (value, datetime.utcnow())
                return value
            except Exception as e:
                _LOGGER.error("Error getting context %s: %s", context_type, e)
                
                # Return cached value if available, regardless of age
                if context_type in self.context_cache:
                    return self.context_cache[context_type][0]
                    
        # No value available
        return None
    
    def get_all_context(self, ignore_cache: bool = False) -> Dict[ContextType, Any]:
        """Get all available context values.
        
        Args:
            ignore_cache: If True, ignore cached values
            
        Returns:
            Dictionary of context types to values
        """
        result = {}
        
        # Add static context
        result.update(self.static_context)
        
        # Add dynamic context
        for context_type in self.context_callbacks:
            value = self.get_context(context_type, ignore_cache)
            if value is not None:
                result[context_type] = value
                
        return result


class AdaptiveParameterManager:
    """Manages adaptive parameters that adjust to system conditions."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = AdaptiveParameterManager()
        return cls._instance
    
    def __init__(self):
        """Initialize parameter manager."""
        # We'll defer actual storage operations until we know the storage system is ready
        self._storage_ready = False
        self._initialization_attempted = False
        
        # Initialize context provider
        self.context_provider = ContextProvider()
        
        # Parameter profiles
        self.profiles: Dict[str, ParameterProfile] = {}
        
        # Currently active profile
        self.active_profile_name: Optional[str] = None
        
        # Parameter constraints and defaults
        self.parameter_constraints = copy.deepcopy(DEFAULT_PARAMETER_CONSTRAINTS)
        
        # Current parameter values
        self.current_parameters: Dict[str, Any] = {}
        
        # Parameter history for learning
        self.parameter_history: List[Dict[str, Any]] = []
        self.max_history_size = 100
        
        # System status
        self.last_adaptation: Optional[datetime] = None
        self.adaptation_count = 0
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # We'll initialize defaults right away, but defer loading from storage
        self._setup_default_profiles()
        self._initialize_current_parameters()
        self._register_standard_context_providers()
        
        _LOGGER.info("Adaptive parameter manager initialized with default profiles")
    
    def ensure_storage_ready(self):
        """Ensure storage is initialized and ready to use.
        
        Returns:
            True if storage is ready, False otherwise
        """
        if self._storage_ready:
            return True
            
        if not self._initialization_attempted:
            self._initialization_attempted = True
            try:
                # Check if the storage manager is initialized and has storage
                if (PERSISTENT_STATISTICS_MANAGER.enabled and 
                    hasattr(PERSISTENT_STATISTICS_MANAGER, 'storage') and 
                    PERSISTENT_STATISTICS_MANAGER.storage is not None and
                    hasattr(PERSISTENT_STATISTICS_MANAGER.storage, 'get_metadata_file')):
                    
                    self._storage_ready = True
                    # Now that storage is ready, load data
                    self._load_data()
                    _LOGGER.info("Successfully connected to persistent storage")
                    return True
                else:
                    _LOGGER.warning("Storage not yet initialized - using defaults")
                    return False
            except Exception as e:
                _LOGGER.error("Error checking storage readiness: %s", e)
                return False
        return False
    
    def initialize_storage(self, base_path: Path = None) -> bool:
        """Explicitly initialize storage if not already initialized.
        
        Args:
            base_path: Optional base path to use for storage
        
        Returns:
            True if initialization was successful or storage was already initialized
        """
        if self._storage_ready:
            return True
            
        try:
            # Check if PERSISTENT_STATISTICS_MANAGER is already initialized
            if PERSISTENT_STATISTICS_MANAGER.enabled:
                self._storage_ready = True
                self._load_data()
                _LOGGER.info("Connected to already initialized storage")
                return True
                
            # If base_path is provided, try to initialize
            if base_path and not PERSISTENT_STATISTICS_MANAGER.enabled:
                PERSISTENT_STATISTICS_MANAGER.initialize(base_path)
                self._storage_ready = True
                self._load_data()
                _LOGGER.info("Initialized storage with base path: %s", base_path)
                return True
                
            return False
        except Exception as e:
            _LOGGER.error("Failed to initialize storage: %s", e)
            return False
    
    def _load_data(self) -> None:
        """Load parameter data from storage."""
        try:
            if not self.ensure_storage_ready():
                return
                
            # Get metadata file with profiles and parameters
            data = PERSISTENT_STATISTICS_MANAGER.storage.get_metadata_file("adaptive_parameters")
            
            if not data:
                _LOGGER.debug("No adaptive parameter data found in storage")
                return
                
            # Load profiles
            for profile_data in data.get("profiles", []):
                try:
                    profile = ParameterProfile.from_dict(profile_data)
                    self.profiles[profile.name] = profile
                    _LOGGER.debug("Loaded profile: %s", profile.name)
                except Exception as e:
                    _LOGGER.error("Error loading profile %s: %s", 
                                profile_data.get("name", "unknown"), e)
            
            # Load active profile
            self.active_profile_name = data.get("active_profile")
            
            # Load parameter constraints
            constraints = data.get("parameter_constraints")
            if constraints:
                self.parameter_constraints.update(constraints)
                
            # Load parameter history
            self.parameter_history = data.get("parameter_history", [])
            
            # Load system status
            last_adaptation = data.get("last_adaptation")
            if last_adaptation:
                try:
                    self.last_adaptation = datetime.fromisoformat(last_adaptation)
                except ValueError:
                    self.last_adaptation = None
                    
            self.adaptation_count = data.get("adaptation_count", 0)
            
            _LOGGER.info("Loaded %d profiles from storage", len(self.profiles))
            
        except Exception as e:
            _LOGGER.error("Error loading parameter data: %s", e)
    
    def _save_data(self) -> None:
        """Save parameter data to storage."""
        try:
            if not self.ensure_storage_ready():
                _LOGGER.warning("Cannot save parameters - storage not ready")
                return
                
            data = {
                "profiles": [profile.to_dict() for profile in self.profiles.values()],
                "active_profile": self.active_profile_name,
                "parameter_constraints": self.parameter_constraints,
                "parameter_history": self.parameter_history[-self.max_history_size:],
                "last_adaptation": self.last_adaptation.isoformat() if self.last_adaptation else None,
                "adaptation_count": self.adaptation_count,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            PERSISTENT_STATISTICS_MANAGER.storage.save_metadata_file("adaptive_parameters", data)
            _LOGGER.debug("Saved parameter data to storage")
        except Exception as e:
            _LOGGER.error("Error saving parameter data: %s", e)
    
    def _setup_default_profiles(self) -> None:
        """Set up default parameter profiles."""
        # Normal operation profile
        normal = ParameterProfile("normal", "Balanced parameters for normal operation")
        normal.parameters = {
            "min_scan_interval": 10,
            "max_scan_interval": 900,
            "default_scan_interval": 30,
            "max_batch_size": 20,
            "correlation_threshold": 0.6,
            "pattern_detection_sensitivity": 0.5,
            "max_parallel_requests": 3,
            "registers_per_second_limit": 100,
            "memory_cache_size_mb": 50,
            "self_healing_aggressiveness": 0.5
        }
        normal.update_context_match(ContextType.SYSTEM_LOAD, {
            "min": 0.0,
            "max": 0.7,
            "optimal": 0.3,
            "weight": 1.0
        })
        normal.update_context_match(ContextType.ERROR_RATE, {
            "min": 0.0,
            "max": 0.1,
            "optimal": 0.0,
            "weight": 1.0
        })
        normal.source = "system"
        self.profiles["normal"] = normal
        
        # High load profile
        high_load = ParameterProfile("high_load", "Conservative parameters for high system load")
        high_load.parameters = {
            "min_scan_interval": 20,
            "max_scan_interval": 1800,
            "default_scan_interval": 60,
            "max_batch_size": 10,
            "correlation_threshold": 0.7,
            "pattern_detection_sensitivity": 0.3,
            "max_parallel_requests": 2,
            "registers_per_second_limit": 50,
            "memory_cache_size_mb": 30,
            "self_healing_aggressiveness": 0.3
        }
        high_load.update_context_match(ContextType.SYSTEM_LOAD, {
            "min": 0.6,
            "max": 1.0,
            "optimal": 0.7,
            "weight": 1.0
        })
        high_load.update_context_match(ContextType.ERROR_RATE, {
            "min": 0.0,
            "max": 0.2,
            "optimal": 0.0,
            "weight": 0.5
        })
        high_load.source = "system"
        self.profiles["high_load"] = high_load
        
        # Error recovery profile
        error_recovery = ParameterProfile("error_recovery", "Recovery mode for high error rates")
        error_recovery.parameters = {
            "min_scan_interval": 15,
            "max_scan_interval": 600,
            "default_scan_interval": 45,
            "max_batch_size": 5,
            "correlation_threshold": 0.8,
            "pattern_detection_sensitivity": 0.3,
            "max_parallel_requests": 1,
            "registers_per_second_limit": 30,
            "memory_cache_size_mb": 40,
            "self_healing_aggressiveness": 0.8
        }
        error_recovery.update_context_match(ContextType.ERROR_RATE, {
            "min": 0.1,
            "max": 1.0,
            "optimal": 0.15,
            "weight": 1.0
        })
        error_recovery.source = "system"
        self.profiles["error_recovery"] = error_recovery
        
        # Night mode profile
        night_mode = ParameterProfile("night_mode", "Optimized for nighttime operation")
        night_mode.parameters = {
            "min_scan_interval": 15,
            "max_scan_interval": 1800,
            "default_scan_interval": 60,
            "max_batch_size": 30,
            "correlation_threshold": 0.5,
            "pattern_detection_sensitivity": 0.7,
            "max_parallel_requests": 4,
            "registers_per_second_limit": 200,
            "memory_cache_size_mb": 80,
            "self_healing_aggressiveness": 0.7
        }
        night_mode.update_context_match(ContextType.TIME_OF_DAY, {
            "optimal_hours": [22, 23, 0, 1, 2, 3, 4, 5],
            "weight": 1.0
        })
        night_mode.update_context_match(ContextType.SYSTEM_LOAD, {
            "min": 0.0,
            "max": 0.5,
            "optimal": 0.1,
            "weight": 0.5
        })
        night_mode.source = "system"
        self.profiles["night_mode"] = night_mode
        
        # Set normal as the active profile
        self.active_profile_name = "normal"
    
    def _initialize_current_parameters(self) -> None:
        """Initialize current parameters from defaults or active profile."""
        with self.lock:
            # Start with constraints defaults
            for param, constraints in self.parameter_constraints.items():
                self.current_parameters[param] = constraints.get("default")
                
            # Apply active profile if available
            if self.active_profile_name and self.active_profile_name in self.profiles:
                profile = self.profiles[self.active_profile_name]
                for param, value in profile.parameters.items():
                    if param in self.parameter_constraints:
                        # Apply constraints
                        min_val = self.parameter_constraints[param].get("min")
                        max_val = self.parameter_constraints[param].get("max")
                        
                        if min_val is not None and value < min_val:
                            value = min_val
                        if max_val is not None and value > max_val:
                            value = max_val
                            
                    self.current_parameters[param] = value
    
    def _register_standard_context_providers(self) -> None:
        """Register standard context providers."""
        # Time of day
        self.context_provider.register_context_provider(
            ContextType.TIME_OF_DAY,
            lambda: datetime.utcnow().hour
        )
        
        # Day of week (0=Monday, 6=Sunday)
        self.context_provider.register_context_provider(
            ContextType.DAY_OF_WEEK,
            lambda: datetime.utcnow().weekday()
        )
    
    def register_context_provider(self, 
                               context_type: ContextType, 
                               provider_callback: Callable[[], Any]) -> None:
        """Register a callback to provide context data.
        
        Args:
            context_type: Type of context
            provider_callback: Function that returns the current context value
        """
        self.context_provider.register_context_provider(context_type, provider_callback)
    
    def set_static_context(self, context_type: ContextType, value: Any) -> None:
        """Set a static context value.
        
        Args:
            context_type: Type of context
            value: Context value
        """
        self.context_provider.set_static_context(context_type, value)
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get the current value of a parameter.
        
        Args:
            name: Parameter name
            default: Default value if parameter not found
            
        Returns:
            Parameter value
        """
        # Try to ensure storage is ready (non-blocking)
        self.ensure_storage_ready()
        
        with self.lock:
            return self.current_parameters.get(name, default)
    
    def set_parameter(self, name: str, value: Any) -> bool:
        """Set a parameter value.
        
        Args:
            name: Parameter name
            value: Parameter value
            
        Returns:
            True if parameter was set successfully
        """
        with self.lock:
            # Check constraints
            if name in self.parameter_constraints:
                min_val = self.parameter_constraints[name].get("min")
                max_val = self.parameter_constraints[name].get("max")
                
                if min_val is not None and value < min_val:
                    value = min_val
                if max_val is not None and value > max_val:
                    value = max_val
            
            # Set the parameter
            self.current_parameters[name] = value
            
            # Add to history
            self._add_to_parameter_history(name, value)
            
            # Save if storage is ready
            if self._storage_ready:
                self._save_data()
                
            return True
    
    def set_parameter_constraints(self, name: str, constraints: Dict[str, Any]) -> bool:
        """Set constraints for a parameter.
        
        Args:
            name: Parameter name
            constraints: Dictionary with min, max, default values
            
        Returns:
            True if constraints were set successfully
        """
        with self.lock:
            if name in self.parameter_constraints:
                self.parameter_constraints[name].update(constraints)
            else:
                self.parameter_constraints[name] = constraints
                
            # Ensure current parameter respects new constraints
            if name in self.current_parameters:
                value = self.current_parameters[name]
                min_val = self.parameter_constraints[name].get("min")
                max_val = self.parameter_constraints[name].get("max")
                
                if min_val is not None and value < min_val:
                    self.current_parameters[name] = min_val
                if max_val is not None and value > max_val:
                    self.current_parameters[name] = max_val
                    
            # Save changes if storage is ready
            if self._storage_ready:
                self._save_data()
            
            return True
    
    def create_profile(self, name: str, description: str = "", source: str = "user") -> ParameterProfile:
        """Create a new parameter profile.
        
        Args:
            name: Profile name
            description: Profile description
            source: Profile source (user, system, learned)
            
        Returns:
            Created profile
        """
        with self.lock:
            if name in self.profiles:
                # Update existing profile
                profile = self.profiles[name]
                profile.description = description
                profile.source = source
            else:
                # Create new profile
                profile = ParameterProfile(name, description)
                profile.source = source
                self.profiles[name] = profile
                
                # Initialize with current parameters
                profile.parameters = copy.deepcopy(self.current_parameters)
                
            # Save changes if storage is ready
            if self._storage_ready:
                self._save_data()
            
            return profile
    
    def delete_profile(self, name: str) -> bool:
        """Delete a parameter profile.
        
        Args:
            name: Profile name
            
        Returns:
            True if profile was deleted
        """
        with self.lock:
            if name not in self.profiles:
                return False
                
            # Don't delete active profile
            if name == self.active_profile_name:
                return False
                
            # Delete profile
            del self.profiles[name]
            
            # Save changes if storage is ready
            if self._storage_ready:
                self._save_data()
            
            return True
    
    def activate_profile(self, name: str) -> bool:
        """Activate a specific profile.
        
        Args:
            name: Profile name
            
        Returns:
            True if profile was activated
        """
        with self.lock:
            if name not in self.profiles:
                return False
                
            # Set as active profile
            self.active_profile_name = name
            profile = self.profiles[name]
            
            # Apply profile parameters
            for param, value in profile.parameters.items():
                if param in self.parameter_constraints:
                    # Apply constraints
                    min_val = self.parameter_constraints[param].get("min")
                    max_val = self.parameter_constraints[param].get("max")
                    
                    if min_val is not None and value < min_val:
                        value = min_val
                    if max_val is not None and value > max_val:
                        value = max_val
                        
                self.current_parameters[param] = value
                
            # Update usage stats
            profile.usage_count += 1
            
            # Record adaptation
            self.last_adaptation = datetime.utcnow()
            self.adaptation_count += 1
            
            # Save changes if storage is ready
            if self._storage_ready:
                self._save_data()
            
            _LOGGER.info("Activated parameter profile: %s", name)
            
            return True
    
    def get_best_profile_for_context(self) -> Tuple[Optional[str], float]:
        """Find the best profile for current context.
        
        Returns:
            Tuple of (profile_name, match_score)
        """
        with self.lock:
            # Get current context
            current_context = self.context_provider.get_all_context()
            
            if not current_context:
                # No context available
                return self.active_profile_name, 0.0
                
            best_profile = None
            best_score = -1.0
            
            for name, profile in self.profiles.items():
                score = profile.get_context_match_score(current_context)
                if score > best_score:
                    best_score = score
                    best_profile = name
                    
            return best_profile, best_score
    
    def adapt_parameters(self, force: bool = False) -> Dict[str, Any]:
        """Adapt parameters to current context.
        
        Args:
            force: Force adaptation even if not needed
            
        Returns:
            Dictionary with adaptation results
        """
        # Try to ensure storage is ready (non-blocking)
        self.ensure_storage_ready()
        
        with self.lock:
            start_time = time.time()
            
            # Check if adaptation is needed
            adaptation_needed = force
            
            if not adaptation_needed and self.last_adaptation:
                # Check if enough time has passed since last adaptation
                time_since_last = datetime.utcnow() - self.last_adaptation
                if time_since_last.total_seconds() > 3600:  # 1 hour
                    adaptation_needed = True
            
            result = {
                "timestamp": datetime.utcnow().isoformat(),
                "adaptation_needed": adaptation_needed,
                "profiles_evaluated": len(self.profiles),
                "current_profile": self.active_profile_name,
            }
            
            if not adaptation_needed:
                result["message"] = "No adaptation needed"
                return result
                
            # Find best profile for current context
            best_profile, match_score = self.get_best_profile_for_context()
            
            result["best_profile"] = best_profile
            result["match_score"] = match_score
            
            # If best profile is different from current, activate it
            if best_profile and best_profile != self.active_profile_name and match_score > 0.6:
                self.activate_profile(best_profile)
                result["adapted"] = True
                result["previous_profile"] = self.active_profile_name
                result["new_profile"] = best_profile
            else:
                result["adapted"] = False
                result["message"] = "No better profile found"
                
            # Record adaptation attempt regardless
            self.last_adaptation = datetime.utcnow()
            self.adaptation_count += 1
            
            result["duration_seconds"] = round(time.time() - start_time, 3)
            
            # Save changes if storage is ready
            if self._storage_ready:
                self._save_data()
            
            return result
    
    def learn_from_performance(self, 
                             performance_metrics: Dict[str, float], 
                             time_period: Dict[str, datetime]) -> Dict[str, Any]:
        """Learn from system performance and adjust parameters.
        
        Args:
            performance_metrics: Dictionary of performance metrics
            time_period: Dictionary with start and end times
            
        Returns:
            Dictionary with learning results
        """
        # Try to ensure storage is ready (non-blocking)
        self.ensure_storage_ready()
        
        with self.lock:
            start_time = time.time()
            
            # Evaluate current profile effectiveness
            effectiveness = self._evaluate_metrics(performance_metrics)
            
            result = {
                "timestamp": datetime.utcnow().isoformat(),
                "time_period": {
                    "start": time_period.get("start").isoformat() if time_period.get("start") else None,
                    "end": time_period.get("end").isoformat() if time_period.get("end") else None,
                },
                "current_profile": self.active_profile_name,
                "effectiveness_score": effectiveness
            }
            
            # Update profile effectiveness score
            if self.active_profile_name in self.profiles:
                profile = self.profiles[self.active_profile_name]
                
                # Use exponential moving average to update score
                old_score = profile.effectiveness_score
                profile.effectiveness_score = (old_score * 0.7) + (effectiveness * 0.3)
                
                result["previous_score"] = old_score
                result["new_score"] = profile.effectiveness_score
                
                # If effectiveness is poor, try to learn improvements
                if effectiveness < 0.4:
                    improvements = self._learn_improvements(performance_metrics)
                    
                    if improvements:
                        # Create a learned profile if significant improvements found
                        learned_profile = self._create_learned_profile(improvements)
                        result["learned_profile"] = learned_profile
                        result["improvements"] = improvements
                    else:
                        result["message"] = "No significant improvements found"
                else:
                    result["message"] = "Current profile is performing well"
            
            result["duration_seconds"] = round(time.time() - start_time, 3)
            
            # Save changes if storage is ready
            if self._storage_ready:
                self._save_data()
            
            return result
    
    def _evaluate_metrics(self, metrics: Dict[str, float]) -> float:
        """Evaluate performance metrics to determine effectiveness.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Effectiveness score (0.0 to 1.0)
        """
        # Example metrics evaluation - actual implementation would use more sophisticated logic
        score_components = []
        
        if "error_rate" in metrics:
            # Lower error rate is better
            error_score = 1.0 - min(metrics["error_rate"], 1.0)
            score_components.append(error_score * 2.0)  # Weight errors more heavily
            
        if "response_time_ms" in metrics:
            # Response time - normalize to 0-1 range
            # Assume 0ms is perfect, 500ms+ is poor
            response_score = max(0, 1.0 - (metrics["response_time_ms"] / 500.0))
            score_components.append(response_score)
            
        if "successful_polls" in metrics and "total_polls" in metrics:
            # Success rate
            if metrics["total_polls"] > 0:
                success_rate = metrics["successful_polls"] / metrics["total_polls"]
                score_components.append(success_rate)
                
        if "memory_usage" in metrics:
            # Memory usage - normalize to 0-1 range
            # Assume 0% is perfect, 90%+ is poor
            memory_score = max(0, 1.0 - (metrics["memory_usage"] / 90.0))
            score_components.append(memory_score)
            
        if "cpu_usage" in metrics:
            # CPU usage - normalize to 0-1 range
            # Assume 0% is perfect, 90%+ is poor
            cpu_score = max(0, 1.0 - (metrics["cpu_usage"] / 90.0))
            score_components.append(cpu_score)
            
        if "polling_efficiency" in metrics:
            # Polling efficiency - already 0-100 range
            efficiency_score = metrics["polling_efficiency"] / 100.0
            score_components.append(efficiency_score)
            
        # Calculate overall score
        if score_components:
            return sum(score_components) / len(score_components)
        else:
            return 0.5  # Neutral score if no metrics available
    
    def _learn_improvements(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Learn parameter improvements based on performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Dictionary of suggested parameter improvements
        """
        improvements = {}
        
        # Error rate is high - adjust for reliability
        if metrics.get("error_rate", 0) > 0.1:
            improvements["max_batch_size"] = max(5, self.get_parameter("max_batch_size", 20) * 0.8)
            improvements["max_parallel_requests"] = max(1, self.get_parameter("max_parallel_requests", 3) - 1)
            improvements["min_scan_interval"] = self.get_parameter("min_scan_interval", 10) * 1.2
            improvements["self_healing_aggressiveness"] = min(1.0, self.get_parameter("self_healing_aggressiveness", 0.5) * 1.3)
            
        # Response time is high - adjust for speed
        if metrics.get("response_time_ms", 0) > 300:
            improvements["max_batch_size"] = max(5, self.get_parameter("max_batch_size", 20) * 0.9)
            improvements["registers_per_second_limit"] = max(20, self.get_parameter("registers_per_second_limit", 100) * 0.9)
            
        # Memory usage is high - adjust for efficiency
        if metrics.get("memory_usage", 0) > 70:
            improvements["memory_cache_size_mb"] = max(10, self.get_parameter("memory_cache_size_mb", 50) * 0.8)
            
        # CPU usage is high - adjust for efficiency
        if metrics.get("cpu_usage", 0) > 70:
            improvements["max_parallel_requests"] = max(1, self.get_parameter("max_parallel_requests", 3) - 1)
            improvements["registers_per_second_limit"] = max(20, self.get_parameter("registers_per_second_limit", 100) * 0.8)
            
        # Polling efficiency is low - adjust scanning
        if metrics.get("polling_efficiency", 100) < 50:
            improvements["pattern_detection_sensitivity"] = min(0.9, self.get_parameter("pattern_detection_sensitivity", 0.5) * 1.2)
            improvements["correlation_threshold"] = max(0.3, min(0.9, self.get_parameter("correlation_threshold", 0.6) * 0.9))
        
        return improvements
    
    def _create_learned_profile(self, improvements: Dict[str, Any]) -> str:
        """Create a new profile with learned improvements.
        
        Args:
            improvements: Dictionary of parameter improvements
            
        Returns:
            Name of the created profile
        """
        # Create new profile name
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
        profile_name = f"learned_{timestamp}"
        
        # Create profile
        profile = self.create_profile(
            profile_name,
            f"Automatically learned profile ({timestamp})",
            source="learned"
        )
        
        # Start with current parameters
        if self.active_profile_name in self.profiles:
            profile.parameters = copy.deepcopy(self.profiles[self.active_profile_name].parameters)
        else:
            profile.parameters = copy.deepcopy(self.current_parameters)
            
        # Apply improvements
        for param, value in improvements.items():
            profile.parameters[param] = value
            
        # Copy context match from current profile
        if self.active_profile_name in self.profiles:
            profile.context_match = copy.deepcopy(self.profiles[self.active_profile_name].context_match)
            
        # Save changes if storage is ready
        if self._storage_ready:
            self._save_data()
        
        return profile_name
    
    def _add_to_parameter_history(self, name: str, value: Any) -> None:
        """Add parameter change to history.
        
        Args:
            name: Parameter name
            value: New value
        """
        history_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "parameter": name,
            "value": value,
            "profile": self.active_profile_name
        }
        
        # Add to history
        self.parameter_history.append(history_entry)
        
        # Trim history if needed
        if len(self.parameter_history) > self.max_history_size:
            self.parameter_history = self.parameter_history[-self.max_history_size:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of parameter management system.
        
        Returns:
            Status information
        """
        # Try to ensure storage is ready (non-blocking)
        self.ensure_storage_ready()
        
        with self.lock:
            return {
                "active_profile": self.active_profile_name,
                "profile_count": len(self.profiles),
                "storage_ready": self._storage_ready,
                "profiles": {
                    name: {
                        "description": profile.description,
                        "effectiveness": profile.effectiveness_score,
                        "usage_count": profile.usage_count,
                        "source": profile.source
                    }
                    for name, profile in self.profiles.items()
                },
                "last_adaptation": self.last_adaptation.isoformat() if self.last_adaptation else None,
                "adaptation_count": self.adaptation_count,
                "current_context": {
                    context_type.value: value
                    for context_type, value in self.context_provider.get_all_context().items()
                },
                "parameter_count": len(self.current_parameters)
            }
    
    def get_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific profile.
        
        Args:
            name: Profile name
            
        Returns:
            Profile data or None if not found
        """
        # Try to ensure storage is ready (non-blocking)
        self.ensure_storage_ready()
        
        with self.lock:
            if name not in self.profiles:
                return None
                
            return self.profiles[name].to_dict()
    
    def get_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get all profiles.
        
        Returns:
            Dictionary of profile name to profile data
        """
        # Try to ensure storage is ready (non-blocking)
        self.ensure_storage_ready()
        
        with self.lock:
            return {
                name: profile.to_dict()
                for name, profile in self.profiles.items()
            }
    
    def get_all_parameters(self) -> Dict[str, Any]:
        """Get all current parameters.
        
        Returns:
            Dictionary of parameter name to value
        """
        # Try to ensure storage is ready (non-blocking)
        self.ensure_storage_ready()
        
        with self.lock:
            return copy.deepcopy(self.current_parameters)
    
    def get_parameter_history(self) -> List[Dict[str, Any]]:
        """Get parameter change history.
        
        Returns:
            List of parameter changes
        """
        # Try to ensure storage is ready (non-blocking)
        self.ensure_storage_ready()
        
        with self.lock:
            return copy.deepcopy(self.parameter_history)
    
    def reset_to_defaults(self) -> bool:
        """Reset to default profiles and parameters.
        
        Returns:
            True if reset was successful
        """
        with self.lock:
            # Clear existing profiles and parameters
            self.profiles = {}
            self.current_parameters = {}
            
            # Re-initialize with defaults
            self._setup_default_profiles()
            self._initialize_current_parameters()
            
            # Save changes if storage is ready
            if self._storage_ready:
                self._save_data()
                
            _LOGGER.info("Reset parameter manager to default settings")
            return True


# Global singleton instance
PARAMETER_MANAGER = AdaptiveParameterManager.get_instance()