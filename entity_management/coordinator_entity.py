"""Base coordinator entity for Modbus Local Gateway."""

import logging
import time
from typing import Any, Dict, Optional
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.core import callback

from ..context import ModbusContext
from ..coordinator import ModbusCoordinator
from ..statistics import StatisticMetric
from ..const import CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL
from ..statistics import STATISTICS_MANAGER

_LOGGER = logging.getLogger(__name__)

class ModbusCoordinatorEntity(CoordinatorEntity):
    """Base class for Modbus entities with coordinator."""

    def __init__(
        self,
        coordinator: ModbusCoordinator,
        modbus_context: ModbusContext,
        device_info,
    ) -> None:
        """Initialize the entity."""
        super().__init__(coordinator)
        self._attr_device_info = device_info
        self._modbus_context = modbus_context
        
        # Cache for attributes to reduce CPU usage
        self._cached_attributes = None
        self._last_attributes_update = 0
        
        # Calculate dynamic cache duration based on scan interval
        scan_interval = self._get_scan_interval()
        self._attributes_cache_duration = max(3, scan_interval // 3)  # 1/3 of scan interval, min 3 seconds
        
        # Track last value for change detection
        self._last_native_value = None
        
        # Set up unique ID and name
        description = getattr(modbus_context, "desc", None)
        entity_key = getattr(description, "key", None) if description else None
        
        if entity_key:
            # Generate unique ID from gateway and entity key
            gateway_id = getattr(coordinator, "gateway", "unknown")
            self._attr_unique_id = f"{gateway_id}_{entity_key}"
            
            # Use name from description or fallback to key
            self._attr_name = getattr(description, "name", entity_key)

    def _get_scan_interval(self) -> int:
        """Get the scan interval from various sources."""
        # 1. Try entity-specific scan interval from device YAML
        if hasattr(self, '_modbus_context') and self._modbus_context:
            desc = getattr(self._modbus_context, "desc", None)
            if desc:
                entity_scan_interval = getattr(desc, "scan_interval", None)
                if entity_scan_interval is not None:
                    return entity_scan_interval
        
        # 2. Get from config entry (main source from config_flow.py)
        if hasattr(self, "coordinator") and self.coordinator:
            config_entry = getattr(self.coordinator, "config_entry", None)
            if config_entry:
                # Try config entry data first (this is what config_flow.py sets)
                scan_interval = config_entry.data.get(CONF_SCAN_INTERVAL)
                if scan_interval is not None:
                    return scan_interval
                
                # Fallback to options if not in data
                scan_interval = config_entry.options.get(CONF_SCAN_INTERVAL)
                if scan_interval is not None:
                    return scan_interval
        
        # 3. Try coordinator's update_interval as backup
        if hasattr(self, "coordinator") and self.coordinator:
            coordinator_interval = getattr(self.coordinator, "update_interval", None)
            if coordinator_interval is not None:
                if hasattr(coordinator_interval, 'total_seconds'):
                    return int(coordinator_interval.total_seconds())
                elif isinstance(coordinator_interval, (int, float)):
                    return int(coordinator_interval)
        
        # 4. Final fallback to const.py default
        return DEFAULT_SCAN_INTERVAL

    def _generate_extra_attributes(self) -> Dict[str, Any]:
        """Generate extra state attributes (called only when cache expires)."""
        try:
            attrs = {}
            
            # Get modbus context
            modbus_ctx = getattr(self, "_modbus_context", None)
            if not modbus_ctx:
                return {"debug": "no_modbus_context"}
            
            # Get description
            desc = getattr(modbus_ctx, "desc", None) if modbus_ctx else None
            scan_interval = self._get_scan_interval()
            
            # Add core Modbus attributes
            if desc:
                attrs.update({
                    "slave_id": getattr(modbus_ctx, "slave_id", None),
                    "register_address": getattr(desc, "register_address", None),
                    "data_type": getattr(desc, "data_type", None),
                    "scan_interval": scan_interval,  # Now properly detected
                })
            
            # Add comprehensive statistics from enhanced statistics.py
            if modbus_ctx and hasattr(modbus_ctx, "statistics"):
                try:
                    stats = modbus_ctx.statistics.get_statistics()
                    if stats:
                        attrs.update({
                            "stats_poll_count": stats.get("poll_count", 0),
                            "stats_error_count": stats.get("error_count", 0),
                            "stats_success_rate": f"{stats.get('success_rate', 0):.1f}%",
                            "stats_last_success": stats.get("last_success_time"),
                            "stats_last_error": stats.get("last_error_time"),
                            "stats_insufficient_data": stats.get("insufficient_data", True),
                            "stats_value_changes": stats.get("value_changes", 0),
                            "stats_avg_response_time": f"{stats.get('avg_response_time', 0):.2f}ms",
                            "stats_recent_values": len(stats.get("recent_values", [])),
                            "stats_last_value": stats.get("recent_values", [None])[-1] if stats.get("recent_values") else None,
                        })
                except Exception as e:
                    attrs["stats_error"] = str(e)
            
            # Add cache info for debugging
            attrs["cache_duration"] = self._attributes_cache_duration
            
            return attrs
            
        except Exception as e:
            return {"debug_error": str(e)}

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return cached extra state attributes (CPU efficient)."""
        current_time = time.time()
        
        # Use cache if it's still valid
        if (self._cached_attributes is not None and 
            current_time - self._last_attributes_update < self._attributes_cache_duration):
            return self._cached_attributes
        
        # Generate new attributes and cache them
        self._cached_attributes = self._generate_extra_attributes()
        self._last_attributes_update = current_time
        
        return self._cached_attributes
    
    def _get_current_value(self):
        """Get current value - should be overridden by entity types."""
        # Default implementation tries common attributes
        return getattr(self, "native_value", None) or getattr(self, "is_on", None) or getattr(self, "current_option", None)
        
    @callback    
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        try:
            # Get current value using entity-specific method
            current_value = self._get_current_value()
            old_value = self._last_native_value
            
            # Track value changes for statistics
            value_changed = current_value != old_value
            current_time = time.time()
            
            # Record the poll with comprehensive information
            if hasattr(self._modbus_context, "statistics"):
                # Determine if this was an error (value is None usually indicates error)
                poll_error = current_value is None
                
                self._modbus_context.statistics.record_poll(
                    value_changed=value_changed,
                    timestamp=current_time,
                    error=poll_error,
                    response_time=None,  # Could be enhanced later
                    value=current_value
                )
            
            if value_changed:
                entity_id = getattr(self, 'name', None) or getattr(self, 'entity_id', 'unknown')
                _LOGGER.debug("Updating Modbus entity %s to %s", entity_id, current_value)
                self._last_native_value = current_value
                
                # Invalidate cache when value changes (for fresh stats)
                self._cached_attributes = None
            
            # Get cached attributes (CPU efficient)
            attrs = self.extra_state_attributes
            self._attr_extra_state_attributes = attrs
            
            # Force state write
            self.async_write_ha_state()
            
            # Call parent class to update the entity state
            super()._handle_coordinator_update()
            
        except Exception as err:
            _LOGGER.error("Unable to update Modbus entity %s: %s", 
                         getattr(self, 'name', 'unknown'), err)
            
            # Record error in statistics
            if hasattr(self._modbus_context, "statistics"):
                self._modbus_context.statistics.record_poll(
                    value_changed=False,
                    timestamp=time.time(),
                    error=True,
                    response_time=None,
                    value=None
                )

def register_entity_for_advanced_statistics(entity):
    """Register an entity with the statistics manager."""
    if hasattr(entity, "entity_id"):
        entity_id = entity.entity_id
        # Get scan interval from entity
        scan_interval = getattr(entity, "_get_scan_interval", lambda: 30)()
        
        # Register with statistics manager
        STATISTICS_MANAGER.register_entity(entity_id, scan_interval)

        # Process any already collected data
        if hasattr(entity, "_modbus_context") and hasattr(entity._modbus_context, "statistics"):
            stats = entity._modbus_context.statistics
            if hasattr(stats, "recent_values") and stats.recent_values:
                # Add recent values to correlation tracking
                for value in stats.recent_values:
                    STATISTICS_MANAGER.record_value(entity_id, value)
        
        return True
    return False