"""Diagnostics support for Modbus Local Gateway."""
from __future__ import annotations

import logging
import sys
import platform
import functools
from datetime import datetime
from typing import Any, Dict, List, Optional

# Safe top-level imports
from homeassistant.components.diagnostics import async_redact_data
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.const import CONF_HOST, CONF_PORT, __version__ as HA_VERSION

from .const import DOMAIN, CONF_SLAVE_ID

_LOGGER = logging.getLogger(__name__)

# Keys that should be redacted from the diagnostics data
REDACT_KEYS = {
    "username", "password", "api_key", "token", "credentials", "mac",
    "serial_number", "unique_id", "id", "identifier", "uuid"
}


def _safe_get_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get an attribute from an object with a default value."""
    try:
        return getattr(obj, attr, default)
    except Exception as ex:
        _LOGGER.debug("Failed to get attribute '%s': %s", attr, ex)
        return default


def _make_json_serializable(data: Any) -> Any:
    """Convert data to JSON serializable format."""
    if isinstance(data, dict):
        return {k: _make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [_make_json_serializable(item) for item in data]
    elif isinstance(data, datetime):
        return data.isoformat()
    elif hasattr(data, '__dict__'):
        return str(data)
    elif callable(data):
        return str(data)
    else:
        return data


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> Dict[str, Any]:
    """
    Return diagnostics for a config entry.
    
    This function gathers comprehensive diagnostics information about the
    Modbus Local Gateway integration for troubleshooting purposes.
    
    Args:
        hass: The Home Assistant instance.
        entry: The config entry to get diagnostics for.
        
    Returns:
        A dictionary containing diagnostics information.
    """
    # Lazy import helpers to avoid circular imports
    from .helpers import get_gateway_key
    
    diagnostics_data = {}
    
    try:
        gateway_key: str = get_gateway_key(entry=entry, with_slave=True)
        
        # Run the blocking system info call in an executor thread to avoid blocking the event loop
        try:
            system_info = await hass.async_add_executor_job(_get_system_info)
        except Exception as ex:
            _LOGGER.exception("Failed to get system info: %s", ex)
            system_info = {"error": f"Failed to get system info: {ex}"}

        diagnostics_data.update({
            "system_info": system_info,
            "gateway_key": gateway_key,
        })

        # Safely get entry data
        try:
            diagnostics_data["entry"] = async_redact_data(entry.as_dict(), REDACT_KEYS)
        except Exception as ex:
            _LOGGER.exception("Failed to get entry data: %s", ex)
            diagnostics_data["entry"] = {"error": f"Failed to get entry data: {ex}"}

        # Check if coordinator exists
        domain_data = hass.data.get(DOMAIN, {})
        if gateway_key not in domain_data:
            diagnostics_data.update({
                "error": "Coordinator not found. It may be starting up or has failed to set up.",
                "available_keys": list(domain_data.keys()),
            })
            return _make_json_serializable(diagnostics_data)
        
        # Import needed classes only when needed
        from .coordinator import ModbusCoordinator
        from .tcp_client import AsyncModbusTcpClientGateway
        
        coordinator: ModbusCoordinator = domain_data[gateway_key]
        client = _safe_get_attr(coordinator, "client", None)
        
        # Get all diagnostic sections with error handling
        diagnostics_data.update({
            "connection": _get_connection_info(client, entry),
            "adaptive_timeout_stats": _get_adaptive_timeout_info(client),
            "coordinator": _get_coordinator_info(coordinator),
            "throughput_limiter_stats": _get_throughput_limiter_info(coordinator),
            "device_info": await _get_device_info(hass, entry),
            "entity_info": await _get_entity_registry_info(hass, entry),
            "data_info": _get_data_info(coordinator),
            # Enhanced diagnostics
            "entity_analysis": _get_entity_analysis(coordinator),
            "batch_efficiency": _get_batch_efficiency_analysis(coordinator),
            "health_assessment": _get_health_assessment(coordinator),
            "configuration_analysis": _get_configuration_analysis(coordinator, entry),
            "register_mapping": _get_register_mapping(coordinator),
            "recommendations": _get_recommendations(coordinator),
            "troubleshooting": _get_troubleshooting_guide(coordinator),
        })

    except Exception as ex:
        _LOGGER.exception("Critical error in diagnostics collection: %s", ex)
        diagnostics_data["critical_error"] = f"Failed to collect diagnostics: {ex}"
        
    # Try to write diagnostics file
    try:
        import json
        from pathlib import Path
        
        # Write diagnostics to www directory for HTTP access
        www_dir = Path(hass.config.config_dir) / "www"
        www_dir.mkdir(exist_ok=True)
        
        diagnostics_file = www_dir / "modbus_diagnostics.json"
        
        # Convert to JSON serializable format first
        final_data = _make_json_serializable(diagnostics_data)
        
        def _write_diagnostics_file(path, data):
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        
        await hass.async_add_executor_job(_write_diagnostics_file, diagnostics_file, final_data)
        
        _LOGGER.info("Modbus diagnostics written to: %s", diagnostics_file)
        
    except Exception as ex:
        _LOGGER.error("Failed to write modbus diagnostics file: %s", ex)
        
    return _make_json_serializable(diagnostics_data)


def _get_system_info() -> Dict[str, Any]:
    """
    Get system information for diagnostics.
    
    Note: This function performs blocking I/O via `platform.platform()` and
    MUST be called via `hass.async_add_executor_job` to avoid blocking
    the Home Assistant event loop.
    """
    try:
        return {
            "python_version": sys.version.split()[0],
            "python_platform": platform.platform(),
            "home_assistant_version": HA_VERSION,
            "time_utc": datetime.utcnow().isoformat(),
            "timezone": str(datetime.now().astimezone().tzinfo.tzname(None)) if datetime.now().astimezone().tzinfo else "Unknown",
        }
    except Exception as ex:
        _LOGGER.exception("Error getting system info: %s", ex)
        return {"error": f"Failed to get system info: {ex}"}


def _get_connection_info(
    client: Optional[Any], 
    entry: ConfigEntry
) -> Dict[str, Any]:
    """Get connection information with better error handling."""
    try:
        info = {
            "host": entry.data.get(CONF_HOST, "Unknown") if entry else "Unknown",
            "port": entry.data.get(CONF_PORT, "Unknown") if entry else "Unknown", 
            "slave_id": entry.data.get(CONF_SLAVE_ID, "Unknown") if entry else "Unknown",
            "is_connected": False,
            "client_available": client is not None,
        }
        
        if client:
            try:
                info.update({
                    "is_connected": _safe_get_attr(client, "connected", False),
                    "framer": str(_safe_get_attr(client, "framer", "Unknown")),
                })
            except Exception as ex:
                _LOGGER.debug("Error getting client info: %s", ex)
                info["client_error"] = str(ex)
                
        return info
    except Exception as ex:
        _LOGGER.exception("Error getting connection info: %s", ex)
        return {
            "host": "Error",
            "port": "Error", 
            "slave_id": "Error",
            "is_connected": False,
            "client_available": False,
            "error": f"Failed to get connection info: {ex}"
        }


def _get_adaptive_timeout_info(client: Optional[Any]) -> Dict[str, Any]:
    """
    Get adaptive timeout statistics from the client.
    
    Args:
        client: The Modbus client instance.
        
    Returns:
        Dictionary with adaptive timeout metrics.
    """
    try:
        if not client:
            return {"status": "Client not available."}
        
        return {
            "current_dynamic_timeout_s": f"{_safe_get_attr(client, 'dynamic_timeout', 0.0):.3f}",
            "learned_smoothed_rtt_s": f"{_safe_get_attr(client, '_smoothed_rtt', 0.0):.3f}",
            "learned_rtt_variance_s": f"{_safe_get_attr(client, '_rtt_var', 0.0):.3f}",
            "smoothing_factor_alpha": _safe_get_attr(client, '_alpha', 'N/A'),
            "smoothing_factor_beta": _safe_get_attr(client, '_beta', 'N/A'),
        }
    except Exception as ex:
        _LOGGER.exception("Error getting adaptive timeout info: %s", ex)
        return {"error": f"Failed to get adaptive timeout info: {ex}"}


def _get_coordinator_info(coordinator: Any) -> Dict[str, Any]:
    """
    Get coordinator state information.
    
    Args:
        coordinator: The ModbusCoordinator instance.
        
    Returns:
        Dictionary with coordinator state details.
    """
    try:
        last_update_ts = _safe_get_attr(coordinator, "last_update_success_timestamp", None)
        update_interval = _safe_get_attr(coordinator, "update_interval", None)
        entities = _safe_get_attr(coordinator, "entities", [])
        
        return {
            "last_update_success": _safe_get_attr(coordinator, "last_update_success", None),
            "last_update_timestamp": (
                last_update_ts if isinstance(last_update_ts, str)
                else last_update_ts.isoformat() if last_update_ts else None
            ),
            "update_interval_s": update_interval.total_seconds() if update_interval else "N/A",
            "max_read_size": _safe_get_attr(coordinator, "_max_read_size", "N/A"),
            "entity_count": len(entities) if entities else 0,
            "initialized": _safe_get_attr(coordinator, "_initialized", False),
            "gateway": _safe_get_attr(coordinator, "gateway", "Unknown"),
        }
    except Exception as ex:
        _LOGGER.exception("Error getting coordinator info: %s", ex)
        return {"error": f"Failed to get coordinator info: {ex}"}


def _get_throughput_limiter_info(coordinator: Any) -> Dict[str, Any]:
    """
    Get throughput limiter (batch manager) information.
    
    Args:
        coordinator: The ModbusCoordinator instance.
        
    Returns:
        Dictionary with throughput limiter details.
    """
    try:
        batch_managers = _safe_get_attr(coordinator, "_batch_managers", {})
        initialized = _safe_get_attr(coordinator, "_initialized", False)
        
        # Handle uninitialized state
        if not initialized:
            return {
                "status": "Coordinator initializing - batch managers will be created after first data update",
                "initialized": False,
                "batch_managers_count": 0,
                "pending_entities": len(_safe_get_attr(coordinator, "entities", []))
            }
        
        if not batch_managers:
            return {
                "status": "No batch managers found after initialization",
                "initialized": True,
                "batch_managers_count": 0
            }
        
        batch_info = {}
        for key, batch_manager in batch_managers.items():
            if batch_manager:
                try:
                    # Try to get batch manager methods safely
                    current_batch_size = None
                    last_error_timestamp = None
                    is_in_error_state = None
                    
                    if hasattr(batch_manager, 'get_current_batch_size'):
                        try:
                            current_batch_size = batch_manager.get_current_batch_size()
                        except Exception:
                            pass
                    
                    if hasattr(batch_manager, 'get_last_error_timestamp'):
                        try:
                            last_error_ts = batch_manager.get_last_error_timestamp()
                            last_error_timestamp = last_error_ts.isoformat() if last_error_ts else None
                        except Exception:
                            pass
                    
                    if hasattr(batch_manager, 'is_in_error_state'):
                        try:
                            is_in_error_state = batch_manager.is_in_error_state()
                        except Exception:
                            pass
                    
                    batch_info[str(key)] = {
                        "current_batch_size": current_batch_size,
                        "last_error_timestamp": last_error_timestamp,
                        "is_in_error_state": is_in_error_state,
                        "congestion_window": _safe_get_attr(batch_manager, "congestion_window", "N/A"),
                        "consecutive_successes": _safe_get_attr(batch_manager, "consecutive_successes", "N/A"),
                        "batches_count": len(_safe_get_attr(batch_manager, "batches", [])),
                    }
                except Exception as ex:
                    batch_info[str(key)] = {"error": f"Failed to get batch manager info: {ex}"}
        
        return {
            "status": "Active",
            "initialized": True,
            "batch_managers_count": len(batch_managers),
            "batch_managers": batch_info,
        }
    except Exception as ex:
        _LOGGER.exception("Error getting throughput limiter info: %s", ex)
        return {"error": f"Failed to get throughput limiter info: {ex}"}


async def _get_device_info(
    hass: HomeAssistant, 
    entry: ConfigEntry
) -> Dict[str, Any]:
    """Get device information from the registry."""
    try:
        device_registry = dr.async_get(hass)
        devices: List[Dict[str, Any]] = []
        
        for device in dr.async_entries_for_config_entry(device_registry, entry.entry_id):
            try:
                devices.append({
                    "name": device.name,
                    "model": device.model,
                    "manufacturer": device.manufacturer,
                    "sw_version": device.sw_version,
                    "hw_version": device.hw_version,
                    "entry_type": str(device.entry_type) if device.entry_type else None,
                    "id": "redacted",
                })
            except Exception as ex:
                _LOGGER.debug("Error processing device %s: %s", device.id, ex)
                devices.append({"error": f"Failed to process device: {ex}"})
        
        return {"devices": devices, "device_count": len(devices)}
    except Exception as ex:
        _LOGGER.exception("Error getting device info: %s", ex)
        return {"error": f"Failed to get device info: {ex}"}


async def _get_entity_registry_info(
    hass: HomeAssistant, 
    entry: ConfigEntry
) -> Dict[str, Any]:
    """Get entity information from the registry."""
    try:
        entity_registry = er.async_get(hass)
        entities: List[Dict[str, Any]] = []
        
        for entity_entry in er.async_entries_for_config_entry(entity_registry, entry.entry_id):
            try:
                entities.append({
                    "entity_id": entity_entry.entity_id,
                    "name": entity_entry.name,
                    "platform": entity_entry.platform,
                    "domain": entity_entry.domain,
                    "disabled": entity_entry.disabled,
                    "entity_category": str(entity_entry.entity_category) if entity_entry.entity_category else None,
                    "device_id": "redacted" if entity_entry.device_id else None,
                    "has_unique_id": bool(entity_entry.unique_id),
                    "original_name": entity_entry.original_name,
                })
            except Exception as ex:
                _LOGGER.debug("Error processing entity %s: %s", entity_entry.entity_id, ex)
                entities.append({"error": f"Failed to process entity: {ex}"})
        
        return {
            "entity_count": len(entities),
            "entities": entities,
        }
    except Exception as ex:
        _LOGGER.exception("Error getting entity registry info: %s", ex)
        return {"error": f"Failed to get entity registry info: {ex}"}


def _get_data_info(coordinator: Any) -> Dict[str, Any]:
    """
    Get metadata about the data stored in the coordinator.
    
    Args:
        coordinator: The ModbusCoordinator instance.
        
    Returns:
        Dictionary with data metadata.
    """
    try:
        data_info = {}
        type_counts = {}
        
        coordinator_data = _safe_get_attr(coordinator, "data", {})
        
        if coordinator_data:
            for key, value in coordinator_data.items():
                try:
                    value_type = type(value).__name__
                    data_info[str(key)] = {
                        "type": value_type, 
                        "has_value": value is not None,
                        "value_str": str(value)[:100] if value is not None else None  # Truncate long values
                    }
                    type_counts[value_type] = type_counts.get(value_type, 0) + 1
                except Exception as ex:
                    data_info[str(key)] = {"error": f"Failed to process data: {ex}"}
                    
        return {
            "total_keys_with_data": len(data_info),
            "data_type_counts": type_counts,
            "keys": data_info,
        }
    except Exception as ex:
        _LOGGER.exception("Error getting data info: %s", ex)
        return {"error": f"Failed to get data info: {ex}"}


def _get_entity_analysis(coordinator: Any) -> Dict[str, Any]:
    """Analyze entity distribution and characteristics."""
    try:
        entities = _safe_get_attr(coordinator, "entities", [])
        
        if not entities:
            return {"error": "No entities available for analysis"}
        
        # Distribution analysis
        slave_distribution = {}
        scan_interval_distribution = {}
        data_type_distribution = {}
        
        for entity in entities:
            try:
                # Slave ID distribution
                slave_id = _safe_get_attr(entity, "slave_id", "unknown")
                slave_distribution[str(slave_id)] = slave_distribution.get(str(slave_id), 0) + 1
                
                # Scan interval distribution
                scan_interval = _safe_get_attr(entity, "scan_interval", "unknown")
                scan_interval_distribution[str(scan_interval)] = scan_interval_distribution.get(str(scan_interval), 0) + 1
                
                # Data type distribution
                desc = _safe_get_attr(entity, "desc", None)
                if desc:
                    data_type = _safe_get_attr(desc, "data_type", "unknown")
                    data_type_distribution[str(data_type)] = data_type_distribution.get(str(data_type), 0) + 1
            except Exception as ex:
                _LOGGER.debug("Error analyzing entity: %s", ex)
        
        return {
            "total_entities": len(entities),
            "slave_distribution": slave_distribution,
            "scan_interval_distribution": scan_interval_distribution,
            "data_type_distribution": data_type_distribution,
            "unique_slaves": len(slave_distribution),
            "unique_intervals": len(scan_interval_distribution),
            "unique_data_types": len(data_type_distribution),
        }
    except Exception as ex:
        _LOGGER.exception("Error in entity analysis: %s", ex)
        return {"error": f"Failed to analyze entities: {ex}"}


def _get_batch_efficiency_analysis(coordinator: Any) -> Dict[str, Any]:
    """Analyze batch efficiency and optimization opportunities."""
    try:
        entities = _safe_get_attr(coordinator, "entities", [])
        batch_managers = _safe_get_attr(coordinator, "_batch_managers", {})
        initialized = _safe_get_attr(coordinator, "_initialized", False)
        
        if not entities:
            return {"error": "No entities available for batch analysis"}
        
        # Handle uninitialized coordinator
        if not initialized:
            return {
                "status": "initializing",
                "message": "Batch analysis will be available after first data update",
                "total_entities": len(entities),
                "overall_register_efficiency_percent": 0,
                "batch_managers_count": 0,
                "efficiency_rating": "Pending Initialization",
                "initialization_pending": True
            }
        
        total_entities = len(entities)
        total_batches = 0
        batch_analysis = {}
        overall_efficiency = 0
        
        for key, bm in batch_managers.items():
            if bm:
                try:
                    batches = _safe_get_attr(bm, "batches", [])
                    batch_count = len(batches)
                    total_batches += batch_count
                    
                    if batches:
                        batch_sizes = []
                        register_utilization = []
                        
                        for batch in batches:
                            if batch:
                                try:
                                    # Calculate batch span
                                    batch_start = _safe_get_attr(batch[0], "start", 0)
                                    batch_end = _safe_get_attr(batch[-1], "end", 0)
                                    batch_span = batch_end - batch_start + 1 if batch_end >= batch_start else 0
                                    batch_sizes.append(batch_span)
                                    
                                    # Calculate register utilization
                                    actual_registers = sum(_safe_get_attr(block, "count", 0) for block in batch)
                                    utilization = (actual_registers / batch_span * 100) if batch_span > 0 else 0
                                    register_utilization.append(utilization)
                                except Exception:
                                    pass
                        
                        avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0
                        avg_utilization = sum(register_utilization) / len(register_utilization) if register_utilization else 0
                        
                        batch_analysis[str(key)] = {
                            "batch_count": batch_count,
                            "avg_batch_size": round(avg_batch_size, 2),
                            "max_batch_size": max(batch_sizes) if batch_sizes else 0,
                            "min_batch_size": min(batch_sizes) if batch_sizes else 0,
                            "avg_register_utilization_percent": round(avg_utilization, 1),
                            "congestion_window": _safe_get_attr(bm, "congestion_window", "N/A"),
                        }
                        
                        overall_efficiency += avg_utilization
                except Exception as ex:
                    batch_analysis[str(key)] = {"error": f"Failed to analyze batch manager: {ex}"}
        
        # Calculate overall metrics
        entities_per_batch = total_entities / total_batches if total_batches > 0 else 0
        avg_efficiency = overall_efficiency / len(batch_managers) if batch_managers else 0
        
        return {
            "status": "active",
            "total_batches": total_batches,
            "entities_per_batch_avg": round(entities_per_batch, 2),
            "overall_register_efficiency_percent": round(avg_efficiency, 1),
            "batch_managers_count": len(batch_managers),
            "batch_analysis_by_slave_type": batch_analysis,
            "efficiency_rating": _calculate_efficiency_rating(entities_per_batch, avg_efficiency),
            "initialization_pending": False
        }
    except Exception as ex:
        _LOGGER.exception("Error in batch efficiency analysis: %s", ex)
        return {"error": f"Failed to analyze batch efficiency: {ex}"}


def _calculate_efficiency_rating(entities_per_batch: float, register_efficiency: float) -> str:
    """Calculate an efficiency rating based on metrics."""
    try:
        # Scoring: entities per batch (weight 40%) + register efficiency (weight 60%)
        batch_score = min(entities_per_batch * 20, 40)  # Max 40 points
        efficiency_score = register_efficiency * 0.6    # Max 60 points
        total_score = batch_score + efficiency_score
        
        if total_score >= 80:
            return "Excellent"
        elif total_score >= 60:
            return "Good"
        elif total_score >= 40:
            return "Fair"
        else:
            return "Poor"
    except Exception:
        return "Unknown"


def _get_health_assessment(coordinator: Any) -> Dict[str, Any]:
    """Calculate overall health score and status."""
    try:
        health_factors = {}
        score_components = {}
        
        # Check initialization status first
        initialized = _safe_get_attr(coordinator, "_initialized", False)
        
        # Connection health (25 points max)
        client = _safe_get_attr(coordinator, "client", None)
        if client and _safe_get_attr(client, "connected", False):
            connection_score = 25
            health_factors["connection"] = "Connected"
        else:
            connection_score = 0
            health_factors["connection"] = "Disconnected"
        score_components["connection"] = connection_score
        
        # Update success (25 points max)
        last_success = _safe_get_attr(coordinator, "last_update_success", None)
        if last_success:
            update_score = 25
            health_factors["updates"] = "Recent success"
        else:
            update_score = 0
            health_factors["updates"] = "No recent success"
        score_components["updates"] = update_score
        
        # Entity configuration (25 points max)
        entities = _safe_get_attr(coordinator, "entities", [])
        entity_count = len(entities)
        if entity_count > 0:
            entity_score = min(entity_count * 2, 25)  # 2 points per entity, max 25
            health_factors["entities"] = f"{entity_count} entities configured"
        else:
            entity_score = 0
            health_factors["entities"] = "No entities configured"
        score_components["entities"] = entity_score
        
        # Handle initialization status in batch efficiency (25 points max)
        if not initialized:
            batch_score = 0
            health_factors["batching"] = "Coordinator initializing - batch managers pending"
        else:
            batch_managers = _safe_get_attr(coordinator, "_batch_managers", {})
            if batch_managers and entity_count > 0:
                total_batches = sum(len(_safe_get_attr(bm, "batches", [])) for bm in batch_managers.values() if bm)
                if total_batches > 0:
                    efficiency = min((entity_count / total_batches) * 5, 25)  # Efficiency factor
                    batch_score = efficiency
                    health_factors["batching"] = f"Efficiency: {efficiency/25*100:.0f}%"
                else:
                    batch_score = 0
                    health_factors["batching"] = "No active batches"
            else:
                batch_score = 0
                health_factors["batching"] = "No batch managers"
        score_components["batching"] = batch_score
        
        # Calculate total score
        total_score = sum(score_components.values())
        
        # Adjust status based on initialization
        if not initialized:
            status = "Initializing"
            status_emoji = "🟡"
        elif total_score >= 80:
            status = "Excellent"
            status_emoji = "🟢"
        elif total_score >= 60:
            status = "Good"
            status_emoji = "🟡"
        elif total_score >= 40:
            status = "Fair"
            status_emoji = "🟠"
        else:
            status = "Poor"
            status_emoji = "🔴"
        
        return {
            "overall_score": round(total_score, 1),
            "status": status,
            "status_emoji": status_emoji,
            "score_components": score_components,
            "health_factors": health_factors,
            "max_possible_score": 100,
            "initialized": initialized,
        }
    except Exception as ex:
        _LOGGER.exception("Error in health assessment: %s", ex)
        return {"error": f"Failed to assess health: {ex}"}


def _get_configuration_analysis(coordinator: Any, entry: ConfigEntry) -> Dict[str, Any]:
    """Analyze configuration for potential issues."""
    try:
        entities = _safe_get_attr(coordinator, "entities", [])
        issues = []
        warnings = []
        info = []
        
        # Check entity count
        entity_count = len(entities)
        if entity_count == 0:
            issues.append("No entities configured - integration will not provide any data")
        elif entity_count > 100:
            warnings.append(f"Large number of entities ({entity_count}) may impact performance")
        else:
            info.append(f"Entity count ({entity_count}) is within normal range")
        
        # Check scan intervals
        scan_intervals = []
        for entity in entities:
            interval = _safe_get_attr(entity, "scan_interval", None)
            if interval:
                scan_intervals.append(interval)
        
        if scan_intervals:
            fast_intervals = [i for i in scan_intervals if i < 5]
            very_fast_intervals = [i for i in scan_intervals if i < 2]
            
            if very_fast_intervals:
                issues.append(f"{len(very_fast_intervals)} entities have very fast scan intervals (<2s) which may overload the device")
            elif len(fast_intervals) > len(scan_intervals) * 0.5:
                warnings.append(f"{len(fast_intervals)} entities have fast scan intervals (<5s)")
            
            unique_intervals = set(scan_intervals)
            if len(unique_intervals) > 5:
                warnings.append(f"Many different scan intervals ({len(unique_intervals)}) may fragment polling efficiency")
        
        # Check connection settings
        host = entry.data.get(CONF_HOST)
        port = entry.data.get(CONF_PORT)
        slave_id = entry.data.get(CONF_SLAVE_ID)
        
        if not host:
            issues.append("No host configured")
        if not port:
            warnings.append("No port configured, using default")
        if slave_id is None:
            warnings.append("No slave ID configured")
        
        # Check batch configuration
        max_read_size = _safe_get_attr(coordinator, "_max_read_size", None)
        if max_read_size:
            if max_read_size > 125:
                warnings.append(f"Large max_read_size ({max_read_size}) may cause timeouts with some devices")
            elif max_read_size < 10:
                warnings.append(f"Small max_read_size ({max_read_size}) may reduce efficiency")
        
        return {
            "issues": issues,
            "warnings": warnings,
            "info": info,
            "configuration_summary": {
                "host": host,
                "port": port,
                "slave_id": slave_id,
                "max_read_size": max_read_size,
                "entity_count": entity_count,
                "unique_scan_intervals": len(set(scan_intervals)) if scan_intervals else 0,
            }
        }
    except Exception as ex:
        _LOGGER.exception("Error in configuration analysis: %s", ex)
        return {"error": f"Failed to analyze configuration: {ex}"}


def _get_register_mapping(coordinator: Any) -> Dict[str, Any]:
    """Create a map of register usage."""
    try:
        entities = _safe_get_attr(coordinator, "entities", [])
        
        if not entities:
            return {"error": "No entities available for register mapping"}
        
        register_map = {
            "holding_registers": [],
            "input_registers": [],
            "coils": [],
            "discrete_inputs": []
        }
        
        address_ranges = {}
        
        for entity in entities:
            try:
                desc = _safe_get_attr(entity, "desc", None)
                if desc:
                    data_type = _safe_get_attr(desc, "data_type", "unknown").lower()
                    address = _safe_get_attr(desc, "register_address", None)
                    count = _safe_get_attr(desc, "register_count", 1)
                    entity_key = _safe_get_attr(desc, "key", "unknown")
                    
                    if address is not None:
                        register_info = {
                            "address": address,
                            "count": count,
                            "end_address": address + count - 1,
                            "entity_key": entity_key,
                            "slave_id": _safe_get_attr(entity, "slave_id", "unknown")
                        }
                        
                        # Categorize by data type
                        if "holding" in data_type:
                            register_map["holding_registers"].append(register_info)
                            key = "holding"
                        elif "input" in data_type:
                            register_map["input_registers"].append(register_info)
                            key = "input"
                        elif "coil" in data_type:
                            register_map["coils"].append(register_info)
                            key = "coils"
                        elif "discrete" in data_type:
                            register_map["discrete_inputs"].append(register_info)
                            key = "discrete"
                        else:
                            key = "unknown"
                        
                        # Track address ranges
                        if key not in address_ranges:
                            address_ranges[key] = {"min": address, "max": address + count - 1}
                        else:
                            address_ranges[key]["min"] = min(address_ranges[key]["min"], address)
                            address_ranges[key]["max"] = max(address_ranges[key]["max"], address + count - 1)
            except Exception as ex:
                _LOGGER.debug("Error processing entity for register mapping: %s", ex)
        
        # Sort register lists by address
        for reg_type in register_map:
            register_map[reg_type].sort(key=lambda x: x["address"])
        
        # Calculate statistics
        total_registers = sum(len(registers) for registers in register_map.values())
        
        return {
            "register_map": register_map,
            "address_ranges": address_ranges,
            "total_registers_mapped": total_registers,
            "register_type_counts": {
                reg_type: len(registers) for reg_type, registers in register_map.items()
            }
        }
    except Exception as ex:
        _LOGGER.exception("Error creating register mapping: %s", ex)
        return {"error": f"Failed to create register mapping: {ex}"}


def _get_recommendations(coordinator: Any) -> Dict[str, Any]:
    """Generate actionable recommendations."""
    try:
        entities = _safe_get_attr(coordinator, "entities", [])
        batch_managers = _safe_get_attr(coordinator, "_batch_managers", {})
        client = _safe_get_attr(coordinator, "client", None)
        initialized = _safe_get_attr(coordinator, "_initialized", False)
        
        recommendations = {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": [],
            "optimization": []
        }
        
        # Handle initialization status
        if not initialized:
            recommendations["low_priority"].append({
                "category": "Status",
                "issue": "Coordinator is initializing",
                "recommendation": "Batch managers will be created after first data update",
                "action": "Wait for first polling cycle to complete"
            })
        
        # Connection recommendations
        if not client or not _safe_get_attr(client, "connected", False):
            recommendations["high_priority"].append({
                "category": "Connection",
                "issue": "Device not connected",
                "recommendation": "Check network connectivity and device settings",
                "action": "Verify host, port, and network connectivity"
            })
        
        # Entity configuration recommendations
        entity_count = len(entities)
        if entity_count == 0:
            recommendations["high_priority"].append({
                "category": "Configuration",
                "issue": "No entities configured",
                "recommendation": "Add Modbus entities to start monitoring data",
                "action": "Configure entities in the integration settings"
            })
        
        # Scan interval recommendations
        scan_intervals = [_safe_get_attr(e, "scan_interval", None) for e in entities if _safe_get_attr(e, "scan_interval", None)]
        if scan_intervals:
            very_fast = [i for i in scan_intervals if i < 2]
            if very_fast:
                recommendations["medium_priority"].append({
                    "category": "Performance",
                    "issue": f"{len(very_fast)} entities have very fast scan intervals (<2s)",
                    "recommendation": "Increase scan intervals to reduce device load",
                    "action": "Change scan intervals to 5-30 seconds for most entities"
                })
        
        # Batch efficiency recommendations (only if initialized)
        if initialized and batch_managers:
            total_batches = sum(len(_safe_get_attr(bm, "batches", [])) for bm in batch_managers.values() if bm)
            if entity_count > 0 and total_batches > 0:
                efficiency = entity_count / total_batches
                if efficiency < 2:
                    recommendations["optimization"].append({
                        "category": "Batching",
                        "issue": f"Low batch efficiency ({efficiency:.1f} entities per batch)",
                        "recommendation": "Consider grouping nearby register addresses",
                        "action": "Review register addresses and group related entities"
                    })
                elif efficiency > 10:
                    recommendations["optimization"].append({
                        "category": "Batching", 
                        "issue": "Very high batch efficiency - consider if this is optimal",
                        "recommendation": "Monitor for potential timeout issues",
                        "action": "Watch for read errors and consider smaller batches if needed"
                    })
        
        # Success recommendations
        last_success = _safe_get_attr(coordinator, "last_update_success", None)
        if last_success is False:
            recommendations["high_priority"].append({
                "category": "Data",
                "issue": "Recent update failures detected",
                "recommendation": "Check device responsiveness and network stability",
                "action": "Review logs for specific error messages"
            })
        elif last_success is True and entity_count > 0 and initialized:
            recommendations["low_priority"].append({
                "category": "Status",
                "issue": "System is working well",
                "recommendation": "Continue monitoring performance",
                "action": "No immediate action required"
            })
        
        return recommendations
    except Exception as ex:
        _LOGGER.exception("Error generating recommendations: %s", ex)
        return {"error": f"Failed to generate recommendations: {ex}"}


def _get_troubleshooting_guide(coordinator: Any) -> Dict[str, Any]:
    """Provide troubleshooting guidance."""
    try:
        entities = _safe_get_attr(coordinator, "entities", [])
        client = _safe_get_attr(coordinator, "client", None)
        last_success = _safe_get_attr(coordinator, "last_update_success", None)
        initialized = _safe_get_attr(coordinator, "_initialized", False)
        
        # Quick health checks
        health_checks = {
            "device_reachable": client and _safe_get_attr(client, "connected", False),
            "entities_configured": len(entities) > 0,
            "recent_success": last_success is True,
            "batch_managers_active": len(_safe_get_attr(coordinator, "_batch_managers", {})) > 0,
            "coordinator_initialized": initialized,
        }
        
        # Common issues and solutions
        common_issues = [
            {
                "symptom": "No data in entities",
                "possible_causes": [
                    "Device not responding",
                    "Wrong register addresses",
                    "Network connectivity issues",
                    "Incorrect slave ID"
                ],
                "solutions": [
                    "Check device power and network connection",
                    "Verify register addresses in device documentation",
                    "Test connectivity with ping or telnet",
                    "Confirm slave ID matches device configuration"
                ]
            },
            {
                "symptom": "Intermittent connection issues",
                "possible_causes": [
                    "Network instability",
                    "Device overload",
                    "Firewall blocking connections",
                    "Too frequent polling"
                ],
                "solutions": [
                    "Check network stability and bandwidth",
                    "Increase scan intervals to reduce load",
                    "Configure firewall to allow Modbus traffic",
                    "Monitor device CPU and memory usage"
                ]
            },
            {
                "symptom": "Slow response times",
                "possible_causes": [
                    "Large batch sizes",
                    "Network latency",
                    "Device processing limitations",
                    "Inefficient register layout"
                ],
                "solutions": [
                    "Reduce max_read_size setting",
                    "Check network latency and quality",
                    "Distribute polling across more time",
                    "Group related registers together"
                ]
            }
        ]
        
        # Performance optimization tips
        optimization_tips = [
            "Group entities with similar scan intervals together",
            "Use scan intervals appropriate for data change frequency",
            "Place related registers at consecutive addresses when possible",
            "Monitor batch efficiency and adjust max_read_size accordingly",
            "Consider using different slave IDs for logical device separation",
            "Regularly review entity configurations for optimization opportunities"
        ]
        
        return {
            "health_checks": health_checks,
            "common_issues": common_issues,
            "optimization_tips": optimization_tips,
            "next_steps": _generate_next_steps(health_checks, last_success, len(entities), initialized)
        }
    except Exception as ex:
        _LOGGER.exception("Error creating troubleshooting guide: %s", ex)
        return {"error": f"Failed to create troubleshooting guide: {ex}"}


def _generate_next_steps(health_checks: Dict[str, bool], last_success: bool, entity_count: int, initialized: bool) -> List[str]:
    """Generate specific next steps based on current state."""
    next_steps = []
    
    if not health_checks.get("device_reachable", False):
        next_steps.append("🔴 URGENT: Check device connectivity - verify host, port, and network")
    
    if not health_checks.get("entities_configured", False):
        next_steps.append("🟡 Configure entities to start monitoring Modbus data")
    
    if not initialized:
        next_steps.append("🟡 Coordinator initializing - batch managers will be created after first update")
    
    if not health_checks.get("recent_success", False) and entity_count > 0:
        next_steps.append("🟠 Investigate recent update failures - check logs for error details")
    
    if initialized and not health_checks.get("batch_managers_active", False) and entity_count > 0:
        next_steps.append("🟡 Batch managers not active after initialization - restart integration if needed")
    
    if all(health_checks.values()) and last_success and initialized:
        next_steps.append("✅ System appears healthy - monitor performance metrics")
    
    if not next_steps:
        next_steps.append("ℹ️ Review recommendations section for optimization opportunities")
    
    return next_steps