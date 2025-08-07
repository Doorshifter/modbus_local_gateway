"""Services for Modbus Local Gateway."""
import logging
import voluptuous as vol
import os
from datetime import datetime
import json
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv
from homeassistant.config_entries import ConfigEntry
from typing import Dict, Any

from .const import DOMAIN
from .statistics.manager import STATISTICS_MANAGER
from .statistics.self_healing import SELF_HEALING_SYSTEM
from .statistics.resource_adaptation import RESOURCE_ADAPTER
from .statistics.performance_optimization import PERFORMANCE_OPTIMIZER

_LOGGER = logging.getLogger(__name__)

async def async_setup_services(hass: HomeAssistant):
    """Set up services for Modbus Local Gateway integration."""
    
    async def set_scan_interval(call: ServiceCall):
        """Set scan interval for a Modbus entity or group of entities."""
        entity_id = call.data.get("entity_id")
        scan_interval = call.data.get("scan_interval")
        
        if not entity_id or not scan_interval:
            _LOGGER.error("Both entity_id and scan_interval are required")
            return
        
        # Get all gateway coordinators
        coordinators = {}
        for key, data in hass.data.get(DOMAIN, {}).items():
            if hasattr(data, "entities"):
                coordinators[key] = data
        
        # Find the entity in available coordinators
        entity_found = False
        for gateway_key, coordinator in coordinators.items():
            for entity in coordinator.entities:
                if getattr(entity, "entity_id", None) == entity_id or (
                    isinstance(entity_id, list) and entity.entity_id in entity_id
                ):
                    entity_found = True
                    old_interval = entity.scan_interval
                    entity.scan_interval = scan_interval
                    _LOGGER.info(
                        "Updated scan interval for %s from %s to %s seconds",
                        entity.entity_id, old_interval, scan_interval
                    )
                    
            # Reschedule polling jobs if we found entities for this coordinator
            if entity_found:
                coordinator._schedule_dynamic_updates()
        
        if not entity_found:
            _LOGGER.warning("No entities found matching %s", entity_id)
    
    async def analyze_correlations(call: ServiceCall):
        """Analyze correlations between Modbus entities."""
        force = call.data.get("force", True)
        threshold = call.data.get("threshold", 0.6)
        
        # Run analysis in executor to avoid blocking
        def _do_analysis():
            _LOGGER.info("Starting correlation analysis (threshold=%.2f)", threshold)
            clusters = STATISTICS_MANAGER.analyze_correlations(force=force, threshold=threshold)
            _LOGGER.info("Correlation analysis complete: %d clusters found", len(clusters))
            
            # Log cluster details at debug level
            for cluster_id, entities in clusters.items():
                _LOGGER.debug("Cluster %s: %s", cluster_id, entities)
                
            return clusters
        
        try:
            clusters = await hass.async_add_executor_job(_do_analysis)
            
            # Format nice response message
            message = f"Analysis complete: {len(clusters)} clusters created."
            if clusters:
                message += f" Largest cluster contains {max(len(entities) for entities in clusters.values())} entities."
            return {"success": True, "clusters": len(clusters), "message": message}
        except Exception as e:
            _LOGGER.error("Error in correlation analysis: %s", e)
            return {"success": False, "error": str(e)}
            
    async def configure_pattern_detection(call: ServiceCall):
        """Configure entity influence in pattern detection."""
        entity_id = call.data.get("entity_id")
        importance = call.data.get("importance", 5)
        
        if not entity_id:
            _LOGGER.error("Entity ID is required")
            return
            
        # Set entity importance for pattern detection
        STATISTICS_MANAGER.set_entity_importance(entity_id, importance)
        _LOGGER.info(
            "Entity %s configured with importance factor %d for pattern detection",
            entity_id, importance
        )
        
    async def optimize_scan_intervals(call: ServiceCall):
        """Optimize scan intervals for all entities."""
        # Optional parameters
        apply_changes = call.data.get("apply_changes", False)
        max_registers_per_second = call.data.get("max_registers_per_second")
        min_interval = call.data.get("min_interval")
        max_interval = call.data.get("max_interval")
        
        # Update config if provided
        if hasattr(STATISTICS_MANAGER, "_interval_manager"):
            if max_registers_per_second is not None:
                STATISTICS_MANAGER._interval_manager.max_registers_per_second = max_registers_per_second
                
            if min_interval is not None:
                STATISTICS_MANAGER._interval_manager.min_interval = min_interval
                
            if max_interval is not None:
                STATISTICS_MANAGER._interval_manager.max_interval = max_interval
        
        # Get optimized intervals
        if hasattr(STATISTICS_MANAGER, "optimize_all_intervals"):
            optimized_intervals = STATISTICS_MANAGER.optimize_all_intervals()
            
            if apply_changes:
                # Actually update the scan intervals
                update_count = 0
                for entity_id, interval in optimized_intervals.items():
                    success = STATISTICS_MANAGER.set_scan_interval(entity_id, interval)
                    if success:
                        update_count += 1
                
                _LOGGER.info(
                    "Applied optimized scan intervals to %d entities", 
                    update_count
                )
            
            # Return the optimized intervals
            return {
                "optimized_intervals": optimized_intervals,
                "system_stats": STATISTICS_MANAGER.get_system_statistics()
            }
        else:
            # Fallback to coordinator-based optimization
            try:
                # Get all coordinators
                coordinators = []
                for coordinator in hass.data.get(DOMAIN, {}).values():
                    if hasattr(coordinator, "entities") and hasattr(coordinator, "apply_recommended_intervals"):
                        coordinators.append(coordinator)
                
                if not coordinators:
                    _LOGGER.warning("No compatible Modbus coordinators found")
                    return {"error": "No compatible coordinators found"}
                
                results = {}
                
                for coordinator in coordinators:
                    try:
                        if apply_changes and hasattr(coordinator, "apply_recommended_intervals"):
                            result = await coordinator.apply_recommended_intervals()
                            results[coordinator.gateway] = result
                        else:
                            # Just return info about entities
                            results[coordinator.gateway] = {
                                "message": "Changes not applied (dry run)",
                                "entities": len(coordinator.entities) if hasattr(coordinator, "entities") else 0
                            }
                    except Exception as e:
                        results[coordinator.gateway] = {"error": str(e)}
                
                return results
            except Exception as e:
                _LOGGER.error("Error optimizing scan intervals: %s", e)
                return {"error": str(e)}
    
    # Services integrated from service_extension.py
    async def run_self_healing(call: ServiceCall) -> Dict[str, Any]:
        """Run self-healing system."""
        full_check = call.data.get("full_check", True)
        
        # Run self-healing in executor to avoid blocking
        return await hass.async_add_executor_job(
            SELF_HEALING_SYSTEM.check_and_heal_system, full_check
        )
        
    async def update_resource_adaptation(call: ServiceCall) -> Dict[str, Any]:
        """Update resource adaptation settings."""
        force_update = call.data.get("force_update", False)
        
        if force_update:
            # Force recalculation of system capabilities
            return await hass.async_add_executor_job(
                RESOURCE_ADAPTER.measure_system_capabilities
            )
        else:
            # Get current adaptation settings
            return {
                "resources": await hass.async_add_executor_job(
                    RESOURCE_ADAPTER.get_resource_history
                ),
                "recommendations": await hass.async_add_executor_job(
                    RESOURCE_ADAPTER.get_throughput_recommendation
                )
            }
    
    async def export_statistics(call: ServiceCall) -> Dict[str, Any]:
        """Export statistics data to JSON file."""
        target_path = call.data.get("target_path")
        data_type = call.data.get("data_type", "all")
        
        # Default path if not provided
        if not target_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_dir = hass.config.path()
            target_path = os.path.join(config_dir, f"modbus_statistics_export_{timestamp}.json")
        
        # This will be implemented with the statistics manager integration
        if hasattr(STATISTICS_MANAGER, "export_data"):
            try:
                data = STATISTICS_MANAGER.export_data()
                
                # Filter data if specific type requested
                if data_type != "all" and data_type in data:
                    export_data = {data_type: data[data_type]}
                else:
                    export_data = data
                
                # Save to file
                with open(target_path, "w") as f:
                    json.dump(export_data, f, indent=2)
                    
                return {
                    "success": True,
                    "message": f"Statistics exported to {target_path}",
                    "file_path": target_path,
                    "data_type": data_type,
                    "size_bytes": os.path.getsize(target_path)
                }
            except Exception as e:
                _LOGGER.error("Error exporting statistics: %s", e)
                return {"success": False, "error": str(e)}
        else:
            return {
                "success": False,
                "message": "Export functionality requires statistics manager integration",
                "target_path": target_path,
                "data_type": data_type
            }
    
    async def clear_statistics(call: ServiceCall) -> Dict[str, Any]:
        """Clear statistics data."""
        data_type = call.data.get("data_type", "all")
        
        # Check if storage manager has clear method
        if hasattr(STATISTICS_MANAGER, "storage") and hasattr(STATISTICS_MANAGER.storage, "clear_storage"):
            try:
                success = await hass.async_add_executor_job(
                    STATISTICS_MANAGER.storage.clear_storage, data_type
                )
                if success:
                    return {
                        "success": True,
                        "message": f"Successfully cleared {data_type} statistics data"
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Failed to clear {data_type} statistics data"
                    }
            except Exception as e:
                _LOGGER.error("Error clearing statistics: %s", e)
                return {"success": False, "error": str(e)}
        else:
            return {
                "success": False,
                "message": "Clear functionality requires statistics manager integration",
                "data_type": data_type
            }
    
    async def get_statistics_storage_info(call: ServiceCall) -> Dict[str, Any]:
        """Get information about statistics storage."""
        if hasattr(STATISTICS_MANAGER, "get_storage_info"):
            try:
                return await hass.async_add_executor_job(STATISTICS_MANAGER.get_storage_info)
            except Exception as e:
                _LOGGER.error("Error getting storage info: %s", e)
                return {"success": False, "error": str(e)}
        else:
            return {
                "success": False,
                "message": "Storage info functionality requires statistics manager integration"
            }
    
    # NEW PERFORMANCE OPTIMIZATION SERVICES
    async def run_performance_optimization(call: ServiceCall) -> Dict[str, Any]:
        """Run performance optimization."""
        optimization_type = call.data.get("type", "quick")
        strategy = call.data.get("strategy")
        
        # Run in executor to avoid blocking
        if strategy:
            return await hass.async_add_executor_job(
                PERFORMANCE_OPTIMIZER.run_strategy, strategy
            )
        elif optimization_type == "full":
            return await hass.async_add_executor_job(
                PERFORMANCE_OPTIMIZER.run_full_optimization
            )
        else:
            return await hass.async_add_executor_job(
                PERFORMANCE_OPTIMIZER.run_quick_optimization
            )
    
    async def get_optimization_statistics(call: ServiceCall) -> Dict[str, Any]:
        """Get performance optimization statistics."""
        # Run in executor to avoid blocking
        return await hass.async_add_executor_job(
            PERFORMANCE_OPTIMIZER.get_statistics
        )
    
    async def update_optimization_strategy(call: ServiceCall) -> Dict[str, Any]:
        """Update performance optimization strategy parameters."""
        strategy = call.data.get("strategy")
        parameters = call.data.get("parameters", {})
        
        if not strategy or not parameters:
            return {"success": False, "error": "Missing strategy or parameters"}
            
        # Run in executor to avoid blocking
        return await hass.async_add_executor_job(
            PERFORMANCE_OPTIMIZER.update_strategy_parameters, strategy, parameters
        )
    
    async def get_parameter_profiles(call: ServiceCall) -> Dict[str, Any]:
        """Get parameter profiles."""
        # Get specific profile or all
        profile_name = call.data.get("profile_name")
        
        return await hass.async_add_executor_job(
            STATISTICS_MANAGER.get_parameter_profiles, profile_name
        )
    
    async def create_parameter_profile(call: ServiceCall) -> Dict[str, Any]:
        """Create a parameter profile."""
        name = call.data.get("name")
        description = call.data.get("description", "")
        parameters = call.data.get("parameters", {})
        
        return await hass.async_add_executor_job(
            STATISTICS_MANAGER.create_parameter_profile,
            name, description, parameters
        )
    
    async def update_parameter_profile(call: ServiceCall) -> Dict[str, Any]:
        """Update a parameter profile."""
        name = call.data.get("name")
        data = {k: v for k, v in call.data.items() if k != "name"}
        
        return await hass.async_add_executor_job(
            STATISTICS_MANAGER.update_parameter_profile,
            name, data
        )
    
    async def delete_parameter_profile(call: ServiceCall) -> Dict[str, Any]:
        """Delete a parameter profile."""
        name = call.data.get("name")
        
        return await hass.async_add_executor_job(
            STATISTICS_MANAGER.delete_parameter_profile,
            name
        )
    
    async def activate_parameter_profile(call: ServiceCall) -> Dict[str, Any]:
        """Activate a parameter profile."""
        name = call.data.get("name")
        
        return await hass.async_add_executor_job(
            STATISTICS_MANAGER.activate_parameter_profile,
            name
        )
    
    async def adapt_parameters(call: ServiceCall) -> Dict[str, Any]:
        """Adapt parameters to current context."""
        force = call.data.get("force", False)
        
        return await hass.async_add_executor_job(
            STATISTICS_MANAGER.adapt_parameters,
            force
        )
    
    async def get_parameter_status(call: ServiceCall) -> Dict[str, Any]:
        """Get parameter system status."""
        return await hass.async_add_executor_job(
            STATISTICS_MANAGER.get_parameter_status
        )
    
    async def set_parameter(call: ServiceCall) -> Dict[str, Any]:
        """Set a specific parameter value."""
        name = call.data.get("name")
        value = call.data.get("value")
        
        return await hass.async_add_executor_job(
            STATISTICS_MANAGER.set_parameter,
            name, value
        )
    
    # Add registrations to async_setup_services
    hass.services.async_register(
        DOMAIN,
        "get_parameter_profiles",
        get_parameter_profiles,
        schema=vol.Schema({
            vol.Optional("profile_name"): cv.string,
        })
    )
    
    hass.services.async_register(
        DOMAIN,
        "create_parameter_profile",
        create_parameter_profile,
        schema=vol.Schema({
            vol.Required("name"): cv.string,
            vol.Optional("description"): cv.string,
            vol.Optional("parameters"): vol.Schema({}),
        })
    )
    
    hass.services.async_register(
        DOMAIN,
        "update_parameter_profile",
        update_parameter_profile,
        schema=vol.Schema({
            vol.Required("name"): cv.string,
            vol.Optional("description"): cv.string,
            vol.Optional("parameters"): vol.Schema({}),
            vol.Optional("context_match"): vol.Schema({}),
        })
    )
    
    hass.services.async_register(
        DOMAIN,
        "delete_parameter_profile",
        delete_parameter_profile,
        schema=vol.Schema({
            vol.Required("name"): cv.string,
        })
    )
    
    hass.services.async_register(
        DOMAIN,
        "activate_parameter_profile",
        activate_parameter_profile,
        schema=vol.Schema({
            vol.Required("name"): cv.string,
        })
    )
    
    hass.services.async_register(
        DOMAIN,
        "adapt_parameters",
        adapt_parameters,
        schema=vol.Schema({
            vol.Optional("force", default=False): cv.boolean,
        })
    )
    
    hass.services.async_register(
        DOMAIN,
        "get_parameter_status",
        get_parameter_status,
        schema=vol.Schema({})
    )
    
    hass.services.async_register(
        DOMAIN,
        "set_parameter",
        set_parameter,
        schema=vol.Schema({
            vol.Required("name"): cv.string,
            vol.Required("value"): vol.Any(
                cv.positive_int, cv.positive_float, cv.string, cv.boolean
            ),
        })
    )
    # Register original services
    hass.services.async_register(
        DOMAIN,
        "set_scan_interval",
        set_scan_interval,
        schema=vol.Schema({
            vol.Required("entity_id"): cv.entity_id,
            vol.Required("scan_interval"): vol.All(vol.Coerce(int), vol.Range(min=1, max=3600)),
        }),
    )
    
    hass.services.async_register(
        DOMAIN,
        "analyze_correlations",
        analyze_correlations,
        schema=vol.Schema({
            vol.Optional("force", default=True): cv.boolean,
            vol.Optional("threshold", default=0.6): vol.All(
                vol.Coerce(float), vol.Range(min=0.1, max=0.99)
            ),
        }),
    )

    hass.services.async_register(
        DOMAIN,
        "configure_pattern_detection",
        configure_pattern_detection,
        schema=vol.Schema({
            vol.Required("entity_id"): cv.entity_id,
            vol.Optional("importance", default=5): vol.All(
                vol.Coerce(int), vol.Range(min=1, max=10)
            ),
        }),
    )
    
    hass.services.async_register(
        DOMAIN,
        "optimize_scan_intervals",
        optimize_scan_intervals,
        schema=vol.Schema({
            vol.Optional("apply_changes", default=False): cv.boolean,
            vol.Optional("max_registers_per_second"): vol.Any(None, cv.positive_int),
            vol.Optional("min_interval"): vol.Any(None, cv.positive_int),
            vol.Optional("max_interval"): vol.Any(None, cv.positive_int),
        }),
    )
    
    # Register the services moved from service_extension.py
    hass.services.async_register(
        DOMAIN, 
        "run_self_healing", 
        run_self_healing, 
        schema=vol.Schema({
            vol.Optional("full_check", default=True): cv.boolean,
        })
    )
    
    hass.services.async_register(
        DOMAIN, 
        "resource_adaptation", 
        update_resource_adaptation, 
        schema=vol.Schema({
            vol.Optional("force_update", default=False): cv.boolean,
        })
    )
    
    hass.services.async_register(
        DOMAIN, 
        "export_statistics", 
        export_statistics, 
        schema=vol.Schema({
            vol.Optional("target_path"): cv.string,
            vol.Optional("data_type", default="all"): vol.In([
                "all", "entity_stats", "patterns", "clusters", 
                "correlation", "interval_history"
            ]),
        })
    )
    
    hass.services.async_register(
        DOMAIN, 
        "clear_statistics", 
        clear_statistics, 
        schema=vol.Schema({
            vol.Optional("data_type", default="all"): vol.In([
                "all", "entity_stats", "patterns", "clusters", 
                "correlation", "interval_history"
            ]),
        })
    )
    
    hass.services.async_register(
        DOMAIN, 
        "get_statistics_storage_info", 
        get_statistics_storage_info, 
        schema=vol.Schema({})
    )
    
    # Register NEW performance optimization services
    hass.services.async_register(
        DOMAIN,
        "run_performance_optimization",
        run_performance_optimization,
        schema=vol.Schema({
            vol.Optional("type"): vol.In(["quick", "full"]),
            vol.Optional("strategy"): vol.In([
                "interval", "cluster", "pattern"
            ])
        })
    )
    
    hass.services.async_register(
        DOMAIN,
        "get_optimization_statistics",
        get_optimization_statistics,
        schema=vol.Schema({})
    )
    
    hass.services.async_register(
        DOMAIN,
        "update_optimization_strategy",
        update_optimization_strategy,
        schema=vol.Schema({
            vol.Required("strategy"): vol.In([
                "interval", "cluster", "pattern"
            ]),
            vol.Required("parameters"): vol.Schema({
                vol.Optional("min_interval"): cv.positive_int,
                vol.Optional("max_interval"): cv.positive_int,
                vol.Optional("aggressiveness"): vol.All(
                    vol.Coerce(float), vol.Range(min=0.1, max=1.0)
                ),
                vol.Optional("correlation_threshold"): vol.All(
                    vol.Coerce(float), vol.Range(min=0.1, max=0.9)
                ),
                vol.Optional("max_patterns"): vol.All(
                    vol.Coerce(int), vol.Range(min=2, max=20)
                ),
                vol.Optional("pattern_stability_threshold"): vol.All(
                    vol.Coerce(float), vol.Range(min=0.1, max=1.0)
                ),
                vol.Optional("max_entities_per_run"): cv.positive_int,
                vol.Optional("min_cluster_size"): cv.positive_int,
                vol.Optional("max_cluster_size"): cv.positive_int,
            })
        })
    )
    
    # Return True ONLY ONCE at the end, after all services are registered
    return True