"""
Validation framework for Modbus optimization system.

This module provides comprehensive validation and consistency checks to ensure
the system maintains data integrity and operates correctly.
"""

import json
import logging
import os
import time
import asyncio
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple, Union, Callable

from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_call_later

from .storage import StatisticsStorageManager
from .data_validation import StorageValidator

_LOGGER = logging.getLogger(__name__)


class ValidationStage(Enum):
    """Validation system stage enum to track lifecycle."""
    INITIALIZING = "initializing"  # During HA startup
    READY = "ready"                # After HA fully started
    SHUTDOWN = "shutdown"          # During shutdown


class ValidationResult:
    """Represents the result of a validation check."""
    
    def __init__(self, component: str):
        """Initialize validation result.
        
        Args:
            component: Name of the component being validated
        """
        self.component = component
        self.valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        self.status = "pass"
        self.details: Dict[str, Any] = {}
        self.timestamp = datetime.utcnow().isoformat()
    
    def add_error(self, message: str) -> None:
        """Add an error message and mark validation as failed."""
        self.errors.append(message)
        self.valid = False
        self.status = "fail"
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        if self.valid and not self.status == "fail":
            self.status = "warn"
    
    def add_info(self, message: str) -> None:
        """Add an informational message."""
        self.info.append(message)
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "component": self.component,
            "valid": self.valid,
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "details": self.details,
            "timestamp": self.timestamp
        }


class ValidationManager:
    """Manages system validation and health checks with lifecycle awareness."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = ValidationManager()
        return cls._instance
    
    def __init__(self):
        """Initialize validation manager."""
        self.storage = StatisticsStorageManager.get_instance()
        self.storage_validator = StorageValidator(self.storage.base_path)
        self.last_validation_time = 0
        self.validation_results: Dict[str, ValidationResult] = {}
        self.overall_status = "unknown"
        self.hass = None
        self.current_stage = ValidationStage.INITIALIZING
        
        # Scheduled tasks and callbacks
        self._scheduled_validations = {}
        self._deferred_validations = []
        self._startup_validation_scheduled = False
        
        # Define required files
        self.required_files = [
            "meta.json",
            "entity_stats.json",
            "patterns.json",
            "clusters.json"
        ]
        
        # Define grace periods
        self.entity_grace_period_days = 7  # Keep removed entity data for 7 days
        
        # Initialize validation result cache
        for component in ["file_structure", "entity_consistency", "clusters", 
                          "scan_intervals", "data_quality", "overall"]:
            self.validation_results[component] = ValidationResult(component)
    
    def set_hass(self, hass: HomeAssistant) -> None:
        """Set Home Assistant instance for async operations.
        
        Args:
            hass: Home Assistant instance
        """
        self.hass = hass
        if self.storage_validator:
            self.storage_validator.set_hass(hass)
            
        # Register with HA lifecycle events
        if hasattr(hass, 'bus'):
            # Listen for the start event to know when HA is fully started
            hass.bus.async_listen_once(
                'homeassistant_started', 
                self._async_handle_ha_started
            )
            
            # Register for stop event to cleanup
            hass.bus.async_listen_once(
                'homeassistant_stop',
                self._async_handle_ha_stop
            )
        
        # Check if required files exist, create them if missing
        # Use the async method to avoid blocking
        hass.async_create_task(self.async_initialize_storage_if_needed())
    
    async def _async_handle_ha_started(self, _):
        """Handle Home Assistant fully started event."""
        _LOGGER.info("Home Assistant started - switching to READY validation stage")
        self.current_stage = ValidationStage.READY
        
        # Process any deferred validations
        if self._deferred_validations:
            _LOGGER.info("Processing %d deferred validation tasks", len(self._deferred_validations))
            
            # Schedule the first full validation after 10 seconds
            if not self._startup_validation_scheduled:
                self._startup_validation_scheduled = True
                async_call_later(self.hass, 10, self._run_startup_validation)
    
    async def _run_startup_validation(self, _=None):
        """Run startup validation after HA is fully started."""
        _LOGGER.info("Running startup validation")
        
        # Get current entity IDs - fetch all entity IDs from Home Assistant
        current_entity_ids = []
        if self.hass and hasattr(self.hass, 'states'):
            current_entity_ids = list(self.hass.states.async_entity_ids())
            
        # Run validation in executor
        await self.async_validate_all(current_entity_ids)
        
        _LOGGER.info("Startup validation completed with status: %s", self.overall_status)
    
    async def _async_handle_ha_stop(self, _):
        """Handle Home Assistant stopping."""
        _LOGGER.info("Home Assistant stopping - cleaning up validation tasks")
        self.current_stage = ValidationStage.SHUTDOWN
        
        # Cancel any pending validations
        for validation_name, cancellable in self._scheduled_validations.items():
            if isinstance(cancellable, asyncio.Task) and not cancellable.done():
                _LOGGER.debug("Cancelling pending validation: %s", validation_name)
                cancellable.cancel()
    
    async def async_initialize_storage_if_needed(self):
        """Initialize storage files asynchronously if they don't exist."""
        # Check if base directory exists and required files are present
        missing_files = []
        
        # Run the check in executor
        def _check_files():
            nonlocal missing_files
            for filename in self.required_files:
                file_path = self.storage.base_path / filename
                if not os.path.exists(file_path):
                    missing_files.append(filename)
            return missing_files
            
        missing_files = await self.hass.async_add_executor_job(_check_files)
        
        if missing_files:
            _LOGGER.info("First-time setup: Creating required storage files")
            try:
                result = await self.async_create_missing_files()
                _LOGGER.info("Created files: %s", result["created_files"])
                if result["failed_creations"]:
                    _LOGGER.error("Failed to create some files: %s", result["failed_creations"])
            except Exception as e:
                _LOGGER.error("Failed to initialize storage files: %s", e)
    
    def _initialize_storage_if_needed(self):
        """Initialize storage files if they don't exist (synchronous version).
        
        This is the blocking version - avoid calling from the event loop.
        """
        # Check if base directory exists and required files are present
        missing_files = []
        for filename in self.required_files:
            file_path = self.storage.base_path / filename
            if not file_path.exists():
                missing_files.append(filename)
        
        if missing_files:
            _LOGGER.info("First-time setup: Creating required storage files")
            try:
                result = self.create_missing_files()
                _LOGGER.info("Created files: %s", result["created_files"])
                if result["failed_creations"]:
                    _LOGGER.error("Failed to create some files: %s", result["failed_creations"])
            except Exception as e:
                _LOGGER.error("Failed to initialize storage files: %s", e)
    
    async def async_validate_all(self, current_entity_ids: List[str] = None) -> Dict[str, Any]:
        """Run all validation checks asynchronously.
        
        Args:
            current_entity_ids: List of currently active entity IDs
            
        Returns:
            Dictionary with validation results
        """
        if self.current_stage == ValidationStage.INITIALIZING:
            _LOGGER.info("Deferring validation until Home Assistant is fully started")
            result = {
                "overall_status": "deferred",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Validation deferred until Home Assistant startup is complete"
            }
            # Add to deferred validations list
            self._deferred_validations.append(("validate_all", current_entity_ids))
            return result
            
        # Run in executor to avoid blocking the event loop
        if self.hass:
            return await self.hass.async_add_executor_job(self.validate_all, current_entity_ids)
        
        # Fallback to synchronous if no hass available
        return self.validate_all(current_entity_ids)
    
    def validate_all(self, current_entity_ids: List[str] = None) -> Dict[str, Any]:
        """Run all validation checks (synchronous version).
        
        Args:
            current_entity_ids: List of currently active entity IDs
            
        Returns:
            Dictionary with validation results
        """
        # This method contains blocking I/O - should be run in an executor
        start_time = time.time()
        self.last_validation_time = start_time
        
        # Reset overall status
        self.overall_status = "pass"
        
        # Make sure storage is initialized
        self._initialize_storage_if_needed()
        
        # File structure validation
        file_result = self.validate_file_structure()
        if file_result.status == "fail":
            self.overall_status = "fail"
        elif file_result.status == "warn" and self.overall_status != "fail":
            self.overall_status = "warn"
        
        # Skip other checks if file structure validation failed critically
        if file_result.status == "fail" and len(file_result.errors) > 0:
            _LOGGER.warning(
                "Skipping further validation due to file structure errors: %s",
                file_result.errors
            )
            
            # Attempt to fix the file structure issues automatically
            if any(error.startswith("Required file missing:") for error in file_result.errors):
                _LOGGER.info("Attempting to fix missing files automatically")
                try:
                    result = self.create_missing_files()
                    _LOGGER.info("Created files: %s", result["created_files"])
                    if not result["failed_creations"]:
                        _LOGGER.info("All required files created successfully, validation can proceed")
                        # Re-validate file structure
                        file_result = self.validate_file_structure()
                        if file_result.status != "fail":
                            _LOGGER.info("File structure validation now passed, continuing with validation")
                            # Continue with validation
                        else:
                            # Prepare simplified result
                            self.validation_results["overall"] = ValidationResult("overall")
                            self.validation_results["overall"].status = "fail"
                            self.validation_results["overall"].add_error(
                                "File structure validation still failed after attempting to fix"
                            )
                            self.validation_results["overall"].details = {
                                "file_structure_errors": file_result.errors,
                                "validation_time_ms": int((time.time() - start_time) * 1000)
                            }
                            
                            return self._prepare_validation_report()
                except Exception as e:
                    _LOGGER.error("Failed to fix file structure issues: %s", e)
                    # Prepare simplified result
                    self.validation_results["overall"] = ValidationResult("overall")
                    self.validation_results["overall"].status = "fail"
                    self.validation_results["overall"].add_error(
                        f"Failed to fix file structure issues: {e}"
                    )
                    self.validation_results["overall"].details = {
                        "file_structure_errors": file_result.errors,
                        "validation_time_ms": int((time.time() - start_time) * 1000)
                    }
                    
                    return self._prepare_validation_report()
            else:
                # Prepare simplified result
                self.validation_results["overall"] = ValidationResult("overall")
                self.validation_results["overall"].status = "fail"
                self.validation_results["overall"].add_error(
                    "File structure validation failed, fix these issues first"
                )
                self.validation_results["overall"].details = {
                    "file_structure_errors": file_result.errors,
                    "validation_time_ms": int((time.time() - start_time) * 1000)
                }
                
                return self._prepare_validation_report()
        
        # Entity consistency validation
        if current_entity_ids is not None:
            entity_result = self.validate_entity_consistency(current_entity_ids)
            if entity_result.status == "fail":
                self.overall_status = "fail"
            elif entity_result.status == "warn" and self.overall_status != "fail":
                self.overall_status = "warn"
        else:
            # Skip entity validation if no entity IDs provided
            entity_result = ValidationResult("entity_consistency")
            entity_result.status = "skip"
            entity_result.add_info("Skipped entity validation (no entity IDs provided)")
            self.validation_results["entity_consistency"] = entity_result
        
        # Cluster validation
        cluster_result = self.validate_clusters()
        if cluster_result.status == "fail":
            self.overall_status = "fail"
        elif cluster_result.status == "warn" and self.overall_status != "fail":
            self.overall_status = "warn"
        
        # Scan interval validation
        interval_result = self.validate_scan_intervals()
        if interval_result.status == "fail":
            self.overall_status = "fail"
        elif interval_result.status == "warn" and self.overall_status != "fail":
            self.overall_status = "warn"
        
        # Data quality validation
        quality_result = self.validate_data_quality()
        if quality_result.status == "fail":
            self.overall_status = "fail"
        elif quality_result.status == "warn" and self.overall_status != "fail":
            self.overall_status = "warn"
        
        # Prepare overall result
        overall = ValidationResult("overall")
        overall.status = self.overall_status
        
        if self.overall_status == "fail":
            overall.valid = False
            error_count = sum(len(r.errors) for r in self.validation_results.values())
            overall.add_error(f"System validation failed with {error_count} errors")
        elif self.overall_status == "warn":
            warning_count = sum(len(r.warnings) for r in self.validation_results.values())
            overall.add_warning(f"System validation passed with {warning_count} warnings")
        else:
            overall.add_info("All validation checks passed successfully")
        
        # Add timing information
        overall.details["validation_time_ms"] = int((time.time() - start_time) * 1000)
        
        # Count issues by category
        overall.details["issue_counts"] = {
            "errors": {k: len(v.errors) for k, v in self.validation_results.items() if v.errors},
            "warnings": {k: len(v.warnings) for k, v in self.validation_results.items() if v.warnings}
        }
        
        self.validation_results["overall"] = overall
        
        return self._prepare_validation_report()
    
    async def async_read_json_file(self, file_path: Path) -> Tuple[bool, Dict, str]:
        """Read JSON from file asynchronously."""
        if self.hass:
            return await self.hass.async_add_executor_job(self._blocking_read_json_file, file_path)
        return self._blocking_read_json_file(file_path)
    
    def _blocking_read_json_file(self, file_path: Path) -> Tuple[bool, Dict, str]:
        """Read JSON from file - blocking version.
        
        Args:
            file_path: Path to file
            
        Returns:
            Tuple of (success, data, error_message)
        """
        try:
            if not os.path.exists(file_path):
                return False, {}, f"File not found: {file_path}"
            
            with open(file_path, "r") as f:
                data = json.load(f)
            return True, data, ""
        except json.JSONDecodeError as e:
            return False, {}, f"JSON decode error: {e}"
        except IOError as e:
            return False, {}, f"IO error: {e}"
        except Exception as e:
            return False, {}, f"Unexpected error: {e}"
    
    async def async_validate_file_structure(self) -> ValidationResult:
        """Validate file structure asynchronously."""
        # Skip during initialization to avoid blocking the event loop
        if self.current_stage == ValidationStage.INITIALIZING:
            result = ValidationResult("file_structure")
            result.status = "deferred"
            result.add_info("File structure validation deferred until startup is complete")
            return result
            
        if self.hass:
            return await self.hass.async_add_executor_job(self.validate_file_structure)
        return self.validate_file_structure()
    
    def validate_file_structure(self) -> ValidationResult:
        """Validate file structure.
        
        Returns:
            ValidationResult object
        """
        # This method contains blocking I/O and should be run in an executor
        result = ValidationResult("file_structure")
        
        # Check if base directory exists
        if not self.storage.base_path.exists():
            result.add_error(f"Storage directory not found: {self.storage.base_path}")
            result.details["recommended_action"] = "Create directory and initialize storage"
            self.validation_results["file_structure"] = result
            return result
        
        # Check required files
        missing_files = []
        for filename in self.required_files:
            file_path = self.storage.base_path / filename
            if not file_path.exists():
                missing_files.append(filename)
        
        if missing_files:
            for filename in missing_files:
                result.add_error(f"Required file missing: {filename}")
            result.details["missing_files"] = missing_files
            result.details["recommended_action"] = "Initialize storage or restore from backup"
            self.validation_results["file_structure"] = result
            return result
        
        # Validate meta file
        meta_valid, meta_errors = self.storage_validator.validate_meta()
        if not meta_valid:
            for error in meta_errors:
                result.add_error(f"Meta file error: {error}")
            result.details["meta_errors"] = meta_errors
        
        # Validate all files
        validation_results = self.storage_validator.validate_all_files()
        
        # Check if the overall validation failed
        if not validation_results["overall_valid"]:
            # Add file-specific errors or warnings
            critical_errors = False
            
            for file_type, file_result in validation_results["files"].items():
                if not file_result["valid"]:
                    # Check if this is a required file or not
                    file_path = f"{file_type}.json"
                    if file_path in self.required_files:
                        # Required file has errors - add as error
                        for error in file_result["errors"]:
                            result.add_error(f"{file_type} error: {error}")
                        critical_errors = True
                    elif "No schema defined" in str(file_result["errors"]):
                        # Non-required file with no schema - add as warning
                        for error in file_result["errors"]:
                            result.add_warning(f"{file_type} warning: {error}")
                    else:
                        # Other errors in non-required file - add as warning
                        for error in file_result["errors"]:
                            result.add_warning(f"{file_type} warning: {error}")
            
            result.details["validation_results"] = validation_results
            
            # If there are critical errors in required files, return early
            if critical_errors and result.errors:
                self.validation_results["file_structure"] = result
                return result
        else:
            result.add_info("All files passed schema validation")
        
        # Check version consistency
        success, meta_data, error = self._blocking_read_json_file(self.storage.meta_path)
        if success:
            meta_version = meta_data.get("version")
            
            # Check versions in other files
            version_inconsistencies = []
            
            for file_type, filename in meta_data.get("files", {}).items():
                file_path = self.storage.base_path / filename
                success, file_data, _ = self._blocking_read_json_file(file_path)
                if success:
                    if isinstance(file_data, dict) and "version" in file_data:
                        file_version = file_data["version"]
                        if file_version != meta_version:
                            version_inconsistencies.append(
                                f"{file_type}: expected {meta_version}, got {file_version}"
                            )
            
            if version_inconsistencies:
                for inconsistency in version_inconsistencies:
                    result.add_warning(f"Version inconsistency: {inconsistency}")
                result.details["version_inconsistencies"] = version_inconsistencies
            else:
                result.add_info(f"All files have consistent version: {meta_version}")
                result.details["version"] = meta_version
        
        # Add file size information
        try:
            file_sizes = {}
            total_size = 0
            for file_type, filename in meta_data.get("files", {}).items():
                file_path = self.storage.base_path / filename
                if file_path.exists():
                    size = file_path.stat().st_size
                    file_sizes[file_type] = size
                    total_size += size
            
            result.details["file_sizes_bytes"] = file_sizes
            result.details["total_size_bytes"] = total_size
            result.details["total_size_kb"] = round(total_size / 1024, 1)
            
        except Exception as e:
            result.add_warning(f"Failed to get file sizes: {e}")
        
        self.validation_results["file_structure"] = result
        return result
    
    async def async_validate_entity_consistency(self, current_entity_ids: List[str]) -> ValidationResult:
        """Validate entity consistency asynchronously."""
        if self.hass:
            return await self.hass.async_add_executor_job(
                self.validate_entity_consistency, current_entity_ids
            )
        return self.validate_entity_consistency(current_entity_ids)
    
    def validate_entity_consistency(self, current_entity_ids: List[str]) -> ValidationResult:
        """Validate entity consistency between storage and HA.
        
        Args:
            current_entity_ids: List of current entity IDs in Home Assistant
            
        Returns:
            ValidationResult object
        """
        # This method contains blocking I/O and should be run in an executor
        result = ValidationResult("entity_consistency")
        
        # Get stored entity IDs
        stored_entities = set(self.storage.get_entity_stats().keys())
        current_entities = set(current_entity_ids)
        
        # Find orphaned entities (in storage but not in HA)
        orphaned_entities = stored_entities - current_entities
        
        # Find new entities (in HA but not in storage)
        new_entities = current_entities - stored_entities
        
        if orphaned_entities:
            # Check if entities are within grace period
            entities_to_remove = []
            entities_in_grace = []
            
            entity_stats = self.storage.get_entity_stats()
            now = datetime.utcnow()
            
            for entity_id in orphaned_entities:
                entity_data = entity_stats.get(entity_id, {})
                last_updated_str = entity_data.get("last_updated")
                
                try:
                    if last_updated_str:
                        last_updated = datetime.fromisoformat(last_updated_str)
                        days_inactive = (now - last_updated).days
                        
                        if days_inactive > self.entity_grace_period_days:
                            entities_to_remove.append(entity_id)
                        else:
                            entities_in_grace.append({
                                "entity_id": entity_id,
                                "days_inactive": days_inactive,
                                "days_until_removal": self.entity_grace_period_days - days_inactive
                            })
                    else:
                        # No last_updated info, consider for removal
                        entities_to_remove.append(entity_id)
                except (ValueError, TypeError):
                    # Invalid date format, consider for removal
                    entities_to_remove.append(entity_id)
            
            if entities_to_remove:
                result.add_warning(
                    f"Found {len(entities_to_remove)} orphaned entities that should be removed"
                )
                result.details["entities_to_remove"] = entities_to_remove
            
            if entities_in_grace:
                result.add_info(
                    f"Found {len(entities_in_grace)} recently removed entities in grace period"
                )
                result.details["entities_in_grace_period"] = entities_in_grace
        
        if new_entities:
            result.add_info(f"Found {len(new_entities)} new entities to analyze")
            result.details["new_entities"] = list(new_entities)
        
        # Check if entity references in clusters are valid
        clusters = self.storage.get_clusters()
        invalid_cluster_references = []
        
        for cluster_id, cluster_entities in clusters.items():
            for entity_id in cluster_entities:
                if entity_id not in current_entities and entity_id not in orphaned_entities:
                    invalid_cluster_references.append({
                        "cluster_id": cluster_id,
                        "entity_id": entity_id
                    })
        
        if invalid_cluster_references:
            result.add_error(
                f"Found {len(invalid_cluster_references)} invalid entity references in clusters"
            )
            result.details["invalid_cluster_references"] = invalid_cluster_references
        
        # Summary statistics
        result.details["entity_counts"] = {
            "total_current": len(current_entities),
            "total_stored": len(stored_entities),
            "new": len(new_entities),
            "orphaned": len(orphaned_entities),
            "common": len(current_entities & stored_entities)
        }
        
        # Overall assessment
        if not result.errors and not result.warnings:
            result.add_info("Entity consistency check passed")
        
        self.validation_results["entity_consistency"] = result
        return result
    
    async def async_validate_clusters(self) -> ValidationResult:
        """Validate cluster assignments asynchronously."""
        if self.hass:
            return await self.hass.async_add_executor_job(self.validate_clusters)
        return self.validate_clusters()
    
    def validate_clusters(self) -> ValidationResult:
        """Validate cluster assignments.
        
        Returns:
            ValidationResult object
        """
        # This method contains blocking I/O and should be run in an executor
        result = ValidationResult("clusters")
        
        # Get clusters
        clusters = self.storage.get_clusters()
        if not clusters:
            result.add_warning("No cluster data found")
            result.details["recommended_action"] = "Run correlation analysis"
            self.validation_results["clusters"] = result
            return result
        
        # Get entity stats
        entity_stats = self.storage.get_entity_stats()
        all_entities = set(entity_stats.keys())
        
        # Check cluster structure
        empty_clusters = []
        undersized_clusters = []  # Less than 2 members
        unnamed_clusters = []
        
        # Check if all entities are assigned to clusters
        clustered_entities = set()
        for cluster_id, members in clusters.items():
            clustered_entities.update(members)
            
            if not members:
                empty_clusters.append(cluster_id)
            elif len(members) < 2:
                undersized_clusters.append({
                    "cluster_id": cluster_id,
                    "size": len(members),
                    "members": members
                })
            
            # Check if cluster has proper naming
            if cluster_id.startswith("cluster_") and cluster_id[8:].isdigit():
                # This is a default naming scheme, it's okay
                pass
            elif "unnamed" in cluster_id.lower() or "default" in cluster_id.lower():
                unnamed_clusters.append(cluster_id)
        
        # Find unclustered entities
        unclustered_entities = all_entities - clustered_entities
        
        # Add warnings for issues
        if empty_clusters:
            result.add_warning(f"Found {len(empty_clusters)} empty clusters")
            result.details["empty_clusters"] = empty_clusters
        
        if undersized_clusters:
            result.add_warning(f"Found {len(undersized_clusters)} undersized clusters (fewer than 2 members)")
            result.details["undersized_clusters"] = undersized_clusters
        
        if unnamed_clusters:
            result.add_warning(f"Found {len(unnamed_clusters)} unnamed clusters")
            result.details["unnamed_clusters"] = unnamed_clusters
        
        if unclustered_entities:
            result.add_warning(f"Found {len(unclustered_entities)} unclustered entities")
            result.details["unclustered_entities"] = list(unclustered_entities)
            result.details["recommended_action"] = "Run correlation analysis or assign to default cluster"
        
        # Summary statistics
        result.details["cluster_stats"] = {
            "total_clusters": len(clusters),
            "total_clustered_entities": len(clustered_entities),
            "unclustered_entities": len(unclustered_entities),
            "average_cluster_size": (
                len(clustered_entities) / len(clusters) if clusters else 0
            )
        }
        
        # Add distribution information
        cluster_sizes = [len(members) for members in clusters.values()]
        if cluster_sizes:
            result.details["cluster_sizes"] = {
                "min": min(cluster_sizes),
                "max": max(cluster_sizes),
                "avg": sum(cluster_sizes) / len(cluster_sizes)
            }
        
        # Overall assessment
        if not result.errors and not result.warnings:
            result.add_info("Cluster validation passed successfully")
        
        self.validation_results["clusters"] = result
        return result
    
    async def async_validate_scan_intervals(self) -> ValidationResult:
        """Validate scan interval assignments asynchronously."""
        if self.hass:
            return await self.hass.async_add_executor_job(self.validate_scan_intervals)
        return self.validate_scan_intervals()
    
    def validate_scan_intervals(self) -> ValidationResult:
        """Validate scan interval assignments.
        
        Returns:
            ValidationResult object
        """
        # This method contains blocking I/O and should be run in an executor
        result = ValidationResult("scan_intervals")
        
        # Get entity stats
        entity_stats = self.storage.get_entity_stats()
        
        # Define valid scan interval range
        min_valid_interval = 1  # 1 second
        max_valid_interval = 3600  # 1 hour
        
        # Check each entity for valid scan interval
        missing_intervals = []
        invalid_intervals = []
        stale_optimizations = []
        
        now = datetime.utcnow()
        stale_threshold = 7  # days
        
        for entity_id, stats in entity_stats.items():
            recommended_interval = stats.get("recommended_scan_interval")
            
            if recommended_interval is None:
                missing_intervals.append(entity_id)
                continue
            
            # Check if interval is in valid range
            if recommended_interval < min_valid_interval or recommended_interval > max_valid_interval:
                invalid_intervals.append({
                    "entity_id": entity_id,
                    "interval": recommended_interval
                })
            
            # Check if optimization is stale
            last_updated_str = stats.get("last_updated")
            if last_updated_str:
                try:
                    last_updated = datetime.fromisoformat(last_updated_str)
                    days_since_update = (now - last_updated).days
                    
                    if days_since_update > stale_threshold:
                        stale_optimizations.append({
                            "entity_id": entity_id,
                            "days_since_update": days_since_update
                        })
                except (ValueError, TypeError):
                    # Invalid date format
                    pass
        
        # Add warnings for issues
        if missing_intervals:
            result.add_warning(f"Found {len(missing_intervals)} entities without scan intervals")
            result.details["missing_intervals"] = missing_intervals
        
        if invalid_intervals:
            result.add_warning(f"Found {len(invalid_intervals)} entities with invalid scan intervals")
            result.details["invalid_intervals"] = invalid_intervals
        
        if stale_optimizations:
            result.add_warning(f"Found {len(stale_optimizations)} entities with stale optimizations")
            result.details["stale_optimizations"] = stale_optimizations
            result.details["recommended_action"] = "Run scan interval optimization"
        
        # Get interval distribution
        intervals = [
            stats.get("recommended_scan_interval", 0) 
            for stats in entity_stats.values()
            if stats.get("recommended_scan_interval") is not None
        ]
        
        if intervals:
            # Calculate distribution statistics
            result.details["interval_stats"] = {
                "min": min(intervals),
                "max": max(intervals),
                "avg": sum(intervals) / len(intervals),
                "count": len(intervals)
            }
            
            # Group by common intervals
            interval_distribution = {}
            for interval in intervals:
                interval_distribution[interval] = interval_distribution.get(interval, 0) + 1
            
            # Sort by frequency
            top_intervals = sorted(
                interval_distribution.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            result.details["common_intervals"] = [
                {"interval": interval, "count": count} 
                for interval, count in top_intervals
            ]
        
        # Overall assessment
        if not result.errors and not result.warnings:
            result.add_info("Scan interval validation passed successfully")
        
        self.validation_results["scan_intervals"] = result
        return result
    
    async def async_validate_data_quality(self) -> ValidationResult:
        """Validate data quality asynchronously."""
        if self.hass:
            return await self.hass.async_add_executor_job(self.validate_data_quality)
        return self.validate_data_quality()
    
    def validate_data_quality(self) -> ValidationResult:
        """Validate data quality for analysis.
        
        Returns:
            ValidationResult object
        """
        # This method contains blocking I/O and should be run in an executor
        result = ValidationResult("data_quality")
        
        # Get entity stats
        entity_stats = self.storage.get_entity_stats()
        
        # Define thresholds
        min_change_count = 5  # Minimum number of changes for reliable analysis
        black_hole_threshold = 1  # Max changes per 24h to consider a black hole
        
        # Check entities for data quality issues
        insufficient_data = []
        black_holes = []
        
        for entity_id, stats in entity_stats.items():
            # Check for insufficient data
            change_count = stats.get("change_count_24h", 0)
            
            if change_count < min_change_count:
                insufficient_data.append({
                    "entity_id": entity_id,
                    "change_count": change_count
                })
            
            # Check for black holes
            if change_count <= black_hole_threshold:
                is_blackhole = stats.get("is_blackhole", False)
                if is_blackhole:
                    black_holes.append({
                        "entity_id": entity_id,
                        "change_count": change_count
                    })
        
        # Add warnings for issues
        if insufficient_data:
            result.add_warning(
                f"Found {len(insufficient_data)} entities with insufficient data for analysis"
            )
            result.details["insufficient_data"] = insufficient_data
        
        if black_holes:
            result.add_warning(f"Found {len(black_holes)} black hole entities")
            result.details["black_holes"] = black_holes
            result.details["recommended_action"] = "Consider excluding black holes from clustering"
        
        # Check data age
        patterns_data = self.storage.get_patterns()
        if patterns_data:
            meta = patterns_data.get("meta", {})
            last_detection_str = meta.get("last_detection")
            
            if last_detection_str:
                try:
                    last_detection = datetime.fromisoformat(last_detection_str)
                    now = datetime.utcnow()
                    hours_since_detection = (now - last_detection).total_seconds() / 3600
                    
                    result.details["pattern_age_hours"] = round(hours_since_detection, 1)
                    
                    if hours_since_detection > 24:
                        result.add_warning(
                            f"Pattern data is {round(hours_since_detection, 1)} hours old"
                        )
                except (ValueError, TypeError):
                    # Invalid date format
                    pass
        
        # Overall assessment
        if not result.errors and not result.warnings:
            result.add_info("Data quality validation passed successfully")
        elif black_holes and len(black_holes) > len(entity_stats) * 0.3:
            # If more than 30% are black holes, add an error
            result.add_error(
                f"High proportion of black holes: {len(black_holes)}/{len(entity_stats)} entities"
            )
        
        self.validation_results["data_quality"] = result
        return result
    
    def _prepare_validation_report(self) -> Dict[str, Any]:
        """Prepare validation report for external use."""
        return {
            "overall_status": self.overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "validation_stage": self.current_stage.value,
            "results": {
                component: result.as_dict()
                for component, result in self.validation_results.items()
            }
        }
    
    async def async_fix_issues(self, issues_to_fix: List[str] = None) -> Dict[str, Any]:
        """Fix validation issues automatically asynchronously.
        
        Args:
            issues_to_fix: List of issue types to fix, or None for all fixable issues
            
        Returns:
            Dictionary with fix results
        """
        # Skip during initialization
        if self.current_stage == ValidationStage.INITIALIZING:
            return {
                "status": "deferred",
                "message": "Issue fixes deferred until Home Assistant startup is complete"
            }
            
        if self.hass:
            return await self.hass.async_add_executor_job(self.fix_issues, issues_to_fix)
        # Fallback to synchronous if no hass available
        return self.fix_issues(issues_to_fix)
    
    def fix_issues(self, issues_to_fix: List[str] = None) -> Dict[str, Any]:
        """Fix validation issues automatically.
        
        Args:
            issues_to_fix: List of issue types to fix, or None for all fixable issues
            
        Returns:
            Dictionary with fix results
        """
        results = {
            "fixed_issues": [],
            "failed_fixes": [],
            "skipped_issues": []
        }
        
        # Default to fixing all issues if none specified
        if issues_to_fix is None:
            issues_to_fix = ["orphaned_entities", "empty_clusters", "undersized_clusters"]
        
        # Fix orphaned entities
        if "orphaned_entities" in issues_to_fix:
            try:
                entity_result = self.validation_results.get("entity_consistency")
                if entity_result and "entities_to_remove" in entity_result.details:
                    entities_to_remove = entity_result.details["entities_to_remove"]
                    
                    if entities_to_remove:
                        # Get current entity stats
                        entity_stats = self.storage.get_entity_stats()
                        
                        # Remove orphaned entities
                        for entity_id in entities_to_remove:
                            if entity_id in entity_stats:
                                del entity_stats[entity_id]
                        
                        # Save updated entity stats
                        self.storage._entity_stats_cache = entity_stats
                        self.storage.save_entity_stats("dummy", {}, force=True)  # Force save
                        
                        results["fixed_issues"].append({
                            "type": "orphaned_entities",
                            "count": len(entities_to_remove),
                            "details": entities_to_remove
                        })
                    else:
                        results["skipped_issues"].append({
                            "type": "orphaned_entities",
                            "reason": "No orphaned entities to remove"
                        })
                else:
                    results["skipped_issues"].append({
                        "type": "orphaned_entities",
                        "reason": "No validation result available"
                    })
            except Exception as e:
                results["failed_fixes"].append({
                    "type": "orphaned_entities",
                    "error": str(e)
                })
        
        # Fix empty clusters
        if "empty_clusters" in issues_to_fix:
            try:
                cluster_result = self.validation_results.get("clusters")
                if cluster_result and "empty_clusters" in cluster_result.details:
                    empty_clusters = cluster_result.details["empty_clusters"]
                    
                    if empty_clusters:
                        # Get current clusters
                        clusters = self.storage.get_clusters()
                        
                        # Remove empty clusters
                        for cluster_id in empty_clusters:
                            if cluster_id in clusters:
                                del clusters[cluster_id]
                        
                        # Save updated clusters
                        self.storage.save_clusters(clusters, force=True)
                        
                        results["fixed_issues"].append({
                            "type": "empty_clusters",
                            "count": len(empty_clusters),
                            "details": empty_clusters
                        })
                    else:
                        results["skipped_issues"].append({
                            "type": "empty_clusters",
                            "reason": "No empty clusters to remove"
                        })
                else:
                    results["skipped_issues"].append({
                        "type": "empty_clusters",
                        "reason": "No validation result available"
                    })
            except Exception as e:
                results["failed_fixes"].append({
                    "type": "empty_clusters",
                    "error": str(e)
                })
        
        # Fix undersized clusters
        if "undersized_clusters" in issues_to_fix:
            try:
                cluster_result = self.validation_results.get("clusters")
                if cluster_result and "undersized_clusters" in cluster_result.details:
                    undersized_clusters = cluster_result.details["undersized_clusters"]
                    
                    if undersized_clusters:
                        # Get current clusters
                        clusters = self.storage.get_clusters()
                        
                        # Get or create default cluster
                        if "default_cluster" not in clusters:
                            clusters["default_cluster"] = []
                        
                        # Move entities from undersized clusters to default
                        moved_entities = []
                        for cluster_info in undersized_clusters:
                            cluster_id = cluster_info["cluster_id"]
                            if cluster_id in clusters and cluster_id != "default_cluster":
                                # Move members to default cluster
                                members = clusters[cluster_id]
                                clusters["default_cluster"].extend(members)
                                moved_entities.extend(members)
                                
                                # Remove undersized cluster
                                del clusters[cluster_id]
                        
                        # Save updated clusters
                        self.storage.save_clusters(clusters, force=True)
                        
                        results["fixed_issues"].append({
                            "type": "undersized_clusters",
                            "count": len(undersized_clusters),
                            "moved_entities": moved_entities
                        })
                    else:
                        results["skipped_issues"].append({
                            "type": "undersized_clusters",
                            "reason": "No undersized clusters to fix"
                        })
                else:
                    results["skipped_issues"].append({
                        "type": "undersized_clusters",
                        "reason": "No validation result available"
                    })
            except Exception as e:
                results["failed_fixes"].append({
                    "type": "undersized_clusters",
                    "error": str(e)
                })
        
        return results
    
    async def async_create_missing_files(self) -> Dict[str, Any]:
        """Create missing required files asynchronously."""
        if self.hass:
            return await self.hass.async_add_executor_job(self.create_missing_files)
        return self.create_missing_files()
    
    def create_missing_files(self) -> Dict[str, Any]:
        """Create missing required files with empty structures.
        
        Returns:
            Dictionary with creation results
        """
        results = {
            "created_files": [],
            "failed_creations": []
        }
        
        # Ensure base directory exists
        try:
            self.storage.base_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            results["failed_creations"].append({
                "file": "base_directory",
                "error": str(e)
            })
            return results
        
        # Create meta.json if missing
        if not self.storage.meta_path.exists():
            try:
                meta_data = {
                    "version": "1.0.0",
                    "created": datetime.utcnow().isoformat(),
                    "last_update": datetime.utcnow().isoformat(),
                    "files": {
                        "entity_stats": "entity_stats.json",
                        "patterns": "patterns.json",
                        "clusters": "clusters.json",
                        "correlation": "correlation_matrix.json",
                        "interval_history": "interval_history.json"
                    }
                }
                
                with open(self.storage.meta_path, "w") as f:
                    json.dump(meta_data, f, indent=2)
                
                results["created_files"].append("meta.json")
            except Exception as e:
                results["failed_creations"].append({
                    "file": "meta.json",
                    "error": str(e)
                })
        
        # Create entity_stats.json if missing
        entity_stats_path = self.storage.base_path / "entity_stats.json"
        if not entity_stats_path.exists():
            try:
                with open(entity_stats_path, "w") as f:
                    json.dump({}, f, indent=2)
                
                results["created_files"].append("entity_stats.json")
            except Exception as e:
                results["failed_creations"].append({
                    "file": "entity_stats.json",
                    "error": str(e)
                })
        
        # Create patterns.json if missing
        patterns_path = self.storage.base_path / "patterns.json"
        if not patterns_path.exists():
            try:
                patterns_data = {
                    "current_pattern": None,
                    "pattern_transition_time": None,
                    "patterns": {},
                    "meta": {
                        "last_detection": datetime.utcnow().isoformat(),
                        "entity_count": 0
                    }
                }
                
                with open(patterns_path, "w") as f:
                    json.dump(patterns_data, f, indent=2)
                
                results["created_files"].append("patterns.json")
            except Exception as e:
                results["failed_creations"].append({
                    "file": "patterns.json",
                    "error": str(e)
                })
        
        # Create clusters.json if missing
        clusters_path = self.storage.base_path / "clusters.json"
        if not clusters_path.exists():
            try:
                with open(clusters_path, "w") as f:
                    json.dump({"default_cluster": []}, f, indent=2)
                
                results["created_files"].append("clusters.json")
            except Exception as e:
                results["failed_creations"].append({
                    "file": "clusters.json",
                    "error": str(e)
                })
        
        # Create correlation_matrix.json if missing
        correlation_path = self.storage.base_path / "correlation_matrix.json"
        if not correlation_path.exists():
            try:
                with open(correlation_path, "w") as f:
                    json.dump({}, f, indent=2)
                
                results["created_files"].append("correlation_matrix.json")
            except Exception as e:
                results["failed_creations"].append({
                    "file": "correlation_matrix.json",
                    "error": str(e)
                })
        
        # Create interval_history.json if missing
        interval_history_path = self.storage.base_path / "interval_history.json"
        if not interval_history_path.exists():
            try:
                with open(interval_history_path, "w") as f:
                    json.dump({}, f, indent=2)
                
                results["created_files"].append("interval_history.json")
            except Exception as e:
                results["failed_creations"].append({
                    "file": "interval_history.json",
                    "error": str(e)
                })
        
        return results

# Create singleton instance
VALIDATION_MANAGER = ValidationManager.get_instance()