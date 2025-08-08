"""
Data validation system for Modbus optimization data.

This module provides validation and integrity checks for stored data
to prevent corruption and ensure consistency.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)

class ValidationError(Exception):
    """Validation error for Modbus optimization data."""
    pass

class StorageValidator:
    """Validator for persistent storage data."""
    
    def __init__(self, base_path: Path):
        """Initialize the validator.
        
        Args:
            base_path: Base directory for storage files
        """
        self.base_path = base_path
        self.meta_path = base_path / "meta.json"
        self.hass = None
        
        # Schema definitions
        self.schemas = {
            "meta": {
                "required_fields": ["version", "created", "last_update", "files"],
                "field_types": {
                    "version": str,
                    "created": str,
                    "last_update": str,
                    "files": dict
                }
            },
            "entity_stats": {
                "required_fields": [],  # Dynamic content
                "field_types": {}  # Dynamic content
            },
            "patterns": {
                "required_fields": ["current_pattern", "patterns", "meta"],
                "field_types": {
                    "current_pattern": (int, type(None)),
                    "patterns": dict,
                    "meta": dict
                }
            },
            "clusters": {
                "required_fields": [],  # Dynamic content
                "field_types": {}  # Dynamic content
            },
            "correlation_matrix": {
                "required_fields": [],  # Dynamic content
                "field_types": {}  # Dynamic content
            },
            # Add a schema for "correlation" to fix the validation error
            "correlation": {
                "required_fields": [],  # Dynamic content
                "field_types": {}  # Dynamic content
            },
            "interval_history": {
                "required_fields": [],  # Dynamic content
                "field_types": {}  # Dynamic content
            }
        }
    
    def set_hass(self, hass):
        """Set Home Assistant instance for async operations."""
        self.hass = hass
    
    def validate_file(self, file_type: str, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a data structure against its schema.
        
        Args:
            file_type: Type of file/data to validate
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        schema = self.schemas.get(file_type)
        
        if not schema:
            errors.append(f"No schema defined for {file_type}")
            return False, errors
        
        # Check required fields
        for field in schema["required_fields"]:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        for field, expected_type in schema["field_types"].items():
            if field in data:
                # Handle case where multiple types are valid
                if isinstance(expected_type, tuple):
                    if not isinstance(data[field], expected_type):
                        errors.append(f"Field {field} has wrong type: {type(data[field]).__name__}, expected one of {[t.__name__ for t in expected_type]}")
                elif not isinstance(data[field], expected_type):
                    errors.append(f"Field {field} has wrong type: {type(data[field]).__name__}, expected {expected_type.__name__}")
        
        # Extra validation for specific file types
        if file_type == "patterns" and "patterns" in data:
            # Check pattern structure
            for pattern_id, pattern_info in data["patterns"].items():
                if not isinstance(pattern_info, dict):
                    errors.append(f"Pattern {pattern_id} is not a dictionary")
                    continue
                
                if "defining_criteria" not in pattern_info:
                    errors.append(f"Pattern {pattern_id} is missing defining_criteria")
        
        return len(errors) == 0, errors
    
    async def async_validate_meta(self) -> Tuple[bool, List[str]]:
        """Validate meta file asynchronously."""
        if self.hass:
            return await self.hass.async_add_executor_job(self.validate_meta)
        return self.validate_meta()
    
    def validate_meta(self) -> Tuple[bool, List[str]]:
        """Validate the meta file."""
        errors = []
        
        try:
            if not self.meta_path.exists():
                errors.append(f"Meta file missing: {self.meta_path}")
                return False, errors
                
            with open(self.meta_path, "r") as f:
                meta_data = json.load(f)
                
            # Validate meta file structure
            is_valid, validation_errors = self.validate_file("meta", meta_data)
            errors.extend(validation_errors)
            
            # Check if referenced files exist
            if "files" in meta_data:
                for file_type, filename in meta_data["files"].items():
                    file_path = self.base_path / filename
                    if not file_path.exists():
                        errors.append(f"Referenced file missing: {file_path}")
            
            return len(errors) == 0, errors
            
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in meta file: {str(e)}")
            return False, errors
            
        except Exception as e:
            errors.append(f"Error validating meta file: {str(e)}")
            return False, errors
    
    async def async_validate_all_files(self) -> Dict[str, Any]:
        """Validate all storage files asynchronously."""
        if self.hass:
            return await self.hass.async_add_executor_job(self.validate_all_files)
        return self.validate_all_files()
    
    def validate_all_files(self) -> Dict[str, Any]:
        """Validate all storage files.
        
        Returns:
            Dictionary with validation results for each file
        """
        results = {
            "overall_valid": True,
            "files": {}
        }
        
        # Start with meta file
        meta_valid, meta_errors = self.validate_meta()
        results["files"]["meta"] = {
            "valid": meta_valid,
            "errors": meta_errors
        }
        
        if not meta_valid:
            results["overall_valid"] = False
            return results
        
        # Get list of files from meta
        try:
            with open(self.meta_path, "r") as f:
                meta_data = json.load(f)
                
            for file_type, filename in meta_data.get("files", {}).items():
                file_path = self.base_path / filename
                
                if file_path.exists():
                    try:
                        with open(file_path, "r") as f:
                            file_data = json.load(f)
                            
                        is_valid, validation_errors = self.validate_file(file_type, file_data)
                        results["files"][file_type] = {
                            "valid": is_valid,
                            "errors": validation_errors,
                            "size_bytes": file_path.stat().st_size
                        }
                        
                        if not is_valid:
                            results["overall_valid"] = False
                            
                    except json.JSONDecodeError as e:
                        results["files"][file_type] = {
                            "valid": False,
                            "errors": [f"Invalid JSON: {str(e)}"],
                            "size_bytes": file_path.stat().st_size if file_path.exists() else 0
                        }
                        results["overall_valid"] = False
                        
                    except Exception as e:
                        results["files"][file_type] = {
                            "valid": False,
                            "errors": [f"Error validating file: {str(e)}"],
                            "size_bytes": file_path.stat().st_size if file_path.exists() else 0
                        }
                        results["overall_valid"] = False
                else:
                    results["files"][file_type] = {
                        "valid": False,
                        "errors": [f"File does not exist: {filename}"],
                        "size_bytes": 0
                    }
                    results["overall_valid"] = False
                    
        except Exception as e:
            results["error"] = f"Failed to process files: {str(e)}"
            results["overall_valid"] = False
        
        return results
    
    async def async_validate_entity_references(self, current_entity_ids: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """Validate entity references asynchronously."""
        if self.hass:
            return await self.hass.async_add_executor_job(
                self.validate_entity_references, current_entity_ids)
        return self.validate_entity_references(current_entity_ids)
    
    def validate_entity_references(self, current_entity_ids: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """Validate entity references in stored data.
        
        Args:
            current_entity_ids: List of currently valid entity IDs
            
        Returns:
            Tuple of (is_valid, issues_dict)
        """
        issues = {
            "orphaned_entities": [],
            "missing_entities": [],
            "entity_stats_file": None,
            "clusters_file": None
        }
        
        # Check if entity stats file exists
        entity_stats_path = self.base_path / "entity_stats.json"
        if entity_stats_path.exists():
            try:
                with open(entity_stats_path, "r") as f:
                    entity_stats = json.load(f)
                    
                # Find entities in storage that no longer exist
                stored_entity_ids = set(entity_stats.keys())
                orphaned_entities = stored_entity_ids - set(current_entity_ids)
                
                if orphaned_entities:
                    issues["orphaned_entities"] = list(orphaned_entities)
                    
                # Find current entities missing from storage
                missing_entities = set(current_entity_ids) - stored_entity_ids
                if missing_entities:
                    issues["missing_entities"] = list(missing_entities)
                    
                issues["entity_stats_file"] = {
                    "path": str(entity_stats_path),
                    "entity_count": len(entity_stats)
                }
                    
            except Exception as e:
                issues["entity_stats_error"] = str(e)
        else:
            issues["entity_stats_file"] = "Missing"
            issues["missing_entities"] = current_entity_ids
        
        # Check clusters file
        clusters_path = self.base_path / "clusters.json"
        if clusters_path.exists():
            try:
                with open(clusters_path, "r") as f:
                    clusters = json.load(f)
                
                # Check entities in clusters
                clustered_entities = set()
                for cluster_id, entities in clusters.items():
                    clustered_entities.update(entities)
                
                # Find entities in clusters that no longer exist
                orphaned_cluster_entities = clustered_entities - set(current_entity_ids)
                if orphaned_cluster_entities:
                    issues["orphaned_cluster_entities"] = list(orphaned_cluster_entities)
                
                issues["clusters_file"] = {
                    "path": str(clusters_path),
                    "cluster_count": len(clusters),
                    "clustered_entities": len(clustered_entities)
                }
                
            except Exception as e:
                issues["clusters_error"] = str(e)
        else:
            issues["clusters_file"] = "Missing"
        
        # Determine if there are any critical issues
        critical_issues = (
            len(issues.get("orphaned_entities", [])) > 0 or
            issues.get("entity_stats_error") is not None or
            issues.get("clusters_error") is not None
        )
        
        return not critical_issues, issues
    
    async def async_fix_orphaned_entities(self, current_entity_ids: List[str]) -> Dict[str, Any]:
        """Fix orphaned entities asynchronously."""
        if self.hass:
            return await self.hass.async_add_executor_job(
                self.fix_orphaned_entities, current_entity_ids)
        return self.fix_orphaned_entities(current_entity_ids)
    
    def fix_orphaned_entities(self, current_entity_ids: List[str]) -> Dict[str, Any]:
        """Remove orphaned entities from storage.
        
        Args:
            current_entity_ids: List of currently valid entity IDs
            
        Returns:
            Dictionary with fix results
        """
        results = {
            "entity_stats_removed": [],
            "cluster_references_removed": []
        }
        
        # Fix entity stats
        entity_stats_path = self.base_path / "entity_stats.json"
        if entity_stats_path.exists():
            try:
                with open(entity_stats_path, "r") as f:
                    entity_stats = json.load(f)
                
                # Find and remove orphaned entities
                stored_entity_ids = set(entity_stats.keys())
                orphaned_entities = stored_entity_ids - set(current_entity_ids)
                
                if orphaned_entities:
                    for entity_id in orphaned_entities:
                        if entity_id in entity_stats:
                            del entity_stats[entity_id]
                            results["entity_stats_removed"].append(entity_id)
                    
                    # Write back the fixed file
                    with open(entity_stats_path, "w") as f:
                        json.dump(entity_stats, f, indent=2)
            except Exception as e:
                results["entity_stats_error"] = str(e)
        
        # Fix clusters
        clusters_path = self.base_path / "clusters.json"
        if clusters_path.exists():
            try:
                with open(clusters_path, "r") as f:
                    clusters = json.load(f)
                
                # Remove orphaned entities from clusters
                modified = False
                for cluster_id, entities in list(clusters.items()):
                    valid_entities = [e for e in entities if e in current_entity_ids]
                    if len(valid_entities) != len(entities):
                        removed = set(entities) - set(valid_entities)
                        results["cluster_references_removed"].extend(removed)
                        clusters[cluster_id] = valid_entities
                        modified = True
                    
                    # Remove empty clusters
                    if not valid_entities:
                        del clusters[cluster_id]
                        modified = True
                
                # Write back the fixed file if modified
                if modified:
                    with open(clusters_path, "w") as f:
                        json.dump(clusters, f, indent=2)
            except Exception as e:
                results["clusters_error"] = str(e)
        
        return results

# Initialize the validator
def get_validator(base_path: Path) -> StorageValidator:
    """Get a StorageValidator instance for the given path."""
    return StorageValidator(base_path)