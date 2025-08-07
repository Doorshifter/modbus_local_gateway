"""
Validation framework for Modbus optimization system.

This module provides comprehensive validation and consistency checks to ensure
the system maintains data integrity and operates correctly.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple, Union

from .storage import StatisticsStorageManager
from .data_validation import StorageValidator

_LOGGER = logging.getLogger(__name__)

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
    """Manages system validation and health checks."""
    
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
    
    def validate_all(self, current_entity_ids: List[str] = None) -> Dict[str, Any]:
        """Run all validation checks.
        
        Args:
            current_entity_ids: List of currently active entity IDs
            
        Returns:
            Dictionary with validation results
        """
        start_time = time.time()
        self.last_validation_time = start_time
        
        # Reset overall status
        self.overall_status = "pass"
        
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
    
    def validate_file_structure(self) -> ValidationResult:
        """Validate file structure.
        
        Returns:
            ValidationResult object
        """
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
        if not validation_results["overall_valid"]:
            # Add file-specific errors
            for file_type, file_result in validation_results["files"].items():
                if not file_result["valid"]:
                    for error in file_result["errors"]:
                        result.add_error(f"{file_type} error: {error}")
            
            result.details["validation_results"] = validation_results
        else:
            result.add_info("All files passed schema validation")
        
        # Check version consistency
        try:
            with open(self.storage.meta_path, "r") as f:
                meta_data = json.load(f)
                meta_version = meta_data.get("version")
            
            # Check versions in other files
            version_inconsistencies = []
            
            for file_type, filename in meta_data.get("files", {}).items():
                file_path = self.storage.base_path / filename
                if file_path.exists():
                    try:
                        with open(file_path, "r") as f:
                            file_data = json.load(f)
                            if isinstance(file_data, dict) and "version" in file_data:
                                file_version = file_data["version"]
                                if file_version != meta_version:
                                    version_inconsistencies.append(
                                        f"{file_type}: expected {meta_version}, got {file_version}"
                                    )
                    except (json.JSONDecodeError, IOError):
                        # Already reported in schema validation
                        pass
            
            if version_inconsistencies:
                for inconsistency in version_inconsistencies:
                    result.add_warning(f"Version inconsistency: {inconsistency}")
                result.details["version_inconsistencies"] = version_inconsistencies
            else:
                result.add_info(f"All files have consistent version: {meta_version}")
                result.details["version"] = meta_version
        
        except (json.JSONDecodeError, IOError) as e:
            # This would have been caught in meta validation
            pass
        
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
    
    def validate_entity_consistency(self, current_entity_ids: List[str]) -> ValidationResult:
        """Validate entity consistency between storage and HA.
        
        Args:
            current_entity_ids: List of current entity IDs in Home Assistant
            
        Returns:
            ValidationResult object
        """
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
    
    def validate_clusters(self) -> ValidationResult:
        """Validate cluster assignments.
        
        Returns:
            ValidationResult object
        """
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
    
    def validate_scan_intervals(self) -> ValidationResult:
        """Validate scan interval assignments.
        
        Returns:
            ValidationResult object
        """
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
    
    def validate_data_quality(self) -> ValidationResult:
        """Validate data quality for analysis.
        
        Returns:
            ValidationResult object
        """
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
            "results": {
                component: result.as_dict()
                for component, result in self.validation_results.items()
            }
        }
    
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