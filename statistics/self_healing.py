"""
Self-healing system for Modbus optimization.

This module provides automatic detection and recovery from various issues
to maintain system reliability without manual intervention.
"""

import logging
import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple, Union
import threading

_LOGGER = logging.getLogger(__name__)

class SelfHealingSystem:
    """Implements self-healing capabilities for the system."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = SelfHealingSystem()
        return cls._instance
    
    def __init__(self):
        """Initialize the self-healing system."""
        self.healing_history = []
        self.max_history = 100  # Keep last 100 healing operations
        self.last_full_check = 0
        self.check_interval = 3600  # 1 hour between checks
        self.healing_lock = threading.Lock()
        
        # Storage path - will be set after integration with STATISTICS_MANAGER
        self.storage_path = None
        
    def initialize(self, storage_path: Path) -> None:
        """Initialize storage path from statistics manager."""
        self.storage_path = storage_path
        
    def check_and_heal_system(self, full_check: bool = False) -> Dict[str, Any]:
        """Check for issues and heal automatically."""
        current_time = time.time()
        
        # For full checks, respect interval
        if full_check and current_time - self.last_full_check < self.check_interval:
            return {"status": "skipped", "reason": "Last full check too recent"}
            
        if full_check:
            self.last_full_check = current_time
            
        # Ensure we don't have concurrent healing operations
        with self.healing_lock:
            start_time = time.time()
            results = {
                "start_time": datetime.utcnow().isoformat(),
                "full_check": full_check,
                "issues_detected": 0,
                "issues_fixed": 0,
                "checks_performed": [],
                "healing_operations": []
            }
            
            try:
                # Always check file structure
                file_result = self._check_and_heal_file_structure()
                results["checks_performed"].append("file_structure")
                
                if file_result["issues_detected"] > 0:
                    results["issues_detected"] += file_result["issues_detected"]
                    results["issues_fixed"] += file_result["issues_fixed"]
                    results["healing_operations"].extend(file_result["operations"])
                    
                    # Skip other checks if file structure had critical issues
                    if file_result["critical_issues"]:
                        results["status"] = "partial"
                        results["message"] = "Critical file structure issues detected - skipping further checks"
                        self._record_healing_operation(results)
                        return results
                
                # Check data integrity
                data_result = self._check_and_heal_data_integrity()
                results["checks_performed"].append("data_integrity")
                
                if data_result["issues_detected"] > 0:
                    results["issues_detected"] += data_result["issues_detected"]
                    results["issues_fixed"] += data_result["issues_fixed"]
                    results["healing_operations"].extend(data_result["operations"])
                
                # For full checks, do more comprehensive validation
                if full_check:
                    # Check entity consistency when we have a coordinator reference
                    entity_result = self._check_and_heal_entity_consistency()
                    results["checks_performed"].append("entity_consistency")
                    
                    if entity_result["issues_detected"] > 0:
                        results["issues_detected"] += entity_result["issues_detected"]
                        results["issues_fixed"] += entity_result["issues_fixed"]
                        results["healing_operations"].extend(entity_result["operations"])
                    
                results["status"] = "success"
                results["duration_seconds"] = time.time() - start_time
                
                if results["issues_fixed"] > 0:
                    results["message"] = f"Healed {results['issues_fixed']} of {results['issues_detected']} issues"
                else:
                    results["message"] = "No issues requiring healing were detected"
                
            except Exception as e:
                results["status"] = "error"
                results["message"] = f"Error during healing: {str(e)}"
                results["error"] = str(e)
                _LOGGER.exception("Error during self-healing: %s", e)
            
            # Record this healing operation
            self._record_healing_operation(results)
            
            return results
    
    def _check_and_heal_file_structure(self) -> Dict[str, Any]:
        """Check and heal file structure issues."""
        result = {
            "issues_detected": 0,
            "issues_fixed": 0,
            "critical_issues": False,
            "operations": []
        }
        
        if not self.storage_path:
            result["critical_issues"] = True
            result["operations"].append({
                "type": "storage_path_missing",
                "status": "error",
                "message": "Storage path not initialized"
            })
            return result
        
        # Check if storage directory exists
        if not self.storage_path.exists():
            result["issues_detected"] += 1
            result["critical_issues"] = True
            
            try:
                # Create storage directory
                self.storage_path.mkdir(parents=True, exist_ok=True)
                result["issues_fixed"] += 1
                result["operations"].append({
                    "type": "create_directory",
                    "path": str(self.storage_path),
                    "status": "success"
                })
            except Exception as e:
                result["operations"].append({
                    "type": "create_directory",
                    "path": str(self.storage_path),
                    "status": "error",
                    "error": str(e)
                })
                return result  # Stop if we can't create directory
        
        # Check if meta.json exists
        meta_path = self.storage_path / "meta.json"
        if not meta_path.exists():
            result["issues_detected"] += 1
            result["critical_issues"] = True
            
            try:
                # Create meta.json
                meta_data = {
                    "version": "1.0.0",
                    "created": datetime.utcnow().isoformat(),
                    "last_update": datetime.utcnow().isoformat(),
                    "files": {
                        "entity_stats": "entity_stats.json",
                        "patterns": "patterns.json",
                        "clusters": "clusters.json"
                    }
                }
                
                with open(meta_path, "w") as f:
                    json.dump(meta_data, f, indent=2)
                    
                result["issues_fixed"] += 1
                result["operations"].append({
                    "type": "create_file",
                    "path": str(meta_path),
                    "status": "success"
                })
            except Exception as e:
                result["operations"].append({
                    "type": "create_file",
                    "path": str(meta_path),
                    "status": "error",
                    "error": str(e)
                })
                return result  # Stop if we can't create meta.json
        
        # Get list of required files from meta.json
        try:
            with open(meta_path, "r") as f:
                meta_data = json.load(f)
                
            # Check each file listed in meta.json
            for file_type, filename in meta_data.get("files", {}).items():
                file_path = self.storage_path / filename
                
                if not file_path.exists():
                    result["issues_detected"] += 1
                    
                    try:
                        # Create empty structure based on file type
                        if file_type == "entity_stats":
                            with open(file_path, "w") as f:
                                json.dump({}, f, indent=2)
                        elif file_type == "patterns":
                            with open(file_path, "w") as f:
                                json.dump({
                                    "current_pattern": None,
                                    "patterns": {},
                                    "meta": {
                                        "last_detection": datetime.utcnow().isoformat()
                                    }
                                }, f, indent=2)
                        elif file_type == "clusters":
                            with open(file_path, "w") as f:
                                json.dump({"default_cluster": []}, f, indent=2)
                        else:
                            # Generic empty file
                            with open(file_path, "w") as f:
                                json.dump({}, f, indent=2)
                                
                        result["issues_fixed"] += 1
                        result["operations"].append({
                            "type": "create_file",
                            "path": str(file_path),
                            "status": "success"
                        })
                    except Exception as e:
                        result["operations"].append({
                            "type": "create_file",
                            "path": str(file_path),
                            "status": "error",
                            "error": str(e)
                        })
                        
        except Exception as e:
            result["operations"].append({
                "type": "read_meta",
                "path": str(meta_path),
                "status": "error",
                "error": str(e)
            })
            result["critical_issues"] = True
        
        return result
    
    def _check_and_heal_entity_consistency(self) -> Dict[str, Any]:
        """Check and heal entity consistency issues."""
        result = {
            "issues_detected": 0,
            "issues_fixed": 0,
            "operations": []
        }
        
        # This method requires the statistics manager to be integrated
        # We'll implement this when we integrate with the existing coordinator
        result["operations"].append({
            "type": "entity_consistency_check",
            "status": "skipped",
            "message": "Method not yet integrated with coordinator system"
        })
        
        return result
    
    def _check_and_heal_data_integrity(self) -> Dict[str, Any]:
        """Check and heal data integrity issues."""
        result = {
            "issues_detected": 0,
            "issues_fixed": 0,
            "operations": []
        }
        
        if not self.storage_path:
            result["operations"].append({
                "type": "data_integrity_check",
                "status": "error",
                "message": "Storage path not initialized"
            })
            return result
        
        # Check each known file for JSON validity
        known_files = ["entity_stats.json", "patterns.json", "clusters.json"]
        for filename in known_files:
            file_path = self.storage_path / filename
            if file_path.exists():
                try:
                    # Try to parse JSON
                    with open(file_path, "r") as f:
                        json.load(f)
                except json.JSONDecodeError:
                    # File is corrupt
                    result["issues_detected"] += 1
                    
                    try:
                        # Create empty structure based on file type
                        if filename == "entity_stats.json":
                            with open(file_path, "w") as f:
                                json.dump({}, f, indent=2)
                        elif filename == "patterns.json":
                            with open(file_path, "w") as f:
                                json.dump({
                                    "current_pattern": None,
                                    "patterns": {},
                                    "meta": {
                                        "last_detection": datetime.utcnow().isoformat()
                                    }
                                }, f, indent=2)
                        elif filename == "clusters.json":
                            with open(file_path, "w") as f:
                                json.dump({"default_cluster": []}, f, indent=2)
                                
                        result["issues_fixed"] += 1
                        result["operations"].append({
                            "type": "repair_corrupt_json",
                            "path": str(file_path),
                            "status": "success"
                        })
                    except Exception as e:
                        result["operations"].append({
                            "type": "repair_corrupt_json",
                            "path": str(file_path),
                            "status": "error",
                            "error": str(e)
                        })
        
        return result
    
    def _record_healing_operation(self, operation: Dict[str, Any]) -> None:
        """Record a healing operation in history."""
        with self.healing_lock:
            self.healing_history.append(operation)
            
            # Keep history bounded
            while len(self.healing_history) > self.max_history:
                self.healing_history.pop(0)
    
    def get_healing_history(self) -> List[Dict[str, Any]]:
        """Get history of healing operations."""
        with self.healing_lock:
            return self.healing_history.copy()
    
    def heal_entity_data(self, entity_id: str, issue_type: str) -> Dict[str, Any]:
        """Attempt to heal entity-specific issues."""
        result = {
            "entity_id": entity_id,
            "issue_type": issue_type,
            "success": False,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            if issue_type == "missing_data":
                # This method requires integration with the statistics manager
                result["message"] = "Method not yet integrated with statistics manager"
                
            elif issue_type == "corrupted_data":
                # This method requires integration with the statistics manager
                result["message"] = "Method not yet integrated with statistics manager"
                
            else:
                result["message"] = f"Unknown issue type: {issue_type}"
                
        except Exception as e:
            result["success"] = False
            result["message"] = f"Error healing entity: {str(e)}"
            result["error"] = str(e)
            _LOGGER.exception("Error healing entity %s: %s", entity_id, e)
            
        # Record this operation
        self._record_healing_operation({
            "type": "entity_healing",
            "entity_id": entity_id,
            "issue_type": issue_type,
            "success": result["success"],
            "timestamp": result["timestamp"]
        })
            
        return result


# Global instance
SELF_HEALING_SYSTEM = SelfHealingSystem.get_instance()