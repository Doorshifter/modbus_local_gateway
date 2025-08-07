"""
Persistent storage for statistics data.

This module manages persistent storage of statistics data, ensuring
data is available across restarts.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
import threading

_LOGGER = logging.getLogger(__name__)

class StatisticsStorageManager:
    """Manages persistent storage of statistics data."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize storage manager.
        
        Args:
            base_path: Base directory for storage, or None to use default
        """
        if base_path is None:
            # Use default path in Home Assistant config directory
            config_dir = Path(os.path.expanduser("~/.homeassistant"))
            if not config_dir.exists():
                # Try another common location
                config_dir = Path("/config")
            
            self.base_path = config_dir / "modbus_statistics"
        else:
            self.base_path = base_path
            
        # Ensure directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.meta_path = self.base_path / "meta.json"
        self.entity_stats_path = self.base_path / "entity_stats.json"
        self.patterns_path = self.base_path / "patterns.json"
        self.clusters_path = self.base_path / "clusters.json"
        
        # Cache
        self._entity_stats_cache = None
        self._patterns_cache = None
        self._clusters_cache = None
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Initialize storage
        self._initialize_storage()
    
    def _initialize_storage(self) -> None:
        """Initialize storage files if they don't exist."""
        # Create meta.json if it doesn't exist
        if not self.meta_path.exists():
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
            
            try:
                with open(self.meta_path, "w") as f:
                    json.dump(meta_data, f, indent=2)
            except Exception as e:
                _LOGGER.error("Failed to create meta.json: %s", e)
        
        # Create entity_stats.json if it doesn't exist
        if not self.entity_stats_path.exists():
            try:
                with open(self.entity_stats_path, "w") as f:
                    json.dump({}, f, indent=2)
            except Exception as e:
                _LOGGER.error("Failed to create entity_stats.json: %s", e)
        
        # Create patterns.json if it doesn't exist
        if not self.patterns_path.exists():
            patterns_data = {
                "current_pattern": None,
                "patterns": {},
                "meta": {
                    "last_detection": datetime.utcnow().isoformat()
                }
            }
            
            try:
                with open(self.patterns_path, "w") as f:
                    json.dump(patterns_data, f, indent=2)
            except Exception as e:
                _LOGGER.error("Failed to create patterns.json: %s", e)
        
        # Create clusters.json if it doesn't exist
        if not self.clusters_path.exists():
            try:
                with open(self.clusters_path, "w") as f:
                    json.dump({"default_cluster": []}, f, indent=2)
            except Exception as e:
                _LOGGER.error("Failed to create clusters.json: %s", e)
    
    def get_entity_stats(self) -> Dict[str, Any]:
        """Get entity statistics.
        
        Returns:
            Dictionary with entity statistics
        """
        with self._lock:
            if self._entity_stats_cache is None:
                try:
                    with open(self.entity_stats_path, "r") as f:
                        self._entity_stats_cache = json.load(f)
                except Exception as e:
                    _LOGGER.error("Failed to load entity stats: %s", e)
                    self._entity_stats_cache = {}
                    
            return self._entity_stats_cache.copy()
    
    def save_entity_stats(self, entity_id: str, stats: Dict[str, Any], force: bool = False) -> bool:
        """Save entity statistics.
        
        Args:
            entity_id: Entity ID
            stats: Statistics to save
            force: If True, force save even if entity_id is None
            
        Returns:
            True if successful
        """
        if not entity_id and not force:
            return False
            
        with self._lock:
            if self._entity_stats_cache is None:
                self._entity_stats_cache = self.get_entity_stats()
                
            if entity_id:
                # Update specific entity
                self._entity_stats_cache[entity_id] = stats
                
            try:
                with open(self.entity_stats_path, "w") as f:
                    json.dump(self._entity_stats_cache, f, indent=2)
                    
                # Update meta last_update
                self._update_meta()
                    
                return True
            except Exception as e:
                _LOGGER.error("Failed to save entity stats: %s", e)
                return False
    
    def get_patterns(self) -> Dict[str, Any]:
        """Get patterns.
        
        Returns:
            Dictionary with pattern data
        """
        with self._lock:
            if self._patterns_cache is None:
                try:
                    with open(self.patterns_path, "r") as f:
                        self._patterns_cache = json.load(f)
                except Exception as e:
                    _LOGGER.error("Failed to load patterns: %s", e)
                    self._patterns_cache = {
                        "current_pattern": None,
                        "patterns": {},
                        "meta": {
                            "last_detection": datetime.utcnow().isoformat()
                        }
                    }
                    
            return self._patterns_cache.copy()
    
    def save_patterns(self, patterns: Dict[str, Any], force: bool = False) -> bool:
        """Save patterns.
        
        Args:
            patterns: Pattern data to save
            force: If True, force save even if patterns is None
            
        Returns:
            True if successful
        """
        if not patterns and not force:
            return False
            
        with self._lock:
            self._patterns_cache = patterns
                
            try:
                with open(self.patterns_path, "w") as f:
                    json.dump(patterns, f, indent=2)
                    
                # Update meta last_update
                self._update_meta()
                    
                return True
            except Exception as e:
                _LOGGER.error("Failed to save patterns: %s", e)
                return False
    
    def get_clusters(self) -> Dict[str, List[str]]:
        """Get clusters.
        
        Returns:
            Dictionary with cluster data (cluster_id -> entity_ids)
        """
        with self._lock:
            if self._clusters_cache is None:
                try:
                    with open(self.clusters_path, "r") as f:
                        self._clusters_cache = json.load(f)
                except Exception as e:
                    _LOGGER.error("Failed to load clusters: %s", e)
                    self._clusters_cache = {"default_cluster": []}
                    
            return self._clusters_cache.copy()
    
    def save_clusters(self, clusters: Dict[str, List[str]], force: bool = False) -> bool:
        """Save clusters.
        
        Args:
            clusters: Cluster data to save
            force: If True, force save even if clusters is None
            
        Returns:
            True if successful
        """
        if not clusters and not force:
            return False
            
        with self._lock:
            self._clusters_cache = clusters
                
            try:
                with open(self.clusters_path, "w") as f:
                    json.dump(clusters, f, indent=2)
                    
                # Update meta last_update
                self._update_meta()
                    
                return True
            except Exception as e:
                _LOGGER.error("Failed to save clusters: %s", e)
                return False
    
    def get_metadata_file(self, file_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata from a generic metadata file.
        
        Args:
            file_name: Base name of the file (without extension)
            
        Returns:
            Dictionary with metadata or None if not found
        """
        file_path = self.base_path / f"{file_name}.json"
        
        if not file_path.exists():
            return None
            
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            _LOGGER.error("Failed to load metadata file %s: %s", file_name, e)
            return None
    
    def save_metadata_file(self, file_name: str, data: Dict[str, Any]) -> bool:
        """Save metadata to a generic metadata file.
        
        Args:
            file_name: Base name of the file (without extension)
            data: Dictionary with metadata
            
        Returns:
            True if successful
        """
        file_path = self.base_path / f"{file_name}.json"
        
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
                
            # Update meta last_update
            self._update_meta()
                
            return True
        except Exception as e:
            _LOGGER.error("Failed to save metadata file %s: %s", file_name, e)
            return False
    
    def _update_meta(self) -> None:
        """Update meta file with last update timestamp."""
        if not self.meta_path.exists():
            return
            
        try:
            with open(self.meta_path, "r") as f:
                meta_data = json.load(f)
                
            meta_data["last_update"] = datetime.utcnow().isoformat()
            
            with open(self.meta_path, "w") as f:
                json.dump(meta_data, f, indent=2)
                
        except Exception as e:
            _LOGGER.error("Failed to update meta.json: %s", e)
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about storage.
        
        Returns:
            Dictionary with storage information
        """
        meta_data = None
        if self.meta_path.exists():
            try:
                with open(self.meta_path, "r") as f:
                    meta_data = json.load(f)
            except Exception as e:
                _LOGGER.error("Failed to load meta.json: %s", e)
        
        # Count stored items
        entity_stats = self.get_entity_stats()
        patterns_data = self.get_patterns()
        clusters = self.get_clusters()
        
        # Get file sizes
        files = {}
        for file_path in [self.meta_path, self.entity_stats_path, self.patterns_path, self.clusters_path]:
            if file_path.exists():
                try:
                    files[file_path.name] = {
                        "size_bytes": file_path.stat().st_size,
                        "size_kb": round(file_path.stat().st_size / 1024, 1),
                        "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    }
                except Exception:
                    files[file_path.name] = {"error": "Failed to get file info"}
        
        return {
            "storage_location": str(self.base_path),
            "version": meta_data["version"] if meta_data else None,
            "created": meta_data["created"] if meta_data else None,
            "last_update": meta_data["last_update"] if meta_data else None,
            "stats_count": len(entity_stats),
            "patterns_count": len(patterns_data.get("patterns", {})),
            "clusters_count": len(clusters),
            "files": files
        }


class PersistentStatisticsManager:
    """Manager for persistent statistics."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = PersistentStatisticsManager()
        return cls._instance
    
    def __init__(self):
        """Initialize persistent statistics manager."""
        self.storage = StatisticsStorageManager()
        self._initialized = False
        self.pattern_detector = None  # Will be initialized later
        self.correlation_manager = None  # Will be initialized later
        self._last_integration_analysis = 0
        self._integration_interval = 3600  # Run integration analysis hourly
        
        # Cross-system insights
        self._pattern_correlations = {}  # pattern_id -> list of correlation clusters
        self._cluster_patterns = {}  # cluster_id -> list of patterns
    
    def initialize(self) -> bool:
        """Initialize the manager."""
        if self._initialized:
            return True
            
        try:
            # Check if storage is accessible
            entity_stats = self.storage.get_entity_stats()
            self._initialized = True
            return True
        except Exception as e:
            _LOGGER.error("Failed to initialize persistent statistics: %s", e)
            return False
            
    def initialize_with_components(self, pattern_detector, correlation_manager) -> bool:
        """Initialize with pattern detector and correlation manager components.
        
        Args:
            pattern_detector: Pattern detection component
            correlation_manager: Correlation management component
            
        Returns:
            True if initialization was successful
        """
        self.pattern_detector = pattern_detector
        self.correlation_manager = correlation_manager
        
        # Load stored pattern correlation data if available
        self._load_integration_data()
        
        return self.initialize()
        
    def save_entity_stats(self, entity_id: str, stats: Dict[str, Any]) -> bool:
        """Save entity statistics.
        
        Args:
            entity_id: Entity ID
            stats: Statistics to save
            
        Returns:
            True if successful
        """
        return self.storage.save_entity_stats(entity_id, stats)
        
    def get_entity_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get entity statistics.
        
        Returns:
            Dictionary with entity statistics
        """
        return self.storage.get_entity_stats()
        
    def save_patterns(self, patterns: Dict[str, Any]) -> bool:
        """Save patterns.
        
        Args:
            patterns: Pattern data to save
            
        Returns:
            True if successful
        """
        return self.storage.save_patterns(patterns)
        
    def get_patterns(self) -> Dict[str, Any]:
        """Get patterns.
        
        Returns:
            Dictionary with pattern data
        """
        return self.storage.get_patterns()
        
    def save_clusters(self, clusters: Dict[str, List[str]]) -> bool:
        """Save clusters.
        
        Args:
            clusters: Cluster data to save
            
        Returns:
            True if successful
        """
        return self.storage.save_clusters(clusters)
        
    def get_clusters(self) -> Dict[str, List[str]]:
        """Get clusters.
        
        Returns:
            Dictionary with cluster data
        """
        return self.storage.get_clusters()
        
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about storage.
        
        Returns:
            Dictionary with storage information
        """
        return self.storage.get_storage_info()
        
    def export_data(self) -> Dict[str, Any]:
        """Export all statistics data.
        
        Returns:
            Dictionary with all statistics data
        """
        return {
            "entity_stats": self.storage.get_entity_stats(),
            "patterns": self.storage.get_patterns(),
            "clusters": self.storage.get_clusters(),
            "storage_info": self.storage.get_storage_info()
        }
        
    def perform_integration_analysis(self, timestamp=None) -> Dict[str, Any]:
        """Perform integration analysis between pattern and correlation systems.
        
        Args:
            timestamp: Analysis timestamp (default: current time)
            
        Returns:
            Dictionary with analysis results
        """
        if timestamp is None:
            timestamp = time.time()
        
        if not self.pattern_detector or not self.correlation_manager:
            return {"error": "Components not initialized"}
            
        self._last_integration_analysis = timestamp
        
        _LOGGER.info("Running integrated pattern-correlation analysis")
        
        # Get current state from both systems
        pattern_info = self.pattern_detector.get_statistics()
        clusters = self.correlation_manager.get_clusters()
        
        # Reset cross-system insights
        self._pattern_correlations = {}
        self._cluster_patterns = {}
        
        # Analyze patterns in each correlation cluster
        for cluster_id, entity_ids in clusters.items():
            if len(entity_ids) < 2:  # Skip singleton clusters
                continue
                
            # Find patterns common to this cluster
            cluster_patterns = self._find_patterns_for_cluster(cluster_id, entity_ids)
            
            if cluster_patterns:
                self._cluster_patterns[cluster_id] = cluster_patterns
                
                # Update reverse mapping
                for pattern_id in cluster_patterns:
                    if pattern_id not in self._pattern_correlations:
                        self._pattern_correlations[pattern_id] = []
                    
                    if cluster_id not in self._pattern_correlations[pattern_id]:
                        self._pattern_correlations[pattern_id].append(cluster_id)
        
        # Save integration data
        self._save_integration_data()
        
        results = {
            "timestamp": timestamp,
            "clusters_analyzed": len(clusters),
            "patterns_with_correlations": len(self._pattern_correlations),
            "clusters_with_patterns": len(self._cluster_patterns),
            "total_relationships": sum(len(clusters) for clusters in self._pattern_correlations.values())
        }
        
        _LOGGER.info("Integrated analysis complete. Found %d pattern-cluster relationships",
                   results["total_relationships"])
                   
        return results
    
    def _find_patterns_for_cluster(self, cluster_id: str, entity_ids: List[str]) -> List[int]:
        """Find patterns that are common to entities in a correlation cluster.
        
        Args:
            cluster_id: Correlation cluster identifier
            entity_ids: List of entity IDs in the cluster
            
        Returns:
            List of pattern IDs that are common to this cluster
        """
        if not self.pattern_detector:
            return []
            
        # Basic implementation - current pattern if the entities are active
        # In a full implementation, this would analyze historical pattern data
        current_pattern = self.pattern_detector.current_pattern
        if current_pattern is not None:
            # Check if the entities in the cluster are active in this pattern
            pattern = self.pattern_detector.patterns.get(current_pattern)
            if pattern:
                # Count how many entities from this cluster are active in the pattern
                active_entities = 0
                for entity_id in entity_ids:
                    if entity_id in pattern.characteristic_values:
                        active_entities += 1
                
                # If a significant portion of cluster entities are active in this pattern
                if active_entities >= min(2, len(entity_ids) // 2):
                    return [current_pattern]
        
        return []
    
    def get_integrated_insights(self, entity_id: str) -> Dict[str, Any]:
        """Get comprehensive insights for an entity combining pattern and correlation data.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Dictionary with integrated insights
        """
        insights = {
            "entity_id": entity_id,
            "correlation": {},
            "patterns": {},
            "integrated_insights": {},
        }
        
        if not self.correlation_manager or not self.pattern_detector:
            return insights
        
        # Get correlation data
        cluster = self.correlation_manager.get_cluster_for_entity(entity_id)
        if cluster:
            insights["correlation"]["cluster"] = cluster
            insights["correlation"]["cluster_members"] = self.correlation_manager.get_clusters().get(cluster, [])
            
            # Add correlation strength with other entities
            insights["correlation"]["strengths"] = {}
            for other_entity in insights["correlation"]["cluster_members"]:
                if other_entity != entity_id:
                    corr = self.correlation_manager.get_correlation(entity_id, other_entity)
                    if corr is not None:
                        insights["correlation"]["strengths"][other_entity] = round(corr, 2)
        
        # Get pattern data
        current_pattern = self.pattern_detector.current_pattern
        if current_pattern is not None:
            pattern_info = self.pattern_detector.get_current_pattern_info()
            insights["patterns"]["current_pattern"] = current_pattern
            insights["patterns"]["confidence"] = pattern_info.get("pattern_confidence")
            insights["patterns"]["stability"] = pattern_info.get("stability")
            insights["patterns"]["defining_criteria"] = pattern_info.get("defining_criteria", [])
        
        # Add integrated insights
        if cluster and current_pattern is not None:
            # Check if this pattern is associated with the entity's cluster
            cluster_patterns = self._cluster_patterns.get(cluster, [])
            pattern_in_cluster = current_pattern in cluster_patterns
            insights["integrated_insights"]["pattern_in_cluster"] = pattern_in_cluster
            
            # Calculate value confidence based on correlation and pattern
            if pattern_in_cluster:
                # Entities in the same cluster sharing a pattern have higher confidence
                confidence = 0.7
                
                # Adjust based on correlation strength and pattern confidence
                if "strengths" in insights["correlation"] and insights["correlation"]["strengths"]:
                    avg_correlation = sum(insights["correlation"]["strengths"].values()) / len(insights["correlation"]["strengths"])
                    pattern_confidence = insights["patterns"].get("confidence", 50) / 100
                    
                    confidence = (0.5 * avg_correlation) + (0.5 * pattern_confidence)
                    
                insights["integrated_insights"]["value_confidence"] = round(confidence, 2)
            else:
                insights["integrated_insights"]["value_confidence"] = 0.5  # Default confidence
        
        return insights
    
    def get_pattern_correlation_insights(self, pattern_id: Optional[int] = None) -> Dict[str, Any]:
        """Get correlation insights for patterns.
        
        Args:
            pattern_id: Optional pattern ID to filter results
            
        Returns:
            Dictionary with pattern correlation insights
        """
        if pattern_id is not None:
            # Single pattern insights
            return {
                "pattern_id": pattern_id,
                "associated_clusters": self._pattern_correlations.get(pattern_id, []),
                "entity_count": self._get_entity_count_in_pattern(pattern_id),
            }
        else:
            # All patterns overview
            results = {
                "patterns": {},
                "cluster_patterns": self._cluster_patterns,
            }
            
            for pattern_id in self._pattern_correlations:
                results["patterns"][str(pattern_id)] = {
                    "associated_clusters": self._pattern_correlations[pattern_id],
                    "entity_count": self._get_entity_count_in_pattern(pattern_id),
                }
                
            return results
    
    def _get_entity_count_in_pattern(self, pattern_id: int) -> int:
        """Get number of entities in a pattern.
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Number of entities
        """
        if not self.pattern_detector or pattern_id not in self.pattern_detector.patterns:
            return 0
            
        pattern = self.pattern_detector.patterns[pattern_id]
        return len(pattern.characteristic_values)
    
    def _load_integration_data(self) -> None:
        """Load pattern correlation data from storage."""
        try:
            data = self.storage.get_metadata_file("pattern_correlation_data")
            if data:
                if "pattern_correlations" in data:
                    self._pattern_correlations = data["pattern_correlations"]
                
                if "cluster_patterns" in data:
                    self._cluster_patterns = data["cluster_patterns"]
                    
                if "last_integration_analysis" in data:
                    self._last_integration_analysis = data["last_integration_analysis"]
                    
                _LOGGER.info("Loaded pattern correlation data from storage")
        except Exception as e:
            _LOGGER.warning("Failed to load pattern correlation data: %s", e)
    
    def _save_integration_data(self) -> bool:
        """Save pattern correlation data to storage."""
        try:
            data = {
                "pattern_correlations": self._pattern_correlations,
                "cluster_patterns": self._cluster_patterns,
                "last_integration_analysis": self._last_integration_analysis,
                "meta": {
                    "last_update": datetime.utcnow().isoformat(),
                }
            }
            
            success = self.storage.save_metadata_file("pattern_correlation_data", data)
            return success
        except Exception as e:
            _LOGGER.error("Error saving pattern correlation data: %s", e)
            return False

    def initialize_advanced_interval_manager(self) -> None:
        """Initialize and connect the advanced interval manager with other components."""
        from .advanced_interval_manager import ADVANCED_INTERVAL_MANAGER
        
        # Set components
        ADVANCED_INTERVAL_MANAGER.set_components(
            self.pattern_detector,
            self.correlation_manager
        )
        
        # Store reference
        self.advanced_interval_manager = ADVANCED_INTERVAL_MANAGER
        
        _LOGGER.info("Advanced interval manager initialized and connected")

# Global instance
PERSISTENT_STATISTICS_MANAGER = PersistentStatisticsManager.get_instance()