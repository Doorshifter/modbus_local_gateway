"""
Storage manager for persistent statistics.
Handles reading and writing statistics to disk.
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

_LOGGER = logging.getLogger(__name__)

class StatisticsStorageManager:
    """Manager for reading and writing statistics data to disk."""

    def __init__(self, base_path: Path):
        """Initialize the storage manager.
        
        Args:
            base_path: Base path for storing statistics
        """
        self.base_path = base_path
        self.meta_path = base_path / "meta.json"
        self.patterns_path = base_path / "patterns.json"
        self.clusters_path = base_path / "clusters.json"
        self.insights_path = base_path / "insights.json"
        self.anomalies_path = base_path / "anomalies.json"
        
        # Create directory if it doesn't exist
        if not os.path.exists(base_path):
            try:
                os.makedirs(base_path)
                _LOGGER.info("Created statistics directory at %s", base_path)
            except Exception as e:
                _LOGGER.error("Could not create statistics directory: %s", e)
        
    def _blocking_read_json_file(self, file_path: Path) -> Dict:
        """Read JSON from file - BLOCKING VERSION.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON data or empty dict if file doesn't exist
        """
        try:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            _LOGGER.error("Error reading from %s: %s", file_path, e)
            return {}
            
    def _blocking_write_json_file(self, file_path: Path, data: Dict) -> bool:
        """Write JSON to file - BLOCKING VERSION.
        
        Args:
            file_path: Path to JSON file
            data: Data to write
            
        Returns:
            True if successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            _LOGGER.error("Error writing to %s: %s", file_path, e)
            return False
    
    async def async_get_patterns(self, hass) -> Dict:
        """Get pattern data asynchronously.
        
        Returns:
            Pattern data
        """
        return await hass.async_add_executor_job(self._blocking_get_patterns)
    
    def _blocking_get_patterns(self) -> Dict:
        """Get pattern data - BLOCKING VERSION.
        
        Returns:
            Pattern data
        """
        return self._blocking_read_json_file(self.patterns_path)
        
    def get_patterns(self) -> Dict:
        """Get pattern data (blocking - use async_get_patterns in event loop).
        
        Returns:
            Pattern data
        """
        # Warning: This is a blocking call and should not be used in the event loop
        return self._blocking_get_patterns()
        
    async def async_save_patterns(self, hass, patterns_data: Dict) -> bool:
        """Save pattern data asynchronously.
        
        Args:
            patterns_data: Pattern data to save
            
        Returns:
            True if successful
        """
        return await hass.async_add_executor_job(
            self._blocking_save_patterns, patterns_data
        )
        
    def _blocking_save_patterns(self, patterns_data: Dict) -> bool:
        """Save pattern data - BLOCKING VERSION.
        
        Args:
            patterns_data: Pattern data to save
            
        Returns:
            True if successful
        """
        return self._blocking_write_json_file(self.patterns_path, patterns_data)
        
    def save_patterns(self, patterns_data: Dict) -> bool:
        """Save pattern data (blocking - use async_save_patterns in event loop).
        
        Args:
            patterns_data: Pattern data to save
            
        Returns:
            True if successful
        """
        # Warning: This is a blocking call and should not be used in the event loop
        return self._blocking_save_patterns(patterns_data)
            
    async def async_get_clusters(self, hass) -> Dict:
        """Get cluster data asynchronously.
        
        Returns:
            Cluster data
        """
        return await hass.async_add_executor_job(self._blocking_get_clusters)
    
    def _blocking_get_clusters(self) -> Dict:
        """Get cluster data - BLOCKING VERSION.
        
        Returns:
            Cluster data
        """
        return self._blocking_read_json_file(self.clusters_path)
        
    def get_clusters(self) -> Dict:
        """Get cluster data (blocking - use async_get_clusters in event loop).
        
        Returns:
            Cluster data
        """
        # Warning: This is a blocking call and should not be used in the event loop
        return self._blocking_get_clusters()
        
    async def async_save_clusters(self, hass, clusters_data: Dict) -> bool:
        """Save cluster data asynchronously.
        
        Args:
            clusters_data: Cluster data to save
            
        Returns:
            True if successful
        """
        return await hass.async_add_executor_job(
            self._blocking_save_clusters, clusters_data
        )
        
    def _blocking_save_clusters(self, clusters_data: Dict) -> bool:
        """Save cluster data - BLOCKING VERSION.
        
        Args:
            clusters_data: Cluster data to save
            
        Returns:
            True if successful
        """
        return self._blocking_write_json_file(self.clusters_path, clusters_data)
        
    def save_clusters(self, clusters_data: Dict) -> bool:
        """Save cluster data (blocking - use async_save_clusters in event loop).
        
        Args:
            clusters_data: Cluster data to save
            
        Returns:
            True if successful
        """
        # Warning: This is a blocking call and should not be used in the event loop
        return self._blocking_save_clusters(clusters_data)
            
    def save_metadata_file(self, filename: str, data: Any) -> bool:
        """Save metadata to a JSON file.
        
        Args:
            filename: Filename without extension
            data: Data to save
            
        Returns:
            True if successful
        """
        # Warning: This is a blocking call and should not be used in the event loop
        path = self.base_path / f"{filename}.json"
        return self._blocking_write_json_file(path, data)
        
    async def async_save_metadata_file(self, hass, filename: str, data: Any) -> bool:
        """Save metadata to a JSON file asynchronously.
        
        Args:
            filename: Filename without extension
            data: Data to save
            
        Returns:
            True if successful
        """
        path = self.base_path / f"{filename}.json"
        return await hass.async_add_executor_job(
            self._blocking_write_json_file, path, data
        )
        
    def get_metadata_file(self, filename: str) -> Dict:
        """Get metadata from a JSON file.
        
        Args:
            filename: Filename without extension
            
        Returns:
            Metadata or empty dict if file doesn't exist
        """
        # Warning: This is a blocking call and should not be used in the event loop
        path = self.base_path / f"{filename}.json"
        return self._blocking_read_json_file(path)
        
    async def async_get_metadata_file(self, hass, filename: str) -> Dict:
        """Get metadata from a JSON file asynchronously.
        
        Args:
            filename: Filename without extension
            
        Returns:
            Metadata or empty dict if file doesn't exist
        """
        path = self.base_path / f"{filename}.json"
        return await hass.async_add_executor_job(
            self._blocking_read_json_file, path
        )
    
    def _blocking_get_storage_info(self) -> Dict:
        """Get storage information - BLOCKING VERSION.
        
        Returns:
            Storage information
        """
        result = {
            "storage_path": str(self.base_path),
            "files": {},
            "total_size_bytes": 0,
            "patterns_count": 0,
            "clusters_count": 0
        }
        
        # Get metadata about the storage
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                try:
                    meta = json.load(f)
                    result.update(meta)
                except:
                    pass
                    
        # Get pattern info
        patterns_data = self._blocking_get_patterns()
        result["patterns_count"] = len(patterns_data)
        
        # Get clusters info
        clusters = self._blocking_get_clusters()
        result["clusters_count"] = len(clusters)
        
        # Get file stats
        for filename in os.listdir(self.base_path) if os.path.exists(self.base_path) else []:
            file_path = self.base_path / filename
            if os.path.isfile(file_path):
                try:
                    file_size = os.path.getsize(file_path)
                    result["files"][filename] = {
                        "size_bytes": file_size,
                        "last_modified": os.path.getmtime(file_path)
                    }
                    result["total_size_bytes"] += file_size
                except:
                    pass
                    
        return result
    
    def get_storage_info(self) -> Dict:
        """Get storage information (blocking - use async_get_storage_info in event loop).
        
        Returns:
            Storage information
        """
        # Warning: This is a blocking call and should not be used in the event loop
        return self._blocking_get_storage_info()
        
    async def async_get_storage_info(self, hass) -> Dict:
        """Get storage information asynchronously.
        
        Returns:
            Storage information
        """
        return await hass.async_add_executor_job(self._blocking_get_storage_info)
                    
    def get_interval_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get interval history for entities.
        
        Returns:
            Dictionary of entity ID to interval history
        """
        # Warning: This is a blocking call and should not be used in the event loop
        data = self.get_metadata_file("interval_history")
        if data:
            return data
        return {}
        
    async def async_get_interval_history(self, hass) -> Dict[str, List[Dict[str, Any]]]:
        """Get interval history for entities asynchronously.
        
        Returns:
            Dictionary of entity ID to interval history
        """
        data = await self.async_get_metadata_file(hass, "interval_history")
        if data:
            return data
        return {}

    def save_interval_history(self, interval_history: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Save interval history.
        
        Args:
            interval_history: Dictionary of entity ID to interval history
            
        Returns:
            True if successful
        """
        # Warning: This is a blocking call and should not be used in the event loop
        return self.save_metadata_file("interval_history", interval_history)
        
    async def async_save_interval_history(self, hass, interval_history: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Save interval history asynchronously.
        
        Args:
            interval_history: Dictionary of entity ID to interval history
            
        Returns:
            True if successful
        """
        return await self.async_save_metadata_file(hass, "interval_history", interval_history)


class PatternDetector:
    """Simple placeholder for pattern detection."""
    
    def __init__(self):
        """Initialize the pattern detector."""
        self._patterns = {}
        
    def detect_patterns(self, entity_id: str, data: List[Any]) -> Dict:
        """Detect patterns in entity data."""
        # Placeholder implementation
        return {
            "periodic": False,
            "predictable": False,
            "has_pattern": False,
            "confidence": 0.0
        }


class CorrelationManager:
    """Simple placeholder for correlation management."""
    
    def __init__(self):
        """Initialize the correlation manager."""
        self._correlations = {}
        
    def add_correlation(self, entity1: str, entity2: str, value: float) -> None:
        """Add correlation between entities."""
        key = f"{entity1}:{entity2}"
        self._correlations[key] = {
            "value": value,
            "timestamp": time.time()
        }
        
    def get_correlations(self) -> Dict:
        """Get all correlations."""
        return self._correlations


class PersistentStatisticsManager:
    """Manager for persistent statistics across restarts."""
    
    def __init__(self):
        """Initialize the manager."""
        self.storage = None
        self.enabled = False
        
        # Add required components with placeholders
        self._correlation_manager = CorrelationManager()
        self._pattern_detector = PatternDetector()
        self._interval_manager = None
        self._anomaly_detector = None
        self._resource_adapter = None
        self._self_healing_system = None
        
    @property
    def correlation_manager(self):
        """Get the correlation manager."""
        if self._correlation_manager is None:
            self._correlation_manager = CorrelationManager()
        return self._correlation_manager
    
    @property
    def pattern_detector(self):
        """Get the pattern detector."""
        if self._pattern_detector is None:
            self._pattern_detector = PatternDetector()
        return self._pattern_detector
        
    def initialize(self, storage_path: Path) -> None:
        """Initialize the manager with storage path.
        
        Args:
            storage_path: Path to store statistics
        """
        self.storage = StatisticsStorageManager(storage_path)
        self.enabled = True
        _LOGGER.info("Persistent statistics initialized with storage path: %s", storage_path)
        
    def initialize_advanced_interval_manager(self):
        """Initialize the advanced interval manager."""
        try:
            from .advanced_interval_manager import ADVANCED_INTERVAL_MANAGER
            self._interval_manager = ADVANCED_INTERVAL_MANAGER
            _LOGGER.info("Advanced interval manager initialized")
        except ImportError as e:
            _LOGGER.warning("Could not initialize advanced interval manager: %s", e)
        
    def get_storage_info(self) -> Dict:
        """Get storage information.
        
        Returns:
            Storage information
        """
        if not self.enabled or not self.storage:
            return {"status": "disabled"}
        return self.storage.get_storage_info()
    
    async def async_get_storage_info(self, hass) -> Dict:
        """Get storage information asynchronously.
        
        Returns:
            Storage information
        """
        if not self.enabled or not self.storage:
            return {"status": "disabled"}
        return await self.storage.async_get_storage_info(hass)


# Global singleton instance
PERSISTENT_STATISTICS_MANAGER = PersistentStatisticsManager()