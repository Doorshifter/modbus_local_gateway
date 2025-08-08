"""
Storage manager for persistent statistics.
Handles reading and writing statistics to disk.
"""
import json
import logging
import os
import time
import hashlib
import asyncio
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
        self.hass = None
        
        # Add cache for storage info to reduce I/O operations
        self._storage_info_cache = {}
        self._cache_timestamp = 0
        self._cache_ttl = 60  # Cache TTL in seconds
        
        # Create directory if it doesn't exist - non-blocking
        try:
            # This is safe to do during import as it's lightweight
            os.makedirs(base_path, exist_ok=True)
            _LOGGER.debug("Statistics directory exists at %s", base_path)
        except Exception as e:
            _LOGGER.error("Could not create statistics directory: %s", e)
    
    def set_hass(self, hass):
        """Set Home Assistant instance for async operations."""
        self.hass = hass
        
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
    
    def _blocking_list_directory(self, directory_path: Path) -> List[str]:
        """List directory contents - BLOCKING VERSION.
        
        Args:
            directory_path: Path to directory
            
        Returns:
            List of filenames in directory or empty list if directory doesn't exist
        """
        try:
            if os.path.exists(directory_path):
                return os.listdir(directory_path)
            return []
        except Exception as e:
            _LOGGER.error("Error listing directory %s: %s", directory_path, e)
            return []
    
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
        _LOGGER.debug("Called blocking get_patterns() - consider using async_get_patterns instead")
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
        _LOGGER.debug("Called blocking save_patterns() - consider using async_save_patterns instead")
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
        _LOGGER.debug("Called blocking get_clusters() - consider using async_get_clusters instead")
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
        _LOGGER.debug("Called blocking save_clusters() - consider using async_save_clusters instead")
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
        _LOGGER.debug("Called blocking save_metadata_file() - consider using async_save_metadata_file instead")
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
        _LOGGER.debug("Called blocking get_metadata_file() - consider using async_get_metadata_file instead")
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
    
    async def async_get_storage_info(self, hass) -> Dict:
        """Get storage information asynchronously.
        
        Returns:
            Storage information
        """
        # Check if cache is valid
        current_time = time.time()
        if self._storage_info_cache and (current_time - self._cache_timestamp) < self._cache_ttl:
            return self._storage_info_cache
            
        # Run blocking operation in executor
        result = await hass.async_add_executor_job(self._blocking_get_storage_info)
        
        # Update cache
        self._storage_info_cache = result
        self._cache_timestamp = current_time
        
        return result
    
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
            "clusters_count": 0,
            "timestamp": time.time()
        }
        
        # Get metadata about the storage
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r") as f:
                    meta = json.load(f)
                    result.update(meta)
            except Exception as e:
                _LOGGER.error("Error reading meta file: %s", e)
                    
        # Get pattern info
        patterns_data = self._blocking_get_patterns()
        result["patterns_count"] = len(patterns_data.get("patterns", {}))
        
        # Get clusters info
        clusters = self._blocking_get_clusters()
        result["clusters_count"] = len(clusters)
        
        # List directory contents - THIS IS THE PROBLEMATIC PART
        # We use the dedicated method to avoid calling os.listdir directly
        file_list = self._blocking_list_directory(self.base_path)
        for filename in file_list:
            file_path = self.base_path / filename
            if os.path.isfile(file_path):
                try:
                    file_size = os.path.getsize(file_path)
                    result["files"][filename] = {
                        "size_bytes": file_size,
                        "last_modified": os.path.getmtime(file_path),
                        "last_modified_str": time.ctime(os.path.getmtime(file_path))
                    }
                    result["total_size_bytes"] += file_size
                except Exception as e:
                    _LOGGER.debug("Error getting stats for %s: %s", filename, e)
                    
        return result
    
    def get_storage_info(self) -> Dict:
        """Get storage information (blocking).
        
        WARNING: This is a blocking call and should not be used in the event loop.
        Use async_get_storage_info() instead when in an async context.
        
        Returns:
            Storage information
        """
        _LOGGER.warning("Called blocking get_storage_info() - use async_get_storage_info instead")
        
        # Check if cache is valid
        current_time = time.time()
        if self._storage_info_cache and (current_time - self._cache_timestamp) < self._cache_ttl:
            return self._storage_info_cache
            
        # Get storage info (blocking)
        result = self._blocking_get_storage_info()
        
        # Update cache
        self._storage_info_cache = result
        self._cache_timestamp = current_time
        
        return result
                    
    async def async_get_interval_history(self, hass) -> Dict[str, List[Dict[str, Any]]]:
        """Get interval history for entities asynchronously.
        
        Returns:
            Dictionary of entity ID to interval history
        """
        data = await self.async_get_metadata_file(hass, "interval_history")
        if data:
            return data
        return {}

    def get_interval_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get interval history for entities.
        
        Returns:
            Dictionary of entity ID to interval history
        """
        # Warning: This is a blocking call and should not be used in the event loop
        _LOGGER.debug("Called blocking get_interval_history() - consider using async_get_interval_history instead")
        data = self.get_metadata_file("interval_history")
        if data:
            return data
        return {}
        
    async def async_save_interval_history(self, hass, interval_history: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Save interval history asynchronously.
        
        Args:
            interval_history: Dictionary of entity ID to interval history
            
        Returns:
            True if successful
        """
        return await self.async_save_metadata_file(hass, "interval_history", interval_history)

    def save_interval_history(self, interval_history: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Save interval history.
        
        Args:
            interval_history: Dictionary of entity ID to interval history
            
        Returns:
            True if successful
        """
        # Warning: This is a blocking call and should not be used in the event loop
        _LOGGER.debug("Called blocking save_interval_history() - consider using async_save_interval_history instead")
        return self.save_metadata_file("interval_history", interval_history)


class PersistentStatisticsManager:
    """Manager for persistent statistics across restarts."""
    
    _initialization_lock = asyncio.Lock()
    _initialized = False
    
    def __init__(self):
        """Initialize the manager."""
        self.storage = None
        self.enabled = False
        self.hass = None
        
        # Initialize component references as None
        self._correlation_manager = None
        self._pattern_detector = None
        self._interval_manager = None
        self._anomaly_detector = None
        self._resource_adapter = None
        self._self_healing_system = None
        
    def set_hass(self, hass):
        """Set Home Assistant instance for async operations."""
        self.hass = hass
        if self.storage:
            self.storage.set_hass(hass)
    
    async def async_initialize(self, force: bool = False) -> bool:
        """Initialize the manager asynchronously.
        
        Args:
            force: Force initialization even if already initialized
            
        Returns:
            bool: True if initialization was successful
        """
        if PersistentStatisticsManager._initialized and not force:
            return True
        
        async with self._initialization_lock:
            if PersistentStatisticsManager._initialized and not force:
                return True
            
            # Get configuration path
            if not self.hass:
                _LOGGER.warning("Home Assistant instance not set, using default path")
                config_path = Path("/config")
            else:
                config_path = Path(self.hass.config.path())
                
            # Set up storage path
            storage_path = config_path / "modbus_state"
            
            # Initialize storage
            if self.hass:
                # Run in executor to avoid blocking
                def _init_storage():
                    self.storage = StatisticsStorageManager(storage_path)
                    self.storage.set_hass(self.hass)
                    self.enabled = True
                    _LOGGER.info("Persistent statistics initialized with storage path: %s", storage_path)
                    return True
                
                result = await self.hass.async_add_executor_job(_init_storage)
            else:
                # Synchronous fallback
                self.storage = StatisticsStorageManager(storage_path)
                self.enabled = True
                _LOGGER.info("Persistent statistics initialized with storage path: %s", storage_path)
                result = True
            
            if result:
                # Try to initialize components in executor
                if self.hass:
                    await self.hass.async_add_executor_job(self._try_initialize_components)
                else:
                    self._try_initialize_components()
                    
                PersistentStatisticsManager._initialized = True
                
            return result
        
    def initialize(self, storage_path: Path) -> bool:
        """Initialize the manager with storage path.
        
        Args:
            storage_path: Path to store statistics
            
        Returns:
            bool: True if initialization was successful
        """
        self.storage = StatisticsStorageManager(storage_path)
        if self.hass:
            self.storage.set_hass(self.hass)
        self.enabled = True
        _LOGGER.info("Persistent statistics initialized with storage path: %s", storage_path)
        
        # Try to initialize components
        self._try_initialize_components()
        
        PersistentStatisticsManager._initialized = True
        return True
        
    def _try_initialize_components(self):
        """Try to initialize optional components."""
        # Try to initialize correlation manager
        try:
            from .correlation import EntityCorrelationManager
            self._correlation_manager = EntityCorrelationManager()
            _LOGGER.debug("Correlation manager initialized")
        except ImportError:
            _LOGGER.debug("Correlation manager not available - module not found")
        
        # Try to initialize pattern detector
        try:
            from .pattern_detection import PatternDetector
            self._pattern_detector = PatternDetector()
            _LOGGER.debug("Pattern detector initialized")
        except ImportError:
            _LOGGER.debug("Pattern detector not available - module not found")
        
    def initialize_advanced_interval_manager(self):
        """Initialize the advanced interval manager."""
        try:
            from .advanced_interval_manager import ADVANCED_INTERVAL_MANAGER
            self._interval_manager = ADVANCED_INTERVAL_MANAGER
            _LOGGER.info("Advanced interval manager initialized")
        except ImportError as e:
            _LOGGER.warning("Could not initialize advanced interval manager: %s", e)
    
    async def async_get_storage_info(self, hass) -> Dict:
        """Get storage information asynchronously.
        
        Returns:
            Storage information
        """
        if not self.enabled or not self.storage:
            return {"status": "disabled"}
            
        return await self.storage.async_get_storage_info(hass)
    
    def get_storage_info(self) -> Dict:
        """Get storage information.
        
        WARNING: This is a blocking call and should not be used in the event loop.
        Use async_get_storage_info() instead when in an async context.
        
        Returns:
            Storage information
        """
        if not self.enabled or not self.storage:
            return {"status": "disabled"}
            
        # This is blocking and should not be called from the event loop
        return self.storage.get_storage_info()
    
    @property
    def correlation_manager(self):
        """Get the correlation manager.
        
        Returns:
            Correlation manager or None if not available
        """
        return self._correlation_manager
    
    @property
    def pattern_detector(self):
        """Get the pattern detector.
        
        Returns:
            Pattern detector or None if not available
        """
        return self._pattern_detector
    
    async def async_save_entity_stats(self, hass, entity_id: str, stats: Dict[str, Any]) -> bool:
        """Save statistics data for an entity asynchronously."""
        if not self.enabled or not self.storage:
            return False
            
        # Get current entity stats
        entity_stats = await self.storage.async_get_metadata_file(hass, "entity_stats") or {}
        
        # Update stats for this entity
        entity_stats[entity_id] = stats
        
        # Save updated entity stats
        return await self.storage.async_save_metadata_file(hass, "entity_stats", entity_stats)
    
    def save_entity_stats(self, entity_id: str, stats: Dict[str, Any]) -> bool:
        """Save statistics data for an entity.
        
        WARNING: This is a blocking call and should not be used in the event loop.
        Use async_save_entity_stats() instead when in an async context.
        """
        if not self.enabled or not self.storage:
            return False
            
        # Get current entity stats
        entity_stats = self.storage.get_metadata_file("entity_stats") or {}
        
        # Update stats for this entity
        entity_stats[entity_id] = stats
        
        # Save updated entity stats
        return self.storage.save_metadata_file("entity_stats", entity_stats)
    
    # ADD THESE NEW METHODS
    
    async def async_get_entity_stats(self, hass, entity_id: str = None) -> Union[Dict[str, Dict], Dict, None]:
        """Get entity statistics asynchronously.
        
        Args:
            hass: Home Assistant instance
            entity_id: Optional entity ID to get stats for. If None, returns all stats.
            
        Returns:
            Entity statistics or None if not found
        """
        if not self.enabled or not self.storage:
            return {} if entity_id is None else None
            
        entity_stats = await self.storage.async_get_metadata_file(hass, "entity_stats") or {}
        
        if entity_id is not None:
            return entity_stats.get(entity_id)
        return entity_stats
    
    def get_entity_stats(self, entity_id: str = None) -> Union[Dict[str, Dict], Dict, None]:
        """Get entity statistics.
        
        WARNING: This is a blocking call and should not be used in the event loop.
        Use async_get_entity_stats() instead when in an async context.
        
        Args:
            entity_id: Optional entity ID to get stats for. If None, returns all stats.
            
        Returns:
            Entity statistics or None if not found
        """
        _LOGGER.debug("Called blocking get_entity_stats() - consider using async_get_entity_stats instead")
        if not self.enabled or not self.storage:
            return {} if entity_id is None else None
            
        entity_stats = self.storage.get_metadata_file("entity_stats") or {}
        
        if entity_id is not None:
            return entity_stats.get(entity_id)
        return entity_stats
    
    async def async_get_patterns(self, hass) -> Dict:
        """Get pattern data asynchronously.
        
        Args:
            hass: Home Assistant instance
            
        Returns:
            Pattern data
        """
        if not self.enabled or not self.storage:
            return {}
            
        return await self.storage.async_get_patterns(hass)
    
    def get_patterns(self) -> Dict:
        """Get pattern data.
        
        WARNING: This is a blocking call and should not be used in the event loop.
        Use async_get_patterns() instead when in an async context.
        
        Returns:
            Pattern data
        """
        if not self.enabled or not self.storage:
            return {}
            
        return self.storage.get_patterns()
    
    async def async_get_clusters(self, hass) -> Dict:
        """Get cluster data asynchronously.
        
        Args:
            hass: Home Assistant instance
            
        Returns:
            Cluster data
        """
        if not self.enabled or not self.storage:
            return {}
            
        return await self.storage.async_get_clusters(hass)
    
    def get_clusters(self) -> Dict:
        """Get cluster data.
        
        WARNING: This is a blocking call and should not be used in the event loop.
        Use async_get_clusters() instead when in an async context.
        
        Returns:
            Cluster data
        """
        if not self.enabled or not self.storage:
            return {}
            
        return self.storage.get_clusters()


# Global singleton instance
PERSISTENT_STATISTICS_MANAGER = PersistentStatisticsManager()