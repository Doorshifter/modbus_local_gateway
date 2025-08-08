"""
Storage manager for Modbus statistics and optimization data.

This module provides persistent storage for:
- Entity statistics
- Pattern detection data
- Correlation analysis results
- Interval optimization history
"""

import json
import logging
import os
import shutil
import time
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set

from .extensible_storage import (
    CURRENT_STORAGE_VERSION, DataChunk, ChunkManager, EntityIndex, 
    StorageMigrator, DEFAULT_CHUNK_SIZE, DEFAULT_MAX_CHUNK_SIZE_KB
)

_LOGGER = logging.getLogger(__name__)

# Constants for file paths and versions
DEFAULT_BASE_PATH = "modbus_state"
META_FILENAME = "meta.json"

class StatisticsStorageManager:
    """Manages persistent storage for Modbus statistics and optimization data."""
    
    _instance = None
    _initialization_lock = asyncio.Lock()
    _initialized = False
    
    @classmethod
    def get_instance(cls, config_dir: Optional[str] = None):
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = StatisticsStorageManager(config_dir)
        return cls._instance
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the storage manager.
        
        Args:
            config_dir: Optional configuration directory path. If None, 
                       uses default Home Assistant config directory.
        """
        # Determine base storage directory
        if config_dir:
            self.base_path = Path(config_dir) / DEFAULT_BASE_PATH
        else:
            # Use Home Assistant config directory if available
            config_dir = os.environ.get("HASS_CONFIG", None)
            if config_dir:
                self.base_path = Path(config_dir) / DEFAULT_BASE_PATH
            else:
                # Fallback to current directory
                self.base_path = Path.cwd() / DEFAULT_BASE_PATH
        
        # File paths for storage
        self.meta_path = self.base_path / META_FILENAME
        self.entity_stats_path = self.base_path / "entity_stats.json"
        self.patterns_path = self.base_path / "patterns.json"
        self.clusters_path = self.base_path / "clusters.json"
        self.correlation_path = self.base_path / "correlation_matrix.json"
        self.interval_history_path = self.base_path / "interval_history.json"
        
        # Home Assistant instance
        self._hass = None
        
        # In-memory caches
        self._entity_stats_cache = {}
        self._patterns_cache = {}
        self._clusters_cache = {}
        self._correlation_cache = {}
        self._interval_history_cache = {}
        
        # Cache timestamps
        self._last_entity_stats_update = 0
        self._last_patterns_update = 0
        self._last_clusters_update = 0
        self._last_correlation_update = 0
        self._last_interval_history_update = 0
        
        # Last write timestamps
        self._last_write_times = {
            "entity_stats": 0,
            "patterns": 0,
            "clusters": 0,
            "correlation": 0,
            "interval_history": 0,
        }
        
        # Enhanced storage components
        self.metadata_path = self.base_path / "metadata"
        self.chunks_path = self.base_path / "chunks"
        self.indexes_path = self.base_path / "indexes"
        self.archives_path = self.base_path / "archives"
        
        # Component managers for enhanced storage
        self.chunk_manager = None
        self.entity_index = None
        
        # Storage metadata for enhanced version
        self.meta = None
        self.storage_meta = {
            "version": CURRENT_STORAGE_VERSION,
            "created": datetime.utcnow().isoformat(),
            "last_update": datetime.utcnow().isoformat(),
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "max_chunk_size_kb": DEFAULT_MAX_CHUNK_SIZE_KB,
            "last_compaction": None,
            "last_backup": None
        }
    
    def set_hass(self, hass):
        """Set Home Assistant instance for async operations."""
        self._hass = hass
    
    async def async_initialize(self, force: bool = False) -> bool:
        """Initialize storage asynchronously.
        
        Args:
            force: Force initialization even if already initialized
            
        Returns:
            bool: True if initialization was successful
        """
        if StatisticsStorageManager._initialized and not force:
            return True
        
        async with self._initialization_lock:
            if StatisticsStorageManager._initialized and not force:
                return True
                
            _LOGGER.info("Starting async initialization of statistics storage")
            
            # Create necessary directories asynchronously
            await self._async_ensure_directories()
            
            # Initialize enhanced storage components
            self.chunk_manager = ChunkManager(self.chunks_path)
            self.entity_index = EntityIndex(self.indexes_path / "entity_index.json")
            
            # Load metadata asynchronously
            await self._async_load_metadata()
            
            # Load minimal data needed for startup
            await self._async_load_minimal_data()
            
            StatisticsStorageManager._initialized = True
            _LOGGER.info("Statistics storage initialized at %s", self.base_path)
            return True

    async def _async_ensure_directories(self):
        """Ensure all necessary directories exist."""
        if not self._hass:
            # Create directories synchronously as fallback
            for directory in [self.base_path, self.metadata_path, 
                             self.chunks_path, self.indexes_path, 
                             self.archives_path]:
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    _LOGGER.error("Failed to create directory %s: %s", directory, e)
            return
            
        # Create directories in executor
        def _create_directories():
            for directory in [self.base_path, self.metadata_path, 
                             self.chunks_path, self.indexes_path, 
                             self.archives_path]:
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    _LOGGER.error("Failed to create directory %s: %s", directory, e)
                    
        await self._hass.async_add_executor_job(_create_directories)
    
    async def _async_load_metadata(self):
        """Load metadata asynchronously."""
        if not self._hass:
            # Use synchronous loading as fallback
            self._load_meta_sync()
            return
        
        # Load original metadata
        if await self._async_path_exists(self.meta_path):
            try:
                self.meta = await self._async_load_json_file(self.meta_path)
            except Exception as e:
                _LOGGER.error("Error loading metadata: %s", e)
                self.meta = self._create_default_meta()
        else:
            self.meta = self._create_default_meta()
            
        # Load enhanced storage metadata
        meta_path = self.metadata_path / "storage_meta.json"
        
        if await self._async_path_exists(meta_path):
            try:
                self.storage_meta = await self._async_load_json_file(meta_path)
            except Exception as e:
                _LOGGER.error("Error loading enhanced storage metadata: %s", e)
    
    def _create_default_meta(self):
        """Create default metadata structure."""
        return {
            "version": "1.0.0",
            "created": datetime.utcnow().isoformat(),
            "last_update": datetime.utcnow().isoformat(),
            "files": {
                "entity_stats": "entity_stats.json",
                "patterns": "patterns.json",
                "clusters": "clusters.json",
                "correlation": "correlation_matrix.json",
                "interval_history": "interval_history.json",
            }
        }
    
    def _load_meta_sync(self):
        """Load metadata synchronously."""
        if self.meta_path.exists():
            try:
                with open(self.meta_path, "r") as f:
                    self.meta = json.load(f)
            except Exception as e:
                _LOGGER.error("Error loading metadata: %s", e)
                self.meta = self._create_default_meta()
        else:
            self.meta = self._create_default_meta()
    
    async def _async_load_minimal_data(self):
        """Load minimal data needed for startup."""
        # Only load entity_stats and patterns
        if await self._async_path_exists(self.entity_stats_path):
            self._entity_stats_cache = await self._async_load_json_file(
                self.entity_stats_path, {}
            )
            self._last_entity_stats_update = time.time()
            
        if await self._async_path_exists(self.patterns_path):
            self._patterns_cache = await self._async_load_json_file(
                self.patterns_path, {}
            )
            self._last_patterns_update = time.time()
    
    async def _async_path_exists(self, path):
        """Check if path exists asynchronously."""
        if not self._hass:
            return path.exists()
            
        def _check_exists():
            return path.exists()
            
        return await self._hass.async_add_executor_job(_check_exists)
    
    async def _async_load_json_file(self, path, default=None):
        """Load JSON file asynchronously."""
        if default is None:
            default = {}
            
        if not self._hass:
            return self._load_json_file_sync(path, default)
            
        def _load_file():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                _LOGGER.warning("Failed to load %s: %s", path, e)
                return default
                
        return await self._hass.async_add_executor_job(_load_file)
    
    def _load_json_file_sync(self, path, default=None):
        """Load JSON file synchronously."""
        if default is None:
            default = {}
            
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            _LOGGER.warning("Failed to load %s: %s", path, e)
            return default
    
    async def async_save_json_file(self, path, data):
        """Save JSON file asynchronously."""
        if not self._hass:
            return self._save_json_file_sync(path, data)
            
        def _save_file():
            try:
                # Ensure parent directory exists
                path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write to temp file first
                temp_path = path.with_suffix(".tmp")
                with open(temp_path, "w") as f:
                    json.dump(data, f, indent=2)
                    
                # Replace original file atomically
                shutil.move(temp_path, path)
                return True
            except Exception as e:
                _LOGGER.error("Failed to save %s: %s", path, e)
                # Clean up temp file if it exists
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                return False
                
        return await self._hass.async_add_executor_job(_save_file)
    
    def _save_json_file_sync(self, path, data):
        """Save JSON file synchronously."""
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temp file first
            temp_path = path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
                
            # Replace original file atomically
            shutil.move(temp_path, path)
            return True
        except Exception as e:
            _LOGGER.error("Failed to save %s: %s", path, e)
            # Clean up temp file if it exists
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            return False
    
    async def async_get_entity_stats(self, entity_id: Optional[str] = None) -> Union[Dict[str, Any], Any, None]:
        """Get entity statistics asynchronously."""
        # Check if we have data in cache
        if not self._entity_stats_cache and self._hass:
            # Load data on demand
            if await self._async_path_exists(self.entity_stats_path):
                self._entity_stats_cache = await self._async_load_json_file(self.entity_stats_path, {})
                self._last_entity_stats_update = time.time()
            
        # Return requested data
        if entity_id:
            return self._entity_stats_cache.get(entity_id)
        return self._entity_stats_cache.copy()
    
    def get_entity_stats(self, entity_id: Optional[str] = None) -> Union[Dict[str, Any], Any, None]:
        """Get entity statistics synchronously."""
        # If data is not in cache and not initialized yet, return empty data
        if not self._entity_stats_cache and not StatisticsStorageManager._initialized:
            if entity_id:
                return None
            return {}
            
        # Load data if needed
        if not self._entity_stats_cache and self.entity_stats_path.exists():
            try:
                with open(self.entity_stats_path, "r") as f:
                    self._entity_stats_cache = json.load(f)
                self._last_entity_stats_update = time.time()
            except Exception as e:
                _LOGGER.error("Error loading entity stats: %s", e)
                self._entity_stats_cache = {}
        
        # Return requested data
        if entity_id:
            return self._entity_stats_cache.get(entity_id)
        return self._entity_stats_cache.copy()
    
    async def async_save_entity_stats(self, entity_id: str, stats: Dict[str, Any], force: bool = False) -> bool:
        """Save entity statistics asynchronously."""
        if entity_id != "dummy" or force:
            # Update cache
            if entity_id != "dummy":
                self._entity_stats_cache[entity_id] = stats
            
            # Save if forced or due for write
            now = time.time()
            if force or now - self._last_write_times.get("entity_stats", 0) >= 3600:
                success = await self.async_save_json_file(self.entity_stats_path, self._entity_stats_cache)
                if success:
                    self._last_write_times["entity_stats"] = now
                return success
        return True
    
    def save_entity_stats(self, entity_id: str, stats: Dict[str, Any], force: bool = False) -> bool:
        """Save entity statistics synchronously.
        
        If Home Assistant instance is available, schedules async save instead.
        """
        if entity_id != "dummy" or force:
            # Update cache
            if entity_id != "dummy":
                self._entity_stats_cache[entity_id] = stats
            
            # Schedule async save if we have hass
            if self._hass:
                self._hass.async_create_task(
                    self.async_save_entity_stats(entity_id, stats, force)
                )
                return True
            
            # Otherwise save synchronously if forced
            now = time.time()
            if force or now - self._last_write_times.get("entity_stats", 0) >= 3600:
                success = self._save_json_file_sync(self.entity_stats_path, self._entity_stats_cache)
                if success:
                    self._last_write_times["entity_stats"] = now
                return success
        return True
    
    # === PATTERNS METHODS ===
    
    async def async_get_patterns(self) -> Dict[str, Any]:
        """Get patterns data asynchronously."""
        if not self._patterns_cache and self._hass:
            # Load data if needed
            if await self._async_path_exists(self.patterns_path):
                self._patterns_cache = await self._async_load_json_file(self.patterns_path, {})
                self._last_patterns_update = time.time()
                
        return self._patterns_cache.copy()
    
    def get_patterns(self) -> Dict[str, Any]:
        """Get pattern detection data."""
        # If data is not in cache, load it
        if not self._patterns_cache and self.patterns_path.exists():
            try:
                with open(self.patterns_path, "r") as f:
                    self._patterns_cache = json.load(f)
                self._last_patterns_update = time.time()
            except Exception as e:
                _LOGGER.error("Error loading patterns: %s", e)
                self._patterns_cache = {}
        
        return self._patterns_cache.copy()
    
    async def async_save_patterns(self, patterns: Dict[str, Any], force: bool = False) -> bool:
        """Save pattern detection data asynchronously."""
        # Update cache
        self._patterns_cache = patterns
        
        # Save if forced or due for write
        now = time.time()
        if force or now - self._last_write_times.get("patterns", 0) >= 7200:
            success = await self.async_save_json_file(self.patterns_path, patterns)
            if success:
                self._last_write_times["patterns"] = now
            return success
        return True
    
    def save_patterns(self, patterns: Dict[str, Any], force: bool = False) -> bool:
        """Save pattern detection data synchronously."""
        # Update cache
        self._patterns_cache = patterns
        
        # Schedule async save if we have hass
        if self._hass:
            self._hass.async_create_task(
                self.async_save_patterns(patterns, force)
            )
            return True
        
        # Otherwise save synchronously if forced
        now = time.time()
        if force or now - self._last_write_times.get("patterns", 0) >= 7200:
            success = self._save_json_file_sync(self.patterns_path, patterns)
            if success:
                self._last_write_times["patterns"] = now
            return success
        return True
    
    # === CLUSTERS METHODS ===
    
    async def async_get_clusters(self) -> Dict[str, List[str]]:
        """Get clusters data asynchronously."""
        if not self._clusters_cache and self._hass:
            # Load data if needed
            if await self._async_path_exists(self.clusters_path):
                self._clusters_cache = await self._async_load_json_file(self.clusters_path, {"default_cluster": []})
                self._last_clusters_update = time.time()
                
        return self._clusters_cache.copy() if self._clusters_cache else {"default_cluster": []}
    
    def get_clusters(self) -> Dict[str, List[str]]:
        """Get cluster definitions."""
        # If data is not in cache, load it
        if not self._clusters_cache and self.clusters_path.exists():
            try:
                with open(self.clusters_path, "r") as f:
                    self._clusters_cache = json.load(f)
                self._last_clusters_update = time.time()
            except Exception as e:
                _LOGGER.error("Error loading clusters: %s", e)
                self._clusters_cache = {"default_cluster": []}
        
        return self._clusters_cache.copy() if self._clusters_cache else {"default_cluster": []}
    
    async def async_save_clusters(self, clusters: Dict[str, List[str]], force: bool = False) -> bool:
        """Save cluster definitions asynchronously."""
        # Update cache
        self._clusters_cache = clusters
        
        # Save if forced or due for write
        now = time.time()
        if force or now - self._last_write_times.get("clusters", 0) >= 86400:
            success = await self.async_save_json_file(self.clusters_path, clusters)
            if success:
                self._last_write_times["clusters"] = now
            return success
        return True
    
    def save_clusters(self, clusters: Dict[str, List[str]], force: bool = False) -> bool:
        """Save cluster definitions synchronously."""
        # Update cache
        self._clusters_cache = clusters
        
        # Schedule async save if we have hass
        if self._hass:
            self._hass.async_create_task(
                self.async_save_clusters(clusters, force)
            )
            return True
        
        # Otherwise save synchronously if forced
        now = time.time()
        if force or now - self._last_write_times.get("clusters", 0) >= 86400:
            success = self._save_json_file_sync(self.clusters_path, clusters)
            if success:
                self._last_write_times["clusters"] = now
            return success
        return True
    
    # === CORRELATION METHODS ===
    
    async def async_get_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get correlation matrix asynchronously."""
        if not self._correlation_cache and self._hass:
            # Load data if needed
            if await self._async_path_exists(self.correlation_path):
                # Check file size first
                def _get_size():
                    if self.correlation_path.exists():
                        return self.correlation_path.stat().st_size
                    return 0
                    
                size = await self._hass.async_add_executor_job(_get_size)
                if size < 1_000_000:  # Skip if > 1MB
                    self._correlation_cache = await self._async_load_json_file(self.correlation_path, {})
                    self._last_correlation_update = time.time()
                
        return self._correlation_cache
    
    def get_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get correlation matrix."""
        # If data is not in cache, load it
        if not self._correlation_cache and self.correlation_path.exists():
            # Skip if file is too large
            try:
                file_size = self.correlation_path.stat().st_size
                if file_size < 1_000_000:  # Skip if > 1MB
                    with open(self.correlation_path, "r") as f:
                        self._correlation_cache = json.load(f)
                    self._last_correlation_update = time.time()
                else:
                    _LOGGER.info("Skipping large correlation matrix file (%d bytes)", file_size)
            except Exception as e:
                _LOGGER.error("Error loading correlation matrix: %s", e)
                self._correlation_cache = {}
        
        return self._correlation_cache
    
    async def async_save_correlation_matrix(self, matrix: Dict[str, Dict[str, float]], force: bool = False) -> bool:
        """Save correlation matrix asynchronously."""
        # Update cache
        self._correlation_cache = matrix
        
        # Save if forced or due for write
        now = time.time()
        if force or now - self._last_write_times.get("correlation", 0) >= 86400:
            success = await self.async_save_json_file(self.correlation_path, matrix)
            if success:
                self._last_write_times["correlation"] = now
            return success
        return True
    
    def save_correlation_matrix(self, matrix: Dict[str, Dict[str, float]], force: bool = False) -> bool:
        """Save correlation matrix synchronously."""
        # Update cache
        self._correlation_cache = matrix
        
        # Schedule async save if we have hass
        if self._hass:
            self._hass.async_create_task(
                self.async_save_correlation_matrix(matrix, force)
            )
            return True
        
        # Otherwise save synchronously if forced
        now = time.time()
        if force or now - self._last_write_times.get("correlation", 0) >= 86400:
            success = self._save_json_file_sync(self.correlation_path, matrix)
            if success:
                self._last_write_times["correlation"] = now
            return success
        return True
    
    # === INTERVAL HISTORY METHODS ===
    
    async def async_get_interval_history(self, entity_id=None, start_time=None, end_time=None):
        """Get interval history asynchronously."""
        if not self._interval_history_cache and self._hass:
            # Load data if needed
            if await self._async_path_exists(self.interval_history_path):
                self._interval_history_cache = await self._async_load_json_file(self.interval_history_path, {})
                self._last_interval_history_update = time.time()
                
        # Return data with filtering
        return self._filter_interval_history(self._interval_history_cache, entity_id, start_time, end_time)
    
    def get_interval_history(self, entity_id=None, start_time=None, end_time=None):
        """Get interval history for entities."""
        # If data is not in cache, load it
        if not self._interval_history_cache and self.interval_history_path.exists():
            try:
                with open(self.interval_history_path, "r") as f:
                    self._interval_history_cache = json.load(f)
                self._last_interval_history_update = time.time()
            except Exception as e:
                _LOGGER.error("Error loading interval history: %s", e)
                self._interval_history_cache = {}
                
        # Return data with filtering
        return self._filter_interval_history(self._interval_history_cache, entity_id, start_time, end_time)
    
    def _filter_interval_history(self, history, entity_id=None, start_time=None, end_time=None):
        """Filter interval history based on parameters."""
        if not history:
            return {"entities": {}} if not entity_id else {entity_id: []}
        
        # Filter for specific entity if requested
        if entity_id:
            if isinstance(history, dict) and "entities" in history:
                entities_data = history.get("entities", {})
                result = {entity_id: entities_data.get(entity_id, {})}
            else:
                result = {entity_id: history.get(entity_id, [])}
            
            # Apply time filters if needed
            if start_time or end_time:
                for ent_id, data in result.items():
                    if isinstance(data, dict) and "history" in data:
                        filtered_history = [
                            item for item in data["history"]
                            if (not start_time or item.get("timestamp", 0) >= start_time) and
                               (not end_time or item.get("timestamp", 0) <= end_time)
                        ]
                        data["history"] = filtered_history
                    elif isinstance(data, list):
                        filtered_data = [
                            item for item in data
                            if (not start_time or item.get("timestamp", 0) >= start_time) and
                               (not end_time or item.get("timestamp", 0) <= end_time)
                        ]
                        result[ent_id] = filtered_data
            
            return result
        
        # Return all history
        return history
    
    async def async_save_interval_history(self, history: Dict[str, Any], force: bool = False) -> bool:
        """Save interval optimization history asynchronously."""
        # Update cache
        self._interval_history_cache = history
        
        # Save if forced or due for write
        now = time.time()
        if force or now - self._last_write_times.get("interval_history", 0) >= 14400:
            success = await self.async_save_json_file(self.interval_history_path, history)
            if success:
                self._last_write_times["interval_history"] = now
            return success
        return True
    
    def save_interval_history(self, history: Dict[str, Any], force: bool = False) -> bool:
        """Save interval optimization history synchronously."""
        # Update cache
        self._interval_history_cache = history
        
        # Schedule async save if we have hass
        if self._hass:
            self._hass.async_create_task(
                self.async_save_interval_history(history, force)
            )
            return True
        
        # Otherwise save synchronously if forced
        now = time.time()
        if force or now - self._last_write_times.get("interval_history", 0) >= 14400:
            success = self._save_json_file_sync(self.interval_history_path, history)
            if success:
                self._last_write_times["interval_history"] = now
            return success
        return True
    
    async def async_get_storage_info(self, hass) -> Dict[str, Any]:
        """Get storage system information asynchronously."""
        # Get basic info that's safe to access without blocking
        info = {
            "version": self.storage_meta.get("version", self.meta.get("version", "unknown") if self.meta else "unknown"),
            "created": self.storage_meta.get("created"),
            "last_update": self.storage_meta.get("last_update"),
            "stats_count": len(self._entity_stats_cache),
            "patterns_count": len(self._patterns_cache) if self._patterns_cache else 0,
            "clusters_count": len(self._clusters_cache) if self._clusters_cache else 0,
            "storage_location": str(self.base_path),
            "initialization_complete": StatisticsStorageManager._initialized,
            "files": {}
        }
        
        # Get file info in executor
        def _get_file_info():
            files = {}
            try:
                for name, path in [
                    ("entity_stats", self.entity_stats_path),
                    ("patterns", self.patterns_path),
                    ("clusters", self.clusters_path),
                    ("correlation", self.correlation_path),
                    ("interval_history", self.interval_history_path),
                ]:
                    if path.exists():
                        files[name] = {
                            "size_kb": round(path.stat().st_size / 1024, 1),
                            "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                            "exists": True
                        }
                    else:
                        files[name] = {"exists": False}
            except Exception as e:
                _LOGGER.debug("Error getting file info: %s", e)
            return files
            
        info["files"] = await hass.async_add_executor_job(_get_file_info)
        
        return info
        
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the storage system."""
        # Basic info that doesn't require file I/O
        info = {
            "version": self.storage_meta.get("version", self.meta.get("version", "unknown") if self.meta else "unknown"),
            "created": self.storage_meta.get("created"),
            "last_update": self.storage_meta.get("last_update"),
            "stats_count": len(self._entity_stats_cache),
            "patterns_count": len(self._patterns_cache) if self._patterns_cache else 0,
            "clusters_count": len(self._clusters_cache) if self._clusters_cache else 0,
            "storage_location": str(self.base_path),
            "initialization_complete": StatisticsStorageManager._initialized,
            "files": {}
        }
        
        # Add file information
        try:
            for name, path in [
                ("entity_stats", self.entity_stats_path),
                ("patterns", self.patterns_path),
                ("clusters", self.clusters_path),
                ("correlation", self.correlation_path),
                ("interval_history", self.interval_history_path),
            ]:
                if path.exists():
                    try:
                        info["files"][name] = {
                            "size_kb": round(path.stat().st_size / 1024, 1),
                            "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                            "exists": True
                        }
                    except Exception:
                        info["files"][name] = {"exists": True}
                else:
                    info["files"][name] = {"exists": False}
        except Exception as e:
            _LOGGER.debug("Error getting file info: %s", e)
            
        return info

# Create singleton instance without initialization
PERSISTENT_STATISTICS_MANAGER = StatisticsStorageManager.get_instance()