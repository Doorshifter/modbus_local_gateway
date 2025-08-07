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
from datetime import datetime
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
        
        # File paths for original storage (for backward compatibility)
        self.meta_path = self.base_path / META_FILENAME
        self.entity_stats_path = self.base_path / "entity_stats.json"
        self.patterns_path = self.base_path / "patterns.json"
        self.clusters_path = self.base_path / "clusters.json"
        self.correlation_path = self.base_path / "correlation_matrix.json"
        self.interval_history_path = self.base_path / "interval_history.json"
        
        # Create directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize meta data
        self.meta = self._load_or_create_meta()
        
        # In-memory cache - kept for backward compatibility
        self._entity_stats_cache = {}
        self._patterns_cache = {}
        self._clusters_cache = {}
        self._correlation_cache = {}
        self._interval_history_cache = {}
        
        # Last write timestamps
        self._last_write_times = {
            "entity_stats": 0,
            "patterns": 0,
            "clusters": 0,
            "correlation": 0,
            "interval_history": 0,
        }
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Enhanced storage components
        self.metadata_path = self.base_path / "metadata"
        self.chunks_path = self.base_path / "chunks"
        self.indexes_path = self.base_path / "indexes"
        self.archives_path = self.base_path / "archives"
        
        # Ensure directories exist
        self.metadata_path.mkdir(exist_ok=True)
        self.chunks_path.mkdir(exist_ok=True)
        self.indexes_path.mkdir(exist_ok=True)
        self.archives_path.mkdir(exist_ok=True)
        
        # Component managers for enhanced storage
        self.chunk_manager = ChunkManager(self.chunks_path)
        self.entity_index = EntityIndex(self.indexes_path / "entity_index.json")
        
        # Storage metadata for enhanced version
        self.storage_meta: Dict[str, Any] = {
            "version": CURRENT_STORAGE_VERSION,
            "created": datetime.utcnow().isoformat(),
            "last_update": datetime.utcnow().isoformat(),
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "max_chunk_size_kb": DEFAULT_MAX_CHUNK_SIZE_KB,
            "last_compaction": None,
            "last_backup": None
        }
        
        # Load initial data
        self._initialize_storage()
        
        _LOGGER.info(
            "Statistics storage initialized at %s (version %s)", 
            self.base_path, 
            self.storage_meta.get("version", self.meta.get("version", "unknown"))
        )
    
    def _load_or_create_meta(self) -> Dict[str, Any]:
        """Load meta data or create if it doesn't exist."""
        if self.meta_path.exists():
            try:
                with open(self.meta_path, "r") as f:
                    meta = json.load(f)
                return meta
            except (json.JSONDecodeError, IOError) as e:
                _LOGGER.warning("Failed to load meta data: %s", e)
        
        # Create new meta data
        meta = {
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
        
        # Save new meta data
        try:
            with open(self.meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except IOError as e:
            _LOGGER.error("Failed to create meta data: %s", e)
        
        return meta
    
    def _initialize_storage(self):
        """Initialize storage system."""
        # First, check if this is a new versioned storage installation
        meta_path = self.metadata_path / "storage_meta.json"
        
        if meta_path.exists():
            # Load existing metadata for enhanced storage
            try:
                with open(meta_path, "r") as f:
                    self.storage_meta = json.load(f)
                    
                # Check version and migrate if needed
                current_version = self.storage_meta.get("version", "1.0.0")
                if current_version != CURRENT_STORAGE_VERSION:
                    # Migrate storage
                    migrator = StorageMigrator(self.base_path)
                    success, from_version, to_version = migrator.check_and_migrate(current_version)
                    
                    if success:
                        self.storage_meta["version"] = to_version
                        self.storage_meta["migrated_from"] = from_version
                        self.storage_meta["migration_time"] = datetime.utcnow().isoformat()
                        
                        # Save updated metadata
                        self._save_storage_metadata()
                    else:
                        _LOGGER.error(
                            "Failed to migrate storage from %s to %s", 
                            from_version, CURRENT_STORAGE_VERSION
                        )
            except Exception as e:
                _LOGGER.error("Error loading storage metadata: %s", e)
                # Create new metadata
                self._save_storage_metadata()
        else:
            # Check if we need to migrate from the old format
            if self.meta_path.exists():
                # Migrate from old format
                try:
                    migrator = StorageMigrator(self.base_path)
                    success, from_version, to_version = migrator.check_and_migrate("1.0.0")
                    
                    if success:
                        self.storage_meta["version"] = to_version
                        self.storage_meta["migrated_from"] = from_version
                        self.storage_meta["migration_time"] = datetime.utcnow().isoformat()
                        
                        # Save updated metadata
                        self._save_storage_metadata()
                    else:
                        _LOGGER.error(
                            "Failed to migrate storage from old format to %s", 
                            CURRENT_STORAGE_VERSION
                        )
                        
                        # Fall back to traditional loading
                        self._load_original_storage()
                        return
                except Exception as e:
                    _LOGGER.error("Error during storage migration: %s", e)
                    # Fall back to traditional loading
                    self._load_original_storage()
                    return
            else:
                # New installation
                self._save_storage_metadata()
        
        # Check if auto-compaction is due
        self._check_auto_compaction()
        
        # Also load original format for backward compatibility
        self._load_original_storage()
        
    def _load_original_storage(self):
        """Load all existing data into memory from original storage format."""
        # Load entity statistics
        if self.entity_stats_path.exists():
            try:
                with open(self.entity_stats_path, "r") as f:
                    self._entity_stats_cache = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                _LOGGER.warning("Failed to load entity statistics: %s", e)
                self._entity_stats_cache = {}
        
        # Load patterns
        if self.patterns_path.exists():
            try:
                with open(self.patterns_path, "r") as f:
                    self._patterns_cache = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                _LOGGER.warning("Failed to load patterns: %s", e)
                self._patterns_cache = {}
        
        # Load clusters
        if self.clusters_path.exists():
            try:
                with open(self.clusters_path, "r") as f:
                    self._clusters_cache = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                _LOGGER.warning("Failed to load clusters: %s", e)
                self._clusters_cache = {}
        
        # Load correlation matrix (skip if too large, will load on demand)
        if self.correlation_path.exists() and self.correlation_path.stat().st_size < 1_000_000:
            try:
                with open(self.correlation_path, "r") as f:
                    self._correlation_cache = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                _LOGGER.warning("Failed to load correlation matrix: %s", e)
                self._correlation_cache = {}
        
        # Load interval history
        if self.interval_history_path.exists():
            try:
                with open(self.interval_history_path, "r") as f:
                    self._interval_history_cache = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                _LOGGER.warning("Failed to load interval history: %s", e)
                self._interval_history_cache = {}
    
    def _save_storage_metadata(self) -> bool:
        """Save enhanced storage metadata."""
        try:
            meta_path = self.metadata_path / "storage_meta.json"
            
            # Update timestamp
            self.storage_meta["last_update"] = datetime.utcnow().isoformat()
            
            with open(meta_path, "w") as f:
                json.dump(self.storage_meta, f, indent=2)
                
            return True
        except Exception as e:
            _LOGGER.error("Error saving storage metadata: %s", e)
            return False
    
    def _check_auto_compaction(self) -> None:
        """Check if auto-compaction should run."""
        last_compaction_str = self.storage_meta.get("last_compaction")
        
        # Skip if never compacted (allow system to stabilize first)
        if not last_compaction_str:
            return
            
        try:
            last_compaction = datetime.fromisoformat(last_compaction_str)
            now = datetime.utcnow()
            
            # Check if it's time for auto-compaction (7 days since last)
            if (now - last_compaction).days >= 7:
                # Schedule compaction (don't block initialization)
                threading.Thread(
                    target=self.compact_storage,
                    daemon=True
                ).start()
        except (ValueError, TypeError) as e:
            _LOGGER.error("Error checking auto-compaction: %s", e)
    
    def _atomic_write(self, path: Path, data: Dict[str, Any]) -> bool:
        """Write data to file atomically."""
        # Create temporary file
        temp_path = path.with_suffix(".tmp")
        
        try:
            # Write to temporary file
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
            
            # Rename temporary file to target file (atomic operation)
            shutil.move(temp_path, path)
            
            # Update meta data
            self.meta["last_update"] = datetime.utcnow().isoformat()
            with open(self.meta_path, "w") as f:
                json.dump(self.meta, f, indent=2)
                
            return True
        except IOError as e:
            _LOGGER.error("Failed to write %s: %s", path, e)
            
            # Clean up temporary file
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except IOError:
                    pass
                
            return False
    
    def _should_write(self, data_type: str, min_interval: int = 3600) -> bool:
        """Check if enough time has passed since the last write."""
        now = time.time()
        if now - self._last_write_times.get(data_type, 0) >= min_interval:
            self._last_write_times[data_type] = now
            return True
        return False
    
    def get_entity_stats(self, entity_id: Optional[str] = None) -> Union[Dict[str, Any], Any, None]:
        """Get statistics for a specific entity or all entities.
        
        Args:
            entity_id: Optional entity ID. If None, returns all statistics.
            
        Returns:
            Statistics data for the entity or all entities
        """
        # First try to get from chunked storage if we have an entity index
        if self.entity_index:
            with self._lock:
                if entity_id:
                    # Get single entity
                    chunk_id = self.entity_index.get_chunk_id(entity_id)
                    if chunk_id:
                        # Get chunk
                        chunk = self.chunk_manager.get_chunk(chunk_id)
                        return chunk.get(entity_id)
                    else:
                        # Fall back to original storage
                        return self._entity_stats_cache.get(entity_id)
                else:
                    # Get all entities
                    entities_by_chunk = self.entity_index.get_entities_by_chunk()
                    
                    # If we have no chunk data yet, fall back to original storage
                    if not entities_by_chunk:
                        return self._entity_stats_cache
                    
                    # Load each chunk and extract entities
                    all_stats = {}
                    for chunk_id, entities in entities_by_chunk.items():
                        chunk = self.chunk_manager.get_chunk(chunk_id)
                        
                        for entity_id in entities:
                            entity_data = chunk.get(entity_id)
                            if entity_data:
                                all_stats[entity_id] = entity_data
                    
                    # Add any entities from original storage that aren't in chunks
                    for entity_id, stats in self._entity_stats_cache.items():
                        if entity_id not in all_stats:
                            all_stats[entity_id] = stats
                                
                    return all_stats
        
        # Fall back to original storage
        if entity_id:
            return self._entity_stats_cache.get(entity_id)
        return self._entity_stats_cache
    
    def save_entity_stats(self, entity_id: str, stats: Dict[str, Any], force: bool = False) -> bool:
        """Save statistics for a specific entity.
        
        Args:
            entity_id: Entity ID
            stats: Statistics data
            force: Force immediate write to disk, otherwise may batch writes
            
        Returns:
            True if saved successfully, False otherwise
        """
        # Update cache
        self._entity_stats_cache[entity_id] = stats
        
        # Save to enhanced storage if available
        with self._lock:
            # Get or assign chunk ID
            chunk_id = self.entity_index.get_chunk_id(entity_id)
            
            if not chunk_id:
                # Assign to a chunk using hash distribution
                import hashlib
                hash_value = int(hashlib.md5(entity_id.encode()).hexdigest(), 16)
                chunk_id = str(hash_value % 100)  # Use 100 possible chunks
                
                # Update index
                self.entity_index.set_chunk_id(entity_id, chunk_id)
                
            # Get chunk and update
            chunk = self.chunk_manager.get_chunk(chunk_id)
            chunk.set(entity_id, stats)
            
            # Save index if dirty
            if self.entity_index.dirty:
                self.entity_index.save()
            
            # Update storage metadata
            self.storage_meta["last_update"] = datetime.utcnow().isoformat()
            
            # Save if forced or due for write
            if force or self._should_write("entity_stats", min_interval=3600):  # 1 hour
                # Save chunk
                chunk.save(compress=chunk.should_compress())
                
                # Also update original format for backward compatibility
                self._atomic_write(self.entity_stats_path, self._entity_stats_cache)
                
                # Save storage metadata
                self._save_storage_metadata()
        
        return True
    
    def save_patterns(self, patterns: Dict[str, Any], force: bool = False) -> bool:
        """Save pattern detection data.
        
        Args:
            patterns: Pattern data
            force: Force immediate write to disk, otherwise may batch writes
            
        Returns:
            True if saved successfully, False otherwise
        """
        # Update cache
        self._patterns_cache = patterns
        
        # Save to metadata directory
        try:
            patterns_path = self.metadata_path / "patterns.json"
            
            with open(patterns_path, "w") as f:
                json.dump(patterns, f, indent=2)
        except Exception as e:
            _LOGGER.warning("Failed to save patterns to enhanced storage: %s", e)
        
        # Write to original location if forced or enough time passed
        if force or self._should_write("patterns", min_interval=7200):  # 2 hours
            return self._atomic_write(self.patterns_path, self._patterns_cache)
        
        return True
    
    def get_patterns(self) -> Dict[str, Any]:
        """Get pattern detection data."""
        # First try enhanced storage
        patterns_path = self.metadata_path / "patterns.json"
        if patterns_path.exists():
            try:
                with open(patterns_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                _LOGGER.warning("Failed to load patterns from enhanced storage: %s", e)
        
        # Fall back to original cache
        return self._patterns_cache
    
    def save_clusters(self, clusters: Dict[str, List[str]], force: bool = False) -> bool:
        """Save cluster definitions.
        
        Args:
            clusters: Cluster data
            force: Force immediate write to disk, otherwise may batch writes
            
        Returns:
            True if saved successfully, False otherwise
        """
        # Update cache
        self._clusters_cache = clusters
        
        # Save to metadata directory
        try:
            clusters_path = self.metadata_path / "clusters.json"
            
            with open(clusters_path, "w") as f:
                json.dump(clusters, f, indent=2)
        except Exception as e:
            _LOGGER.warning("Failed to save clusters to enhanced storage: %s", e)
        
        # Write to original location if forced or enough time passed
        if force or self._should_write("clusters", min_interval=86400):  # 24 hours
            return self._atomic_write(self.clusters_path, self._clusters_cache)
        
        return True
    
    def get_clusters(self) -> Dict[str, List[str]]:
        """Get cluster definitions."""
        # First try enhanced storage
        clusters_path = self.metadata_path / "clusters.json"
        if clusters_path.exists():
            try:
                with open(clusters_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                _LOGGER.warning("Failed to load clusters from enhanced storage: %s", e)
        
        # Fall back to original cache
        return self._clusters_cache
    
    def save_correlation_matrix(self, matrix: Dict[str, Dict[str, float]], force: bool = False) -> bool:
        """Save correlation matrix.
        
        Args:
            matrix: Correlation matrix
            force: Force immediate write to disk, otherwise may batch writes
            
        Returns:
            True if saved successfully, False otherwise
        """
        # Update cache
        self._correlation_cache = matrix
        
        # Save to metadata directory
        try:
            correlation_path = self.metadata_path / "correlation_matrix.json"
            
            with open(correlation_path, "w") as f:
                json.dump(matrix, f, indent=2)
        except Exception as e:
            _LOGGER.warning("Failed to save correlation matrix to enhanced storage: %s", e)
        
        # Write to original location if forced or enough time passed
        if force or self._should_write("correlation", min_interval=86400):  # 24 hours
            return self._atomic_write(self.correlation_path, self._correlation_cache)
        
        return True
    
    def get_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get correlation matrix."""
        # First try enhanced storage
        correlation_path = self.metadata_path / "correlation_matrix.json"
        if correlation_path.exists():
            try:
                with open(correlation_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                _LOGGER.warning("Failed to load correlation matrix from enhanced storage: %s", e)
        
        # Load from original path if not in cache
        if not self._correlation_cache and self.correlation_path.exists():
            try:
                with open(self.correlation_path, "r") as f:
                    self._correlation_cache = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                _LOGGER.warning("Failed to load correlation matrix: %s", e)
                self._correlation_cache = {}
        
        return self._correlation_cache
    
    def save_interval_history(self, history: Dict[str, List[Dict[str, Any]]], force: bool = False) -> bool:
        """Save interval optimization history.
        
        Args:
            history: Interval history data
            force: Force immediate write to disk, otherwise may batch writes
            
        Returns:
            True if saved successfully, False otherwise
        """
        # Update cache
        self._interval_history_cache = history
        
        # Save to metadata directory
        try:
            interval_history_path = self.metadata_path / "interval_history.json"
            
            with open(interval_history_path, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            _LOGGER.warning("Failed to save interval history to enhanced storage: %s", e)
        
        # Write to original location if forced or enough time passed
        if force or self._should_write("interval_history", min_interval=14400):  # 4 hours
            return self._atomic_write(self.interval_history_path, self._interval_history_cache)
        
        return True
    
    def get_interval_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get interval optimization history."""
        # First try enhanced storage
        interval_history_path = self.metadata_path / "interval_history.json"
        if interval_history_path.exists():
            try:
                with open(interval_history_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                _LOGGER.warning("Failed to load interval history from enhanced storage: %s", e)
        
        # Fall back to original cache
        return self._interval_history_cache
    
    def clear_storage(self, data_type: Optional[str] = None) -> bool:
        """Clear storage data.
        
        Args:
            data_type: Optional data type to clear. If None, clears all data.
            
        Returns:
            True if cleared successfully, False otherwise
        """
        if data_type == "entity_stats" or data_type is None:
            self._entity_stats_cache = {}
            if self.entity_stats_path.exists():
                try:
                    self.entity_stats_path.unlink()
                except IOError as e:
                    _LOGGER.error("Failed to delete entity stats: %s", e)
                    return False
            
            # Clear chunked entity data
            if self.chunks_path.exists():
                for chunk_file in self.chunks_path.glob("*.json*"):
                    try:
                        chunk_file.unlink()
                    except IOError as e:
                        _LOGGER.error("Failed to delete chunk file: %s", e)
                        return False
            
            # Clear entity index
            if self.entity_index:
                # Reset in-memory index
                self.entity_index.entity_to_chunk = {}
                self.entity_index.dirty = True
                self.entity_index.save()
        
        if data_type == "patterns" or data_type is None:
            self._patterns_cache = {}
            if self.patterns_path.exists():
                try:
                    self.patterns_path.unlink()
                except IOError as e:
                    _LOGGER.error("Failed to delete patterns: %s", e)
                    return False
            
            # Clear enhanced storage patterns
            patterns_path = self.metadata_path / "patterns.json"
            if patterns_path.exists():
                try:
                    patterns_path.unlink()
                except IOError as e:
                    _LOGGER.error("Failed to delete enhanced patterns: %s", e)
                    return False
        
        if data_type == "clusters" or data_type is None:
            self._clusters_cache = {}
            if self.clusters_path.exists():
                try:
                    self.clusters_path.unlink()
                except IOError as e:
                    _LOGGER.error("Failed to delete clusters: %s", e)
                    return False
            
            # Clear enhanced storage clusters
            clusters_path = self.metadata_path / "clusters.json"
            if clusters_path.exists():
                try:
                    clusters_path.unlink()
                except IOError as e:
                    _LOGGER.error("Failed to delete enhanced clusters: %s", e)
                    return False
        
        if data_type == "correlation" or data_type is None:
            self._correlation_cache = {}
            if self.correlation_path.exists():
                try:
                    self.correlation_path.unlink()
                except IOError as e:
                    _LOGGER.error("Failed to delete correlation matrix: %s", e)
                    return False
            
            # Clear enhanced storage correlation matrix
            correlation_path = self.metadata_path / "correlation_matrix.json"
            if correlation_path.exists():
                try:
                    correlation_path.unlink()
                except IOError as e:
                    _LOGGER.error("Failed to delete enhanced correlation matrix: %s", e)
                    return False
        
        if data_type == "interval_history" or data_type is None:
            self._interval_history_cache = {}
            if self.interval_history_path.exists():
                try:
                    self.interval_history_path.unlink()
                except IOError as e:
                    _LOGGER.error("Failed to delete interval history: %s", e)
                    return False
            
            # Clear enhanced storage interval history
            interval_history_path = self.metadata_path / "interval_history.json"
            if interval_history_path.exists():
                try:
                    interval_history_path.unlink()
                except IOError as e:
                    _LOGGER.error("Failed to delete enhanced interval history: %s", e)
                    return False
        
        return True
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the storage system."""
        info = {
            "version": self.storage_meta.get("version", self.meta.get("version", "unknown")),
            "created": self.storage_meta.get("created", self.meta.get("created")),
            "last_update": self.storage_meta.get("last_update", self.meta.get("last_update")),
            "stats_count": len(self._entity_stats_cache),
            "patterns_count": len(self._patterns_cache),
            "clusters_count": len(self._clusters_cache),
            "storage_location": str(self.base_path),
            "files": {},
        }
        
        # Add file information from original storage
        for name, path in [
            ("entity_stats", self.entity_stats_path),
            ("patterns", self.patterns_path),
            ("clusters", self.clusters_path),
            ("correlation", self.correlation_path),
            ("interval_history", self.interval_history_path),
        ]:
            if path.exists():
                info["files"][name] = {
                    "size_kb": round(path.stat().st_size / 1024, 1),
                    "last_modified": datetime.fromtimestamp(
                        path.stat().st_mtime
                    ).isoformat(),
                }
            else:
                info["files"][name] = {"exists": False}
        
        # Add enhanced storage information if available
        enhanced_info = {}
        
        # Entity counts from enhanced storage
        if hasattr(self, "entity_index") and self.entity_index:
            entity_count = len(self.entity_index.get_all_entities())
            enhanced_info["entity_count"] = entity_count
            
            # Get chunk distribution
            chunks_by_size = {}
            entities_by_chunk = self.entity_index.get_entities_by_chunk()
            
            for chunk_id, entities in entities_by_chunk.items():
                size = len(entities)
                if size not in chunks_by_size:
                    chunks_by_size[size] = 0
                chunks_by_size[size] += 1
                
            enhanced_info["chunk_distribution"] = dict(sorted(chunks_by_size.items()))
        
        # Directory sizes
        enhanced_info["directories"] = {}
        for name, path in [
            ("chunks", self.chunks_path),
            ("metadata", self.metadata_path),
            ("indexes", self.indexes_path),
            ("archives", self.archives_path)
        ]:
            if path.exists():
                # Count files and total size
                files = list(path.glob("**/*"))
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                
                enhanced_info["directories"][name] = {
                    "file_count": len(files),
                    "size_kb": round(total_size / 1024, 1),
                }
            else:
                enhanced_info["directories"][name] = {"exists": False}
        
        # Add last compaction and backup times
        enhanced_info["last_compaction"] = self.storage_meta.get("last_compaction")
        enhanced_info["last_backup"] = self.storage_meta.get("last_backup")
        
        # Add storage parameters
        enhanced_info["parameters"] = {
            "chunk_size": self.storage_meta.get("chunk_size", DEFAULT_CHUNK_SIZE),
            "max_chunk_size_kb": self.storage_meta.get("max_chunk_size_kb", DEFAULT_MAX_CHUNK_SIZE_KB)
        }
        
        # Add to main info
        info["enhanced_storage"] = enhanced_info
        
        return info
    
    def compact_storage(self) -> Dict[str, Any]:
        """Compact storage to improve efficiency.
        
        Returns:
            Compaction results
        """
        with self._lock:
            start_time = time.time()
            
            _LOGGER.info("Starting storage compaction")
            
            results = {
                "start_time": datetime.utcnow().isoformat(),
                "chunks_before": 0,
                "chunks_after": 0,
                "entities_processed": 0,
                "orphaned_entities_removed": 0,
                "space_saved_kb": 0,
                "errors": []
            }
            
            try:
                # Count chunks before compaction
                chunks_before = len(list(self.chunks_path.glob("*.json*")))
                results["chunks_before"] = chunks_before
                
                # Get all valid entities
                entity_ids = set(self.entity_index.get_all_entities())
                results["entities_processed"] = len(entity_ids)
                
                # Check for orphaned chunks
                orphaned_chunks = []
                
                # Get chunks referenced in the index
                entities_by_chunk = self.entity_index.get_entities_by_chunk()
                referenced_chunk_ids = set(entities_by_chunk.keys())
                
                # Check for orphaned chunks
                for chunk_path in self.chunks_path.glob("*.json*"):
                    chunk_filename = chunk_path.name
                    if chunk_filename.startswith("entity_stats_"):
                        # Extract chunk ID
                        chunk_id = chunk_filename.replace("entity_stats_", "").replace(".json", "").replace(".gz", "")
                        
                        if chunk_id not in referenced_chunk_ids:
                            orphaned_chunks.append(chunk_path)
                
                # Archive orphaned chunks
                for chunk_path in orphaned_chunks:
                    try:
                        # Move to archives
                        archive_path = self.archives_path / f"{chunk_path.name}.bak"
                        shutil.move(str(chunk_path), str(archive_path))
                        _LOGGER.info("Archived orphaned chunk: %s", chunk_path.name)
                    except Exception as e:
                        _LOGGER.error("Error archiving orphaned chunk %s: %s", chunk_path.name, e)
                        results["errors"].append(f"Failed to archive {chunk_path.name}: {e}")
                
                # Process each chunk for optimization
                space_before = sum(
                    f.stat().st_size for f in self.chunks_path.glob("*.json*")
                ) / 1024
                
                # Optimize each chunk
                for chunk_id, entity_list in entities_by_chunk.items():
                    try:
                        # Skip if chunk is empty
                        if not entity_list:
                            continue
                            
                        # Get chunk
                        chunk = self.chunk_manager.get_chunk(chunk_id)
                        
                        # Check if compression needed
                        if chunk.should_compress():
                            # Compress when saving
                            chunk.save(compress=True)
                    except Exception as e:
                        _LOGGER.error("Error processing chunk %s: %s", chunk_id, e)
                        results["errors"].append(f"Failed to process chunk {chunk_id}: {e}")
                
                # Save all chunks
                self.chunk_manager.save_all_chunks()
                
                # Count chunks after compaction
                chunks_after = len(list(self.chunks_path.glob("*.json*")))
                results["chunks_after"] = chunks_after
                
                # Calculate space saved
                space_after = sum(
                    f.stat().st_size for f in self.chunks_path.glob("*.json*")
                ) / 1024
                results["space_saved_kb"] = round(space_before - space_after, 1)
                
                # Update metadata
                self.storage_meta["last_compaction"] = datetime.utcnow().isoformat()
                self._save_storage_metadata()
                
                results["success"] = True
                results["duration_seconds"] = round(time.time() - start_time, 1)
                
                _LOGGER.info(
                    "Storage compaction complete: %d entities, %.1f KB saved", 
                    results["entities_processed"], results["space_saved_kb"]
                )
                
            except Exception as e:
                _LOGGER.exception("Error during storage compaction: %s", e)
                results["success"] = False
                results["error"] = str(e)
                results["duration_seconds"] = round(time.time() - start_time, 1)
                
            return results
    
    def create_backup(self, target_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Create a backup of the storage.
        
        Args:
            target_dir: Optional target directory
            
        Returns:
            Backup results
        """
        with self._lock:
            start_time = time.time()
            
            results = {
                "start_time": datetime.utcnow().isoformat(),
                "files_backed_up": 0,
                "total_size_kb": 0,
                "errors": []
            }
            
            # Determine target directory
            if not target_dir:
                # Create timestamped backup in archives directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target_dir = self.archives_path / f"backup_{timestamp}"
            
            try:
                # Save all pending changes first
                self._save_pending_changes()
                
                # Create target directory
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # Create subdirectories
                (target_dir / "metadata").mkdir(exist_ok=True)
                (target_dir / "chunks").mkdir(exist_ok=True)
                (target_dir / "indexes").mkdir(exist_ok=True)
                (target_dir / "original").mkdir(exist_ok=True)
                
                # Backup metadata files
                for file_path in self.metadata_path.glob("*.json"):
                    try:
                        target_path = target_dir / "metadata" / file_path.name
                        shutil.copy(file_path, target_path)
                        results["files_backed_up"] += 1
                        results["total_size_kb"] += file_path.stat().st_size / 1024
                    except Exception as e:
                        _LOGGER.error("Error backing up %s: %s", file_path, e)
                        results["errors"].append(f"Failed to backup {file_path.name}: {e}")
                
                # Backup chunks
                for file_path in self.chunks_path.glob("*.json*"):
                    try:
                        target_path = target_dir / "chunks" / file_path.name
                        shutil.copy(file_path, target_path)
                        results["files_backed_up"] += 1
                        results["total_size_kb"] += file_path.stat().st_size / 1024
                    except Exception as e:
                        _LOGGER.error("Error backing up %s: %s", file_path, e)
                        results["errors"].append(f"Failed to backup {file_path.name}: {e}")
                
                # Backup indexes
                for file_path in self.indexes_path.glob("*.json"):
                    try:
                        target_path = target_dir / "indexes" / file_path.name
                        shutil.copy(file_path, target_path)
                        results["files_backed_up"] += 1
                        results["total_size_kb"] += file_path.stat().st_size / 1024
                    except Exception as e:
                        _LOGGER.error("Error backing up %s: %s", file_path, e)
                        results["errors"].append(f"Failed to backup {file_path.name}: {e}")
                
                # Backup original format files for completeness
                for file_path in [
                    self.meta_path, self.entity_stats_path, self.patterns_path, 
                    self.clusters_path, self.correlation_path, self.interval_history_path
                ]:
                    if file_path.exists():
                        try:
                            target_path = target_dir / "original" / file_path.name
                            shutil.copy(file_path, target_path)
                            results["files_backed_up"] += 1
                            results["total_size_kb"] += file_path.stat().st_size / 1024
                        except Exception as e:
                            _LOGGER.error("Error backing up %s: %s", file_path, e)
                            results["errors"].append(f"Failed to backup {file_path.name}: {e}")
                
                # Update metadata
                self.storage_meta["last_backup"] = datetime.utcnow().isoformat()
                self._save_storage_metadata()
                
                results["success"] = True
                results["duration_seconds"] = round(time.time() - start_time, 1)
                results["backup_path"] = str(target_dir)
                results["total_size_kb"] = round(results["total_size_kb"], 1)
                
                _LOGGER.info(
                    "Backup completed: %d files, %.1f KB to %s", 
                    results["files_backed_up"], results["total_size_kb"], target_dir
                )
                
            except Exception as e:
                _LOGGER.exception("Error creating backup: %s", e)
                results["success"] = False
                results["error"] = str(e)
                results["duration_seconds"] = round(time.time() - start_time, 1)
                
            return results
    
    def restore_from_backup(self, source_dir: Path) -> Dict[str, Any]:
        """Restore storage from a backup.
        
        Args:
            source_dir: Backup source directory
            
        Returns:
            Restore results
        """
        with self._lock:
            start_time = time.time()
            
            results = {
                "start_time": datetime.utcnow().isoformat(),
                "files_restored": 0,
                "total_size_kb": 0,
                "errors": []
            }
            
            try:
                # Check if source directory exists
                if not source_dir.exists() or not source_dir.is_dir():
                    results["success"] = False
                    results["error"] = f"Backup directory not found: {source_dir}"
                    return results
                
                # Check for required subdirectories
                if not (source_dir / "metadata").exists() or not (source_dir / "chunks").exists():
                    results["success"] = False
                    results["error"] = "Invalid backup format: missing required directories"
                    return results
                
                # Create backup of current state before restoration
                self.create_backup()
                
                # Clear current storage
                self._clear_current_storage()
                
                # Restore metadata files
                for file_path in (source_dir / "metadata").glob("*.json"):
                    try:
                        target_path = self.metadata_path / file_path.name
                        shutil.copy(file_path, target_path)
                        results["files_restored"] += 1
                        results["total_size_kb"] += file_path.stat().st_size / 1024
                    except Exception as e:
                        _LOGGER.error("Error restoring %s: %s", file_path, e)
                        results["errors"].append(f"Failed to restore {file_path.name}: {e}")
                
                # Restore chunks
                for file_path in (source_dir / "chunks").glob("*.json*"):
                    try:
                        target_path = self.chunks_path / file_path.name
                        shutil.copy(file_path, target_path)
                        results["files_restored"] += 1
                        results["total_size_kb"] += file_path.stat().st_size / 1024
                    except Exception as e:
                        _LOGGER.error("Error restoring %s: %s", file_path, e)
                        results["errors"].append(f"Failed to restore {file_path.name}: {e}")
                
                # Restore indexes
                if (source_dir / "indexes").exists():
                    for file_path in (source_dir / "indexes").glob("*.json"):
                        try:
                            target_path = self.indexes_path / file_path.name
                            shutil.copy(file_path, target_path)
                            results["files_restored"] += 1
                            results["total_size_kb"] += file_path.stat().st_size / 1024
                        except Exception as e:
                            _LOGGER.error("Error restoring %s: %s", file_path, e)
                            results["errors"].append(f"Failed to restore {file_path.name}: {e}")
                
                # Restore original format files if present
                if (source_dir / "original").exists():
                    for file_name in ["meta.json", "entity_stats.json", "patterns.json", 
                                      "clusters.json", "correlation_matrix.json", "interval_history.json"]:
                        file_path = source_dir / "original" / file_name
                        if file_path.exists():
                            try:
                                target_path = self.base_path / file_name
                                shutil.copy(file_path, target_path)
                                results["files_restored"] += 1
                                results["total_size_kb"] += file_path.stat().st_size / 1024
                            except Exception as e:
                                _LOGGER.error("Error restoring %s: %s", file_path, e)
                                results["errors"].append(f"Failed to restore {file_path.name}: {e}")
                
                # Reload storage metadata
                self._reload_metadata()
                
                # Reset instance state
                self.chunk_manager = ChunkManager(self.chunks_path)
                self.entity_index = EntityIndex(self.indexes_path / "entity_index.json")
                
                # Reload original storage
                self._load_original_storage()
                
                results["success"] = True
                results["duration_seconds"] = round(time.time() - start_time, 1)
                results["total_size_kb"] = round(results["total_size_kb"], 1)
                
                _LOGGER.info(
                    "Restore completed: %d files, %.1f KB from %s", 
                    results["files_restored"], results["total_size_kb"], source_dir
                )
                
            except Exception as e:
                _LOGGER.exception("Error restoring from backup: %s", e)
                results["success"] = False
                results["error"] = str(e)
                results["duration_seconds"] = round(time.time() - start_time, 1)
                
            return results
    
    def _clear_current_storage(self) -> None:
        """Clear current storage files."""
        # Clear metadata directory
        for file_path in self.metadata_path.glob("*.json"):
            try:
                file_path.unlink()
            except Exception as e:
                _LOGGER.error("Error deleting %s: %s", file_path, e)
        
        # Clear chunks directory
        for file_path in self.chunks_path.glob("*.json*"):
            try:
                file_path.unlink()
            except Exception as e:
                _LOGGER.error("Error deleting %s: %s", file_path, e)
        
        # Clear indexes directory
        for file_path in self.indexes_path.glob("*.json"):
            try:
                file_path.unlink()
            except Exception as e:
                _LOGGER.error("Error deleting %s: %s", file_path, e)
    
    def _reload_metadata(self) -> None:
        """Reload storage metadata from file."""
        # Load enhanced storage metadata
        meta_path = self.metadata_path / "storage_meta.json"
        
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    self.storage_meta = json.load(f)
            except Exception as e:
                _LOGGER.error("Error reloading metadata: %s", e)
        
        # Also reload original metadata
        if self.meta_path.exists():
            try:
                with open(self.meta_path, "r") as f:
                    self.meta = json.load(f)
            except Exception as e:
                _LOGGER.error("Error reloading original metadata: %s", e)
    
    def _save_pending_changes(self) -> bool:
        """Save all pending changes to disk.
        
        Returns:
            Success status
        """
        with self._lock:
            success = True
            
            # Save all chunks
            if hasattr(self, "chunk_manager"):
                if not self.chunk_manager.save_all_chunks():
                    success = False
                
            # Save entity index
            if hasattr(self, "entity_index"):
                if self.entity_index.dirty and not self.entity_index.save():
                    success = False
                
            # Save metadata
            if not self._save_storage_metadata():
                success = False
                
            # Also save original format files
            self._atomic_write(self.entity_stats_path, self._entity_stats_cache)
            self._atomic_write(self.patterns_path, self._patterns_cache)
            self._atomic_write(self.clusters_path, self._clusters_cache)
            self._atomic_write(self.correlation_path, self._correlation_cache)
            self._atomic_write(self.interval_history_path, self._interval_history_cache)
            
            # Update meta data
            self.meta["last_update"] = datetime.utcnow().isoformat()
            try:
                with open(self.meta_path, "w") as f:
                    json.dump(self.meta, f, indent=2)
            except Exception as e:
                _LOGGER.error("Error saving meta data: %s", e)
                success = False
                
            return success
    
    def cleanup_old_data(self, older_than_days: int = 90) -> Dict[str, Any]:
        """Clean up old data.
        
        Args:
            older_than_days: Remove data older than this many days
            
        Returns:
            Cleanup results
        """
        with self._lock:
            start_time = time.time()
            
            results = {
                "start_time": datetime.utcnow().isoformat(),
                "older_than_days": older_than_days,
                "files_archived": 0,
                "total_size_kb": 0
            }
            
            try:
                # Calculate cutoff date
                cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
                cutoff_timestamp = cutoff_date.timestamp()
                
                # Move old archives to backup directory
                archives_to_remove = []
                
                for file_path in self.archives_path.glob("*.bak"):
                    try:
                        # Check modification time
                        if file_path.stat().st_mtime < cutoff_timestamp:
                            archives_to_remove.append(file_path)
                    except Exception:
                        pass
                
                # Create old archives directory
                old_archives_dir = self.archives_path / "old_archives"
                old_archives_dir.mkdir(exist_ok=True)
                
                # Move old archives
                for file_path in archives_to_remove:
                    try:
                        # Generate target path with timestamp
                        timestamp = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y%m%d")
                        target_name = f"{timestamp}_{file_path.name}"
                        target_path = old_archives_dir / target_name
                        
                        # Move file
                        shutil.move(str(file_path), str(target_path))
                        
                        results["files_archived"] += 1
                        results["total_size_kb"] += file_path.stat().st_size / 1024
                    except Exception as e:
                        _LOGGER.error("Error archiving %s: %s", file_path, e)
                
                results["success"] = True
                results["duration_seconds"] = round(time.time() - start_time, 1)
                results["total_size_kb"] = round(results["total_size_kb"], 1)
                
                _LOGGER.info(
                    "Cleaned up %d old files (%.1f KB) older than %d days", 
                    results["files_archived"], results["total_size_kb"], older_than_days
                )
                
            except Exception as e:
                _LOGGER.exception("Error cleaning up old data: %s", e)
                results["success"] = False
                results["error"] = str(e)
                results["duration_seconds"] = round(time.time() - start_time, 1)
                
            return results
    
    def export_data(self, data_type: Optional[str] = None) -> Dict[str, Any]:
        """Export data for backup or analysis.
        
        Args:
            data_type: Type of data to export (entity_stats, patterns, etc.)
                      If None, exports all data.
                      
        Returns:
            Dictionary with exported data
        """
        # Save any pending changes first
        self._save_pending_changes()
        
        result = {"export_time": datetime.utcnow().isoformat()}
        
        # Add storage info
        result["storage_info"] = {
            "version": self.storage_meta.get("version", "unknown"),
            "location": str(self.base_path),
            "entity_count": len(self.get_entity_stats()),
        }
        
        # Export specific data type or all
        if data_type == "entity_stats" or data_type is None:
            result["entity_stats"] = self.get_entity_stats()
            
        if data_type == "patterns" or data_type is None:
            result["patterns"] = self.get_patterns()
            
        if data_type == "clusters" or data_type is None:
            result["clusters"] = self.get_clusters()
            
        if data_type == "correlation" or data_type is None:
            # Don't include full correlation matrix as it can be very large
            # Just include basic info
            matrix = self.get_correlation_matrix()
            if matrix:
                entity_count = len(matrix)
                result["correlation_info"] = {
                    "entity_count": entity_count,
                    "matrix_size": entity_count * entity_count,
                    "strong_correlations": sum(
                        1 for entity_corr in matrix.values()
                        for corr_val in entity_corr.values()
                        if corr_val > 0.7
                    )
                }
            else:
                result["correlation_info"] = {"entity_count": 0}
                
        if data_type == "interval_history" or data_type is None:
            result["interval_history"] = self.get_interval_history()
        
        return result