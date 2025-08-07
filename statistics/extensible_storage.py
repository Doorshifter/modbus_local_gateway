"""
Extensible storage system for Modbus optimization data.

Provides scalable, versioned storage for statistical data with:
- Automatic data chunking for large datasets
- Data versioning and migration capabilities
- Indexed access for performance
- Configurable storage strategies
"""

import json
import logging
import os
import time
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple, Union
import threading
import gzip
import hashlib
from collections import defaultdict

_LOGGER = logging.getLogger(__name__)

# Storage format version
CURRENT_STORAGE_VERSION = "2.0.0"

# Constants for data management
DEFAULT_CHUNK_SIZE = 1000  # Max entities per chunk file
DEFAULT_MAX_CHUNK_SIZE_KB = 512  # Target max file size in KB
COMPACTION_THRESHOLD = 0.5  # Trigger compaction when fragmentation > 50%
AUTO_COMPACTION_INTERVAL = 7  # Days between auto-compaction runs
DEFAULT_RETENTION_DAYS = 90  # Default retention period for historical data

class StorageMigrator:
    """Handles migrations between storage versions."""
    
    def __init__(self, base_path: Path):
        """Initialize storage migrator.
        
        Args:
            base_path: Base path to storage directory
        """
        self.base_path = base_path
        self.migrations = {
            "1.0.0": self._migrate_1_0_0_to_2_0_0
        }
    
    def check_and_migrate(self, current_version: str) -> Tuple[bool, str, str]:
        """Check if migration is needed and perform if necessary.
        
        Args:
            current_version: Current storage version
            
        Returns:
            Tuple of (success, from_version, to_version)
        """
        if current_version == CURRENT_STORAGE_VERSION:
            return True, current_version, current_version
            
        # Find migration path
        migration_path = self._find_migration_path(current_version, CURRENT_STORAGE_VERSION)
        if not migration_path:
            _LOGGER.error(
                "No migration path found from %s to %s", 
                current_version, CURRENT_STORAGE_VERSION
            )
            return False, current_version, current_version
            
        # Execute migrations in sequence
        current = current_version
        for target_version in migration_path:
            migration_key = current
            
            # Find the migration function
            migration_func = self.migrations.get(migration_key)
            if not migration_func:
                _LOGGER.error(
                    "Missing migration function from %s to next version", 
                    current
                )
                return False, current_version, current
                
            # Execute migration
            try:
                _LOGGER.info("Migrating storage from %s to %s", current, target_version)
                success = migration_func()
                if not success:
                    _LOGGER.error("Migration from %s to %s failed", current, target_version)
                    return False, current_version, current
                
                current = target_version
            except Exception as e:
                _LOGGER.exception("Error during migration from %s: %s", current, e)
                return False, current_version, current
                
        return True, current_version, CURRENT_STORAGE_VERSION
    
    def _find_migration_path(self, from_version: str, to_version: str) -> List[str]:
        """Find a path of migrations from one version to another.
        
        Simple implementation for now - can be enhanced for more complex version paths.
        
        Args:
            from_version: Starting version
            to_version: Target version
            
        Returns:
            List of versions in migration path
        """
        # For now, just return the target version
        # This could be enhanced to find multi-step migration paths
        if from_version in self.migrations:
            return [to_version]
        return []
    
    def _migrate_1_0_0_to_2_0_0(self) -> bool:
        """Migrate from storage version 1.0.0 to 2.0.0.
        
        Returns:
            Success status
        """
        try:
            _LOGGER.info("Starting migration from 1.0.0 to 2.0.0")
            
            # Create metadata directory
            metadata_dir = self.base_path / "metadata"
            metadata_dir.mkdir(exist_ok=True)
            
            # Create chunks directory
            chunks_dir = self.base_path / "chunks"
            chunks_dir.mkdir(exist_ok=True)
            
            # Create indexes directory
            indexes_dir = self.base_path / "indexes"
            indexes_dir.mkdir(exist_ok=True)
            
            # Create archives directory
            archives_dir = self.base_path / "archives"
            archives_dir.mkdir(exist_ok=True)
            
            # Migrate entity stats
            entity_stats_path = self.base_path / "entity_stats.json"
            if entity_stats_path.exists():
                try:
                    with open(entity_stats_path, "r") as f:
                        entity_stats = json.load(f)
                        
                    # Create chunks
                    if entity_stats:
                        self._chunk_entity_data(entity_stats, chunks_dir)
                        
                    # Create index
                    self._create_entity_index(entity_stats.keys(), indexes_dir)
                    
                    # Backup the original file
                    backup_path = self.base_path / "entity_stats.json.bak"
                    shutil.copy(entity_stats_path, backup_path)
                    
                    _LOGGER.info("Migrated entity stats: %d entities", len(entity_stats))
                except Exception as e:
                    _LOGGER.exception("Error migrating entity stats: %s", e)
                    return False
            
            # Migrate other data files - just copy for now
            for filename in ["patterns.json", "clusters.json", "correlation_matrix.json"]:
                file_path = self.base_path / filename
                if file_path.exists():
                    try:
                        # Copy to metadata directory
                        shutil.copy(file_path, metadata_dir / filename)
                        _LOGGER.info("Copied %s to metadata directory", filename)
                    except Exception as e:
                        _LOGGER.exception("Error copying %s: %s", filename, e)
                        return False
            
            # Create storage metadata
            storage_meta = {
                "version": "2.0.0",
                "migrated_from": "1.0.0",
                "migration_time": datetime.utcnow().isoformat(),
                "chunk_size": DEFAULT_CHUNK_SIZE,
                "max_chunk_size_kb": DEFAULT_MAX_CHUNK_SIZE_KB,
                "last_compaction": None,
                "last_backup": None,
                "retention_days": DEFAULT_RETENTION_DAYS
            }
            
            # Save storage metadata
            with open(metadata_dir / "storage_meta.json", "w") as f:
                json.dump(storage_meta, f, indent=2)
                
            _LOGGER.info("Migration completed successfully")
            return True
            
        except Exception as e:
            _LOGGER.exception("Unexpected error during migration: %s", e)
            return False
    
    def _chunk_entity_data(self, entity_stats: Dict[str, Any], chunks_dir: Path) -> None:
        """Split entity data into chunks.
        
        Args:
            entity_stats: Entity statistics data
            chunks_dir: Directory for chunks
        """
        # Group entities into chunks
        chunks = {}
        current_chunk = 0
        current_chunk_size = 0
        current_chunk_data = {}
        
        for entity_id, stats in entity_stats.items():
            # Add to current chunk
            current_chunk_data[entity_id] = stats
            current_chunk_size += 1
            
            # Check if chunk is full
            if current_chunk_size >= DEFAULT_CHUNK_SIZE:
                # Save current chunk
                chunk_filename = f"entity_stats_{current_chunk}.json"
                with open(chunks_dir / chunk_filename, "w") as f:
                    json.dump(current_chunk_data, f)
                
                # Start new chunk
                current_chunk += 1
                current_chunk_size = 0
                current_chunk_data = {}
        
        # Save final chunk if not empty
        if current_chunk_data:
            chunk_filename = f"entity_stats_{current_chunk}.json"
            with open(chunks_dir / chunk_filename, "w") as f:
                json.dump(current_chunk_data, f)
    
    def _create_entity_index(self, entity_ids: Set[str], indexes_dir: Path) -> None:
        """Create entity index file.
        
        Args:
            entity_ids: Set of entity IDs
            indexes_dir: Directory for indexes
        """
        # Create entity to chunk mapping
        entity_index = {}
        
        # For now, we'll create a simple index
        for entity_id in entity_ids:
            # Hash the entity ID to distribute entities across chunks
            hash_value = int(hashlib.md5(entity_id.encode()).hexdigest(), 16) % DEFAULT_CHUNK_SIZE
            chunk_id = hash_value // 50  # Distribute across chunks
            
            entity_index[entity_id] = chunk_id
        
        # Save index
        with open(indexes_dir / "entity_index.json", "w") as f:
            json.dump(entity_index, f)


class DataChunk:
    """Represents a chunk of data with efficient loading and saving."""
    
    def __init__(self, chunk_id: str, file_path: Path, max_size_kb: int = DEFAULT_MAX_CHUNK_SIZE_KB):
        """Initialize data chunk.
        
        Args:
            chunk_id: Unique chunk identifier
            file_path: Path to chunk file
            max_size_kb: Maximum chunk size in KB
        """
        self.chunk_id = chunk_id
        self.file_path = file_path
        self.max_size_kb = max_size_kb
        self.data: Dict[str, Any] = {}
        self.dirty = False
        self.last_access = time.time()
        self.size_kb = 0
        
        # Load if file exists
        if file_path.exists():
            self.load()
    
    def load(self) -> bool:
        """Load chunk data from file.
        
        Returns:
            Success status
        """
        try:
            if not self.file_path.exists():
                return False
                
            # Check if it's a compressed file
            if str(self.file_path).endswith(".gz"):
                with gzip.open(self.file_path, "rt") as f:
                    self.data = json.load(f)
            else:
                with open(self.file_path, "r") as f:
                    self.data = json.load(f)
                    
            # Update size
            self.size_kb = self.file_path.stat().st_size / 1024
            self.last_access = time.time()
            self.dirty = False
            return True
            
        except Exception as e:
            _LOGGER.error("Error loading chunk %s: %s", self.chunk_id, e)
            return False
    
    def save(self, compress: bool = False) -> bool:
        """Save chunk data to file.
        
        Args:
            compress: Whether to compress the chunk
            
        Returns:
            Success status
        """
        if not self.dirty:
            return True
            
        try:
            # Create directory if it doesn't exist
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine output path
            output_path = self.file_path
            if compress and not str(output_path).endswith(".gz"):
                output_path = Path(str(self.file_path) + ".gz")
            elif not compress and str(output_path).endswith(".gz"):
                output_path = Path(str(self.file_path)[:-3])
            
            # Write data
            if compress:
                with gzip.open(output_path, "wt") as f:
                    json.dump(self.data, f)
            else:
                with open(output_path, "w") as f:
                    json.dump(self.data, f)
                    
            # Update stats
            self.size_kb = output_path.stat().st_size / 1024
            self.dirty = False
            
            # Remove old file if path changed
            if output_path != self.file_path and self.file_path.exists():
                self.file_path.unlink()
                self.file_path = output_path
                
            return True
            
        except Exception as e:
            _LOGGER.error("Error saving chunk %s: %s", self.chunk_id, e)
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from chunk.
        
        Args:
            key: Item key
            default: Default value if not found
            
        Returns:
            Item value or default
        """
        self.last_access = time.time()
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set item in chunk.
        
        Args:
            key: Item key
            value: Item value
        """
        self.data[key] = value
        self.dirty = True
        self.last_access = time.time()
    
    def remove(self, key: str) -> bool:
        """Remove item from chunk.
        
        Args:
            key: Item key
            
        Returns:
            True if item was removed
        """
        if key in self.data:
            del self.data[key]
            self.dirty = True
            self.last_access = time.time()
            return True
        return False
    
    def should_compress(self) -> bool:
        """Check if chunk should be compressed.
        
        Returns:
            True if compression is recommended
        """
        # Only compress if size exceeds threshold
        return self.size_kb > self.max_size_kb
    
    def should_split(self) -> bool:
        """Check if chunk should be split.
        
        Returns:
            True if splitting is recommended
        """
        # Split if size significantly exceeds max size
        return self.size_kb > (self.max_size_kb * 1.5) and len(self.data) > 10


class ChunkManager:
    """Manages data chunks with efficient memory usage."""
    
    def __init__(
        self, 
        base_dir: Path, 
        max_chunks_in_memory: int = 10,
        max_chunk_size_kb: int = DEFAULT_MAX_CHUNK_SIZE_KB
    ):
        """Initialize chunk manager.
        
        Args:
            base_dir: Base directory for chunks
            max_chunks_in_memory: Maximum number of chunks to keep in memory
            max_chunk_size_kb: Maximum chunk size in KB
        """
        self.base_dir = base_dir
        self.max_chunks_in_memory = max_chunks_in_memory
        self.max_chunk_size_kb = max_chunk_size_kb
        self.chunks: Dict[str, DataChunk] = {}
        self.lock = threading.RLock()
    
    def get_chunk(self, chunk_id: str, category: str = "entity_stats") -> DataChunk:
        """Get a data chunk, loading it if necessary.
        
        Args:
            chunk_id: Chunk identifier
            category: Data category
            
        Returns:
            DataChunk object
        """
        with self.lock:
            chunk_key = f"{category}_{chunk_id}"
            
            # Return from memory if available
            if chunk_key in self.chunks:
                chunk = self.chunks[chunk_key]
                chunk.last_access = time.time()
                return chunk
            
            # Load chunk
            chunk_path = self.base_dir / category / f"{chunk_id}.json"
            compressed_path = Path(str(chunk_path) + ".gz")
            
            # Check for compressed version first
            if compressed_path.exists():
                chunk_path = compressed_path
            
            # Create chunk
            chunk = DataChunk(
                chunk_key, 
                chunk_path, 
                max_size_kb=self.max_chunk_size_kb
            )
            
            # Add to memory cache
            self.chunks[chunk_key] = chunk
            
            # Evict old chunks if needed
            self._evict_chunks()
            
            return chunk
    
    def save_all_chunks(self) -> bool:
        """Save all dirty chunks.
        
        Returns:
            Success status
        """
        success = True
        with self.lock:
            for chunk_id, chunk in self.chunks.items():
                if chunk.dirty:
                    compress = chunk.should_compress()
                    if not chunk.save(compress=compress):
                        success = False
                        
        return success
    
    def _evict_chunks(self) -> None:
        """Evict least recently used chunks if too many in memory."""
        if len(self.chunks) <= self.max_chunks_in_memory:
            return
            
        # Sort chunks by last access time
        sorted_chunks = sorted(
            self.chunks.items(),
            key=lambda x: x[1].last_access
        )
        
        # Save and remove oldest chunks
        chunks_to_evict = len(self.chunks) - self.max_chunks_in_memory
        for chunk_id, chunk in sorted_chunks[:chunks_to_evict]:
            # Save if dirty
            if chunk.dirty:
                compress = chunk.should_compress()
                chunk.save(compress=compress)
                
            # Remove from memory
            del self.chunks[chunk_id]


class EntityIndex:
    """Maintains index of entities to their storage chunks."""
    
    def __init__(self, index_path: Path):
        """Initialize entity index.
        
        Args:
            index_path: Path to index file
        """
        self.index_path = index_path
        self.entity_to_chunk: Dict[str, str] = {}
        self.dirty = False
        
        # Load index if it exists
        if index_path.exists():
            self._load_index()
    
    def _load_index(self) -> None:
        """Load entity index from file."""
        try:
            with open(self.index_path, "r") as f:
                self.entity_to_chunk = json.load(f)
        except Exception as e:
            _LOGGER.error("Error loading entity index: %s", e)
            self.entity_to_chunk = {}
    
    def save(self) -> bool:
        """Save entity index to file.
        
        Returns:
            Success status
        """
        if not self.dirty:
            return True
            
        try:
            # Create directory if needed
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write index
            with open(self.index_path, "w") as f:
                json.dump(self.entity_to_chunk, f)
                
            self.dirty = False
            return True
        except Exception as e:
            _LOGGER.error("Error saving entity index: %s", e)
            return False
    
    def get_chunk_id(self, entity_id: str) -> Optional[str]:
        """Get chunk ID for an entity.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Chunk ID or None if not found
        """
        return self.entity_to_chunk.get(entity_id)
    
    def set_chunk_id(self, entity_id: str, chunk_id: str) -> None:
        """Set chunk ID for an entity.
        
        Args:
            entity_id: Entity ID
            chunk_id: Chunk ID
        """
        self.entity_to_chunk[entity_id] = chunk_id
        self.dirty = True
    
    def remove_entity(self, entity_id: str) -> bool:
        """Remove entity from index.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            True if entity was removed
        """
        if entity_id in self.entity_to_chunk:
            del self.entity_to_chunk[entity_id]
            self.dirty = True
            return True
        return False
    
    def get_all_entities(self) -> List[str]:
        """Get all entity IDs in the index.
        
        Returns:
            List of entity IDs
        """
        return list(self.entity_to_chunk.keys())
    
    def get_entities_by_chunk(self) -> Dict[str, List[str]]:
        """Get entities grouped by chunk ID.
        
        Returns:
            Dictionary of chunk ID to entity list
        """
        result: Dict[str, List[str]] = defaultdict(list)
        
        for entity_id, chunk_id in self.entity_to_chunk.items():
            result[chunk_id].append(entity_id)
            
        return dict(result)