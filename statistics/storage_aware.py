"""
Base class for components with persistent storage capabilities.

This module provides a base class that can be inherited by components
that need to persist and retrieve data from storage.
"""

import logging
from typing import Dict, Any, Optional, List

_LOGGER = logging.getLogger(__name__)

class StorageAware:
    """Base class for components that need persistent storage."""
    
    def __init__(self, component_name: str):
        """Initialize storage-aware component.
        
        Args:
            component_name: Name of the component (used as storage key)
        """
        self._component_name = component_name
        self._storage_manager = None
        self._last_storage_save = 0
        
    def set_storage_manager(self, storage_manager) -> None:
        """Set the storage manager for this component.
        
        Args:
            storage_manager: Storage manager instance
        """
        self._storage_manager = storage_manager
        self._load_from_storage()
    
    def _load_from_storage(self) -> None:
        """Load component data from storage.
        
        This method should be implemented by subclasses.
        """
        pass
    
    def _save_to_storage(self) -> bool:
        """Save component data to storage.
        
        This method should be implemented by subclasses.
        
        Returns:
            Success status
        """
        return False
    
    def get_metadata_file(self, filename: str) -> Dict[str, Any]:
        """Get a metadata file from storage.
        
        Args:
            filename: Name of the metadata file
            
        Returns:
            File contents or empty dict if not found
        """
        if not self._storage_manager:
            return {}
            
        if hasattr(self._storage_manager, "get_metadata_file"):
            return self._storage_manager.get_metadata_file(filename) or {}
            
        return {}
    
    def save_metadata_file(self, filename: str, data: Dict[str, Any]) -> bool:
        """Save a metadata file to storage.
        
        Args:
            filename: Name of the metadata file
            data: Data to save
            
        Returns:
            Success status
        """
        if not self._storage_manager:
            return False
            
        if hasattr(self._storage_manager, "save_metadata_file"):
            return self._storage_manager.save_metadata_file(filename, data)
            
        return False