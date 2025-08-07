"""Entity correlation and clustering module with integrated storage."""

import time
import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict

from .storage_aware import StorageAware

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "NumPy not available, correlation analysis will use simple methods"
    )

_LOGGER = logging.getLogger(__name__)

class EntityCorrelationManager(StorageAware):
    """Manages correlations between entities and clusters them."""
    
    def __init__(self, correlation_threshold: float = 0.6, min_data_points: int = 10):
        """Initialize correlation manager."""
        # Initialize storage capabilities
        super().__init__("correlation_manager")
        
        self.correlation_threshold = correlation_threshold
        self.min_data_points = min_data_points
        
        # Storage for entity values
        self._entity_values: Dict[str, List[Tuple[float, Any]]] = defaultdict(list)
        
        # Correlation matrix and clusters
        self._correlation_matrix: Dict[str, Dict[str, float]] = {}
        self._clusters: Dict[str, List[str]] = {}
        self._entity_clusters: Dict[str, str] = {}  # entity_id -> cluster_id
        
        # Last analysis timestamp
        self._last_analysis: float = 0
        
    def add_entity_value(self, entity_id: str, value: Any, timestamp: Optional[float] = None) -> None:
        """Record a value for an entity."""
        try:
            # Only store numeric values for correlation
            float_value = float(value) if value is not None else None
            
            if timestamp is None:
                timestamp = time.time()
                
            self._entity_values[entity_id].append((timestamp, float_value))
            
            # Limit history to prevent excessive memory usage
            if len(self._entity_values[entity_id]) > 1000:
                self._entity_values[entity_id] = self._entity_values[entity_id][-1000:]
                
        except (ValueError, TypeError):
            # Skip non-numeric values but log the attempt
            _LOGGER.debug("Non-numeric value for %s: %s", entity_id, value)
            
    def analyze_correlations(self, force: bool = False) -> Dict[str, List[str]]:
        """Analyze correlations and update clusters."""
        current_time = time.time()
        
        # Only analyze once a day unless forced
        if not force and current_time - self._last_analysis < 86400:
            return self._clusters
            
        _LOGGER.info("Running entity correlation analysis")
        self._last_analysis = current_time
        
        # Get entities with sufficient data
        valid_entities = {
            entity_id for entity_id, values in self._entity_values.items() 
            if len(values) >= self.min_data_points and 
            any(v is not None for _, v in values)  # At least some non-None values
        }
        
        if len(valid_entities) < 2:
            _LOGGER.debug("Not enough entities with data for correlation analysis")
            return self._clusters
        
        # Calculate correlation matrix
        self._correlation_matrix = {}
        
        # Get all entity pairs
        entity_pairs = [(a, b) for i, a in enumerate(valid_entities) 
                        for b in list(valid_entities)[i+1:]]
        
        for entity_a, entity_b in entity_pairs:
            corr = self._calculate_correlation(entity_a, entity_b)
            if corr is not None:
                # Initialize dictionaries if needed
                if entity_a not in self._correlation_matrix:
                    self._correlation_matrix[entity_a] = {}
                if entity_b not in self._correlation_matrix:
                    self._correlation_matrix[entity_b] = {}
                    
                # Store correlation value in both directions
                self._correlation_matrix[entity_a][entity_b] = corr
                self._correlation_matrix[entity_b][entity_a] = corr
        
        # Create clusters based on correlation
        self._update_clusters()
        
        # Save to storage
        self._save_to_storage()
        
        _LOGGER.info("Correlation analysis complete. Created %d clusters.", len(self._clusters))
        return self._clusters
        
    def _calculate_correlation(self, entity_a: str, entity_b: str) -> Optional[float]:
        """Calculate correlation between two entities."""
        values_a = self._entity_values[entity_a]
        values_b = self._entity_values[entity_b]
        
        # Need enough overlapping time points to calculate correlation
        if len(values_a) < self.min_data_points or len(values_b) < self.min_data_points:
            return None
        
        # Create time-aligned series by interpolation
        series_a = []
        series_b = []
        
        # Use NumPy if available, otherwise fallback to simple method
        if NUMPY_AVAILABLE:
            # Get all timestamps
            all_timestamps = sorted(set(t for t, _ in values_a + values_b))
            
            # Align timestamps
            for t in all_timestamps:
                # Find closest value for each entity
                closest_a = self._find_closest_value(values_a, t)
                closest_b = self._find_closest_value(values_b, t)
                
                if closest_a is not None and closest_b is not None:
                    series_a.append(closest_a)
                    series_b.append(closest_b)
            
            # Need enough aligned points
            if len(series_a) < self.min_data_points:
                return None
            
            try:
                # Calculate Pearson correlation with NumPy
                corr = np.corrcoef(series_a, series_b)[0, 1]
                return float(corr) if not np.isnan(corr) else None
            except Exception as e:
                _LOGGER.warning("Error calculating correlation: %s", e)
                return None
        else:
            # Simple correlation implementation without NumPy
            return self._calculate_simple_correlation(values_a, values_b)
    
    def _calculate_simple_correlation(self, values_a, values_b):
        """Calculate a simple correlation without NumPy."""
        # Extract aligned timestamps (within 5 minutes of each other)
        aligned_values = []
        
        for t_a, v_a in values_a:
            if v_a is None:
                continue
                
            # Find closest value in series B
            for t_b, v_b in values_b:
                if v_b is None:
                    continue
                    
                # If timestamps are within 5 minutes, consider them aligned
                if abs(t_a - t_b) < 300:
                    aligned_values.append((v_a, v_b))
                    break
        
        if len(aligned_values) < self.min_data_points:
            return None
            
        # Calculate mean of each series
        mean_a = sum(a for a, _ in aligned_values) / len(aligned_values)
        mean_b = sum(b for _, b in aligned_values) / len(aligned_values)
        
        # Calculate correlation coefficient
        num = sum((a - mean_a) * (b - mean_b) for a, b in aligned_values)
        denom_a = sum((a - mean_a) ** 2 for a, _ in aligned_values)
        denom_b = sum((b - mean_b) ** 2 for _, b in aligned_values)
        
        denom = (denom_a * denom_b) ** 0.5
        
        if abs(denom) < 1e-10:  # Avoid division by zero
            return None
            
        return num / denom
    
    def _find_closest_value(self, values: List[Tuple[float, float]], 
                          target_time: float, max_distance: float = 300) -> Optional[float]:
        """Find value closest to target time, within max_distance seconds."""
        closest_value = None
        closest_distance = float('inf')
        
        for timestamp, value in values:
            if value is None:
                continue
                
            distance = abs(timestamp - target_time)
            if distance < closest_distance and distance <= max_distance:
                closest_distance = distance
                closest_value = value
                
        return closest_value
        
    def _update_clusters(self) -> None:
        """Update clusters based on correlation matrix."""
        # Reset clusters
        self._clusters = {}
        self._entity_clusters = {}
        
        # Get all entities in the correlation matrix
        entities = set()
        for entity in self._correlation_matrix:
            entities.add(entity)
        
        # Track assigned entities
        assigned = set()
        cluster_id = 0
        
        # Create clusters
        for entity in entities:
            if entity in assigned:
                continue
                
            # Start new cluster
            cluster = {entity}
            assigned.add(entity)
            
            # Find correlated entities
            for other in entities:
                if other == entity or other in assigned:
                    continue
                    
                if (entity in self._correlation_matrix and 
                    other in self._correlation_matrix[entity] and
                    self._correlation_matrix[entity][other] >= self.correlation_threshold):
                    cluster.add(other)
                    assigned.add(other)
            
            # Only create clusters with at least 2 members
            if len(cluster) > 1:
                cluster_name = f"cluster_{cluster_id}"
                self._clusters[cluster_name] = list(cluster)
                
                # Update entity -> cluster mapping
                for e in cluster:
                    self._entity_clusters[e] = cluster_name
                    
                cluster_id += 1
        
        # Add singletons (entities not in any cluster)
        for entity in entities - assigned:
            singleton_name = f"singleton_{entity}"
            self._clusters[singleton_name] = [entity]
            self._entity_clusters[entity] = singleton_name
    
    def get_clusters(self) -> Dict[str, List[str]]:
        """Return current clusters."""
        return self._clusters.copy()
        
    def get_correlation(self, entity_a: str, entity_b: str) -> Optional[float]:
        """Get correlation between two entities."""
        if entity_a in self._correlation_matrix and entity_b in self._correlation_matrix[entity_a]:
            return self._correlation_matrix[entity_a][entity_b]
        return None
        
    def get_cluster_for_entity(self, entity_id: str) -> Optional[str]:
        """Get cluster ID containing the entity."""
        return self._entity_clusters.get(entity_id)

    def _load_from_storage(self) -> None:
        """Load correlation data from storage."""
        if not self._storage_manager:
            return
            
        # Load correlation matrix
        matrix = self._storage_manager.get_correlation_matrix()
        if matrix:
            try:
                self._correlation_matrix = matrix
                _LOGGER.info("Loaded correlation matrix from storage")
            except Exception as e:
                _LOGGER.warning("Failed to load correlation matrix: %s", e)
        
        # Load clusters
        clusters = self._storage_manager.get_clusters()
        if clusters:
            try:
                self._clusters = clusters
                
                # Rebuild entity -> cluster mapping
                self._entity_clusters = {}
                for cluster_id, entities in clusters.items():
                    for entity_id in entities:
                        self._entity_clusters[entity_id] = cluster_id
                        
                _LOGGER.info("Loaded %d clusters from storage", len(clusters))
            except Exception as e:
                _LOGGER.warning("Failed to load clusters: %s", e)
    
    def _save_to_storage(self) -> bool:
        """Save correlation data to storage."""
        if not self._storage_manager:
            return False
            
        success = True
        
        # Save correlation matrix
        if hasattr(self._storage_manager, "save_correlation_matrix"):
            try:
                self._storage_manager.save_correlation_matrix(self._correlation_matrix)
            except Exception as e:
                _LOGGER.error("Failed to save correlation matrix: %s", e)
                success = False
                
        # Save clusters
        if hasattr(self._storage_manager, "save_clusters"):
            try:
                self._storage_manager.save_clusters(self._clusters)
            except Exception as e:
                _LOGGER.error("Failed to save clusters: %s", e)
                success = False
                
        return success