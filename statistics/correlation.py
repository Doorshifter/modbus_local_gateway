"""
Entity correlation detection and analysis.

This module provides functionality for detecting correlations between entities,
grouping related entities into clusters, and using these relationships to
optimize polling and improve pattern detection.
"""

import logging
import time
import math
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict

_LOGGER = logging.getLogger(__name__)

# Constants for correlation analysis
MIN_SAMPLES_FOR_CORRELATION = 10  # Minimum number of samples to calculate correlation
CORRELATION_THRESHOLD = 0.7  # Default threshold for considering entities correlated
MAX_CLUSTER_SIZE = 20  # Maximum entities in a cluster to prevent over-grouping
MIN_CLUSTER_SIZE = 2  # Minimum entities to form a cluster

class EntityCorrelationManager:
    """Manages correlation detection between entities."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = EntityCorrelationManager()
        return cls._instance
    
    def __init__(self):
        """Initialize correlation manager."""
        self._entity_values = {}  # Dict of entity_id -> list of (timestamp, value) tuples
        self._correlation_matrix = {}  # Dict of (entity_a, entity_b) -> correlation value
        self._clusters = {}  # Dict of cluster_id -> list of entity_ids
        self._entity_to_cluster = {}  # Dict of entity_id -> cluster_id
        self._last_analysis = 0  # Timestamp of last correlation analysis
        self.threshold = CORRELATION_THRESHOLD  # Correlation threshold for clustering
    
    def add_entity_value(self, entity_id: str, value: Any, timestamp: float = None) -> None:
        """Record a new value for an entity.
        
        Args:
            entity_id: Entity identifier
            value: Entity value (must be numeric or convertible to numeric)
            timestamp: Value timestamp (default: current time)
        """
        if timestamp is None:
            timestamp = time.time()
            
        try:
            # Convert value to float for numerical correlation
            numeric_value = float(value)
            
            # Initialize entity in the values dict if needed
            if entity_id not in self._entity_values:
                self._entity_values[entity_id] = []
                
            # Add the new value
            self._entity_values[entity_id].append((timestamp, numeric_value))
            
            # Limit history to prevent memory issues (keep last 100 values)
            if len(self._entity_values[entity_id]) > 100:
                self._entity_values[entity_id] = self._entity_values[entity_id][-100:]
                
        except (ValueError, TypeError):
            # Skip non-numeric values
            pass
    
    def get_correlation(self, entity_a: str, entity_b: str) -> Optional[float]:
        """Get the correlation between two entities.
        
        Args:
            entity_a: First entity ID
            entity_b: Second entity ID
            
        Returns:
            Correlation coefficient (-1 to 1) or None if not available
        """
        # Check cached correlation
        if (entity_a, entity_b) in self._correlation_matrix:
            return self._correlation_matrix[(entity_a, entity_b)]
        
        if (entity_b, entity_a) in self._correlation_matrix:
            return self._correlation_matrix[(entity_b, entity_a)]
            
        # Calculate correlation if we have enough data
        return self._calculate_correlation(entity_a, entity_b)
    
    def _calculate_correlation(self, entity_a: str, entity_b: str) -> Optional[float]:
        """Calculate correlation coefficient between two entities.
        
        Args:
            entity_a: First entity ID
            entity_b: Second entity ID
            
        Returns:
            Correlation coefficient (-1 to 1) or None if not enough data
        """
        # Make sure we have data for both entities
        if entity_a not in self._entity_values or entity_b not in self._entity_values:
            return None
            
        values_a = self._entity_values[entity_a]
        values_b = self._entity_values[entity_b]
        
        # Need minimum number of samples
        if len(values_a) < MIN_SAMPLES_FOR_CORRELATION or len(values_b) < MIN_SAMPLES_FOR_CORRELATION:
            return None
            
        # Extract values with similar timestamps (within 5 seconds)
        paired_values = []
        for ts_a, val_a in values_a:
            for ts_b, val_b in values_b:
                if abs(ts_a - ts_b) < 5:  # Within 5 seconds
                    paired_values.append((val_a, val_b))
                    break
                    
        # Need minimum number of paired samples
        if len(paired_values) < MIN_SAMPLES_FOR_CORRELATION:
            return None
            
        # Calculate Pearson correlation
        x_vals = [p[0] for p in paired_values]
        y_vals = [p[1] for p in paired_values]
        
        try:
            # Calculate means
            mean_x = sum(x_vals) / len(x_vals)
            mean_y = sum(y_vals) / len(y_vals)
            
            # Calculate variances
            var_x = sum((x - mean_x) ** 2 for x in x_vals)
            var_y = sum((y - mean_y) ** 2 for y in y_vals)
            
            # Handle zero variance
            if var_x == 0 or var_y == 0:
                return 0.0
                
            # Calculate covariance
            cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))
            
            # Calculate correlation
            correlation = cov / math.sqrt(var_x * var_y)
            
            # Store in matrix
            self._correlation_matrix[(entity_a, entity_b)] = correlation
            
            return correlation
            
        except (ZeroDivisionError, ValueError):
            return None
    
    def analyze_correlations(self, force: bool = False) -> Dict[str, List[str]]:
        """Analyze correlations and identify clusters of related entities.
        
        Args:
            force: Force analysis even if recently performed
            
        Returns:
            Dictionary mapping cluster IDs to lists of entity IDs
        """
        current_time = time.time()
        
        # Don't analyze too frequently unless forced
        if not force and current_time - self._last_analysis < 3600:  # Once per hour
            return self._clusters
            
        self._last_analysis = current_time
        
        # Step 1: Calculate all pairwise correlations
        entities = list(self._entity_values.keys())
        correlations = []
        
        for i, entity_a in enumerate(entities):
            for j in range(i + 1, len(entities)):
                entity_b = entities[j]
                correlation = self.get_correlation(entity_a, entity_b)
                
                if correlation is not None and abs(correlation) >= self.threshold:
                    # Store significant correlations
                    correlations.append((entity_a, entity_b, abs(correlation)))
                    
        # Step 2: Sort correlations by strength (strongest first)
        correlations.sort(key=lambda x: x[2], reverse=True)
        
        # Step 3: Create clusters using a greedy approach
        self._clusters = {}
        self._entity_to_cluster = {}
        
        for entity_a, entity_b, strength in correlations:
            # Check if either entity is already in a cluster
            cluster_a = self._entity_to_cluster.get(entity_a)
            cluster_b = self._entity_to_cluster.get(entity_b)
            
            if cluster_a is None and cluster_b is None:
                # Create new cluster
                cluster_id = f"cluster_{len(self._clusters)}"
                self._clusters[cluster_id] = [entity_a, entity_b]
                self._entity_to_cluster[entity_a] = cluster_id
                self._entity_to_cluster[entity_b] = cluster_id
                
            elif cluster_a is not None and cluster_b is None:
                # Add entity_b to entity_a's cluster if not too large
                if len(self._clusters[cluster_a]) < MAX_CLUSTER_SIZE:
                    self._clusters[cluster_a].append(entity_b)
                    self._entity_to_cluster[entity_b] = cluster_a
                    
            elif cluster_a is None and cluster_b is not None:
                # Add entity_a to entity_b's cluster if not too large
                if len(self._clusters[cluster_b]) < MAX_CLUSTER_SIZE:
                    self._clusters[cluster_b].append(entity_a)
                    self._entity_to_cluster[entity_a] = cluster_b
                    
            elif cluster_a != cluster_b:
                # Entities are in different clusters - consider merging
                if len(self._clusters[cluster_a]) + len(self._clusters[cluster_b]) <= MAX_CLUSTER_SIZE:
                    # Merge smaller cluster into larger one
                    if len(self._clusters[cluster_a]) >= len(self._clusters[cluster_b]):
                        target_cluster = cluster_a
                        source_cluster = cluster_b
                    else:
                        target_cluster = cluster_b
                        source_cluster = cluster_a
                        
                    # Merge clusters
                    for entity in self._clusters[source_cluster]:
                        self._clusters[target_cluster].append(entity)
                        self._entity_to_cluster[entity] = target_cluster
                        
                    # Remove source cluster
                    del self._clusters[source_cluster]
        
        _LOGGER.info("Correlation analysis complete: found %d clusters", len(self._clusters))
        return self._clusters
    
    def get_clusters(self) -> Dict[str, List[str]]:
        """Get current entity clusters.
        
        Returns:
            Dictionary mapping cluster IDs to lists of entity IDs
        """
        return self._clusters
    
    def get_cluster_for_entity(self, entity_id: str) -> Optional[str]:
        """Get the cluster ID for an entity.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Cluster ID or None if entity is not in a cluster
        """
        return self._entity_to_cluster.get(entity_id)
    
    def get_correlation_matrix(self) -> Dict[Tuple[str, str], float]:
        """Get the full correlation matrix.
        
        Returns:
            Dictionary mapping entity pairs to correlation coefficients
        """
        return self._correlation_matrix
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the correlation system.
        
        Returns:
            Dictionary with correlation statistics
        """
        # Count entities with values
        entities_with_values = len(self._entity_values)
        
        # Count entities in clusters
        entities_in_clusters = sum(len(entities) for entities in self._clusters.values())
        
        # Count correlations by strength
        correlation_by_strength = {
            "strong": 0,    # 0.8-1.0
            "medium": 0,    # 0.5-0.8
            "weak": 0       # threshold-0.5
        }
        
        for corr in self._correlation_matrix.values():
            abs_corr = abs(corr)
            if abs_corr >= 0.8:
                correlation_by_strength["strong"] += 1
            elif abs_corr >= 0.5:
                correlation_by_strength["medium"] += 1
            else:
                correlation_by_strength["weak"] += 1
                
        return {
            "total_entities": entities_with_values,
            "entities_in_clusters": entities_in_clusters,
            "cluster_count": len(self._clusters),
            "correlation_count": len(self._correlation_matrix),
            "correlation_by_strength": correlation_by_strength,
            "current_threshold": self.threshold,
            "last_analysis": self._last_analysis
        }

# Global instance
CORRELATION_MANAGER = EntityCorrelationManager.get_instance()