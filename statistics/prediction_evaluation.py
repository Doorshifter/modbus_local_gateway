"""
Prediction evaluation system for Modbus optimization.

This module tracks and evaluates the accuracy of prediction models used
for interval optimization and pattern detection.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import statistics
import math

_LOGGER = logging.getLogger(__name__)

class PredictionMetrics:
    """Tracks metrics for a prediction model."""
    
    def __init__(self, entity_id: str, window_size: int = 100):
        """Initialize prediction metrics.
        
        Args:
            entity_id: Entity identifier
            window_size: Size of the rolling window for metrics
        """
        self.entity_id = entity_id
        self.window_size = window_size
        
        # For interval prediction
        self.predicted_intervals: List[int] = []
        self.actual_intervals: List[int] = []
        
        # For value change prediction
        self.change_predictions: List[bool] = []  # True if predicted change
        self.actual_changes: List[bool] = []      # True if actual change
        
        # Timestamps
        self.prediction_timestamps: List[float] = []
        
        # Result metrics
        self.hit_rate = 0.0
        self.miss_rate = 0.0
        self.false_positive_rate = 0.0
        self.false_negative_rate = 0.0
        self.mean_absolute_error = 0.0
        self.accuracy_score = 0.0
        
        # Last update
        self.last_update = time.time()
    
    def add_interval_prediction(self, predicted: int, actual: int) -> None:
        """Add an interval prediction result.
        
        Args:
            predicted: Predicted time until next change
            actual: Actual time until next change
        """
        self.predicted_intervals.append(predicted)
        self.actual_intervals.append(actual)
        self.prediction_timestamps.append(time.time())
        
        # Keep window size
        while len(self.predicted_intervals) > self.window_size:
            self.predicted_intervals.pop(0)
            self.actual_intervals.pop(0)
            self.prediction_timestamps.pop(0)
        
        self._calculate_metrics()
    
    def add_change_prediction(self, predicted_change: bool, actual_change: bool) -> None:
        """Add a change prediction result.
        
        Args:
            predicted_change: True if a change was predicted
            actual_change: True if a change actually occurred
        """
        self.change_predictions.append(predicted_change)
        self.actual_changes.append(actual_change)
        self.prediction_timestamps.append(time.time())
        
        # Keep window size
        while len(self.change_predictions) > self.window_size:
            self.change_predictions.pop(0)
            self.actual_changes.pop(0)
            self.prediction_timestamps.pop(0)
        
        self._calculate_metrics()
    
    def _calculate_metrics(self) -> None:
        """Calculate all prediction metrics."""
        self.last_update = time.time()
        
        # Skip if we don't have enough data
        if not self.predicted_intervals and not self.change_predictions:
            return
        
        # Calculate interval prediction metrics if available
        if self.predicted_intervals:
            errors = [
                abs(p - a) for p, a in zip(self.predicted_intervals, self.actual_intervals)
            ]
            self.mean_absolute_error = sum(errors) / len(errors)
        
        # Calculate change prediction metrics if available
        if self.change_predictions:
            true_positives = sum(1 for p, a in zip(self.change_predictions, self.actual_changes) 
                               if p and a)
            true_negatives = sum(1 for p, a in zip(self.change_predictions, self.actual_changes) 
                               if not p and not a)
            false_positives = sum(1 for p, a in zip(self.change_predictions, self.actual_changes) 
                                if p and not a)
            false_negatives = sum(1 for p, a in zip(self.change_predictions, self.actual_changes) 
                                if not p and a)
            
            total = len(self.change_predictions)
            
            # Calculate rates
            self.hit_rate = true_positives / total if total > 0 else 0
            self.miss_rate = false_negatives / total if total > 0 else 0
            self.false_positive_rate = false_positives / total if total > 0 else 0
            self.false_negative_rate = false_negatives / total if total > 0 else 0
            
            # Overall accuracy
            self.accuracy_score = (true_positives + true_negatives) / total if total > 0 else 0
        
        # Combined score (weight more heavily toward not missing changes)
        interval_score = 1.0 - min(1.0, self.mean_absolute_error / 60) if self.mean_absolute_error else 0
        change_score = (2 * self.hit_rate + (1 - self.false_positive_rate)) / 3 if self.change_predictions else 0
        
        # Calculate combined accuracy score
        if self.predicted_intervals and self.change_predictions:
            self.accuracy_score = (interval_score * 0.4) + (change_score * 0.6)
        elif self.predicted_intervals:
            self.accuracy_score = interval_score
        elif self.change_predictions:
            self.accuracy_score = change_score
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the current metrics."""
        metrics = {
            "entity_id": self.entity_id,
            "window_size": self.window_size,
            "sample_count": max(len(self.predicted_intervals), len(self.change_predictions)),
            "accuracy_score": round(self.accuracy_score * 100, 1),
            "hit_rate_percent": round(self.hit_rate * 100, 1),
            "miss_rate_percent": round(self.miss_rate * 100, 1),
            "false_positive_rate_percent": round(self.false_positive_rate * 100, 1),
            "mean_absolute_error_seconds": round(self.mean_absolute_error, 1) if self.mean_absolute_error else None,
            "last_update": datetime.fromtimestamp(self.last_update).isoformat(),
            "confidence_level": self._get_confidence_level(),
        }
        
        # Add recent predictions if available
        if self.predicted_intervals:
            recent_predictions = list(zip(
                self.predicted_intervals[-5:], 
                self.actual_intervals[-5:], 
                self.prediction_timestamps[-5:]
            ))
            metrics["recent_interval_predictions"] = [
                {
                    "predicted": p,
                    "actual": a,
                    "timestamp": datetime.fromtimestamp(ts).isoformat(),
                    "error": abs(p - a)
                } for p, a, ts in recent_predictions
            ]
        
        return metrics
    
    def _get_confidence_level(self) -> str:
        """Get a human-readable confidence level based on accuracy score."""
        if self.accuracy_score >= 0.9:
            return "very_high"
        elif self.accuracy_score >= 0.75:
            return "high"
        elif self.accuracy_score >= 0.6:
            return "moderate"
        elif self.accuracy_score >= 0.4:
            return "low"
        return "very_low"
    
    def should_use_predictions(self) -> bool:
        """Determine if predictions are reliable enough to use."""
        sample_count = max(len(self.predicted_intervals), len(self.change_predictions))
        # Need minimum samples and decent accuracy
        return sample_count >= 10 and self.accuracy_score >= 0.5


class PredictionEvaluator:
    """Centralized evaluator for prediction accuracy."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = PredictionEvaluator()
        return cls._instance
    
    def __init__(self):
        """Initialize the evaluator."""
        self.entity_metrics: Dict[str, PredictionMetrics] = {}
        self.last_metrics_calculation = 0
        self.update_interval = 300  # Calculate overall metrics every 5 minutes
        self.overall_metrics = {
            "average_accuracy": 0.0,
            "best_performer": None,
            "worst_performer": None,
            "total_predictions": 0,
            "entities_tracked": 0,
        }
    
    def get_entity_metrics(self, entity_id: str) -> PredictionMetrics:
        """Get metrics for a specific entity, creating if needed."""
        if entity_id not in self.entity_metrics:
            self.entity_metrics[entity_id] = PredictionMetrics(entity_id)
        return self.entity_metrics[entity_id]
    
    def record_interval_prediction(self, entity_id: str, predicted: int, actual: int) -> None:
        """Record an interval prediction result for an entity."""
        metrics = self.get_entity_metrics(entity_id)
        metrics.add_interval_prediction(predicted, actual)
    
    def record_change_prediction(self, entity_id: str, predicted_change: bool, actual_change: bool) -> None:
        """Record a change prediction result for an entity."""
        metrics = self.get_entity_metrics(entity_id)
        metrics.add_change_prediction(predicted_change, actual_change)
    
    def calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall metrics across all entities."""
        current_time = time.time()
        
        # Only recalculate periodically
        if current_time - self.last_metrics_calculation < self.update_interval:
            return self.overall_metrics
            
        self.last_metrics_calculation = current_time
        
        if not self.entity_metrics:
            return self.overall_metrics
        
        # Calculate averages
        accuracy_scores = [m.accuracy_score for m in self.entity_metrics.values() 
                         if m.accuracy_score > 0]
        
        if accuracy_scores:
            average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
            
            # Find best and worst performers
            best_entity = max(self.entity_metrics.items(), 
                            key=lambda x: x[1].accuracy_score if x[1].accuracy_score > 0 else -1)
            worst_entity = min(self.entity_metrics.items(),
                             key=lambda x: x[1].accuracy_score if x[1].accuracy_score > 0 else float('inf'))
            
            # Total predictions across all entities
            total_predictions = sum(
                max(len(m.predicted_intervals), len(m.change_predictions))
                for m in self.entity_metrics.values()
            )
            
            self.overall_metrics = {
                "average_accuracy": round(average_accuracy * 100, 1),
                "best_performer": {
                    "entity_id": best_entity[0],
                    "accuracy": round(best_entity[1].accuracy_score * 100, 1)
                },
                "worst_performer": {
                    "entity_id": worst_entity[0],
                    "accuracy": round(worst_entity[1].accuracy_score * 100, 1)
                },
                "total_predictions": total_predictions,
                "entities_tracked": len(self.entity_metrics),
                "last_updated": datetime.fromtimestamp(current_time).isoformat(),
            }
        
        return self.overall_metrics
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all entities."""
        self.calculate_overall_metrics()
        
        return {
            "overall": self.overall_metrics,
            "entities": {
                entity_id: metrics.get_metrics()
                for entity_id, metrics in self.entity_metrics.items()
            }
        }
    
    def get_prediction_improvement_suggestions(self) -> Dict[str, List[str]]:
        """Generate improvement suggestions for entities with poor predictions."""
        suggestions = {}
        
        for entity_id, metrics in self.entity_metrics.items():
            if metrics.accuracy_score < 0.6:
                entity_suggestions = []
                
                # Analyze prediction patterns
                if metrics.false_positive_rate > 0.3:
                    entity_suggestions.append(
                        "High false positive rate - prediction model is too sensitive"
                    )
                
                if metrics.false_negative_rate > 0.3:
                    entity_suggestions.append(
                        "High false negative rate - prediction model is missing changes"
                    )
                
                if metrics.mean_absolute_error > 60:
                    entity_suggestions.append(
                        f"High timing error ({metrics.mean_absolute_error:.0f}s) - prediction model timing needs adjustment"
                    )
                
                if entity_suggestions:
                    suggestions[entity_id] = entity_suggestions
        
        return suggestions