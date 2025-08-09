# Module: prediction_evaluation.py

## Functional Purpose
Tracks, evaluates, and summarizes the accuracy of prediction models used for interval optimization and pattern detection in Modbus optimization systems.  
Provides both per-entity and overall metrics to guide model tuning, system diagnostics, and automatic improvement.

## Core Responsibilities
- Maintain rolling-window metrics for prediction accuracy, error, and confidence per entity.
- Track and distinguish between interval/time-to-change predictions and binary change/no-change predictions.
- Calculate hit rate, miss rate, false positive/negative rates, mean absolute error, and overall accuracy score.
- Summarize and aggregate metrics across all entities, finding best/worst performers and overall averages.
- Generate improvement suggestions for entities with poor prediction quality.
- Provide a singleton evaluator for centralized tracking and reporting.

## Key Public Classes and Methods

| Class/Method                                     | Purpose                                                                 | Log Level When Called | Success Indicator                   |
|--------------------------------------------------|-------------------------------------------------------------------------|----------------------|-------------------------------------|
| `PredictionMetrics`                              | Tracks rolling-window prediction performance for a single entity        | (none)               | Metrics maintained, up-to-date      |
| `.add_interval_prediction(predicted, actual)`    | Record a time-to-change prediction and result                           | (none)               | Metrics updated                     |
| `.add_change_prediction(predicted, actual)`      | Record a binary (change/no-change) prediction and result                | (none)               | Metrics updated                     |
| `.get_metrics()`                                 | Return current metrics dict for entity                                  | (none)               | Dict returned                       |
| `.should_use_predictions()`                      | Whether predictions are reliable enough for production use              | (none)               | Bool                                |
| `PredictionEvaluator`                            | Singleton that manages all entity metrics and aggregates                | (none)               | Singleton instance available        |
| `.record_interval_prediction()`                  | Add interval prediction for an entity                                   | (none)               | Metrics for entity updated          |
| `.record_change_prediction()`                    | Add change/no-change prediction for an entity                           | (none)               | Metrics for entity updated          |
| `.calculate_overall_metrics()`                   | Compute global average, best/worst, totals, etc.                        | (none)               | Dict with summary metrics           |
| `.get_all_metrics()`                             | Return all metrics, per-entity and overall                              | (none)               | Dict returned                       |
| `.get_prediction_improvement_suggestions()`      | Suggest improvements for low-performing entities                        | (none)               | Dict of entity_id -> suggestions    |

## Current Logging Philosophy

- No INFO/WARNING/ERROR/DEBUG logs in normal operation.
- Only logger declaration at module level; no runtime logs, even for errors.

## Dependencies (Imports From)
- `logging`, `time`, `datetime`, `statistics`, `math`
- Typing: For type hints

## Used By (Imported By)
- Performance optimization, prediction tuning, diagnostics dashboards, or any analytics module needing to monitor or improve prediction quality.

## Key Data Flows

1. **Input:**
   - Predicted and actual intervals (seconds/timestamps).
   - Predicted and actual change/no-change (bool).
2. **Processing:**
   - Maintains rolling window of results (default 100).
   - Calculates various classification and regression metrics.
   - Aggregates across all entities for system-wide view.
   - Assigns human-readable confidence levels and makes usage recommendations.
   - Generates improvement advice for underperforming entities.
3. **Output:**
   - Per-entity and overall metrics (accuracy, error, hit/miss/FP/FN rates, sample counts).
   - Best/worst performing entities and improvement hints.
   - Dicts for dashboards, reporting, or automated feedback.

## Integration Points

- **Performance/Optimization Loop:**  
  Used to evaluate and improve interval/pattern prediction models and strategies.
- **Diagnostics/Monitoring:**  
  Used to power dashboards, alerts, or automated feedback on prediction health.

## Current Issues/Tech Debt

- No logging for errors or abnormal situations (silent failures).
- No persistence—metrics are lost on restart.
- Accuracy/confidence thresholds are hardcoded and may not generalize to all use cases.
- No thread safety for the singleton or metric updates.
- Only supports basic rolling window, not more advanced time-decay or weighted metrics.