# Module: memory_management.py

## Functional Purpose
Provides memory-efficient rolling buffer utilities for managing, storing, and analyzing historical Modbus optimization and statistical data in-memory.  
Enables automatic calculation of statistics, efficient handling of time-series data, and dynamic downsampling to optimize memory usage for large or long-running systems.

## Core Responsibilities
- **RollingBuffer**: Store a fixed number of recent values in a memory-efficient, FIFO manner, automatically calculating statistics (min, max, mean, stdev, count, etc.) over the buffer.
- **TimestampedBuffer**: Store timestamped values (optionally with metadata), with logic for purging old data, downsampling when nearing capacity, and querying by time range or at a specific timestamp.
- Provide APIs for adding, clearing, retrieving values, and extracting statistics from buffers.
- Enable efficient and loss-tolerant statistical analysis in resource-constrained or long-lived environments.

## Key Public Classes and Methods

| Class/Method                          | Purpose                                                                      | Log Level When Called   | Success Indicator                |
|---------------------------------------|------------------------------------------------------------------------------|------------------------|----------------------------------|
| `RollingBuffer`                       | Generic rolling buffer with stats calculation                                | WARNING (stat error)   | Buffer stores up to max_size     |
| `RollingBuffer.add()`                 | Add value and possibly recalculate stats                                      | (none)                 | Value added, stats updated       |
| `RollingBuffer.get_stats()`           | Return current stats (recalculates if stale)                                 | (none)                 | Dict of stats                    |
| `RollingBuffer.clear()`               | Clear all data and stats                                                     | (none)                 | Buffer/statistics emptied        |
| `RollingBuffer.get_values()`          | Return all buffer values as list                                             | (none)                 | List returned                    |
| `RollingBuffer.count`                 | Number of items currently in buffer                                          | (none)                 | Integer                          |
| `RollingBuffer.full`                  | Whether buffer is at max capacity                                            | (none)                 | Boolean                          |
| `TimestampedBuffer`                   | Rolling buffer for dicts with timestamps and value keys                      | DEBUG (downsampling)   | Buffer stores recent values      |
| `TimestampedBuffer.add()`             | Add timestamped value (with metadata), purge old data, downsample if needed  | DEBUG                  | Value added, buffer managed      |
| `TimestampedBuffer.get_values_in_range()` | Query for values between two timestamps                                 | (none)                 | List of dicts                    |
| `TimestampedBuffer.get_value_at_time()`   | Get/interpolate value at a specific time                                 | (none)                 | Dict or None                     |

## Current Logging Philosophy

### WARNING Level
- When an error occurs during statistics calculation in RollingBuffer:
  - `"Error calculating stats: %s"`

### DEBUG Level
- When downsampling is enabled in TimestampedBuffer:
  - `"Enabling downsampling for buffer"`
  - `"Downsampling reduced buffer size from %d to %d", ...`

### INFO/ERROR Level
- None in this module.

## Dependencies (Imports From)
- `collections.deque`, `statistics`, `time`, `datetime`
- `logging`
- Typing: For generics and type hints

## Used By (Imported By)
- Statistics, correlation, interval management, and pattern detection modules needing memory-efficient rolling history for numeric/time-series/statistical data.

## Key Data Flows

1. **Input:**
   - Numeric or dict values (optionally with timestamps and metadata) added to buffer.
2. **Processing:**
   - FIFO management of values (oldest overwritten when buffer full).
   - Automatic statistics calculation on numeric data.
   - Purge of data older than a set number of hours (TimestampedBuffer).
   - Downsampling of data when buffer nears capacity for time-series.
   - Query for values in range or at specific timestamp (with optional interpolation).
3. **Output:**
   - Most recent N values, statistics (min, max, mean, stdev, count, etc.), and trend info.
   - Cleaned and/or downsampled buffers for efficient memory use.

## Integration Points

- **Historical Data Analysis:**  
  Used as a foundation for storing and analyzing recent history for entities, intervals, or statistical events.
- **Interval/Pattern/Correlation/Optimization:**  
  Underpins time-series and trend analytics by providing fast in-memory access to recent and relevant data.

## Current Issues/Tech Debt

- Downsampling and purging logic is basic; may not be optimal for all usage patterns or large time windows.
- No persistence; buffers exist only in memory.
- No thread safety beyond what is provided by deque (not fully safe in multi-threaded use).
- Stats calculation is triggered by time interval, not by buffer modifications; may not always be up-to-date.
- Interpolation in `get_value_at_time()` is incomplete in this snippet.
- Does not handle non-numeric data in statistics; stats are skipped if buffer is not all numbers.