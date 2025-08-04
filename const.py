"""
Constants for the Modbus Local Gateway integration.

This file defines static values used throughout the integration, including
configuration keys, default values, and tuning parameters for the adaptive
timeout mechanism. Centralizing these values improves maintainability and
clarity.
"""

from homeassistant.const import Platform  # Add this import

DOMAIN = "modbus_local_gateway"

# --- Main Configuration ---
CONF_SLAVE_ID = "slave_id"
CONF_DEFAULT_PORT = 502
CONF_DEFAULT_SLAVE_ID = 1
CONF_PREFIX = "prefix"

MAX_READ_SIZE_DEFAULT = 20
MAX_REGISTERS_PER_SECOND_DEFAULT = 100  # Default for most devices

CONF_SCAN_INTERVAL = "scan_interval"
DEFAULT_SCAN_INTERVAL = 30
CONF_DEFAULT_SCAN_INTERVAL = DEFAULT_SCAN_INTERVAL

# --- Add missing constants that are being imported elsewhere ---
CONF_COORDINATOR = "coordinator"
CONF_DEVICE_DATA = "device_data"
CONF_DEVICE_INFO = "device_info"
CONF_GATEWAY_INFO = "gateway_info"  # Added for storing gateway identifier info


PLATFORMS = [
    Platform.SENSOR,
    Platform.BINARY_SENSOR, 
    Platform.SWITCH, 
    Platform.SELECT, 
    Platform.TEXT, 
    Platform.NUMBER
]

# --- Adaptive Timeout Constants ---
# These values control the behavior of the self-adjusting timeout mechanism
# in the Modbus TCP client.

# Initial state guesses for the adaptive timeout algorithm. These provide a
# starting point before the client has measured actual network performance.
ADAPTIVE_TIMEOUT_INITIAL_RTT = 1.0  # Initial guess for Round-Trip Time (seconds).
ADAPTIVE_TIMEOUT_INITIAL_VAR = 0.5  # Initial guess for RTT variance (seconds).

# Smoothing factors (alpha and beta) for the Exponentially Weighted Moving
# Average (EWMA) calculation. These values, typically between 0 and 1,
# control how quickly the algorithm adapts to new measurements.
# Lower values result in slower, more stable adaptation.
# Higher values result in faster, more responsive adaptation.
ADAPTIVE_TIMEOUT_ALPHA = 0.125  # Smoothing factor for the Round-Trip Time.
ADAPTIVE_TIMEOUT_BETA = 0.25   # Smoothing factor for the RTT variance.

# Hard limits to prevent the dynamic timeout from becoming too extreme.
ADAPTIVE_TIMEOUT_MIN = 2.0  # Minimum allowed timeout (seconds).
ADAPTIVE_TIMEOUT_MAX = 30.0  # Maximum allowed timeout (seconds).