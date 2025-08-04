"""Loading YAML device definitions from disk"""

import glob
import logging
import os.path

from homeassistant.core import HomeAssistant

from custom_components.modbus_local_gateway.const import DOMAIN

from .modbus_device_info import ModbusDeviceInfo

_LOGGER: logging.Logger = logging.getLogger(__name__)


def get_config_files(hass: HomeAssistant) -> dict[str, str]:
    """Get the list of YAML files in the config directory."""

    # Use hass.config.path to resolve CONFIG_DIR
    config_dir = hass.config.path("custom_components/modbus_local_gateway/device_configs")
    filenames: dict[str, str] = {
        os.path.basename(x): os.path.basename(x) for x in glob.glob(f"{config_dir}/*.yaml")
    }

    # Check for extra config files in the custom_components domain directory
    extra_config_dir: str = hass.config.path(DOMAIN)
    if os.path.exists(extra_config_dir) and os.path.isdir(extra_config_dir):
        extra_files: dict[str, str] = {
            os.path.basename(x): os.path.basename(x)
            for x in glob.glob(f"{extra_config_dir}/*.yaml")
        }
        filenames.update(extra_files)

    return filenames


def find_matching_filename(requested: str, available: dict[str, str]) -> str | None:
    """Find a filename in available dict, case-insensitive."""
    requested_lower = requested.lower()
    for actual in available:
        if actual.lower() == requested_lower:
            return actual
    return None


async def load_devices(hass: HomeAssistant) -> dict[str, ModbusDeviceInfo]:
    """Find and load files from disk"""
    filenames: dict[str, str] = await hass.async_add_executor_job(
        get_config_files, hass
    )

    devices: dict[str, ModbusDeviceInfo] = {}
    for filename, path in filenames.items():
        try:
            devices[filename] = await hass.async_add_executor_job(
                create_device_info, hass, path, filenames
            )
        except Exception as err:  # pylint: disable=broad-exception-caught
            _LOGGER.error("Error loading device from YAML file: %s - %s", filename, err)

    return devices


def create_device_info(
    hass: HomeAssistant, filename: str, available_files: dict[str, str] | None = None
) -> ModbusDeviceInfo:
    """Create the ModbusDeviceInfo object"""
    _available_files: dict[str, str] | None = available_files
    if _available_files is None:
        _available_files = get_config_files(hass)
    name: str = os.path.basename(filename)
    actual_filename = find_matching_filename(name, _available_files)
    if not actual_filename:
        _LOGGER.error("File %s not found in available files: %s", name, available_files)
        raise FileNotFoundError(f"File {name} not found in available files.")

    # Pass only the actual base filename to ModbusDeviceInfo
    return ModbusDeviceInfo(fname=actual_filename, hass=hass)