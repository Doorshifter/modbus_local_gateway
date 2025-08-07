from __future__ import annotations

"""Config flow for Modbus Local Gateway integration."""

import logging
from collections.abc import Mapping
from typing import Any, cast

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
    OptionsFlowWithConfigEntry,
)
from homeassistant.const import CONF_FILENAME, CONF_HOST, CONF_PORT
from homeassistant.core import callback

from .const import (
    CONF_DEFAULT_PORT,
    CONF_DEFAULT_SCAN_INTERVAL,
    CONF_DEFAULT_SLAVE_ID,
    CONF_PREFIX,
    CONF_SCAN_INTERVAL,
    CONF_SLAVE_ID,
    DEFAULT_SCAN_INTERVAL,
    DOMAIN,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)


class ConfigFlowHandler(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Modbus Local Gateway."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialise Modbus Local Gateway flow."""
        self.client = None
        self.data = {}

    def is_matching(self, other_flow: ConfigFlowHandler) -> bool:
        """Check if the other flow matches this one."""
        return (
            self.client is not None
            and other_flow.client is not None
            and self.client == other_flow.client
        )
        
    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get the options flow for this handler."""
        return ModbusOptionsFlowHandler(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        host_opts: dict[str, str] = {"default": ""}
        port_opts: dict[str, int] = {"default": CONF_DEFAULT_PORT}
        slave_opts: dict[str, int] = {"default": CONF_DEFAULT_SLAVE_ID}
        prefix_opts: dict[str, str] = {"default": ""}
        scan_interval_opts: dict[str, int] = {"default": DEFAULT_SCAN_INTERVAL}

        if user_input is not None:
            # Log the received scan interval value
            _LOGGER.info(
                "Config flow received scan_interval: %s", 
                user_input.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)
            )
            
            # Import here to avoid circular imports
            from .tcp_client import AsyncModbusTcpClientGateway
            
            self.client = AsyncModbusTcpClientGateway.async_get_client_connection(
                host=user_input[CONF_HOST], port=user_input[CONF_PORT]
            )
            await self.client.connect()
            if self.client.connected:
                self.client.close()
                self.data = user_input
                
                # Ensure scan_interval is saved as integer
                if CONF_SCAN_INTERVAL in user_input:
                    self.data[CONF_SCAN_INTERVAL] = int(user_input[CONF_SCAN_INTERVAL])
                    
                return await self.async_step_device_type()

            errors["base"] = "Gateway connection"
            host_opts["default"] = user_input[CONF_HOST]
            port_opts["default"] = int(user_input[CONF_PORT])
            slave_opts["default"] = int(user_input[CONF_SLAVE_ID])
            prefix_opts["default"] = user_input[CONF_PREFIX]
            scan_interval_opts["default"] = int(user_input.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL))

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_HOST, None, **host_opts): str,
                    vol.Required(CONF_PORT, None, **port_opts): int,
                    vol.Required(CONF_SLAVE_ID, None, **slave_opts): int,
                    vol.Optional(CONF_PREFIX, None, **prefix_opts): str,
                    vol.Optional(CONF_SCAN_INTERVAL, None, **scan_interval_opts): int,
                }
            ),
            errors=errors,
        )

    async def async_step_device_type(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the device type step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            self.data.update(user_input)
            # Set default scan interval if not already set
            if CONF_SCAN_INTERVAL not in self.data:
                self.data[CONF_SCAN_INTERVAL] = DEFAULT_SCAN_INTERVAL
                _LOGGER.info(f"Setting default scan_interval={DEFAULT_SCAN_INTERVAL}")
            else:
                _LOGGER.info(f"Using configured scan_interval={self.data[CONF_SCAN_INTERVAL]}")
                
            return await self.async_create()

        # Import here to avoid circular imports
        from .entity_management.device_loader import load_devices
        from .entity_management.modbus_device_info import ModbusDeviceInfo
        
        devices: dict[str, ModbusDeviceInfo] = await load_devices(self.hass)
        devices_data: dict[str, str] = {
            item[0]: f"{item[1].manufacturer or 'Unknown'} {item[1].model or 'Unknown'}"
            for item in sorted(
                devices.items(),
                key=lambda item: f"{item[1].manufacturer or 'Unknown'}"
                f" {item[1].model or 'Unknown'}",
            )
        }

        return self.async_show_form(
            step_id="device_type",
            data_schema=vol.Schema({vol.Required(CONF_FILENAME): vol.In(devices_data)}),
            errors=errors,
        )

    async def async_create(self) -> ConfigFlowResult:
        """Create the entry if we can"""
        # Import here to avoid circular imports
        from .entity_management.device_loader import create_device_info
        from .entity_management.modbus_device_info import ModbusDeviceInfo
        
        device_info: ModbusDeviceInfo = await self.hass.async_add_executor_job(
            create_device_info, self.hass, self.data[CONF_FILENAME]
        )

        # This title is shown in the main devices list under the Modbus Local Gateway integration
        title: str = " ".join(
            [
                part
                for part in [
                    self.data.get(CONF_PREFIX),
                    device_info.manufacturer,
                    device_info.model,
                ]
                if part
            ]
        )
        
        # Log the final data being saved
        _LOGGER.warning(
            "Creating config entry with scan_interval=%s", 
            self.data.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)
        )
        
        return self.async_create_entry(title=title, data=self.data)

    def async_abort(
        self, *, reason: str, description_placeholders: Mapping[str, str] | None = None
    ) -> ConfigFlowResult:
        """Aborting the setup"""
        if self.client:
            self.client.close()
        return super().async_abort(
            reason=reason, description_placeholders=description_placeholders
        )

    def async_show_progress_done(self, *, next_step_id: str) -> ConfigFlowResult:
        """Setup complete"""
        if self.client:
            self.client.close()
        return super().async_show_progress_done(next_step_id=next_step_id)


class ModbusOptionsFlowHandler(OptionsFlowWithConfigEntry):
    """Handle Modbus options."""

    def __init__(self, config_entry):
        """Initialize options flow."""
        super().__init__(config_entry)
        self.updated_options = {}
        self.new_scan_interval = None
        self.old_scan_interval = None

    async def async_step_init(self, user_input=None):
        """Manage the Modbus options."""
        errors = {}
        
        if user_input is not None:
            # Check if we received the refresh field and map it to scan_interval
            if "refresh" in user_input:
                # Get current values
                self.old_scan_interval = self.config_entry.data.get(
                    CONF_SCAN_INTERVAL, 
                    self.config_entry.options.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)
                )
                self.new_scan_interval = int(user_input["refresh"])
                
                # Log the update
                _LOGGER.warning(
                    "Updating scan_interval from %s to %s for %s",
                    self.old_scan_interval,
                    self.new_scan_interval,
                    self.config_entry.title
                )
                
                # Update the entry's data as well as returning new options
                new_data = dict(self.config_entry.data)
                new_data[CONF_SCAN_INTERVAL] = self.new_scan_interval
                
                # This is the critical line: update the entry's data!
                self.hass.config_entries.async_update_entry(
                    entry=self.config_entry,
                    data=new_data
                )
                
                # Store the options for later return
                self.updated_options = {
                    CONF_SCAN_INTERVAL: self.new_scan_interval
                }
                
                # Now proceed to the reload confirmation step
                return await self.async_step_reload_confirm()

        # Set current/default values - IMPORTANT PART
        # Always prioritize data first (which is what our fix updates)
        current_interval = self.config_entry.data.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)
        
        _LOGGER.warning(
            "Options flow initialized with scan_interval=%s (data=%s, options=%s)",
            current_interval,
            self.config_entry.data.get(CONF_SCAN_INTERVAL),
            self.config_entry.options.get(CONF_SCAN_INTERVAL)
        )
        
        # Show form with "refresh" field to match your translations
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Required(
                    "refresh",  # Use "refresh" here to match your translations
                    default=current_interval
                ): int,
            }),
            description_placeholders={
                "slave_id": self.config_entry.data.get(CONF_SLAVE_ID),
                "device": self.config_entry.title,
            },
        )
    
    async def async_step_reload_confirm(self, user_input=None):
        """Ask user to confirm if they want to reload the integration now."""
        if user_input is not None:
            if user_input.get("reload_choice") == "reload_now":
                # User chose to reload now
                await self.hass.config_entries.async_reload(self.config_entry.entry_id)
                
            # Return the updated options regardless of reload choice
            return self.async_create_entry(title="", data=self.updated_options)
                
        return self.async_show_form(
            step_id="reload_confirm",
            data_schema=vol.Schema({
                vol.Required("reload_choice", default="reload_now"): vol.In({
                    "reload_now": "Yes, reload the integration now",
                    "reload_later": "No, I'll reload manually later"
                }),
            }),
            description_placeholders={
                "old_value": str(self.old_scan_interval),
                "new_value": str(self.new_scan_interval),
                "device": self.config_entry.title,
            }
        )