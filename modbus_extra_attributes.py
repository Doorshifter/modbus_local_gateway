"""Universal mixin for Modbus extra attributes for Home Assistant entities."""

import logging
import time
from typing import Any, Dict, Optional


_LOGGER = logging.getLogger(__name__)

_LOGGER.warning("MIXIN MODULE LOADED: modbus_extra_attributes.py has been imported")

class ModbusExtraAttributesMixin:
    """Mixin to provide comprehensive extra_state_attributes for Modbus entities."""

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return extended state attributes with statistical insights."""
        # BULLETPROOF DEBUG: Log every single step
        _LOGGER.warning("MIXIN START: extra_state_attributes called")
        
        try:
            _LOGGER.warning("MIXIN: Entering try block")
            attrs = {}
            _LOGGER.warning("MIXIN: Created empty attrs dict")
            
            # Step 1: Check modbus context
            _LOGGER.warning("MIXIN: About to check modbus context")
            modbus_ctx = getattr(self, "_modbus_context", None)
            _LOGGER.warning(f"MIXIN STEP 1: modbus_ctx = {modbus_ctx is not None}")
            
            if not modbus_ctx:
                _LOGGER.warning("MIXIN: No modbus context found!")
                return {"debug": "no_modbus_context"}
            
            # Step 2: Check description
            _LOGGER.warning("MIXIN: About to check description")
            desc = getattr(modbus_ctx, "desc", None) if modbus_ctx else None
            _LOGGER.warning(f"MIXIN STEP 2: desc = {desc is not None}")
            
            # Simple test - just add slave_id
            _LOGGER.warning("MIXIN: About to add simple test attribute")
            try:
                slave_id = getattr(modbus_ctx, "slave_id", "unknown")
                attrs["test_slave_id"] = slave_id
                _LOGGER.warning(f"MIXIN: Added test_slave_id = {slave_id}")
            except Exception as e:
                _LOGGER.warning(f"MIXIN: Error adding slave_id: {e}")
            
            # Return early with just the test attribute
            _LOGGER.warning(f"MIXIN FINAL: Returning {len(attrs)} attributes: {list(attrs.keys())}")
            return attrs
            
        except Exception as e:
            _LOGGER.error(f"MIXIN CRITICAL ERROR: {e}", exc_info=True)
            return {"debug_error": str(e)}