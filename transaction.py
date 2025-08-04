"""
Custom Transaction Manager for Modbus Local Gateway.

This module provides a custom TransactionManager that suppresses exception logging
to prevent pollution of Home Assistant logs while maintaining proper Modbus
transaction handling.
"""

import contextlib
from asyncio import InvalidStateError

from pymodbus.exceptions import ModbusIOException
from pymodbus.pdu.pdu import ModbusPDU
from pymodbus.transaction import TransactionManager


class MyTransactionManager(TransactionManager):
    """Custom Transaction Manager to suppress exception logging"""

    def data_received(self, data: bytes) -> None:
        """Catch any protocol exceptions so they don't pollute the HA logs"""
        with contextlib.suppress(ModbusIOException, InvalidStateError):
            super().data_received(data)