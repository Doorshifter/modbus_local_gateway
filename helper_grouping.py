"""
Helper module for grouping Modbus entities by address for efficient batch reads.

Functions:
    group_entities_for_batching
"""

from typing import List, Tuple
from .context import ModbusContext

def group_entities_for_batching(
    entities: List[ModbusContext], max_read_size: int
) -> List[Tuple[int, int, List[ModbusContext]]]:
    """
    Group entities into contiguous register batches for efficient Modbus reading.

    Each batch covers a contiguous address range and does not exceed max_read_size.

    Args:
        entities (List[ModbusContext]): List of ModbusContext entities (must be for the same slave and data type).
        max_read_size (int): Maximum registers/coils per batch.

    Returns:
        List[Tuple[int, int, List[ModbusContext]]]: List of (start_address, count, [entities]) tuples.
    """
    if not entities:
        return []

    # Sort by register address
    sorted_entities = sorted(entities, key=lambda e: e.desc.register_address)
    groups = []
    current_group = []
    group_start = None
    group_end = None

    for entity in sorted_entities:
        start = entity.desc.register_address
        size = entity.desc.register_count or 1
        end = start + size - 1
        if not current_group:
            # Start new group
            current_group = [entity]
            group_start = start
            group_end = end
        else:
            # If this entity fits in the current group (contiguous and not exceeding max batch size)
            if start == group_end + 1 and (end - group_start + 1) <= max_read_size:
                current_group.append(entity)
                group_end = end
            else:
                # Commit current group, start new
                groups.append((group_start, group_end - group_start + 1, current_group))
                current_group = [entity]
                group_start = start
                group_end = end
    if current_group:
        groups.append((group_start, group_end - group_start + 1, current_group))
    return groups