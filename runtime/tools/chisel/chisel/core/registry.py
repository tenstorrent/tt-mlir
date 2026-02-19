# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Registry module for managing and correlating MLIR operations between GOLDEN (CPU) and DEVICE execution contexts.

This module provides the OpGroup and Registry classes which are responsible for:
- Grouping related TTNN operations from different execution contexts (GOLDEN CPU vs DEVICE)
- Tracking tensor locations and their relationships
- Managing operation groups and their inputs/outputs
- Facilitating comparison between CPU golden and device hardware execution paths

Note: Both GOLDEN and DEVICE execution contexts use the same TTNN MLIR module,
      differing only in their execution backend (CPU vs hardware).
"""
from collections import defaultdict
from functools import cache
from typing import Callable, Dict, List, Tuple, Union

from ttmlir.ir import BlockArgument, Operation, OpResult

from ..utils.location import hash_location
from .enums import ExecutionType
from .ops import IRModule, get_op_inputs, get_op_outputs


class OpGroup:
    """
    Groups related TTNN operations from GOLDEN (CPU) and DEVICE execution contexts sharing the same location.

    This class maintains a collection of TTNN operations from both execution contexts that are
    considered equivalent based on their source locations. In cases where operations don't have
    matching counterparts (e.g., due to compiler optimizations or operation fusion), they are
    merged into the next group.

    Args:
        id: Unique identifier for the operation group
        skip_group: If True, this group should be skipped during execution

    Attributes:
        ops (Dict[ExecutionType, List[Operation]]): Maps execution types to their operations
        skip_group (bool): Flag indicating if this group should be skipped
    """

    def __init__(self, id, skip_group=False):
        """Initialize a new operation group with the given ID and skip status."""
        self.id = id
        self.ops: Dict[ExecutionType, List[Operation]] = {
            ExecutionType.GOLDEN: [],
            ExecutionType.DEVICE: [],
        }
        self.skip_group = skip_group

    def add_op(self, op: Operation, execution_type: ExecutionType) -> None:
        self.ops[execution_type].append(op)

    def __getitem__(self, kind: ExecutionType):
        if kind not in self.ops:
            return []
        return self.ops[kind]

    def get_last_op(
        self, kind: ExecutionType, with_output: bool = True
    ) -> Operation | None:
        """
        Returns the last op in the group.
        If the with_output flag is set, returns the last op with output.
        """
        if len(self.ops[kind]) == 0:
            return None
        if not with_output:
            return self.ops[kind][-1]

        for op in self.ops[kind][::-1]:
            if len(get_op_outputs(op)) > 0:
                return op


class Registry:
    """
    Central registry for managing and correlating TTNN operations between execution contexts.

    This class maintains the relationship between TTNN operations in the GOLDEN (CPU reference)
    and DEVICE (hardware target) execution contexts. Both contexts use the same underlying TTNN
    MLIR module, differing only in their execution backend.

    It provides methods to:
    - Load and group TTNN operations from both execution contexts
    - Track tensor locations and their relationships
    - Retrieve operation groups and their inputs/outputs
    - Manage operation execution and comparison

    Args:
        golden_module: The IRModule wrapping TTNN module for CPU/golden execution
        device_module: The IRModule wrapping TTNN module for device/hardware execution
        should_skip_op: Callable that determines if an operation should be skipped

    Note: golden_module and device_module wrap the same TTNN MLIR module with different
          ExecutionType flags (GOLDEN vs DEVICE).
    """

    def __init__(
        self,
        golden_module: IRModule,
        device_module: IRModule,
        should_skip_op: Callable[[Operation], bool] = lambda op: False,
    ) -> None:
        # TODO: check what is actual type of tensors, it is not Operation
        self.tensors: Dict[
            Tuple[int, int], Dict[ExecutionType, OpResult | BlockArgument]
        ] = defaultdict(dict)
        self.tensor_to_location: Dict[ExecutionType, Dict[str, Tuple[int, int]]] = {
            ExecutionType.GOLDEN: {},
            ExecutionType.DEVICE: {},
        }
        self.op_groups: Dict[Tuple[int, int], OpGroup] = {}

        self.modules: Dict[ExecutionType, IRModule] = {
            ExecutionType.GOLDEN: golden_module,
            ExecutionType.DEVICE: device_module,
        }

        self.module_iters = {
            execution_type: iter(enumerate(module.get_function_ops()))
            for execution_type, module in self.modules.items()
        }
        self.last_loaded_loc: Dict[ExecutionType, set] = {
            ExecutionType.GOLDEN: set(),
            ExecutionType.DEVICE: set(),
        }
        self.should_skip = should_skip_op

        for execution_type, module in self.modules.items():
            for arg in module.get_function_inputs():
                self.add_tensor(arg, execution_type)

    def load_all_ops(self) -> None:
        """
        Load and process all operations from both GOLDEN and DEVICE modules.

        This method iterates through all operations in both execution contexts,
        groups them by location, and builds the tensor tracking infrastructure.
        It should be called after initialization to prepare the registry for use.
        """
        for execution_type in [ExecutionType.GOLDEN, ExecutionType.DEVICE]:
            for _, op in self.module_iters[execution_type]:
                self._add_op(op, execution_type, should_skip=self.should_skip(op))
                # Track all tensor outputs from operations
                for output in get_op_outputs(op):
                    self.add_tensor(output, execution_type)
        self._merge_empty_golden_groups()

    def should_compare(
        self,
        op: Operation,
        location_hash: Tuple[int, int],
        execution_type: ExecutionType,
    ):
        # TODO: check if there is need for with_output parameter
        last_op = self.get_last_op(location_hash, execution_type)
        if last_op is None:
            return False
        return last_op == op

    def add_tensor(
        self, tensor: Union[OpResult, BlockArgument], kind: ExecutionType
    ) -> None:
        """
        Register a tensor in the registry and track its location.

        Args:
            tensor: The tensor (OpResult or BlockArgument) to register
            kind: The execution context (GOLDEN or DEVICE)
        """
        location_hash = hash_location(tensor.location)
        # Map tensor name to its location
        self.tensor_to_location[kind][
            tensor.get_name(self.modules[kind].get_asm_state())
        ] = location_hash
        # Store tensor by its location
        self.tensors[location_hash][kind] = tensor

    def get_tensor(self, location, kind: ExecutionType):
        return self.tensors[location][kind]

    def find_op(
        self, location: Tuple[int, int], asm: str, execution_type: ExecutionType
    ) -> Operation | None:
        """
        Find an operation by its location and assembly representation.

        Args:
            location: The location hash to search in
            asm: The assembly representation of the operation to find
            execution_type: The execution context to search in

        Returns:
            The matching operation, or None if not found
        """
        if location not in self.op_groups:
            return None

        # Search through all operations in the group for a matching assembly string
        for op in self.op_groups[location].ops[execution_type]:
            if op.get_asm(enable_debug_info=True) == asm:
                return op
        return None

    def get_group(self, group_id: Tuple[int, int], execution_type: ExecutionType):
        return self.op_groups[group_id][execution_type]

    def get_group_output(
        self, group_id: Tuple[int, int], execution_type: ExecutionType
    ):
        if group_id not in self.tensors:
            print(f"Group {group_id} does not exist")
            return None
        if execution_type not in self.tensors[group_id]:
            print(f"Execution type {execution_type} does not exist in group {group_id}")
            return None
        return self.tensors[group_id][execution_type]

    def get_group_inputs(
        self, group_id: Tuple[int, int], execution_type: ExecutionType
    ) -> List[Operation]:
        """
        Get the input tensors for an operation group.

        This method identifies the input tensors that are consumed but not produced
        within the same operation group.

        Args:
            group_id: The ID of the operation group
            execution_type: The execution context to get inputs for

        Returns:
            List of input tensor operations that are external to the group
        """
        tensors = set()
        # First collect all input tensors from all operations in the group
        for op in self.op_groups[group_id][execution_type]:
            tensors.update([t for t in get_op_inputs(op)])

        # Remove any tensors that are produced within the same group
        for op in self.op_groups[group_id][execution_type]:
            outputs = get_op_outputs(op)
            if len(outputs) == 0:
                continue
            for t in outputs:
                if t not in tensors:
                    continue
                tensors.remove(t)

        return list(tensors)

    @cache
    def get_last_op(
        self,
        group_id: Tuple[int, int],
        execution_type: ExecutionType,
        with_output: bool = True,
    ) -> Operation | None:
        if group_id not in self.op_groups:
            # TODO: check in what cases this happens
            return None
        return self.op_groups[group_id].get_last_op(execution_type, with_output)

    def _add_op(
        self, op: Operation, execution_type: ExecutionType, should_skip: bool = False
    ):
        location_hash = hash_location(op.location)
        # Create a new operation group if one doesn't exist for this location
        if location_hash not in self.op_groups:
            self.op_groups[location_hash] = OpGroup(location_hash)

        # Add the operation to the appropriate group and execution context
        self.op_groups[location_hash].add_op(op, execution_type)
        if should_skip:
            # Mark the entire group for skipping if any operation should be skipped
            self.op_groups[location_hash].skip_group = True

    def _merge_empty_golden_groups(self):
        """
        Merge operation groups that only have GOLDEN operations with the next group.

        This handles cases where operations in the GOLDEN execution context don't have
        corresponding DEVICE operations at the same location (e.g., due to compiler
        optimizations or execution differences between CPU and hardware).

        Since both contexts now use the same TTNN module, this should rarely occur.
        However, we keep this logic to handle edge cases where execution diverges.
        """
        # Groups are keyed by (line, col); sorting gives the order of the golden ops in the function.
        sorted_ids = sorted(self.op_groups.keys())
        idx = 0
        while idx < len(sorted_ids) - 1:  # last group has no “next”
            gid = sorted_ids[idx]
            group = self.op_groups[gid]
            # Only GOLDEN ops?
            if len(group.ops[ExecutionType.DEVICE]) != 0:
                idx += 1
                continue

            next_gid = sorted_ids[idx + 1]
            next_group = self.op_groups[next_gid]

            if len(next_group.ops[ExecutionType.GOLDEN]) == 0:
                idx += 1
                continue

            next_group.ops[ExecutionType.GOLDEN] = (
                group.ops[ExecutionType.GOLDEN] + next_group.ops[ExecutionType.GOLDEN]
            )
            del self.op_groups[gid]
            sorted_ids.pop(idx)
