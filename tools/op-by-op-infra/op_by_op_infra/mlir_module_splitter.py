# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List

from ttmlir.dialects import func, _ttcore_ops_gen

from .utils import ModuleWrapper, OpWrapper, convert_to_module_wrapper


# Ops that are skipped when testing in op-by-op mode, because they shouldn't be executed individually.
SKIPPED_OPS = {
    "ttnn.get_device",  # device acquisition
    "ttnn.deallocate",  # memory cleanup
    "ttnn.paged_update_cache",  # cache management
    "stablehlo.constant",  # wrapped together with ops that use their results
}


class MLIRModuleSplitter:
    """
    Splits MLIR module into constituent ops for sequential execution.

    Module is expected to have a `@main` func and possible other funcs.
    Processing starts with `main` one op at a time, handling special cases:
    - Nested function calls (`func.call`, `ttcore.load_cached`)
    - Composite ops with decomposition functions (`stablehlo.composite`)
    - Skipped ops that can't be executed individually (see SKIPPED_OPS)

    Produces a sequential list of ops that mimics op-by-op execution of the original module.
    """

    # ----- Public methods -----

    def __init__(self) -> None:
        """Constructor."""
        # Original module passed to `split`.
        self._module = None
        # Container for sub operations of original module.
        self._sub_ops: List[OpWrapper] = []
        # Container for sub modules of original module.
        self._sub_modules: List[ModuleWrapper] = []
        # Maps function names to the functions themselves, for easier retrieval.
        self._func_map = {}
        # Name of the model this module originated from.
        self._origin_model = ""

    @convert_to_module_wrapper
    def split(
        self, module: ModuleWrapper, origin_model: str = ""
    ) -> List[ModuleWrapper]:
        """
        Splits `module` and returns list of constituent ops each wrapped in a MLIR
        module (i.e. returns list of sub modules).
        """
        # Each time `split` is called, prepare for new run by forgetting results of
        # previous run and storing new module to work on.
        self._reset(module, origin_model)
        # Run the splitting algorithm on stored module.
        return self._split()

    # -- Convenience read-only properties for easy access --

    @property
    def module(self) -> ModuleWrapper:
        """Returns the original MLIR module passed to the splitter."""
        return self._module

    @property
    def sub_ops(self) -> List[OpWrapper]:
        """
        Returns list of constituent ops.

        Take note that order of ops in this list **is** the order in which graph should
        be run in op by op fashion, regardless of SSA values of ops, which stem from
        original graph, and which no longer hold due to possible nested function calls.
        """
        return self._sub_ops

    @property
    def sub_modules(self) -> List[ModuleWrapper]:
        """Returns list of constituent ops each wrapped in a MLIR module."""
        self._generate_sub_modules()
        return self._sub_modules

    # ----- Private methods -----

    def _reset(self, module: ModuleWrapper, origin_model: str = "") -> None:
        """
        Resets internal state, gets ready for a new run.

        NOTE all references to generated sub ops and modules will be lost after this.
        """
        self._module = module
        self._sub_ops = []
        self._sub_modules = []
        self._func_map = {}
        self._origin_model = origin_model

    def _split(self) -> List[OpWrapper]:
        """Splits the original module into constituent operations."""
        self._build_func_map()
        self._process_func_op(self._main_func)
        return self._sub_ops

    def _build_func_map(self) -> None:
        """
        Builds a mapping of function names to their corresponding func.func operations.
        """
        funcs_list = self._module.operations

        if any(isinstance(op, _ttcore_ops_gen.CPUModuleOp) for op in funcs_list):
            raise ValueError(
                "MLIRModuleSplitter does not support modules containing CPU-hoisted ops."
            )

        # Extract operations from DeviceModuleOp if present.
        if isinstance(funcs_list[0], _ttcore_ops_gen.DeviceModuleOp):
            funcs_list = (
                funcs_list[0]
                .bodyRegion.blocks[0]
                .operations[0]  # builtin.module
                .regions[0]
                .blocks[0]
                .operations
            )

        self._func_map = {func_op.name.value: func_op for func_op in funcs_list}

    @property
    def _main_func(self) -> func.FuncOp:
        assert "main" in self._func_map, f"MLIR module expected to have a `main` func."
        return self._func_map["main"]

    def _generate_sub_modules(self) -> None:
        """
        From sub ops of original module creates and stores list of constituent ops each
        wrapped in a MLIR module.
        """
        self._sub_modules = [op.as_module() for op in self._sub_ops]

    def _process_func_op(self, func_op: func.FuncOp):
        """Processes a single func.func operation and its operations in SSA order."""
        assert (
            len(func_op.regions) == 1
        ), f"Expected func {func_op.name} to have only one region"

        for block in func_op.regions[0].blocks:
            for op in block.operations:
                # `return` marks the end of the function.
                if op.name == "func.return":
                    break

                # Skip ops that can't be executed individually.
                if op.name in SKIPPED_OPS:
                    continue

                # Handle nested call.
                if op.name == "func.call":
                    self._process_call_op(op)
                # Handle const eval function loaded via ttcore.load_cached.
                elif op.name == "ttcore.load_cached":
                    self._process_load_cached_op(op)
                # Wrap raw op and store it. Ops will be turned to modules on demand.
                else:
                    # Handle stablehlo.composite ops that need to have their decomposition functions in the same module.
                    op_str = str(op)
                    if "stablehlo.composite" in op_str:
                        decomposition_func = self._extract_decomposition_func(op_str)
                        op_wrapper = self._module.wrap_op(
                            op, decomposition_func, self._origin_model
                        )
                    else:
                        op_wrapper = self._module.wrap_op(op, None, self._origin_model)

                    self._sub_ops.append(op_wrapper)

    def _process_call_op(self, call_op: func.CallOp) -> None:
        """Processes a func.call operation by processing the function it is calling."""
        callee = str(call_op.callee).replace("@", "")
        assert callee in self._func_map, f"Function {callee} not found in the module."
        self._process_func_op(self._func_map[callee])

    def _process_load_cached_op(self, load_cached_op) -> None:
        """Processes a ttcore.load_cached operation by processing the const eval function it references."""
        # Extract the function name from the operation string.
        # Format: ttcore.load_cached(@function_name, [...])
        op_str = str(load_cached_op)
        func_start = op_str.find("@")
        if func_start == -1:
            raise ValueError(
                f"Could not find function reference in ttcore.load_cached operation: {op_str}"
            )

        func_start += 1  # Skip the '@'
        func_end = op_str.find(",", func_start)
        if func_end == -1:
            func_end = op_str.find(")", func_start)

        callee = op_str[func_start:func_end].strip()
        assert callee in self._func_map, f"Function {callee} not found in the module."
        self._process_func_op(self._func_map[callee])

    def _extract_decomposition_func(self, op_str: str) -> func.FuncOp:
        decomposition_start = op_str.find("decomposition = @")
        decomposition_start += len("decomposition = @")
        decomposition_end = op_str.find("}", decomposition_start)
        callee = op_str[decomposition_start:decomposition_end].strip()
        assert callee in self._func_map, f"Function {callee} not found in the module."
        return self._func_map[callee]
