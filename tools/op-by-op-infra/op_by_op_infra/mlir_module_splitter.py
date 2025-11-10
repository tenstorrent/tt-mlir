# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List

from ttmlir.dialects import func, _ttcore_ops_gen

from .utils import ModuleWrapper, OpWrapper, convert_to_module_wrapper


class MLIRModuleSplitter:
    """
    Class used to split MLIR module into constituent ops.

    Module is expected to consist of one `@main` func and possible other funcs which are
    either called from main or call each other.

    Example:
    ```
    module {
        func.func @main(...) -> ... {
            %0 = dialect.op0(...) -> ...
            %1 = dialect.op1(...) -> ...
            %2 = call @func1(...) -> ...
            return %2
        }

        func.func @func1(...) -> ... {
            ...
            call @func2(...)
            ...
        }

        func.func @func2(...) -> ... {
            ...
        }
    }
    ```

    Processing starts with `main` one op at a time. If `call` to another mlir func is
    encountered (at any point, not just in `main`), processing jumps to that func and
    after that continues to the next op after `call`.

    Processing module this way produces a sequential list of ops which should be run
    in that particular order to mimic op-by-op execution of the original module/graph.
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

    @convert_to_module_wrapper
    def split(self, module: ModuleWrapper) -> List[ModuleWrapper]:
        """
        Splits `module` and returns list of constituent ops each wrapped in a MLIR
        module (i.e. returns list of sub modules).
        """
        # Each time `split` is called, prepare for new run by forgetting results of
        # previous run and storing new module to work on.
        self._reset(module)
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
        return self._sub_modules

    # ----- Private methods -----

    def _reset(self, module: ModuleWrapper) -> None:
        """
        Resets internal state, gets ready for a new run.

        NOTE all references to generated sub ops and modules will be lost after this.
        """
        self._module = module
        self._sub_ops = []
        self._sub_modules = []
        self._func_map = {}

    def _split(self) -> List[ModuleWrapper]:
        """Splits the original module into constituent operations."""
        self._build_func_map()
        self._process_func_op(self._main_func)
        self._generate_sub_modules()
        return self._sub_modules

    def _build_func_map(self) -> None:
        """
        Builds a mapping of function names to their corresponding func.func operations.
        """
        funcs_list = self._module.operations

        if any(isinstance(op, _ttcore_ops_gen.CPUModuleOp) for op in funcs_list):
            raise ValueError(
                "MLIRModuleSplitter does not support modules containing hoisted ops."
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

                # Handle nested call.
                if op.name == "func.call":
                    self._process_call_op(op)
                # Wrap raw op and store it. Ops will be turned to modules on demand.
                else:
                    # Handle stablehlo.composite ops that need to have their decomposition functions in the same module.
                    op_str = str(op)
                    if "stablehlo.composite" in op_str:
                        decomposition_func = self._extract_decomposition_func(op_str)
                        op_wrapper = self._module.wrap_op(op, decomposition_func)
                    else:
                        op_wrapper = self._module.wrap_op(op)

                    self._sub_ops.append(op_wrapper)

    def _process_call_op(self, call_op: func.CallOp) -> None:
        """Processes a func.call operation by processing the function it is calling."""
        callee = str(call_op.callee).replace("@", "")
        assert callee in self._func_map, f"Function {callee} not found in the module."
        self._process_func_op(self._func_map[callee])

    def _extract_decomposition_func(self, op_str: str) -> func.FuncOp:
        decomposition_start = op_str.find("decomposition = @")
        decomposition_start += len("decomposition = @")
        decomposition_end = op_str.find("}", decomposition_start)
        callee = op_str[decomposition_start:decomposition_end].strip()
        assert callee in self._func_map, f"Function {callee} not found in the module."
        return self._func_map[callee]
