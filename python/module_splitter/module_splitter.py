# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from mlir.dialects import func
from mlir.ir import *

from .utils import OpWrapper


class ModuleSplitter(ABC):
    """
    Abstract base class used to split a MLIR module into constituent ops.

    Module is expected to consist of one `@main` func and possible other funcs which are
    either called from main or call each other.

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

    Methods
    -------
    get_module -> Module:
        Returns the original MLIR module passed to the splitter upon creation.

    get_sub_ops -> List[OpWrapper]
        Returns list of constituent ops.

    get_sub_modules -> List[Module]
        Returns list of constituent ops each wrapped in a MLIR module.
    """

    # ----- Public methods -----

    def get_module(self) -> Module:
        """Returns the original MLIR module passed to the splitter upon creation."""
        return self.__module

    def get_sub_ops(self) -> List[OpWrapper]:
        """
        Returns list of constituent ops.

        Take note that order of ops in this list **is** the order in which graph should
        be run in op by op fashion, regardless of SSA values of ops, which stem from
        original graph, and which no longer hold due to possible nested function calls.
        """
        return self.__sub_ops

    def get_sub_modules(self) -> List[Module]:
        """Returns list of constituent ops each wrapped in a MLIR module."""
        return [self._parse_module_str(op.as_module_str()) for op in self.__sub_ops]

    # ----- Protected methods -----

    def __init__(self, module: Module) -> None:
        self.__module: Module = module
        self.__sub_ops: List[OpWrapper] = []
        # Maps function names to the functions themselves, for easier retrieval.
        self.__func_map = {}

        self.__split()

    @staticmethod
    @abstractmethod
    def _parse_module_str(module_str: str) -> Module:
        """
        Within a temporary context registers necessary dialects and parses `module_str`
        returning MLIR Module instance.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    # ----- Private methods -----

    def __split(self) -> None:
        """Splits the original module into constituent operations."""
        self.__build_func_map()
        self.__process_func_op(self.__main_func)

    def __build_func_map(self):
        """
        Builds a map of function names to their corresponding func.func operations.
        """
        self.__func_map = {
            func_op.name.value: func_op for func_op in self.__module.body.operations
        }

    @property
    def __main_func(self) -> func.FuncOp:
        assert "main" in self.__func_map, f"MLIR module expected to have a `main` func."
        return self.__func_map["main"]

    def __process_func_op(self, func_op: func.FuncOp):
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
                    self.__process_call_op(op)
                else:
                    # Add the operation to the list
                    self.__sub_ops.append(OpWrapper(op))

    def __process_call_op(self, call_op: func.CallOp):
        """Processes a func.call operation by processing the function it is calling."""
        callee = str(call_op.callee).replace("@", "")
        assert callee in self.__func_map, f"Function {callee} not found in the module."
        self.__process_func_op(self.__func_map[callee])
