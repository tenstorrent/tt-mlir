# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List, Optional

from ttmlir.dialects import func
from ttmlir.ir import Module, OpAttributeMap

from .utils import OpWrapper, parse_module_str


class MLIRModuleSplitter:
    """
    Used to split a MLIR module into constituent ops.

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
    """

    # ----- Public methods -----

    def __init__(self) -> None:
        """Constructor."""
        # Original module passed to `split` and its attributes.
        self.__module = None
        self.__module_attributes = None

        # Container for sub operations of original module.
        self.__sub_ops: List[OpWrapper] = []
        # Container for sub modules of original module.
        self.__sub_modules: List[Module] = []
        # Maps function names to the functions themselves, for easier retrieval.
        self.__func_map = {}

    def split(self, module: Module | str) -> List[Module]:
        """
        Splits `module` and returns list of constituent ops each wrapped in a MLIR
        module (i.e. returns list of sub modules).
        """
        # Store passed module and its attributes for easier access to them in methods.
        self.__module = parse_module_str(module) if isinstance(module, str) else module
        self.__module_attributes = self.__get_module_attributes()

        # Run the splitting algorithm on stored module.
        self.__split()

        return self.sub_modules

    # -- Convenience read-only properties for easy access --

    @property
    def module(self) -> Module:
        """Returns the original MLIR module passed to the splitter."""
        return self.__module

    @property
    def sub_ops(self) -> List[OpWrapper]:
        """
        Returns list of constituent ops.

        Take note that order of ops in this list **is** the order in which graph should
        be run in op by op fashion, regardless of SSA values of ops, which stem from
        original graph, and which no longer hold due to possible nested function calls.
        """
        return self.__sub_ops

    @property
    def sub_modules(self) -> List[Module]:
        """Returns list of constituent ops each wrapped in a MLIR module."""
        return self.__sub_modules

    # ----- Private methods -----

    def __get_module_attributes(self) -> Optional[OpAttributeMap]:
        """Returns module attributes if any, otherwise None."""
        return (
            self.__module.operation.attributes
            if len(self.__module.operation.attributes) > 0
            else None
        )

    def __split(self) -> None:
        """Splits the original module into constituent operations."""
        self.__build_func_map()
        self.__process_func_op(self.__main_func)
        self.__generate_sub_modules()

    def __build_func_map(self) -> None:
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

    def __generate_sub_modules(self) -> None:
        """
        From sub ops of original module creates and stores list of constituent ops each
        wrapped in a MLIR module.
        """
        self.__sub_modules = [
            parse_module_str(op.as_module_str()) for op in self.__sub_ops
        ]

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
                    # Store captured op along with attributes from original module.
                    self.__sub_ops.append(OpWrapper(op, self.__module_attributes))

    def __process_call_op(self, call_op: func.CallOp):
        """Processes a func.call operation by processing the function it is calling."""
        callee = str(call_op.callee).replace("@", "")
        assert callee in self.__func_map, f"Function {callee} not found in the module."
        self.__process_func_op(self.__func_map[callee])
