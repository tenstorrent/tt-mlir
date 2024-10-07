# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Union, Tuple, Callable
from ttmlir.ir import *
from ttmlir.dialects import ttir, tt, ttkernel, func
from .ttir.ttir_op_test_base import TTIROpTestBase
from .tensor_builder import TensorBuilder


class TTIRBuilder:
    def __init__(self, ctx: Context, location: Location):
        self.ctx = ctx
        self.loc = location
        self.module = Module.create(self.loc)
        # tt.register_dialect(self.ctx)
        # ttir.register_dialect(self.ctx)
        # ttkernel.register_dialect(self.ctx)

        self.tensor_builder = TensorBuilder(self.ctx, self.loc)

    def emit_mlir_function(
        self,
        ttir_op: TTIROpTestBase,
        input_tensor_shapes: Optional[List[Union[Tuple, List]]],
        output_tensor_shapes: Optional[List[Union[Tuple, List]]],
    ) -> func.FuncOp:
        """Creates a function encapsulating the given ttir operation."""
        assert (
            len(input_tensor_shapes) == ttir_op.num_inputs
            and len(output_tensor_shapes) == ttir_op.num_outputs
        ), (
            f"{ttir_op.name} op expects {ttir_op.num_inputs} input and "
            f"{ttir_op.num_outputs} output values"
        )

        input_types = [
            self.tensor_builder.ranked_tensor(shape) for shape in input_tensor_shapes
        ]
        output_types = [
            self.tensor_builder.ranked_tensor(shape) for shape in output_tensor_shapes
        ]

        with self.ctx, self.loc:
            # Function named after the op and block within it.
            func_op = func.FuncOp(ttir_op.name, (input_types, output_types))
            entry_block = func_op.add_entry_block()

            with InsertionPoint(entry_block):
                # Tie block args to op inputs and allocate empty tensors for
                # op to store outputs.
                inputs = entry_block.arguments
                outputs = self.tensor_builder.empty_tensors(
                    [output.shape for output in output_types]
                )

                # Build the op itself.
                op = ttir_op.build(inputs, outputs)

                return_values = self.__coerce_results(op)

                # Return the output tensors from the function.
                func.ReturnOp(return_values)

            return func_op

    def emit_mlir_module(
        self,
        ttir_op: TTIROpTestBase,
        input_tensor_shapes: Optional[List[Union[Tuple, List]]],
        output_tensor_shapes: Optional[List[Union[Tuple, List]]],
    ) -> Module:
        """Wraps mlir function in a module."""
        with InsertionPoint(self.module.body):
            self.emit_mlir_function(ttir_op, input_tensor_shapes, output_tensor_shapes)

        return self.module

    @staticmethod
    def apply(pipeline: Callable, module: Module) -> Module:
        # Apply pipeline in-place on module and return the module for convencience.
        pipeline(module)
        return module

    @staticmethod
    def ttir_to_ttmetal_backend_pipeline(
        module: Module, dump_to_file: bool = True, file_name: str = "test.mlir"
    ) -> Module:
        from ttmlir.passes import ttir_to_ttmetal_backend_pipeline

        ttir_to_ttmetal_backend_pipeline(module)

        if dump_to_file:
            with open(file_name, "w") as f:
                f.write(str(module))

        return module

    @staticmethod
    def ttmetal_to_flatbuffer_file(
        module: Module, file_name: str = "ttrt.ttm", golden_info: str = None
    ) -> None:
        from ttmlir.passes import ttmetal_to_flatbuffer_file

        ttmetal_to_flatbuffer_file(module, file_name)

    def __coerce_results(
        self,
        return_values: Optional[Union[Tuple, Value, OpView, Operation]] = None,
    ) -> List:
        if return_values is None:
            return_values = []
        elif isinstance(return_values, tuple):
            return_values = list(return_values)
        elif isinstance(return_values, Value):
            # Returning a single value is fine, coerce it into a list.
            return_values = [return_values]
        elif isinstance(return_values, OpView):
            # Returning a single operation is fine, coerce its results a list.
            return_values = return_values.operation.results
        elif isinstance(return_values, Operation):
            # Returning a single operation is fine, coerce its results a list.
            return_values = return_values.results
        else:
            return_values = list(return_values)

        return return_values
