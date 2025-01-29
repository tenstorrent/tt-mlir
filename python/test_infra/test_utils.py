# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import inspect
import torch
from typing import Callable, List, Optional

from ttmlir.dialects import func
from ttmlir.ir import *
from ttmlir.passes import (
    ttir_to_ttnn_backend_pipeline,
    ttnn_to_flatbuffer_file,
    ttir_to_ttmetal_backend_pipeline,
    ttmetal_to_flatbuffer_file,
    MLIRModuleLogger,
    ModuleLog,
)

from .ttir_builder import Golden, Operand, Shape, TTIRBuilder, DataType

TT_MLIR_HOME = os.environ.get("TT_MLIR_HOME", "")


# ----- Static helpers used in this file only -----


def _dump_module(module: Module) -> None:
    """Just prints the module to console."""
    print(module)


#  ----- General Purpose Helpers - Could Be Used In Other Files -----


def compile_as_mlir_module(
    test_fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[torch.dtype]] = None,
    module_dump: bool = False,
):
    """
    Define a MLIR module specified as a python function.

    It will wrap `test_fn` in a MLIR FuncOp and then wrap that in a MLIR
    module, and finally tie arguments of that FuncOp to test function inputs. It will
    also pass a `TTIRBuilder` object as the last argument of test function.

    Arguments
    ---------
    test_fn : Callable
        Python function to be converted to MLIR

    inputs_shapes: List[Shape]
        Shapes of the respective ranked tensor inputs of the test function.

    module_dump: bool
        Set to True to print out generated MLIR module.

    golden_dump: bool
        Set to True to dump golden info to flatbuffer file.


    Returns
    -------
    MLIR module containing MLIR op graph defined by `test_fn`

    Example
    -------

    ```python
        def test_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
            return builder.add(in0, in1)

        compile_as_mlir_module(test_add, ((32, 32), (32, 32)))
    ```

    which returns

    ```
        #any = #tt.operand_constraint<...>
        module {
            func.func @test_add(
                %arg0: tensor<32x32xf32>,
                %arg1: tensor<32x32xf32>
            ) -> tensor<32x32xf32> {
                %0 = tensor.empty() : tensor<32x32xf32>
                %1 = "ttir.add"(%arg0, %arg1, %0) ...
                return %1 : tensor<32x32xf32>
            }
        }
    ```

    Check out:
    https://github.com/llvm/llvm-project/blob/main/mlir/test/python/dialects/tensor.py
    """

    ctx = Context()

    # Grab the location of the test function in python for later debugging
    try:
        fname = inspect.getfile(test_fn)
        line_no = inspect.getsourcelines(test_fn)[1]
        loc = Location.file(fname, line_no, 0, ctx)
    except (OSError, TypeError):
        loc = Location.unknown(ctx)

    # Instantiate builder which is passed as the last argument to
    # `test_fn` so the user can use it to build ops.
    builder = TTIRBuilder(ctx, loc)

    # Default to all f32s
    if inputs_types is None:
        inputs_types = [torch.float32] * len(inputs_shapes)

    assert inputs_types is not None and len(inputs_shapes) == len(inputs_types)

    with ctx, loc:
        test_fn_input_types = [
            builder.ranked_tensor_type(shape, builder.get_type_from_torch_dtype(dtype))
            for (shape, dtype) in zip(inputs_shapes, inputs_types)
        ]

        # Wrap everything in a mlir module.
        module = Module.create()

        with InsertionPoint(module.body):
            # Wrap everything in a mlir function.
            @func.func(*test_fn_input_types, name=test_fn.__name__)
            def decorated_func(*inputs):
                # Randomly generate golden tensors for function inputs.
                for index, (operand, dtype) in enumerate(zip(inputs, inputs_types)):
                    builder.generate_input_golden(operand, dtype, index)

                return test_fn(*inputs, builder=builder)

        print(f"`{test_fn.__name__}` sucessfully transformed into a MLIR module.")

        if module_dump:
            _dump_module(module)

        return module, builder


def ttir_to_ttnn(
    module,
    dump_to_file: bool = True,
    output_file_name: str = "test.mlir",
    system_desc_path: Optional[str] = None,
):
    """
    Converts TTIR module to TTNN module and optionally dumps to file.

    Wrapper around `ttir_to_ttnn_backend_pipeline` pybound pass.

    Arguments
    ---------
    dump_to_file: bool
        Flag which indicates that generated TTNN module will be dumped to file.

    output_file_name: str
        Name of the output file.

    Returns
    -------
    MLIR module containing MLIR op graph defined by `module` and instance of TTIRBuilder.
    """

    # Default to the `SYSTEM_DESC_PATH` envvar
    if system_desc_path is None:
        system_desc_path = os.getenv("SYSTEM_DESC_PATH", "")

    # Now, pass it through the TTIR to TTNN pipeline. Module gets
    # modified in place.
    ttir_to_ttnn_backend_pipeline(module, f"system-desc-path={system_desc_path}")

    print("`ttir_to_ttnn_backend_pipeline` passed successfully.")

    # Optionally dump to file.
    if dump_to_file:
        with open(output_file_name, "w") as f:
            f.write(str(module))

    return module


def ttir_to_ttmetal(
    module,
    dump_to_file: bool = True,
    output_file_name: str = "test.mlir",
    system_desc_path: Optional[str] = None,
):
    """
    Converts TTIR module `module` to TTMetal module and optionally dumps to file.

    Wrapper around `ttir_to_ttmetal_backend_pipeline` pybound pass.

    Arguments
    ---------
    module: ???
        TTIR module to convert to TTMetal module

    dump_to_file: bool
        Flag which indicates that generated TTMetal module will be dumped to file.

    output_file_name: str
        Name of the output file.

    Returns
    -------
    MLIR module containing MLIR op graph defined by `module` and instance of TTIRBuilder.
    """

    # Default to the `SYSTEM_DESC_PATH` envvar
    if system_desc_path is None:
        system_desc_path = os.getenv("SYSTEM_DESC_PATH", "")

    # Now, pass it through the TTIR to TTMetal pipeline. Module gets
    # modified in place.
    ttir_to_ttmetal_backend_pipeline(module, f"system-desc-path={system_desc_path}")

    print("`ttir_to_ttmetal_backend_pipeline` passed successfully.")

    # Optionally dump to file.
    if dump_to_file:
        with open(output_file_name, "w") as f:
            f.write(str(module))

    return module


def ttnn_to_flatbuffer(
    module, builder, output_file_name: str = "ttnn_fb.ttnn", module_log=None
):
    """
    Converts TTNN module to flatbuffer and saves to file. Wrapper around
    `ttnn_to_flatbuffer_file` pybound pass.
    """

    # Convert to flatbuffer file.
    if module_log:
        ttnn_to_flatbuffer_file(
            module, output_file_name, builder.get_golden_map(), module_log
        )
    else:
        ttnn_to_flatbuffer_file(module, output_file_name, builder.get_golden_map())

    print("`ttnn_to_flatbuffer_file` passed successfully.")


def ttmetal_to_flatbuffer(
    module,
    builder,
    output_file_name: str = "ttmetal_fb.ttm",
):
    """
    Converts TTMetal module to flatbuffer and saves to file. Wrapper around
    `ttmetal_to_flatbuffer_file` pybound pass.
    """

    # Convert to flatbuffer file.
    ttmetal_to_flatbuffer_file(module, output_file_name, builder.get_golden_map())

    print("`ttmetal_to_flatbuffer_file` passed successfully.")


# ----- Decorators for doing passes and compiling to flatbuffer -----


def compile_to_flatbuffer(
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[torch.dtype]] = None,
    test_name: Optional[str] = None,
    targets: List[str] = ["ttmetal", "ttnn"],
    module_dump: bool = False,
):
    """
    Decorator to run an e2e Python -> Flatbuffer test using the decorated
    function, using the TTNN and/or TTMetal backends.

    This decorator is mainly a wrapper around the following functions, with
    each next function called on the output of the last:

    1. `compile_as_mlir_module`
    2. `ttir_to_tt{nn,metal}`
    3. `tt{nn,metal}_to_flatbuffer`

    The choice of TTNN, TTMetal, or both is controlled by membership of those
    strings in the `targets` parameter.

    Arguments
    ---------

    inputs_shapes: List[Shape]
        Shapes of the respective ranked tensor inputs of the test function.

    test_name: Optional[str]
        The string to be used as the base name for dumped files throughout the
        process. If `None` is provided, then the `__name__` of the decorated
        function will be used.

    targets: List[str]
        A list that can only contain the following strings: 'ttnn' or
        'ttmetal'. Inclusion in this list will signal this decorator to execute
        their respective backend paths. Either, neither, or both are valid inputs.

    module_dump: bool
        Set to True to print out generated MLIR module.

    Example
    -------

    ```python
        @compile_and_convert(((32, 32), (32, 32)), test_name="test_add")
        def test_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
            return builder.add(in0, in1)

        test_add() # NOTE: called without arguments
    ```
    """

    def decorator(test_fn: Callable):

        # Snoop the name of `test_fn` if no override to the test name is provided
        if test_name is None:
            test_base = test_fn.__name__
        else:
            test_base = test_name

        def wrapper():

            # NOTE: since `ttir_to_tt{nn,metal} modifies the module in place,
            # `compile_as_mlir_module` needs to be run twice in the case that
            # both targets are chosen

            if "ttmetal" in targets:
                module, builder = compile_as_mlir_module(
                    test_fn, inputs_shapes, inputs_types
                )
                module = ttir_to_ttmetal(module, builder, test_base + ".mlir")
                ttmetal_to_flatbuffer(module, builder, test_base + ".ttm")

            if "ttnn" in targets:
                module, builder = compile_as_mlir_module(
                    test_fn, inputs_shapes, inputs_types
                )
                module_logger = MLIRModuleLogger()
                module_log = ModuleLog()
                module_logger.attach_context(module.context, module_log)
                module = ttir_to_ttnn(module, builder, test_base + ".mlir")
                ttnn_to_flatbuffer(module, builder, test_base + ".ttnn", module_log)

        return wrapper

    return decorator
