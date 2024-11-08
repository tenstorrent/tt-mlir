# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Callable, Dict, Tuple, List, Optional

import torch
from ttmlir.dialects import func
from ttmlir.ir import *
from ttmlir.passes import (
    ttir_to_ttnn_backend_pipeline,
    ttnn_to_flatbuffer_file,
    ttir_to_ttmetal_backend_pipeline,
    ttmetal_to_flatbuffer_file,
)

from .ttir_builder import Golden, Operand, Shape, TTIRBuilder

TT_MLIR_HOME = os.environ.get("TT_MLIR_HOME", "")


# ----- Static helpers used in this file only -----


def _dump_module(module: Module) -> None:
    """Just prints the module to console."""
    print(module)


# ----- Decorators for doing passes and compiling to flatbuffer -----


def compile_as_mlir_module(
    *inputs_shapes: Tuple[Shape],
    module_dump: bool = False,
):
    """
    Decorator to define a MLIR module specified as a python function.

    It will wrap decorated test function in a MLIR FuncOp and then wrap that in a MLIR
    module, and finally tie arguments of that FuncOp to test function inputs. It will
    also pass a `TTIRBuilder` object as the last argument of test function.

    Arguments
    ---------
    inputs_shapes: Tuple[Shape]
        Shapes of the respective ranked tensor inputs of the test function.

    module_dump: bool
        Set to True to print out generated MLIR module.

    golden_dump: bool
        Set to True to dump golden info to flatbuffer file.


    Returns
    -------
    MLIR module containing MLIR op graph defined by decorated test function.

    Example
    -------

    ```python
        @compile_as_mlir_module((32, 32), (32, 32))
        def test_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
            return builder.add(in0, in1)


        test_add() # NOTE Called without arguments.
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

    def decorator(test_fn: Callable):
        # test_fn should be called with no args.
        def wrapper():
            ctx = Context()
            loc = Location.unknown(ctx)
            # Instantiate builder which is passed as the last argument to
            # `test_fn` so the user can use it to build ops.
            builder = TTIRBuilder(ctx, loc)

            with ctx, loc:
                test_fn_input_types = [
                    builder.ranked_tensor_type(input_shape)
                    for input_shape in inputs_shapes
                ]

                # Wrap everything in a mlir module.
                module = Module.create()

                with InsertionPoint(module.body):
                    # Wrap everything in a mlir function.
                    @func.func(*test_fn_input_types, name=test_fn.__name__)
                    def decorated_func(*inputs):
                        # Randomly generate golden tensors for function inputs.
                        for index, i in enumerate(inputs):
                            builder.generate_input_golden(i, index)

                        return test_fn(*inputs, builder=builder)

                print(
                    f"`{test_fn.__name__}` sucessfully transformed into a MLIR module."
                )

                if module_dump:
                    _dump_module(module)

                return module, builder

        return wrapper

    return decorator


def ttir_to_ttnn(
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
    MLIR module containing MLIR op graph defined by decorated test function and instance of TTIRBuilder.
    """

    # Default to the `SYSTEM_DESC_PATH` envvar
    if system_desc_path is None:
        system_desc_path = os.getenv("SYSTEM_DESC_PATH", "")

    def decorator(fn: Callable):
        def wrapper(*args, **kwargs):
            # First, call the decorated function to get the MLIR module and builder instance
            module, builder = fn(*args, **kwargs)

            # Now, pass it through the TTIR to TTNN pipeline. Module gets
            # modified in place.
            ttir_to_ttnn_backend_pipeline(
                module, f"system-desc-path={system_desc_path}"
            )

            print("`ttir_to_ttnn_backend_pipeline` passed successfully.")

            # Optionally dump to file.
            if dump_to_file:
                with open(output_file_name, "w") as f:
                    f.write(str(module))

            return module, builder

        return wrapper

    return decorator


def ttir_to_ttmetal(
    dump_to_file: bool = True,
    output_file_name: str = "test.mlir",
    system_desc_path: Optional[str] = None,
):
    """
    Converts TTIR module to TTMetal module and optionally dumps to file.

    Wrapper around `ttir_to_ttmetal_backend_pipeline` pybound pass.

    Arguments
    ---------
    dump_to_file: bool
        Flag which indicates that generated TTMetal module will be dumped to file.

    output_file_name: str
        Name of the output file.

    Returns
    -------
    MLIR module containing MLIR op graph defined by decorated test function and instance of TTIRBuilder.
    """

    # Default to the `SYSTEM_DESC_PATH` envvar
    if system_desc_path is None:
        system_desc_path = os.getenv("SYSTEM_DESC_PATH", "")

    def decorator(fn: Callable):
        def wrapper(*args, **kwargs):
            # First, call the decorated function to get the MLIR module.
            module, builder = fn(*args, **kwargs)

            # Now, pass it through the TTIR to TTMetal pipeline. Module gets
            # modified in place.
            ttir_to_ttmetal_backend_pipeline(
                module, f"system-desc-path={system_desc_path}"
            )

            print("`ttir_to_ttmetal_backend_pipeline` passed successfully.")

            # Optionally dump to file.
            if dump_to_file:
                with open(output_file_name, "w") as f:
                    f.write(str(module))

            return module, builder

        return wrapper

    return decorator


def ttnn_to_flatbuffer(
    output_file_name: str = "ttnn_fb.ttnn",
):
    """
    Converts TTNN module to flatbuffer and saves to file, meant to be used as a
    decorator on top of `ttir_to_ttnn` decorator. Take note that `ttir_to_ttnn`
    has to return module instead of file name if decorated with this decorator.

    Wrapper around `ttnn_to_flatbuffer_file` pybound pass.
    """

    def decorator(test_fn: Callable):
        def wrapper(*args, **kwargs):
            # Get the TTNN module by calling the wrapped function.
            module, builder = test_fn(*args, **kwargs)

            # Convert to flatbuffer file.
            ttnn_to_flatbuffer_file(module, output_file_name, builder.get_golden_map())

            print("`ttnn_to_flatbuffer_file` passed successfully.")

        return wrapper

    return decorator


def ttmetal_to_flatbuffer(
    output_file_name: str = "ttmetal_fb.ttmg",
):
    """
    Converts TTMetal module to flatbuffer and saves to file, meant to be used as a
    decorator on top of `ttir_to_ttmetal` decorator. Take note that `ttir_to_ttmetal`
    has to return module instead of file name if decorated with this decorator.

    Wrapper around `ttmetal_to_flatbuffer_file` pybound pass.
    """

    def decorator(test_fn: Callable):
        def wrapper(*args, **kwargs):
            # Get the TTMetal module by calling the wrapped function.
            module, builder = test_fn(*args, **kwargs)

            # Convert to flatbuffer file.
            ttmetal_to_flatbuffer_file(
                module, output_file_name, builder.get_golden_map()
            )

            print("`ttmetal_to_flatbuffer_file` passed successfully.")

        return wrapper

    return decorator
