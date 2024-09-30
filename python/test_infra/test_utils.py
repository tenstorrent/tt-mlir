# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Callable, Dict, Tuple

import torch
from ttmlir.dialects import func
from ttmlir.ir import *
from ttmlir.passes import (
    golden_info_to_flatbuffer_file,
    ttir_to_ttmetal_backend_pipeline,
    ttmetal_to_flatbuffer_file,
)

from .ttir_builder import Golden, Operand, Shape, TTIRBuilder


# ----- Static helpers used in this file only -----


def _dump_module(module: Module) -> None:
    """Just prints the module to console."""
    print(module)


def _dump_goldens(
    goldens: Dict[Operand, Golden], output_file_name: str = "golden_fb.ttmg"
):
    """
    Flattens golden tensors and dumps them to a flatbuffer file.

    Wrapper around `golden_info_to_flatbuffer_file` pybound pass.
    """
    operand_names = []
    flattened_tensor_data = []
    tensor_shapes = []
    # random_tensor_seeds unused currently, but can be used for randomly
    # generated tensors in order not to store entire tensor.
    random_tensor_seeds = []

    for operand, golden in goldens.items():
        operand_names.append(TTIRBuilder._get_name(operand))
        flattened_tensor_data.append(torch.flatten(golden.tensor).tolist())
        tensor_shapes.append(list(golden.tensor.shape))
        random_tensor_seeds.append(golden.seed)

    golden_info_to_flatbuffer_file(
        operand_names, flattened_tensor_data, tensor_shapes, output_file_name
    )

    print(f"Flatbuffer file for GoldenInfo {output_file_name} successfully generated.")


def _run_ttmlir_translate(
    input_file_name: str, output_file_name: str = "ttmetal_fb.ttm"
):
    """
    Util function running `ttmlir-translate` tool on a file containing dumped TTMetal
    module. It produces flatbuffer file `output_file_name`.
    """
    import subprocess

    res = subprocess.run(
        " ".join(
            [
                "build/bin/ttmlir-translate",
                "--ttmetal-to-flatbuffer",
                input_file_name,
                "-o",
                output_file_name,
            ]
        ),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert (
        res.returncode == 0
    ), f"Running ttmlir-translate failed with: {res.stdout.decode('utf-8')}"
    return res


# ----- Decorators for doing passes and compiling to flatbuffer -----


def compile_as_mlir_module(
    *inputs_shapes: Tuple[Shape],
    module_dump: bool = False,
    golden_dump: bool = True,
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
                        for i in inputs:
                            builder.generate_and_store_random_golden(i)

                        return test_fn(*inputs, builder=builder)

                print(
                    f"`{test_fn.__name__}` sucessfully transformed into a MLIR module."
                )

                if module_dump:
                    _dump_module(module)

                if golden_dump:
                    _dump_goldens(builder.goldens, f"golden_{test_fn.__name__}.ttmg")

                return module

        return wrapper

    return decorator


def ttir_to_ttmetal(
    dump_to_file: bool = True,
    output_file_name: str = "test.mlir",
    return_module: bool = False,
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

    return_module: bool
        Flag through which one chooses to return the generated module or name of the
        file in which module was dumped (i.e. `output_file_name`). Exists only to
        accommodate both `ttmetal_to_flatbuffer` and `translate_ttmetal_to_flatbuffer`.
    """

    def decorator(fn: Callable):
        def wrapper(*args, **kwargs):
            # First, call the decorated function to get the MLIR module.
            module = fn(*args, **kwargs)

            assert isinstance(module, Module), (
                f"Make sure this decorator is used on top of "
                f"`compile_as_mlir_module` decorator."
            )

            # Now, pass it through the TTIR to TTMetal pipeline. Module gets
            # modified in place.
            ttir_to_ttmetal_backend_pipeline(module)

            print("`ttir_to_ttmetal_backend_pipeline` passed successfully.")

            # Optionally dump to file.
            if dump_to_file:
                with open(output_file_name, "w") as f:
                    f.write(str(module))

            return module if return_module else output_file_name

        return wrapper

    return decorator


def ttmetal_to_flatbuffer(
    output_file_name: str = "ttmetal_fb.ttmg", golden_info: Dict[Operand, Golden] = None
):
    """
    NOTE NOT WORKING, DO NOT USE.

    Converts TTMetal module to flatbuffer and saves to file, meant to be used as a
    decorator on top of `ttir_to_ttmetal` decorator. Take note that `ttir_to_ttmetal`
    has to return module instead of file name if decorated with this decorator.

    Wrapper around `ttmetal_to_flatbuffer_file` pybound pass.

    TODO Optional golden info is passed to be embedded in flatbuffer as well.

    TODO Decorating a test function with this, i.e. calling
    `ttmetal_to_flatbuffer_file` will result in

    'LLVM ERROR: Building op `emitc.constant` but it isn't known in this MLIRContext:
    the dialect may not be loaded or this operation hasn't been added by the dialect.'

    To circumvent this, `ttmlir-translate` is run on file that
    `ttir_to_ttmetal_backend_pipeline` produces to generate TTMetal flatbuffer file,
    which this decorator was supposed to generate. Use `translate_ttmetal_to_flatbuffer`
    to achieve this, and make `ttir_to_ttmetal` return file name instead of module.
    """

    def decorator(test_fn: Callable):
        def wrapper(*args, **kwargs):
            # Get the TTMetal module by calling the wrapped function.
            module = test_fn(*args, **kwargs)

            assert isinstance(module, Module), (
                f"Make sure `ttir_to_ttmetal` which was decorated with this function "
                f"returns module, not file name."
            )

            # Convert to flatbuffer file.
            ttmetal_to_flatbuffer_file(module, output_file_name)

            print("`ttmetal_to_flatbuffer_file` passed successfully.")

        return wrapper

    return decorator


def translate_ttmetal_to_flatbuffer(output_file_name: str = "ttmetal_fb.ttm"):
    """
    NOTE Substitutes `ttmetal_to_flatbuffer` decorator.

    By running `ttmlir-translate` on input file, it produces TTMetal flatbuffer file
    `output_file_name`, meant to be used as a decorator on top of `ttir_to_ttmetal`
    decorator. Take note that `ttir_to_ttmetal` has to return file name instead of
    module if decorated with this decorator.

    Wrapper around `ttmlir-translate` call.

    Example
    -------

    ```python
    @translate_ttmetal_to_flatbuffer(output_file_name="ttmetal_fb_test_add.ttm")
    @ttir_to_ttmetal(dump_to_file=True, output_file_name="test_add.mlir", return_module=False)
    @compile_as_mlir_module((32, 32), (32, 32))
    def test_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
        # CHECK: %0 = tensor.empty() : tensor<32x32xf32>
        # CHECK: %1 = "ttir.add"(%arg0, %arg1, %0)
        # CHECK: return %1 : tensor<32x32xf32>

        return builder.add(in0, in1)
        ```
    """

    def decorator(fn: Callable):
        def wrapper(*args, **kwargs):
            input_file_name = fn(*args, **kwargs)

            assert isinstance(input_file_name, str) and os.path.isfile(
                input_file_name
            ), (
                f"Make sure `ttir_to_ttmetal` which was decorated with this function "
                f"returns file name, not module."
            )

            res = _run_ttmlir_translate(input_file_name, output_file_name)

            print(
                f"Flatbuffer file for TTMetalBinary {output_file_name} successfully generated."
            )

            return res.returncode

        return wrapper

    return decorator
