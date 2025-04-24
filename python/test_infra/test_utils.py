# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import inspect
import torch
from typing import Callable, List, Optional, Tuple, Union

from ttmlir.dialects import func
from ttmlir.ir import *
from ttmlir.passes import (
    tt_populate_argument_types,
    ttir_to_ttnn_backend_pipeline,
    ttnn_to_flatbuffer_file,
    ttir_to_ttmetal_backend_pipeline,
    ttmetal_to_flatbuffer_file,
    MLIRModuleLogger,
)

from .ttir_builder import Shape, TTIRBuilder, DataType, TypeInfo

TT_MLIR_HOME = os.environ.get("TT_MLIR_HOME", "")

# Default output to the current directory from where this module is being invoked
OUTPUT_PATH = ""

# ----- Static helpers used in this file only -----


def _dump_module(module: Module) -> None:
    """Just prints the module to console."""
    print(module)


#  ----- General Purpose Helpers - Could Be Used In Other Files -----


def set_output_path(path):
    global OUTPUT_PATH
    if not os.path.exists(path):
        raise ValueError(f"The provided path '{path}' is not a valid path.")
    OUTPUT_PATH = path


def get_ttnn_path(output_path, filename):
    ttnn_dir = os.path.join(output_path, "ttnn")
    if not os.path.exists(ttnn_dir):
        os.makedirs(ttnn_dir)
    return os.path.join(ttnn_dir, filename)


def get_ttmetal_path(output_path, filename):
    ttmetal_dir = os.path.join(output_path, "ttmetal")
    if not os.path.exists(ttmetal_dir):
        os.makedirs(ttmetal_dir)
    return os.path.join(ttmetal_dir, filename)


def compile_as_mlir_module(
    test_fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    mesh_shape: Optional[Tuple[int, int]] = None,
    module_dump: bool = False,
    base: Optional[str] = None,
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
                %0 = ttir.empty() : tensor<32x32xf32>
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

    # deliver mesh_shape to TTIRBuilder
    # TTIR itself does not require mesh_shape information; however, it is needed to generate the golden tensor.
    if mesh_shape is not None:
        builder.set_mesh_shape(mesh_shape)

    # Default to all f32s
    if inputs_types is None:
        inputs_types = [torch.float32] * len(inputs_shapes)

    assert inputs_types is not None and len(inputs_shapes) == len(inputs_types)
    with ctx, loc:
        test_fn_input_types = [
            builder.ranked_tensor_type(
                shape,
                builder.get_type_from_torch_dtype(
                    dtype if isinstance(dtype, torch.dtype) else dtype
                ),
            )
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

        base = test_fn.__name__ if base is None else base

        if module_dump:
            with open(base + "_ttir.mlir", "w") as f:
                f.write(str(module))
                _dump_module(module)

        return module, builder


def ttir_to_ttnn(
    module,
    dump_to_file: bool = True,
    output_path: str = "",
    output_file_name: str = "test.mlir",
    system_desc_path: Optional[str] = None,
    mesh_shape: Optional[Tuple[int, int]] = None,
    argument_types_string: Optional[str] = None,
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
    if argument_types_string:
        tt_populate_argument_types(module, argument_types_string)

    # Default to the `SYSTEM_DESC_PATH` envvar
    if system_desc_path is None:
        system_desc_path = os.getenv("SYSTEM_DESC_PATH", "")

    # Generate option string
    options = []
    if system_desc_path:
        options.append(f"system-desc-path={system_desc_path}")
    if mesh_shape and len(mesh_shape) == 2:
        options.append(f"mesh-shape={mesh_shape[0]},{mesh_shape[1]}")
    if argument_types_string:
        options.append("enable-const-eval=true")

    # Now, pass it through the TTIR to TTNN pipeline. Module gets
    # modified in place.
    ttir_to_ttnn_backend_pipeline(module, " ".join(options))

    print("`ttir_to_ttnn_backend_pipeline` passed successfully.")

    # Optionally dump to file.
    if dump_to_file:
        output_file_name = get_ttnn_path(output_path, output_file_name)
        with open(output_file_name, "w") as f:
            f.write(str(module))

    return module


def ttir_to_ttmetal(
    module,
    dump_to_file: bool = True,
    output_path: str = "",
    output_file_name: str = "test.mlir",
    system_desc_path: Optional[str] = None,
    mesh_shape: Optional[Tuple[int, int]] = None,
    argument_types_string: Optional[str] = None,
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
        output_file_name = get_ttmetal_path(output_path, output_file_name)
        with open(output_file_name, "w") as f:
            f.write(str(module))

    return module


def ttnn_to_flatbuffer(
    module,
    builder,
    output_path: str = "",
    output_file_name: str = "ttnn_fb.ttnn",
    module_log=None,
):
    """
    Converts TTNN module to flatbuffer and saves to file. Wrapper around
    `ttnn_to_flatbuffer_file` pybound pass.
    """

    # Convert to flatbuffer file.
    # Take the output_file_name and prefix with the ttnn directory
    output_file_name = get_ttnn_path(output_path, output_file_name)
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
    output_path: str = "",
    output_file_name: str = "ttmetal_fb.ttm",
    module_log=None,
):
    """
    Converts TTMetal module to flatbuffer and saves to file. Wrapper around
    `ttmetal_to_flatbuffer_file` pybound pass.
    """

    # Convert to flatbuffer file.
    # Take the output_file_name and prefix with ttm directory
    output_file_name = get_ttmetal_path(output_path, output_file_name)
    if module_log is not None:
        ttmetal_to_flatbuffer_file(
            module, output_file_name, builder.get_golden_map(), module_log
        )
    else:
        ttmetal_to_flatbuffer_file(module, output_file_name, builder.get_golden_map())

    print("`ttmetal_to_flatbuffer_file` passed successfully.")


def compile_to_flatbuffer(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    test_base: str = "test",
    output_root: str = ".",
    target: str = "ttnn",
    mesh_shape: Optional[Tuple[int, int]] = None,
    module_dump: bool = True,
    argument_types_string: Optional[str] = None,
):
    """
    Compiles a TTIRBuilder function `fn` to TTIR MLIR -> TT{Metal,NN} MLIR -> Flatbuffer

    This decorator is mainly a wrapper around the following functions, with
    each next function called on the output of the last:

    1. `compile_as_mlir_module`
    2. `ttir_to_tt{nn,metal}`
    3. `tt{nn,metal}_to_flatbuffer`

    The choice of TTNN vs. TTMetal is controlled by the `target` parameter

    Arguments
    ---------

    fn: Callable
        The TTIRBuilder function to compile. Must take `builder : TTIRBuilder` as a kwarg

    inputs_shapes: List[Shape]
        Shapes of the respective ranked tensor inputs of the test function.

    inputs_types: Optional[List[torch.dtype]]
        The dtypes to use for the inputs to `fn`. Note that if supplied,
        `len(inputs_shapes) == len(inputs_types)` must be true. Defaults to
        `None`

    test_base: str
        The string to be used as the base name for dumped files throughout the
        process. If `None` is provided, then the `__name__` of `fn` will be used.

    output_root: str
        The path to dump all generated arguments under. If this path doesn't
        exist, it will be created

    target: str
        Either `"ttnn"` or `"ttmetal"`. This controls which backend to use

    mesh_shape: Optional[Tuple[int, int]]
        A list that contains shape of the mesh to be applied on ttir to ttnn
        conversion path. Defaults to `None`

    module_dump: bool
        Set to `True` to print out generated TTIR MLIR module.

    """

    if inputs_types is not None:
        assert len(inputs_shapes) == len(inputs_types)

    from_ttir: Callable
    to_flatbuffer: Callable
    mlir_suffix: str

    if target == "ttnn":
        from_ttir = ttir_to_ttnn
        to_flatbuffer = ttnn_to_flatbuffer
        mlir_suffix = "_ttnn.mlir"
    else:
        from_ttir = ttir_to_ttmetal
        to_flatbuffer = ttmetal_to_flatbuffer
        mlir_suffix = "_ttm.mlir"

    # Compile model to TTIR MLIR
    module, builder = compile_as_mlir_module(
        fn, inputs_shapes, inputs_types, mesh_shape=mesh_shape
    )

    # Compile TTIR MLIR -> TT{Metal,NN} MLIR
    module = from_ttir(
        module,
        module_dump,
        output_root,
        test_base + mlir_suffix,
        system_desc_path=system_desc_path,
        mesh_shape=mesh_shape,
        argument_types_string=argument_types_string,
    )

    module_logger = MLIRModuleLogger()
    module_logger.attach_context(module.context)

    # Compile TT{Metal,NN} MLIR -> flatbuffer
    to_flatbuffer(
        module,
        builder,
        output_root,
        test_base + "." + target,
        module_log=module_logger.module_log,
    )
