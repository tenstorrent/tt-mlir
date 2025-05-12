# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import inspect
import torch
import pytest
from typing import Callable, List, Optional, Tuple, Union, Literal, Dict

from ttmlir.dialects import func
from ttmlir.ir import *
from ttmlir.passmanager import PassManager
from ttmlir.passes import (
    tt_populate_argument_types,
    ttir_to_ttnn_backend_pipeline,
    ttnn_to_flatbuffer_file,
    ttir_to_ttmetal_backend_pipeline,
    ttmetal_to_flatbuffer_file,
    MLIRModuleLogger,
)

from .apis import Shape, TTIRBuilder, TypeInfo

TT_MLIR_HOME = os.environ.get("TT_MLIR_HOME", "")

# Default output to the current directory from where this module is being invoked
OUTPUT_PATH = ""


class Marks:
    """
    Convenience class for adding pytest marks.

    Example
    -------
    >>> skip_test = Marks(pytest.mark.skip)
    >>> test_case | skip_test  # Marks test_case to be skipped
    """

    def __init__(self, *marks):
        """
        Initialize with pytest marks.

        Parameters
        ----------
        *marks : tuple
            Variable number of pytest.mark objects
        """
        self.marks = marks

    def __ror__(self, lhs):
        """
        Apply marks to a test parameter.

        Parameters
        ----------
        lhs : Any
            Test parameter to mark

        Returns
        -------
        pytest.param
            Marked test parameter
        """
        return pytest.param(lhs, marks=self.marks)


#  ----- General Purpose Helpers - Could Be Used In Other Files -----


def shape_str(shape):
    """
    Converts shape tuple to string.

    Parameters
    ----------
    shape : *Union[Tuple[int, ...], List[int]]*
        Shape to convert to string

    Returns
    -------
    str
        String representation of the shape (e.g., '32x32' for shape (32, 32))
    """
    return "x".join(map(str, shape))


def set_output_path(path):
    """
    Sets global output path.

    Parameters
    ----------
    path : str
        Path to set as output directory
    """
    global OUTPUT_PATH
    if not os.path.exists(path):
        raise ValueError(f"The provided path '{path}' is not a valid path.")
    OUTPUT_PATH = path


def get_target_path(output_path, filename, target):
    """
    Gets target file path.

    Parameters
    ----------
    output_path : str
        Base output directory
    filename : str
        Name of the file
    target : str
        Target subdirectory name

    Returns
    -------
    str
        Full path to the target file
    """
    target_dir = os.path.join(output_path, target)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return os.path.join(target_dir, filename)


def create_custom_pipeline_fn(
    pipeline: str, verify: bool = True, print_ir: Union[bool, str] = False
) -> Callable:
    """
    Creates a custom pipeline function.

    Parameters
    ----------
    pipeline : str
        Pipeline string specification
    verify : bool, optional
        Whether to enable verification (default: True)
    print_ir : *Union[bool, str]*, optional
        If True or a path string, enables IR printing (default: False)

    Returns
    -------
    Callable
        Function that runs the custom pipeline on a module
    """

    def wrapper(module, device_register_options):
        register_device = "ttcore-register-device"
        if device_register_options:
            register_device = f"{register_device}{{{device_register_options}}}"

        pipeline_str = f"builtin.module({','.join([register_device, pipeline])})"
        with module.context:
            pm = PassManager.parse(pipeline_str)
            pm.enable_verifier(verify)
            print("Running custom pipeline:", pm)
            if print_ir:
                print_ir_path = print_ir if isinstance(print_ir, str) else None
                pm.enable_ir_printing(tree_printing_dir_path=print_ir_path)
            pm.run(module.operation)

    return wrapper


def build_mlir_module(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    mesh_shape: Optional[Tuple[int, int]] = None,
    module_dump: bool = False,
    base: Optional[str] = None,
    output_root: str = ".",
):
    """
    Define a MLIR module specified as a python function.

    It will wrap `fn` in a MLIR FuncOp and then wrap that in a MLIR
    module, and finally tie arguments of that FuncOp to test function inputs. It will
    also pass a `TTIRBuilder` object as the last argument of test function.

    Parameters
    ----------
    fn : Callable
        Python function to be converted to MLIR

    inputs_shapes : *List[Shape]*
        Shapes of the respective ranked tensor inputs of the test function.

    inputs_types: *Optional[List[Union[torch.dtype, TypeInfo]]]*
        Data types of the input tensors

    mesh_shape: *Optional[Tuple[int, int]]*
        A list that contains shape of the mesh to be applied on ttir to ttnn
        conversion path.

    module_dump : bool
        Set to True to print out generated MLIR module.

    golden_dump : bool
        Set to True to dump golden info to flatbuffer file.

    base : *Optional[str]*
        Output file name

    output_root: str = ".",
        Output file path

    Returns
    -------
    Module
        MLIR module containing MLIR op graph defined by `fn`

    Example
    -------
    >>> def test_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
    ...     return builder.add(in0, in1)
    ...
    >>> build_mlir_module(test_add, ((32, 32), (32, 32)))

    This returns:

    .. code-block:: mlir

        #any = #ttcore.operand_constraint<...>
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

    Check out:
    https://github.com/llvm/llvm-project/blob/main/mlir/test/python/dialects/tensor.py
    """

    ctx = Context()

    # Grab the location of the test function in python for later debugging
    try:
        fname = inspect.getfile(fn)
        line_no = inspect.getsourcelines(fn)[1]
        loc = Location.file(fname, line_no, 0, ctx)
    except (OSError, TypeError):
        loc = Location.unknown(ctx)

    # Instantiate builder which is passed as the last argument to
    # `fn` so the user can use it to build ops.
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
        fn_input_types = [
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
            @func.func(*fn_input_types, name=fn.__name__)
            def decorated_func(*inputs):
                # Randomly generate golden tensors for function inputs.
                input_goldens = []
                for index, (operand, dtype) in enumerate(zip(inputs, inputs_types)):
                    input_goldens.append(
                        builder.generate_input_golden(operand, dtype, index).tensor
                    )
                result = fn(*inputs, builder=builder)
                output_ops = result if hasattr(result, "__iter__") else (result,)
                output_goldens = [builder._get_golden_tensor(op) for op in output_ops]
                builder.set_graph_input_output(input_goldens, output_goldens)
                return result

        print(f"`{fn.__name__}` sucessfully transformed into a MLIR module.")

        base = fn.__name__ if base is None else base

        filename = get_target_path(output_root, base + "_ttir.mlir", "ttir")

        if module_dump:
            with open(filename, "w") as f:
                f.write(str(module))
                print(module)

        return module, builder


def run_pipeline(
    module,
    pipeline_fn: Callable = ttir_to_ttnn_backend_pipeline,
    pipeline_options: Optional[List[str]] = None,
    dump_to_file: bool = True,
    output_file_name: str = "test.mlir",
    system_desc_path: Optional[str] = None,
    mesh_shape: Optional[Tuple[int, int]] = None,
    argument_types_string: Optional[str] = None,
):
    """
    Runs a pipeline over a module and optionally dumps to file.

    Arguments
    ---------
    module :
        TTIR module on which pipeline is run

    pipeline_fn : Callable
        Pipeline function to run. pipeline_fn(module, options)

    pipeline_options : *Optional[List[str]]*
        Pipeline options to be added to the pass

    dump_to_file : bool
        Flag which indicates that generated TTNN module will be dumped to file.

    output_file_name : str
        Name of the output file.

    mesh_shape : *Optional[Tuple[int, int]]*
        A list that contains shape of the mesh to be applied on ttir to ttnn
        conversion path.

    argument_types_string : *Optional[str]*

    Returns
    -------
    MLIR module containing MLIR op graph defined by `module` and pipeline_fn.
    """

    if pipeline_options is None:
        pipeline_options = []

    if argument_types_string:
        tt_populate_argument_types(module, argument_types_string)

    # Default to the `SYSTEM_DESC_PATH` envvar
    if system_desc_path is None:
        system_desc_path = os.getenv("SYSTEM_DESC_PATH", "")

    # Generate option string
    if system_desc_path:
        pipeline_options.append(f"system-desc-path={system_desc_path}")
    if mesh_shape and len(mesh_shape) == 2:
        pipeline_options.append(f"mesh-shape={mesh_shape[0]},{mesh_shape[1]}")
    if argument_types_string:
        pipeline_options.append("enable-const-eval=true")

    # Now, pass it through the pipeline. Module gets modified in place.
    pipeline_fn(module, " ".join(pipeline_options))

    # Optionally dump to file.
    if dump_to_file:
        with open(output_file_name, "w") as f:
            f.write(str(module))

    return module


def ttir_to_ttmetal(
    module,
    dump_to_file: bool = True,
    output_path: str = "",
    output_file_name: str = "test.mlir",
    system_desc_path: Optional[str] = None,
    device_grid_shape: Optional[Tuple[int, int]] = None,
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
    assert mesh_shape is None, "`mesh_shape` is not supported yet for ttmetal backend."

    # Default to the `SYSTEM_DESC_PATH` envvar
    options = []
    if system_desc_path is None:
        system_desc_path = os.getenv("SYSTEM_DESC_PATH", "")
    options.append(f"system-desc-path={system_desc_path}")

    if device_grid_shape is not None:
        options.append(f"override-device-shape={','.join(map(str, device_grid_shape))}")

    # Now, pass it through the TTIR to TTMetal pipeline. Module gets
    # modified in place.
    ttir_to_ttmetal_backend_pipeline(module, " ".join(options))

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
    device_grid_shape: Optional[Tuple[int, int]] = None,
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
    ttmetal_to_flatbuffer_file(
        module,
        output_file_name,
        builder.get_golden_map(),
        module_log if module_log else [],
    )

    print("`ttmetal_to_flatbuffer_file` passed successfully.")


def compile_to_flatbuffer(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    test_base: str = "test",
    output_root: str = ".",
    target: Literal["ttnn", "ttmetal"] = "ttnn",
    mesh_shape: Optional[Tuple[int, int]] = None,
    module_dump: bool = True,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Optional[Union[Callable, str]] = None,
    pipeline_options: Optional[List[str]] = None,
    print_ir: Union[bool, str] = False,
):
    """
    Compiles a TTIRBuilder function `fn` to TTIR MLIR -> TT{Metal,NN} MLIR -> Flatbuffer.

    This decorator is mainly a wrapper around the following functions, with
    each next function called on the output of the last:

    1. `build_mlir_module`
    2. `run_pipeline`
    3. `to_flatbuffer`

    The choice of TTNN vs. TTMetal is controlled by the `target` parameter.

    Parameters
    ----------
    fn : Callable
        The TTIRBuilder function to compile. Must take `builder : TTIRBuilder` as a kwarg.

    inputs_shapes : *List[Shape]*
        Shapes of the respective ranked tensor inputs of the test function.

    inputs_types : *Optional[List[torch.dtype]]*, optional
        The dtypes to use for the inputs to `fn`. Note that if supplied,
        `len(inputs_shapes) == len(inputs_types)` must be true.
        Default is None.

    test_base : str
        The string to be used as the base name for dumped files throughout the
        process. If `None` is provided, then the `__name__` of `fn` will be used.

    output_root : str
        The path to dump all generated arguments under. If this path doesn't
        exist, it will be created.

    target : *Literal["ttnn", "ttmetal"]*
        Either "ttnn" or "ttmetal". This controls which backend to use.

    argument_types_string : *Optional[str]*

    custom_pipeline : *Union[Callable, str]*, optional
        Pipeline function to run.
        Can be either:

        - A Callable: custom_pipeline(module, options)
        - A str: "ttir-lower-to-layout,ttir-bufferization-pipeline"

    mesh_shape : *Optional[Tuple[int, int]]*, optional
        A list that contains shape of the mesh to be applied on ttir to ttnn
        conversion path.
        Default is None.

    module_dump : bool
        Set to True to print out generated TTIR MLIR module.
        Default is False.

    pipeline_options : *Optional[List[str]]*
        Pipeline options to be added to the pass

    print_ir : *Union[bool, str]*, optional
        Set to True to print IR to stdout. Set to dir path to print IR after
        each pass to its own file under that directory.
        Default is False.
    """

    if inputs_types is not None:
        assert len(inputs_shapes) == len(inputs_types)

    if type(custom_pipeline) is str:
        custom_pipeline = create_custom_pipeline_fn(custom_pipeline, print_ir=print_ir)

    if pipeline_options is None:
        pipeline_options = []

    pipeline_fn: Callable
    to_flatbuffer: Callable
    mlir_suffix: str
    target_extension: str

    if target == "ttnn":
        pipeline_fn = (
            custom_pipeline if custom_pipeline else ttir_to_ttnn_backend_pipeline
        )
        to_flatbuffer = ttnn_to_flatbuffer_file
        mlir_suffix = "_ttnn.mlir"
        target_extension = "ttnn"
    elif target == "ttmetal":
        pipeline_fn = (
            custom_pipeline if custom_pipeline else ttir_to_ttmetal_backend_pipeline
        )
        to_flatbuffer = ttmetal_to_flatbuffer_file
        mlir_suffix = "_ttm.mlir"
        target_extension = "ttm"
    else:
        raise ValueError("Unsupported target: " + target)

    # Compile model to TTIR MLIR
    module, builder = build_mlir_module(
        fn,
        inputs_shapes,
        inputs_types,
        mesh_shape=mesh_shape,
        module_dump=module_dump,
        output_root=output_root,
    )

    output_file_mlir = get_target_path(output_root, test_base + mlir_suffix, target)
    output_file_fbb = ".".join([output_file_mlir, target_extension])

    # Compile TTIR MLIR -> TT{Metal,NN} MLIR
    module = run_pipeline(
        module,
        pipeline_fn,
        pipeline_options=pipeline_options,
        dump_to_file=module_dump,
        output_file_name=output_file_mlir,
        system_desc_path=system_desc_path,
        mesh_shape=mesh_shape,
        argument_types_string=argument_types_string,
    )
    print(f"{target} pipeline ran successfully.")

    module_logger = MLIRModuleLogger()
    module_logger.attach_context(module.context)

    # Compile TT{Metal,NN} MLIR -> flatbuffer
    to_flatbuffer(
        module,
        output_file_fbb,
        builder.get_golden_map(),
        module_logger.module_log if module_logger.module_log else [],
    )
    print(f"{target} flatbuffer created successfully.")
