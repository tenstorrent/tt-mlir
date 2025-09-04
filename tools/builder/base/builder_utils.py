# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import inspect
import subprocess
import torch
import pytest
from typing import Callable, List, Optional, Tuple, Union, Literal, Dict
from collections import OrderedDict

from ttmlir.ir import *
from ttmlir.dialects import func, sdy, mpmd
from ttmlir.passmanager import PassManager
from ttmlir.passes import (
    tt_populate_argument_types,
    ttir_to_ttnn_backend_pipeline,
    ttnn_to_flatbuffer_file,
    ttir_to_ttmetal_backend_pipeline,
    ttmetal_to_flatbuffer_file,
    translate_to_cpp,
    MLIRModuleLogger,
    stablehlo_pipeline,
)

from builder.base.builder import *
from builder.ttir.ttir_builder import TTIRBuilder
from builder.stablehlo.stablehlo_builder import StableHLOBuilder

# ----- Exception Classes -----


class TTIRCompileException(Exception):
    """Exception raised when TTIR compilation fails during compile_ttir_to_flatbuffer."""

    pass


class TTIRRuntimeException(Exception):
    """Exception raised when compiled TTIR code fails during runtime execution.

    This exception is reserved for future use when runtime execution is implemented.
    """

    pass


class TTIRGoldenException(Exception):
    """Exception raised when TTIR output doesn't match expected golden results.

    This exception is reserved for future use when golden verification is implemented.
    """

    pass


# ----- Private APIs -----


def _get_target_path(output_path, filename, target):
    target_dir = os.path.join(output_path, target)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return os.path.join(target_dir, filename)


def _emitc_to_executable(module, filepath: str, golden_map, module_cache):
    cpp = translate_to_cpp(module)
    with open(filepath, "w") as f:
        f.write(cpp)


def _create_custom_ttir_pipeline_fn(
    pipeline: str, verify: bool = True, print_ir: Union[bool, str] = False
) -> Callable:
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


def _run_ttir_pipeline(
    module,
    pipeline_fn: Callable,
    pipeline_options: Optional[List[str]] = None,
    dump_to_file: bool = True,
    output_file_name: str = "test.mlir",
    system_desc_path: Optional[str] = None,
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    argument_types_string: Optional[str] = None,
):
    if pipeline_options is None:
        pipeline_options = []

    if argument_types_string:
        tt_populate_argument_types(module, argument_types_string)
        pipeline_options.append("enable-const-eval=true")

    # Default to the `SYSTEM_DESC_PATH` envvar.
    if system_desc_path is None:
        system_desc_path = os.getenv("SYSTEM_DESC_PATH", "")
    pipeline_options.append(f"system-desc-path={system_desc_path}")

    mesh_shape = tuple(mesh_dict.values())
    if len(mesh_shape) != 2:
        raise ValueError(f"Mesh shape must be a tuple of length 2, got: {mesh_shape}")

    pipeline_options.append(f"mesh-shape={mesh_shape[0]},{mesh_shape[1]}")

    # Now, pass it through the pipeline. Module gets modified in place.
    pipeline_fn(module, " ".join(pipeline_options))

    # Optionally dump to file.
    if dump_to_file:
        with open(output_file_name, "w") as f:
            f.write(str(module))

    return module


# ----- Public APIs -----


def build_ttir_module(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    module_dump: bool = False,
    base: Optional[str] = None,
    output_root: str = ".",
) -> Tuple[Module, TTIRBuilder]:
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

    mesh_name: *str*
        Name of the mesh to be used in the module. Default is "mesh".

    mesh_dict: *OrderedDict[str, int]*
        Dictionary that defines the mesh shape, e.g. OrderedDict([("x", 1), ("y", 1)]).

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
    >>> build_ttir_module(test_add, ((32, 32), (32, 32)))

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
    mesh_shape = tuple(mesh_dict.values())
    ttir_builder = TTIRBuilder(ctx, loc, mesh_shape)

    # Default to all f32s
    if inputs_types is None:
        inputs_types = [torch.float32] * len(inputs_shapes)

    if len(inputs_shapes) != len(inputs_types):
        raise ValueError(
            f"inputs_shapes and inputs_types must have the same length: "
            f"{len(inputs_shapes)} != {len(inputs_types)}"
        )

    with ctx, loc:
        fn_input_types = [
            ttir_builder._create_ranked_tensor_type(
                shape,
                ttir_builder._get_type_from_torch_dtype(
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
                        ttir_builder._generate_input_golden(
                            operand, dtype, index
                        ).tensor
                    )
                result = fn(*inputs, ttir_builder)
                output_ops = result if hasattr(result, "__iter__") else (result,)
                output_goldens = [
                    ttir_builder._get_golden_tensor(op) for op in output_ops
                ]
                ttir_builder.set_graph_input_output(input_goldens, output_goldens)
                return result

        print(f"`{fn.__name__}` sucessfully transformed into a MLIR module.")

        base = fn.__name__ if base is None else base

        filename = _get_target_path(output_root, base + "_ttir.mlir", "ttir")

        if module_dump:
            with open(filename, "w") as f:
                f.write(str(module))
                print(module)

        return module, ttir_builder


def compile_ttir_to_flatbuffer(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    test_base: str = "test",
    output_root: str = ".",
    target: Literal["ttnn", "ttmetal", "ttnn-standalone"] = "ttnn",
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    module_dump: bool = True,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Optional[Union[Callable, str]] = None,
    pipeline_options: Optional[List[str]] = None,
    print_ir: Union[bool, str] = False,
) -> str:
    """
    Compiles a TTIRBuilder function `fn` to TTIR MLIR -> TT{Metal,NN} MLIR -> Flatbuffer.

    This decorator is mainly a wrapper around the following functions, with
    each next function called on the output of the last:

    1. `build_ttir_module`
    2. `_run_ttir_pipeline`
    3. `to_target`

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

    target : *Literal["ttnn", "ttmetal", "ttnn-standalone"]*
        Either "ttnn" or "ttmetal". This controls which backend to use.

    argument_types_string : *Optional[str]*

    custom_pipeline : *Union[Callable, str]*, optional
        Pipeline function to run.
        Can be either:

        - A Callable: custom_pipeline(module, options)
        - A str: "ttir-lower-to-layout,ttir-bufferization-pipeline"

    mesh_name : *str*
        Name of the mesh to be used in the module. Default is "mesh".

    mesh_dict : *OrderedDict[str, int]*
        Dictionary that defines the mesh shape, e.g. OrderedDict([("x", 1), ("y", 1)]).

    module_dump : bool
        Set to True to print out generated TTIR MLIR module.
        Default is False.

    pipeline_options : *Optional[List[str]]*
        Pipeline options to be added to the pass

    print_ir : *Union[bool, str]*, optional
        Set to True to print IR to stdout. Set to dir path to print IR after
        each pass to its own file under that directory.
        Default is False.

    Returns
    -------
    str
        The path to the generated TT{Metal,NN} MLIR file.
    """

    if inputs_types is not None:
        if len(inputs_shapes) != len(inputs_types):
            raise ValueError("inputs_shapes and inputs_types must have the same length")

    if type(custom_pipeline) is str:
        custom_pipeline = _create_custom_ttir_pipeline_fn(
            custom_pipeline, print_ir=print_ir
        )

    if pipeline_options is None:
        pipeline_options = []

    pipeline_fn: Callable
    to_target: Callable
    mlir_suffix: str
    target_extension: str

    if target == "ttnn":
        pipeline_fn = (
            custom_pipeline if custom_pipeline else ttir_to_ttnn_backend_pipeline
        )
        to_target = ttnn_to_flatbuffer_file
        mlir_suffix = "_ttnn.mlir"
        target_extension = "ttnn"
    elif target == "ttmetal":
        pipeline_fn = (
            custom_pipeline if custom_pipeline else ttir_to_ttmetal_backend_pipeline
        )
        to_target = ttmetal_to_flatbuffer_file
        mlir_suffix = "_ttm.mlir"
        target_extension = "ttm"
    elif target == "ttnn-standalone":
        ttir_to_ttnn_emitc_pipeline = _create_custom_ttir_pipeline_fn(
            "ttir-to-emitc-pipeline", print_ir=print_ir
        )
        pipeline_fn = (
            custom_pipeline if custom_pipeline else ttir_to_ttnn_emitc_pipeline
        )
        to_target = _emitc_to_executable
        mlir_suffix = "_ttnn.mlir"
        target_extension = "cpp"
    else:
        raise ValueError("Unsupported target: " + target)

    try:
        # Compile model to TTIR MLIR
        module, builder = build_ttir_module(
            fn,
            inputs_shapes,
            inputs_types,
            mesh_dict=mesh_dict,
            module_dump=module_dump,
            output_root=output_root,
        )
    except Exception as e:
        raise TTIRCompileException(e)

    output_file_mlir = _get_target_path(output_root, test_base + mlir_suffix, target)
    output_file_fbb = ".".join([output_file_mlir, target_extension])

    try:
        # Compile TTIR MLIR -> TT{Metal,NN} MLIR
        module = _run_ttir_pipeline(
            module,
            pipeline_fn,
            pipeline_options=pipeline_options,
            dump_to_file=module_dump,
            output_file_name=output_file_mlir,
            system_desc_path=system_desc_path,
            mesh_dict=mesh_dict,
            argument_types_string=argument_types_string,
        )
    except Exception as e:
        raise TTIRCompileException(e)

    print(f"{target} pipeline ran successfully.")

    module_logger = MLIRModuleLogger()
    module_logger.attach_context(module.context)

    try:
        # Compile TT{Metal,NN} MLIR -> flatbuffer
        to_target(
            module,
            output_file_fbb,
            builder.golden_map,
            module_logger.module_log if module_logger.module_log else [],
        )
    except Exception as e:
        raise TTIRCompileException(e)
    print(f"{target} flatbuffer created successfully at: {output_file_fbb}")
    return output_file_mlir


def build_stablehlo_module(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    module_dump: bool = False,
    base: Optional[str] = None,
    output_root: str = ".",
) -> Tuple[Module, StableHLOBuilder]:
    """
    Define a MLIR module specified as a python function.

    It will wrap `fn` in a MLIR FuncOp and then wrap that in a MLIR
    module, and finally tie arguments of that FuncOp to test function inputs. It will
    also pass a `StableHLOBuilder` object as the last argument of test function.

    Parameters
    ----------
    fn : Callable
        Python function to be converted to MLIR

    inputs_shapes : *List[Shape]*
        Shapes of the respective ranked tensor inputs of the test function.

    inputs_types: *Optional[List[Union[torch.dtype, TypeInfo]]]*
        Data types of the input tensors

    mesh_name: *str*
        Name of the mesh to be used in the module. Default is "mesh".

    mesh_dict: *OrderedDict[str, int]*
        Dictionary that defines the mesh shape, e.g. OrderedDict([("x", 1), ("y", 1)]).

    module_dump : bool
        Set to True to print out generated MLIR module.

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
    >>> def test_add(in0: Operand, in1: Operand, builder: StableHLOBuilder):
    ...     return builder.add(in0, in1)
    ...
    >>> build_stablehlo_module(test_add, ((32, 32), (32, 32)))

    This returns:

    .. code-block:: mlir

        #any = #ttcore.operand_constraint<...>
        module {
            func.func @test_add(
                %arg0: tensor<32x32xf32>,
                %arg1: tensor<32x32xf32>
            ) -> tensor<32x32xf32> {
                %0 = "stablehlo.add"(%arg0, %arg1, %0) ...
                return %1 : tensor<32x32xf32>
            }
        }
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
    stablehlo_builder = StableHLOBuilder(ctx, loc)

    # Default to all f32s
    if inputs_types is None:
        inputs_types = [torch.float32] * len(inputs_shapes)

    if len(inputs_shapes) != len(inputs_types):
        raise ValueError(
            f"inputs_shapes and inputs_types must have the same length: "
            f"{len(inputs_shapes)} != {len(inputs_types)}"
        )

    with ctx, loc:
        fn_input_types = [
            stablehlo_builder._create_ranked_tensor_type(
                shape,
                stablehlo_builder._get_type_from_torch_dtype(
                    dtype if isinstance(dtype, torch.dtype) else dtype
                ),
            )
            for (shape, dtype) in zip(inputs_shapes, inputs_types)
        ]

        # Wrap everything in a mlir module.
        module = Module.create()
        module.body.append(
            stablehlo_builder.mesh(
                mesh_name=mesh_name,
                mesh_attr=stablehlo_builder._create_mesh_attr_from_ordered_dict(
                    mesh_dict
                ),
            )
        )

        with InsertionPoint(module.body):
            # Wrap everything in a mlir function.
            @func.func(*fn_input_types, name=fn.__name__)
            def decorated_func(*inputs):
                # Randomly generate golden tensors for function inputs.
                input_goldens = []
                for index, (operand, dtype) in enumerate(zip(inputs, inputs_types)):
                    input_goldens.append(
                        stablehlo_builder._generate_input_golden(
                            operand, dtype, index
                        ).tensor
                    )
                result = fn(*inputs, stablehlo_builder)
                output_ops = result if hasattr(result, "__iter__") else (result,)
                output_goldens = [
                    stablehlo_builder._get_golden_tensor(op) for op in output_ops
                ]
                stablehlo_builder.set_graph_input_output(input_goldens, output_goldens)
                return result

        print(f"`{fn.__name__}` sucessfully transformed into a MLIR module.")

        base = fn.__name__ if base is None else base

        filename = _get_target_path(output_root, base + "_shlo.mlir", "shlo")

        if module_dump:
            with open(filename, "w") as f:
                f.write(str(module))
                print(module)

        return module, stablehlo_builder


# ----- Experimental Public APIs -----


def experimental_build_stablehlo_module(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    mesh_name: List[str] = ["mesh"],
    mesh_dict: List[OrderedDict[str, int]] = [OrderedDict([("x", 1), ("y", 1)])],
    module_dump: bool = False,
    base: Optional[str] = None,
    output_root: str = ".",
) -> Tuple[Module, StableHLOBuilder]:
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
    stablehlo_builder = StableHLOBuilder(ctx, loc)

    # Default to all f32s
    if inputs_types is None:
        inputs_types = [torch.float32] * len(inputs_shapes)

    if len(inputs_shapes) != len(inputs_types):
        raise ValueError(
            f"inputs_shapes and inputs_types must have the same length: "
            f"{len(inputs_shapes)} != {len(inputs_types)}"
        )

    with ctx, loc:
        fn_input_types = [
            stablehlo_builder._create_ranked_tensor_type(
                shape,
                stablehlo_builder._get_type_from_torch_dtype(
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
                        stablehlo_builder._generate_input_golden(
                            operand, dtype, index
                        ).tensor
                    )
                result = fn(*inputs, stablehlo_builder)
                output_ops = result if hasattr(result, "__iter__") else (result,)
                output_goldens = [
                    stablehlo_builder._get_golden_tensor(op) for op in output_ops
                ]
                stablehlo_builder.set_graph_input_output(input_goldens, output_goldens)
                return result

            # Create named meshes and add them to the module
            named_mesh_list = []
            for mesh_name, mesh_dict in zip(mesh_name, mesh_dict):
                named_mesh_attr = stablehlo_builder.experimental_named_mesh_attr(
                    mesh_name,
                    stablehlo_builder._create_mesh_attr_from_ordered_dict(mesh_dict),
                )
                named_mesh_list.append(named_mesh_attr)
            topology_attr = stablehlo_builder.experimental_topology_attr(
                named_mesh_list
            )
            func_op = module.body.operations[-1]
            func_op.attributes["topology"] = topology_attr

        print(f"`{fn.__name__}` sucessfully transformed into a MLIR module.")
        base = fn.__name__ if base is None else base
        filename = _get_target_path(output_root, base + "_shlo.mlir", "shlo")

        if module_dump:
            with open(filename, "w") as f:
                f.write(str(module))
                print(module)

        return module, stablehlo_builder
