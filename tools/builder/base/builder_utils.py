# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import inspect
import subprocess
import torch
from typing import Callable, List, Optional, Tuple, Union, Literal, Dict
from collections import OrderedDict

from ttmlir.ir import *
from ttmlir.dialects import func, ttcore
from ttmlir.passmanager import PassManager
from ttmlir.passes import (
    tt_populate_argument_types,
    ttir_to_ttnn_backend_pipeline,
    ttnn_to_flatbuffer_file,
    ttir_to_ttmetal_backend_pipeline,
    ttmetal_to_flatbuffer_file,
    translate_to_cpp,
    translate_to_python,
    MLIRModuleLogger,
    stablehlo_pipeline,
    stablehlo_to_ttir_pipeline,
    ttir_to_emitpy_pipeline,
)

from builder.base.builder import *
from builder.ttir.ttir_builder import TTIRBuilder
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.d2m.d2m_builder import D2MBuilder

# ----- Exception Classes -----


class TTBuilderCompileException(Exception):
    """Exception raised when builder compilation fails during compile_ttir_to_flatbuffer."""

    pass


class TTBuilderRuntimeException(Exception):
    """Exception raised when compiled builder code fails during runtime execution.

    This exception is reserved for future use when runtime execution is implemented.
    """

    pass


class TTBuilderGoldenException(Exception):
    """Exception raised when builder output doesn't match expected golden results.

    This exception is reserved for future use when golden verification is implemented.
    """

    pass


# ----- Shared Helper Functions -----


def get_metal_tensor_layout(
    ctx: Context,
    logical_shape: Shape,
    tiled=False,
    oobVal=ttcore.OOBVal.Undef,
    memorySpace=ttcore.MemorySpace.DeviceL1,
    grid: Optional[Tuple[int, int]] = None,
    index_map: Optional[AffineMap] = None,
    memory_layout: Optional[
        ttcore.TensorMemoryLayout
    ] = ttcore.TensorMemoryLayout.Sharded,
) -> RankedTensorType:
    """
    Create a metal tensor layout.

    This function creates metal tensor layouts for both TTIR and D2M operations.
    Previously duplicated between TTIRBuilder and D2MBuilder.

    Parameters
    ----------
    ctx : Context
        MLIR context
    logical_shape : Shape
        Logical shape of the tensor
    tiled : bool
        Whether to use tiled layout (32x32 tiles)
    oobVal : ttcore.OOBVal
        Out-of-bounds value handling
    memorySpace : ttcore.MemorySpace
        Memory space (L1, DRAM, etc.)
    grid : Optional[Tuple[int, int]]
        Grid shape for sharding
    index_map : Optional[AffineMap]
        Optional affine map for layout transformation

    Returns
    -------
    RankedTensorType
        The metal tensor type with layout
    """
    # Create grid shape by 1s filling logical rank.
    if grid is None:
        original_rank = len(logical_shape)
        grid_shape = [1] * original_rank
    else:
        grid_shape = list(grid)

    worker_grid = [8, 8]

    # Create layout with original logical shape.
    if index_map is None:
        layout = ttcore.ir.MetalLayoutAttr.get(
            ctx, logical_shape, worker_grid, oobVal, memorySpace, memory_layout
        )
    else:
        layout = ttcore.ir.MetalLayoutAttr.get(
            ctx,
            logical_shape,
            worker_grid,
            oobVal,
            memorySpace,
            memory_layout,
            index_map,
        )

    shard_shape = []
    for l, g in zip(logical_shape, grid_shape):
        assert l % g == 0, f"Logical shape {l} must be divisible by grid shape {g}"
        shard_shape.append(l // g)

    # Get sharded shape w/ proper collapse & alignment logic.
    typed_layout = ttcore.ir.MetalLayoutAttr.maybe_downcast(layout)
    if typed_layout is None:
        raise RuntimeError("Failed to downcast MetalLayoutAttr")
    device_shape = typed_layout.getDeviceShape(
        grid_shape, [32, 32] if tiled else [1, 1]
    )

    elemType = F32Type.get(ctx)

    # For tiled layouts, ensure the device shape accounts for tiles.
    if tiled:
        elemType = ttcore.ir.TileType.get(ctx, 32, 32, ttcore.DataType.Float32)
        if grid is None or grid == (1, 1):
            # For default 1x1 grid, use exact tile count.
            tile_count_h = (logical_shape[-2] + 31) // 32
            tile_count_w = (logical_shape[-1] + 31) // 32
            device_shape[-2] = tile_count_h
            device_shape[-1] = tile_count_w
        else:
            # For explicit grids, calculate proper sharded tile count.
            shard_h, shard_w = shard_shape[-2], shard_shape[-1]
            tiles_per_shard_h = (shard_h + 31) // 32
            tiles_per_shard_w = (shard_w + 31) // 32
            device_shape[-2] = tiles_per_shard_h
            device_shape[-1] = tiles_per_shard_w

    return RankedTensorType.get(device_shape, elemType, layout, Location.unknown(ctx))


# ----- Private APIs -----


def _get_target_path(output_path, builder_dir, filename, target):
    target_dir = os.path.join(output_path, builder_dir, target)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return os.path.join(target_dir, filename)


def _emitc_to_executable(module, filepath: str, golden_map, module_cache):
    py = translate_to_cpp(module)
    with open(filepath, "w") as f:
        f.write(py)


def _emitpy_to_executable(module, filepath: str, golden_map, module_cache):
    cpp = translate_to_python(module)
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
        Set to True to print out generated MLIR module. Default is True.

    base : *Optional[str]*
        Output file name

    output_root: str = ".",
        Output file path

    Returns
    -------
    Tuple[Module, TTIRBuilder]
        A tuple containing the MLIR module and the TTIRBuilder instance

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

    ttir_builder = TTIRBuilder(ctx, loc, mesh_name, mesh_dict)

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

        module = Module.create()
        with InsertionPoint(module.body):

            @func.func(*fn_input_types, name=fn.__name__)
            def decorated_func(*inputs):
                input_goldens: Dict[Operand, BuilderGoldenTensor] = {}
                for index, (operand, dtype) in enumerate(zip(inputs, inputs_types)):
                    input_goldens[operand] = ttir_builder._generate_golden_tensor(
                        operand, dtype
                    )
                ttir_builder._set_goldens(input_goldens)
                ttir_builder._set_input_ordering(inputs)

                result = fn(*inputs, ttir_builder)

                outputs = result if hasattr(result, "__iter__") else (result,)
                output_goldens: Dict[Operand, BuilderGoldenTensor] = {}
                for op in outputs:
                    output_goldens[op] = ttir_builder._get_golden_tensor(op)
                ttir_builder._set_goldens(output_goldens)
                ttir_builder._set_output_ordering(outputs)

                return result

        print(f"`{fn.__name__}` successfully transformed into a MLIR module.")
        base = fn.__name__ if base is None else base
        filename = _get_target_path(
            output_root, "ttir-builder-artifacts", base + "_ttir.mlir", "ttir"
        )

        if module_dump:
            with open(filename, "w") as f:
                f.write(str(module))
                print(module)

        return module, ttir_builder


def build_d2m_module(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    module_dump: bool = False,
    base: Optional[str] = None,
    output_root: str = ".",
) -> Tuple[Module, D2MBuilder]:
    """
    Define a MLIR module specified as a python function using D2M operations.

    This is similar to build_ttir_module but uses D2MBuilder to create D2M operations
    directly, bypassing TTIR. This is useful for tests that need to work with D2M
    operations without going through the full TTIR-to-D2M conversion pipeline.

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
        Set to True to print out generated MLIR module. Default is False.

    base : *Optional[str]*
        Output file name

    output_root: str = ".",
        Output file path

    Returns
    -------
    Tuple[Module, D2MBuilder]
        A tuple containing the MLIR module and the D2MBuilder instance

    Example
    -------
    >>> def test_to_layout(in0: Operand, builder: D2MBuilder):
    ...     return builder.to_layout(in0, target_layout)
    ...
    >>> build_d2m_module(test_to_layout, ((32, 32),))
    """

    ctx = Context()

    # Grab the location of the test function in python for later debugging
    try:
        fname = inspect.getfile(fn)
        line_no = inspect.getsourcelines(fn)[1]
        loc = Location.file(fname, line_no, 0, ctx)
    except (OSError, TypeError):
        loc = Location.unknown(ctx)

    d2m_builder = D2MBuilder(ctx, loc, mesh_name, mesh_dict)

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
            d2m_builder._create_ranked_tensor_type(
                shape,
                d2m_builder._get_type_from_torch_dtype(
                    dtype if isinstance(dtype, torch.dtype) else dtype
                ),
            )
            for (shape, dtype) in zip(inputs_shapes, inputs_types)
        ]

        module = Module.create()
        with InsertionPoint(module.body):

            @func.func(*fn_input_types, name=fn.__name__)
            def decorated_func(*inputs):
                input_goldens: Dict[Operand, BuilderGoldenTensor] = {}
                for index, (operand, dtype) in enumerate(zip(inputs, inputs_types)):
                    input_goldens[operand] = d2m_builder._generate_golden_tensor(
                        operand, dtype
                    )
                d2m_builder._set_goldens(input_goldens)
                d2m_builder._set_input_ordering(inputs)

                result = fn(*inputs, d2m_builder)

                outputs = result if hasattr(result, "__iter__") else (result,)
                output_goldens: Dict[Operand, BuilderGoldenTensor] = {}
                for op in outputs:
                    output_goldens[op] = d2m_builder._get_golden_tensor(op)
                d2m_builder._set_goldens(output_goldens)
                d2m_builder._set_output_ordering(outputs)

                return result

        print(f"`{fn.__name__}` successfully transformed into a MLIR module.")
        base = fn.__name__ if base is None else base
        filename = _get_target_path(
            output_root, "d2m-builder-artifacts", "d2m.mlir", base
        )

        if module_dump:
            with open(filename, "w") as f:
                f.write(str(module))
                print(module)

        return module, d2m_builder


def compile_ttir_to_flatbuffer(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    test_base: str = "test",
    output_root: str = ".",
    target: Literal["ttnn", "ttmetal", "ttnn-standalone", "emitpy"] = "ttnn",
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
        Either "ttnn", "ttmetal", or "ttnn-standalone". This controls which backend to use.

    mesh_name : *str*, optional
        Name of the mesh to be used in the module. Default is "mesh".

    mesh_dict : *OrderedDict[str, int]*, optional
        Dictionary that defines the mesh shape, e.g. OrderedDict([("x", 1), ("y", 1)]).

    argument_types_string : *Optional[str]*, optional
        String defining argument types for constant evaluation.

    argument_types_string : *Optional[str]*

    custom_pipeline : *Union[Callable, str]*, optional
        Pipeline function to run.
        Can be either:

        - A Callable: custom_pipeline(module, options)
        - A str: "ttir-lower-to-layout,ttir-bufferization-pipeline"

    system_desc_path : str, optional
        Path to the system descriptor file

    mesh_name : *str*
        Name of the mesh to be used in the module. Default is "mesh".

    mesh_dict : *OrderedDict[str, int]*
        Dictionary that defines the mesh shape, e.g. OrderedDict([("x", 1), ("y", 1)]).

    module_dump : bool
        Set to True to print out generated TTIR MLIR module.
        Default is False.

    pipeline_options : *Optional[List[str]]*
        Pipeline options to be added to the pass

    print_ir : Union[bool, str], optional
        Controls intermediate IR dumping during compilation.
        - True  →  Print IR to stdout after each pass.
                This is convenient for quick inspection or interactive
                debugging (e.g. with breakpoints), but is unreliable if
                the process crashes or aborts as the output may be truncated or
                lost.
        - str (directory path)  →  Write IR after each pass to a separate file
                under the given directory. This is more reliable than stdout,
                since files are flushed incrementally and preserved up to the
                point of failure. It can give hints about where the pipeline
                crashed.
        Notes:
            - For fatal crashes (e.g. MLIR assertions), neither mode guarantees
            a complete dump. Using a directory at least preserves passes run
            before the crash.
            - For stdout mode, you may need to run Python with unbuffered output
            (e.g. `pytest -s` or `python -u`) and/or use pdb to reliably see
            dumps before a crash.
        Default is False (no IR printed).

    Returns
    -------
    str
        The path to the generated TT{Metal,NN} MLIR file.
    """

    if inputs_types is not None:
        if len(inputs_shapes) != len(inputs_types):
            raise ValueError("inputs_shapes and inputs_types must have the same length")

    # Compile model to TTIR MLIR
    try:
        module, builder = build_ttir_module(
            fn,
            inputs_shapes,
            inputs_types,
            mesh_name=mesh_name,
            mesh_dict=mesh_dict,
            module_dump=module_dump,
            output_root=output_root,
            base=test_base,
        )

        return compile_ttir_module_to_flatbuffer(
            module,
            builder,
            system_desc_path=system_desc_path,
            test_base=test_base,
            output_root=output_root,
            target=target,
            mesh_dict=mesh_dict,
            module_dump=module_dump,
            argument_types_string=argument_types_string,
            custom_pipeline=custom_pipeline,
            pipeline_options=pipeline_options,
            print_ir=print_ir,
        )
    except Exception as e:
        raise TTBuilderCompileException(e)


def compile_d2m_to_flatbuffer(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    test_base: str = "test",
    output_root: str = ".",
    target: Literal["ttnn", "ttmetal", "ttnn-standalone", "emitpy"] = "ttnn",
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    module_dump: bool = True,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Optional[Union[Callable, str]] = None,
    pipeline_options: Optional[List[str]] = None,
    print_ir: Union[bool, str] = False,
) -> str:
    """
    Compiles a D2MBuilder function `fn` to D2M MLIR -> TTMetal MLIR -> Flatbuffer.

    This decorator is a wrapper around:

    1. `build_d2m_module`
    2. `_run_ttir_pipeline`
    3. `to_target`

    The choice of TTNN vs. TTMetal is controlled by the `target` parameter.

    Parameters
    ----------
    fn : Callable
        The D2MBuilder function to compile. Must take `builder : D2MBuilder` as a kwarg.

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
        Either "ttnn", "ttmetal", or "ttnn-standalone". This controls which backend to use.

    mesh_name : *str*, optional
        Name of the mesh to be used in the module. Default is "mesh".

    mesh_dict : *OrderedDict[str, int]*, optional
        Dictionary that defines the mesh shape, e.g. OrderedDict([("x", 1), ("y", 1)]).

    argument_types_string : *Optional[str]*, optional
        String defining argument types for constant evaluation.

    custom_pipeline : *Union[Callable, str]*, optional
        Pipeline function to run.
        Can be either:

        - A Callable: custom_pipeline(module, options)
        - A str: "ttir-lower-to-layout,ttir-bufferization-pipeline"

    system_desc_path : str, optional
        Path to the system descriptor file

    module_dump : bool
        Set to True to print out generated D2M MLIR module.
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

    # Compile model to D2M MLIR
    try:
        module, builder = build_d2m_module(
            fn,
            inputs_shapes,
            inputs_types,
            mesh_name=mesh_name,
            mesh_dict=mesh_dict,
            module_dump=module_dump,
            output_root=output_root,
            base=test_base,
        )

        return compile_ttir_module_to_flatbuffer(
            module,
            builder,
            system_desc_path=system_desc_path,
            test_base=test_base,
            output_root=output_root,
            target=target,
            mesh_dict=mesh_dict,
            module_dump=module_dump,
            argument_types_string=argument_types_string,
            custom_pipeline=custom_pipeline,
            pipeline_options=pipeline_options,
            print_ir=print_ir,
        )
    except Exception as e:
        raise TTBuilderCompileException(e)


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
        Set to True to print out generated MLIR module. Default is True.

    base : *Optional[str]*
        Output file name

    output_root: str = ".",
        Output file path

    Returns
    -------
    *Tuple[Module, StableHLOBuilder]*
        A tuple containing the MLIR module and the StableHLOBuilder instance

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
    stablehlo_builder = StableHLOBuilder(ctx, loc, mesh_name, mesh_dict)

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
        module.body.append(stablehlo_builder._get_mesh(mesh_name))

        with InsertionPoint(module.body):
            # Wrap everything in a mlir function.
            @func.func(*fn_input_types, name=fn.__name__)
            def decorated_func(*inputs):
                input_goldens: Dict[Operand, BuilderGoldenTensor] = {}
                for index, (operand, dtype) in enumerate(zip(inputs, inputs_types)):
                    input_goldens[operand] = stablehlo_builder._generate_golden_tensor(
                        operand, dtype
                    )
                stablehlo_builder._set_goldens(input_goldens)
                stablehlo_builder._set_input_ordering(inputs)

                result = fn(*inputs, stablehlo_builder)

                outputs = result if hasattr(result, "__iter__") else (result,)
                output_goldens: Dict[Operand, BuilderGoldenTensor] = {}
                for op in outputs:
                    output_goldens[op] = stablehlo_builder._get_golden_tensor(op)
                stablehlo_builder._set_goldens(output_goldens)
                stablehlo_builder._set_output_ordering(outputs)

                return result

        print(f"`{fn.__name__}` successfully transformed into a MLIR module.")
        base = fn.__name__ if base is None else base
        filename = _get_target_path(
            output_root, "stablehlo-builder-artifacts", base + "_shlo.mlir", "shlo"
        )

        if module_dump:
            with open(filename, "w") as f:
                f.write(str(module))
                print(module)

        return module, stablehlo_builder


def compile_stablehlo_to_flatbuffer(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    test_base: str = "test",
    output_root: str = ".",
    target: Literal["ttnn", "ttmetal", "ttnn-standalone", "emitpy"] = "ttnn",
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    module_dump: bool = True,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Optional[Union[Callable, str]] = None,
    ttir_pipeline_options: Optional[List[str]] = None,
    shlo_pipeline_options: Optional[List[str]] = None,
    shlo_to_ttir_pipeline_options: Optional[List[str]] = None,
    print_ir: Union[bool, str] = False,
) -> str:
    """
    Compiles a StableHLO function to flatbuffer format.

    This function compiles a StableHLO function through the complete pipeline:
    StableHLO -> TTIR -> TT{Metal,NN} -> Flatbuffer. It first builds a StableHLO
    module, runs the stablehlo pipeline and conversion to TTIR, then compiles
    the TTIR module to the target flatbuffer format.

    Parameters
    ----------
    fn : Callable
        The StableHLO function to compile

    inputs_shapes : *List[Shape]*
        Shapes of the respective ranked tensor inputs of the test function

    inputs_types : *Optional[List[Union[torch.dtype, TypeInfo]]]*, optional
        The dtypes to use for the inputs to `fn`

    system_desc_path : str, optional
        Path to the system descriptor file

    test_base : str, optional
        The string to be used as the test_base name for dumped files

    output_root : str, optional
        The path to dump all generated files under

    target : *Literal["ttnn", "ttmetal", "ttnn-standalone"]*, optional
        The target backend to use. Default is "ttnn"

    mesh_name : str, optional
        Name of the mesh to be used in the module

    mesh_dict : *OrderedDict[str, int]*, optional
        Dictionary that defines the mesh shape

    module_dump : bool, optional
        Set to True to print out generated MLIR modules
        Default is True.

    argument_types_string : *Optional[str]*
        String defining argument types for constant evaluation

    custom_pipeline : *Optional[Union[Callable, str]]*
        Custom pipeline function or string to run instead of default pipeline

    ttir_pipeline_options : *List[str]*
        Additional pipeline options to pass to the TTIR pipeline

    shlo_pipeline_options : *List[str]*
        Additional pipeline options to pass to the StableHLO pipeline

    print_ir : Union[bool, str], optional
        Controls intermediate IR dumping during compilation.
        - True  →  Print IR to stdout after each pass.
                This is convenient for quick inspection or interactive
                debugging (e.g. with breakpoints), but is unreliable if
                the process crashes or aborts as the output may be truncated or
                lost.
        - str (directory path)  →  Write IR after each pass to a separate file
                under the given directory. This is more reliable than stdout,
                since files are flushed incrementally and preserved up to the
                point of failure. It can give hints about where the pipeline
                crashed.
        Notes:
            - For fatal crashes (e.g. MLIR assertions), neither mode guarantees
            a complete dump. Using a directory at least preserves passes run
            before the crash.
            - For stdout mode, you may need to run Python with unbuffered output
            (e.g. `pytest -s` or `python -u`) and/or use pdb to reliably see
            dumps before a crash.
        Default is False (no IR printed).

    Returns
    -------
    str
        The path to the generated TT{Metal,NN} MLIR file.

    Raises
    ------
    ValueError
        If inputs_shapes and inputs_types have different lengths
    """
    if shlo_pipeline_options is None:
        shlo_pipeline_options = []

    if shlo_to_ttir_pipeline_options is None:
        shlo_to_ttir_pipeline_options = []

    if inputs_types is not None:
        if len(inputs_shapes) != len(inputs_types):
            raise ValueError("inputs_shapes and inputs_types must have the same length")

    # Compile model to StableHLO and run stablehlo pipeline to TTIR MLIR
    try:
        module, builder = build_stablehlo_module(
            fn,
            inputs_shapes,
            inputs_types,
            mesh_name=mesh_name,
            mesh_dict=mesh_dict,
            module_dump=module_dump,
            output_root=output_root,
            base=test_base,
        )
    except Exception as e:
        raise TTBuilderCompileException(e)

    stablehlo_pipeline(module, " ".join(shlo_pipeline_options))
    print(f"`{fn.__name__}` successfully ran stablehlo-pipeline.")
    print(module)

    filename = _get_target_path(
        output_root,
        "stablehlo-builder-artifacts",
        test_base + "_shlo_pipeline.mlir",
        "shlo_pipeline",
    )
    if module_dump:
        with open(filename, "w") as f:
            f.write(str(module))

    stablehlo_to_ttir_pipeline(module, " ".join(shlo_to_ttir_pipeline_options))
    print(f"`{fn.__name__}` successfully transformed into a TTIR MLIR module.")
    print(module)

    filename = _get_target_path(
        output_root, "stablehlo-builder-artifacts", test_base + "_ttir.mlir", "ttir"
    )
    if module_dump:
        with open(filename, "w") as f:
            f.write(str(module))

    return compile_ttir_module_to_flatbuffer(
        module,
        builder,
        system_desc_path=system_desc_path,
        test_base=test_base,
        output_root=output_root,
        builder_dir="stablehlo-builder-artifacts",
        target=target,
        mesh_dict=mesh_dict,
        module_dump=module_dump,
        argument_types_string=argument_types_string,
        custom_pipeline=custom_pipeline,
        pipeline_options=ttir_pipeline_options,
        print_ir=print_ir,
    )


def compile_ttir_module_to_flatbuffer(
    module: Module,
    builder: Builder,
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    test_base: str = "test",
    output_root: str = ".",
    builder_dir: str = "ttir-builder-artifacts",
    target: Literal["ttnn", "ttmetal", "ttnn-standalone", "emitpy"] = "ttnn",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    module_dump: bool = True,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Optional[Union[Callable, str]] = None,
    pipeline_options: List[str] = [],
    print_ir: Union[bool, str] = False,
):
    """
    Compiles a TTIR MLIR module to flatbuffer format.

    This decorator takes an existing TTIR MLIR module and compiles it through
    the backend pipeline to generate a flatbuffer file. It supports multiple
    targets including TTNN, TTMetal, and TTNN-standalone. It is mainly a wrapper around the following functions, with
    each next function called on the output of the last:

    1. `_run_ttir_pipeline`
    2. `to_target`

    Parameters
    ----------
    module : Module
        The TTIR MLIR module to compile

    builder : *Union[TTIRBuilder, StableHLOBuilder]*
        The builder instance containing golden reference values

    system_desc_path : str, optional
        Path to the system descriptor file

    test_base : str, optional
        The string to be used as the test_base name for dumped files.

    output_root : str, optional
        The path to dump all generated files under

    target : *Literal["ttnn", "ttmetal", "ttnn-standalone"]*, optional
        The target backend to use. Default is "ttnn"

    mesh_dict : *OrderedDict[str, int]*, optional
        Dictionary that defines the mesh shape.

    module_dump : bool, optional
        Set to True to print out generated MLIR modules. Default is True.

    argument_types_string : *Optional[str]*, optional
        String defining argument types for constant evaluation

    custom_pipeline : *Optional[Union[Callable, str]]*
        Custom pipeline function or string to run instead of default pipeline

    pipeline_options : *List[str]*, optional
        Additional pipeline options to pass to the pipeline

    print_ir : Union[bool, str], optional
        Controls intermediate IR dumping during compilation.
        - True  →  Print IR to stdout after each pass.
                This is convenient for quick inspection or interactive
                debugging (e.g. with breakpoints), but is unreliable if
                the process crashes or aborts as the output may be truncated or
                lost.
        - str (directory path)  →  Write IR after each pass to a separate file
                under the given directory. This is more reliable than stdout,
                since files are flushed incrementally and preserved up to the
                point of failure. It can give hints about where the pipeline
                crashed.
        Notes:
            - For fatal crashes (e.g. MLIR assertions), neither mode guarantees
            a complete dump. Using a directory at least preserves passes run
            before the crash.
            - For stdout mode, you may need to run Python with unbuffered output
            (e.g. `pytest -s` or `python -u`) and/or use pdb to reliably see
            dumps before a crash.
        Default is False (no IR printed).

    Returns
    -------
    str
        The path to the generated target MLIR file

    Raises
    ------
    ValueError
        If an unsupported target is specified
    """
    if type(custom_pipeline) is str:
        custom_pipeline = _create_custom_ttir_pipeline_fn(
            custom_pipeline, print_ir=print_ir
        )

    pipeline_fn: Callable
    to_target: Callable
    filename: str
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
    elif target == "emitpy":
        pipeline_fn = custom_pipeline if custom_pipeline else ttir_to_emitpy_pipeline
        to_target = _emitpy_to_executable
        mlir_suffix = "_ttnn.mlir"
        target_extension = "py"
    else:
        raise ValueError("Unsupported target: " + target)

    output_file_mlir = _get_target_path(
        output_root, builder_dir, test_base + mlir_suffix, target
    )
    output_file_fbb = ".".join([output_file_mlir, target_extension])
    print(output_file_mlir, output_file_fbb)

    # Compile TTIR MLIR -> TT{Metal,NN} MLIR
    try:
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
        raise TTBuilderCompileException(e)

    print(f"{target} pipeline ran successfully.")

    module_logger = MLIRModuleLogger()
    module_logger.attach_context(module.context)

    # Compile TT{Metal,NN} MLIR -> flatbuffer
    try:
        to_target(
            module,
            output_file_fbb,
            builder.golden_map,
            module_logger.module_log if module_logger.module_log else [],
        )
    except Exception as e:
        raise TTBuilderCompileException(e)

    print(f"{target} flatbuffer created successfully at: {output_file_fbb}")

    return output_file_mlir


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
    stablehlo_builder = StableHLOBuilder(ctx, loc, mesh_name, mesh_dict)

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
                input_goldens: Dict[Operand, BuilderGoldenTensor] = {}
                for index, (operand, dtype) in enumerate(zip(inputs, inputs_types)):
                    input_goldens[operand] = stablehlo_builder._generate_golden_tensor(
                        operand, dtype
                    )
                stablehlo_builder._set_goldens(input_goldens)
                stablehlo_builder._set_input_ordering(inputs)

                result = fn(*inputs, stablehlo_builder)

                outputs = result if hasattr(result, "__iter__") else (result,)
                output_goldens: Dict[Operand, BuilderGoldenTensor] = {}
                for op in outputs:
                    output_goldens[op] = stablehlo_builder._get_golden_tensor(op)
                stablehlo_builder._set_goldens(output_goldens)
                stablehlo_builder._set_output_ordering(outputs)

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
        filename = _get_target_path(
            output_root, "stablehlo-builder-artifacts", "shlo.mlir", base
        )

        if module_dump:
            with open(filename, "w") as f:
                f.write(str(module))
                print(module)

        return module, stablehlo_builder
