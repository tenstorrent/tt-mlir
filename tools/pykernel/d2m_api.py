# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import inspect
import functools
import json
import os
from typing import List, Optional

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from _ttmlir_runtime import runtime, binary
except ModuleNotFoundError:
    runtime = None
    binary = None

from ttmlir.ir import *
from ttmlir.passmanager import PassManager
from ttmlir.dialects import (
    ttcore,
    d2m,
    func,
    arith,
)
from ttmlir.passes import ttmetal_to_flatbuffer_bin

from ._src.utils import _discover_dialect_ops, _asindex, _cleanup_source_code
from ._src.d2m_ast import D2MGenericCompiler, syntax
from ._src.stream import Stream


# TODO: add support for collapse intervals and dimension alignments
def create_metal_layout(
    ctx,
    logical_shape: List[int],
    grid: List[int],
    tiled: bool = True,
    memory_space: str = "L1",
    sharded: bool = True,
) -> "ttcore.MetalLayoutAttr":
    """
    Create a MetalLayoutAttr with user-friendly parameters.

    Args:
        ctx: MLIR context
        logical_shape: List of logical tensor dimensions
        grid: Grid shape (e.g., [2, 2])
        tiled: Whether to use tiled layout (default True)
        memory_space: "L1" or "DRAM" (default "L1")
        sharded: Whether to use sharded memory layout (default True)

    Returns:
        ttcore.MetalLayoutAttr with computed device shape
    """
    print(f"[SHAPE DEBUG d2m_api.py:41] create_metal_layout called with:")
    print(f"  logical_shape = {logical_shape} (type: {type(logical_shape)})")
    print(f"  grid = {grid}")
    print(f"  tiled = {tiled}")
    from ttmlir.dialects import ttcore

    # Convert memory_space string to enum
    if memory_space == "L1":
        mem_space = ttcore.MemorySpace.DeviceL1
    elif memory_space == "DRAM":
        mem_space = ttcore.MemorySpace.DeviceDRAM
    else:
        raise ValueError(
            f"Invalid memory_space: {memory_space}. Must be 'L1' or 'DRAM'"
        )

    # Convert sharded bool to enum
    if sharded:
        memory_layout = ttcore.TensorMemoryLayout.Sharded
    else:
        memory_layout = ttcore.TensorMemoryLayout.Interleaved

    # Validate that logical dimensions are divisible by grid dimensions
    for i in range(len(logical_shape)):
        if logical_shape[i] % grid[i] != 0:
            raise ValueError(
                f"Logical dimension {i} ({logical_shape[i]}) must be evenly divisible by grid dimension {i} ({grid[i]})"
            )

    # Use the simple version without collapse intervals
    # This will create default collapse intervals based on the grid rank
    layout = ttcore.ir.MetalLayoutAttr.get(
        ctx,
        logical_shape,
        grid,
        int(ttcore.OOBVal.Undef),
        int(mem_space),
        int(ttcore.TensorMemoryLayout.Sharded),
    )

    return layout


@syntax("!tensor")
class TensorBlock:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def __add__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        return arith.addf(ast_self, rhs)

    def __sub__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        return arith.subf(ast_self, rhs)

    def __mul__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        return arith.mulf(ast_self, rhs)

    def __truediv__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        return arith.divf(ast_self, rhs)

    def __matmul__(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        lhs = ast_self
        assert isinstance(lhs.type, RankedTensorType)
        out_shape = lhs.type.shape
        out_shape[-1] = rhs.type.shape[-1]

        # Preserve the layout attribute from the left-hand side tensor
        # Create a new tensor type with the same layout as the input
        out_type = RankedTensorType.get(
            out_shape, lhs.type.element_type, lhs.type.encoding
        )
        out = d2m.empty(out_type)
        d2m.tile_matmul_block(lhs, rhs, out)
        return out

    def store(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        return d2m.store(ast_self, rhs)


@syntax("!d2m.cb")
class CircularBuffer:
    def pop(ast_self) -> TensorBlock:
        return d2m.pop(d2m.ir.CBType.cast(ast_self.type).getUnderlying(), ast_self)

    def reserve(ast_self) -> TensorBlock:
        return d2m.reserve(d2m.ir.CBType.cast(ast_self.type).getUnderlying(), ast_self)


@syntax("!d2m.mem_tx")
class MemTx:
    def wait(ast_self):
        return d2m.dma_wait(ast_self)


@syntax("dma")
def dma(src, dst, core=None, mcast=None) -> MemTx:
    src_indices = None
    dst_indices = None
    if isinstance(src, tuple):
        src, src_indices = src
    if isinstance(dst, tuple):
        dst, dst_indices = dst
    return d2m.dma(
        src,
        _asindex(src_indices),
        dst,
        _asindex(dst_indices),
        _asindex(core),
        _asindex(mcast),
    )


@syntax("!d2m.semaphore")
class Semaphore:
    def set(ast_self, value, core=None, mcast=None):
        return d2m.semaphore_set(
            ast_self, _asindex(value), _asindex(core), _asindex(mcast)
        )

    def inc(ast_self, value, core=None, mcast=None):
        return d2m.semaphore_inc(
            ast_self, _asindex(value), _asindex(core), _asindex(mcast)
        )

    def wait(ast_self, value, reset=None):
        return d2m.semaphore_wait(
            ast_self, _asindex(value), reset_value=_asindex(reset)
        )


def _collect_captures(f):
    if f.__closure__ is None:
        return {}

    def convert(name, val):
        if isinstance(val, int):
            return val
        elif isinstance(val, Stream):
            return val
        else:
            raise TypeError(f"Unhandled capture for vars of type({type(val)})")

    return {
        n: convert(n, c.cell_contents)
        for n, c in zip(f.__code__.co_freevars, f.__closure__)
    }


def _compile(
    kernel_type=None,
    verbose: bool = False,
    optimize: bool = False,
):
    def _decorator(f):
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            # Code to deal with identation issues
            source_code = _cleanup_source_code(f)

            if verbose:
                # Create easily index-able object to store source code:
                kwargs["_source_code"] = source_code.splitlines()
                kwargs["_verbose"] = True

            m = ast.parse(source_code)
            b = D2MGenericCompiler(
                f.__name__,
                kernel_type,
                _collect_captures(f),
                *args,
                **kwargs,
            )

            if verbose:
                print(ast.dump(m, indent=4) + "\n")

            b.visit(m)

            # Check if generated IR is valid
            if verbose:
                print(b.module)

            b.module.operation.verify()

            return b

        # Make the decorator apply staticmethod for class methods defined using op.py
        _wrapper._decorator_name = kernel_type + "_thread"
        if inspect.ismethod(f):
            return staticmethod(_wrapper)
        return _wrapper

    return _decorator


def compute(verbose: bool = False):
    return _compile(
        kernel_type="compute",
        verbose=verbose,
    )


def datamovement(verbose: bool = False):
    return _compile(
        kernel_type="datamovement",
        verbose=verbose,
    )


class Program:
    def __init__(self, *threads):
        self.threads = threads
        self.args = None
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self


def _affine_map_from_lambda(fn):
    class Dim:
        def __init__(self, position, name):
            self.position = position
            self.name = name

    dims = tuple(
        Dim(name, i) for name, i in enumerate(inspect.signature(fn).parameters)
    )
    num_dims = len(dims)
    results = fn(*dims)
    exprs = []
    for result in results:
        if isinstance(result, Dim):
            exprs.append(AffineDimExpr.get(result.position))
        elif isinstance(result, int):
            assert (
                result == 0
            ), "The only integer constant allowed in an indexing_map is 0"
            exprs.append(AffineConstantExpr.get(result))
        else:
            raise TypeError(
                "Unsupported indexing_map result type `{type(result)}` for result `{result}`"
            )
    num_syms = 0
    return AffineMap.get(num_dims, num_syms, exprs)


def _create_generic_func(
    ctx,
    name,
    stream_func_arg_attrs,
    grid,
    block_factors,
    indexing_maps,
    iterator_types,
    compiled_threads,
    num_outs,
    user_args,  # Original torch tensor arguments
    tiled,
    memory_space,
):
    # Flatten the block factors if need be.
    if (
        isinstance(block_factors, list)
        and len(block_factors) > 0
        and isinstance(block_factors[0], tuple)
    ):
        assert isinstance(block_factors, list)
        assert isinstance(block_factors[0], tuple)
        block_factors = [b for bs in block_factors for b in bs]

    # Some passes still rely on the compute thread being last.
    compiled_threads.sort(key=lambda ct: ct.kernel_type == "compute")

    # Create proper function argument types from original user arguments
    # instead of extracting from CircularBuffer types
    ordered_tensor_args = []
    for arg in user_args:
        shape = arg.shape
        dtype = F32Type.get(ctx)

        # Create MetalLayoutAttr for distributed tensor
        layout = create_metal_layout(ctx, shape, grid, tiled, memory_space)
        tile_shape = [32, 32] if tiled else [1, 1]

        logical_rank = len(shape)
        if len(grid) == 2 and logical_rank == 2:
            grid_shape = list(grid)
        else:
            grid_shape = list(grid) + [1] * (logical_rank - len(grid))

        typed_layout = ttcore.ir.MetalLayoutAttr.maybe_downcast(layout)
        if typed_layout is None:
            raise RuntimeError("Failed to downcast MetalLayoutAttr")
        device_shape = typed_layout.getDeviceShape(grid_shape, tile_shape)

        element_type = (
            ttcore.ir.TileType.get(ctx, 32, 32, ttcore.DataType.Float32)
            if tiled
            else dtype
        )

        tensor_type = RankedTensorType.get(device_shape, element_type, layout)
        ordered_tensor_args.append(tensor_type)

    arg_types = ordered_tensor_args
    ret_type = ordered_tensor_args[-1]
    func_entry = func.FuncOp(name=name, type=(arg_types, [ret_type]))
    func_entry.arg_attrs = stream_func_arg_attrs
    func_bb = func_entry.add_entry_block()
    with InsertionPoint(func_bb):
        inputs = func_bb.arguments[:-num_outs]
        outputs = func_bb.arguments[-num_outs:]
        threads = ArrayAttr.get(
            [
                ct.func_entry.attributes[d2m.ir.ThreadAttr.name]
                for ct in compiled_threads
            ]
        )
        generic = d2m.GenericOp(
            [ret_type],
            inputs,
            outputs,
            ttcore.ir.GridAttr.get(ctx, grid),
            block_factors,
            list(map(_affine_map_from_lambda, indexing_maps)),
            ArrayAttr.get(
                list(
                    ttcore.ir.IteratorTypeAttr.get(
                        ctx, ttcore.IteratorType[i.title()].value
                    )
                    for i in iterator_types
                )
            ),
            threads,
            len(compiled_threads),
        )
        for compiled_thread, generic_region in zip(compiled_threads, generic.regions):
            compiled_thread.func_entry.entry_block.append_to(generic_region)
            if generic_region.blocks[0].operations[-1].name == "func.return":
                generic_region.blocks[0].operations[-1].erase()
        func.ReturnOp(generic.results)


def _copy_symbol_table_globals(module_symbol_table, compiled_threads, f_params):
    f_params_list = list(f_params.keys())
    for ct in compiled_threads:
        for op in ct.module.body:
            if "sym_name" not in op.attributes:
                continue
            sym_name = op.attributes["sym_name"]
            if sym_name.value in f_params and sym_name.value in ct.module_symbol_table:
                clone = op.clone()
                clone.index = IntegerAttr.get(
                    IntegerType.get_signed(32), f_params_list.index(sym_name.value)
                )
                module_symbol_table.insert(clone)


def to_data_type(dtype):
    if dtype == torch.float32:
        return runtime.DataType.Float32
    if dtype == torch.float16:
        return runtime.DataType.Float16
    if dtype == torch.bfloat16:
        return runtime.DataType.BFloat16
    if dtype == torch.uint32:
        return runtime.DataType.UInt32
    if dtype == torch.uint16:
        return runtime.DataType.UInt16
    if dtype == torch.uint8:
        return runtime.DataType.UInt8
    if dtype == torch.int32:
        return runtime.DataType.Int32
    # Data types which are unsupported on ttnn
    if dtype == torch.float64:
        return runtime.DataType.Float64
    if dtype == torch.int64:
        return runtime.DataType.Int64
    if dtype == torch.uint64:
        return runtime.DataType.UInt64
    if dtype == torch.int16:
        return runtime.DataType.Int16
    if dtype == torch.int8:
        return runtime.DataType.Int8
    if dtype == torch.bool:
        return runtime.DataType.Bool
    raise ValueError(f"Torch dtype: {dtype} has no runtime DataType equivalent")


def from_data_type(dtype):
    if dtype == "Float32":
        return torch.float32
    if dtype == "Float16":
        return torch.float16
    if dtype == "BFloat16":
        return torch.bfloat16
    if dtype == "UInt32":
        return torch.uint32
    if dtype == "UInt16":
        return torch.uint16
    if dtype == "UInt8":
        return torch.uint8
    if dtype == "Int32":
        return torch.int32
    # Data types which are unsupported on ttnn
    if dtype == "Float64":
        return torch.float64
    if dtype == "Int64":
        return torch.int64
    if dtype == "UInt64":
        return torch.uint64
    if dtype == "Int16":
        return torch.int16
    if dtype == "Int8":
        return torch.int8
    if dtype == "Bool":
        return torch.bool

    raise ValueError(f"unsupported dtype: {dtype}")


_g_current_system_desc = None


def pykernel_gen(
    grid=None,
    block_factors=None,
    indexing_maps=None,
    iterator_types=None,
    num_outs=1,
    kernel_source_dir=None,
    kernel_source_mode=None,  # Literal["store", "load"]
    memory_space="L1",  # "L1" or "DRAM"
    tiled=True,  # bool for tiled layout
):
    assert grid is not None
    assert num_outs == 1
    assert memory_space in [
        "L1",
        "DRAM",
    ], f"memory_space must be 'L1' or 'DRAM', got '{memory_space}'"
    assert isinstance(tiled, bool), f"tiled must be a boolean, got {type(tiled)}"
    assert (iterator_types is None) or (
        indexing_maps is not None
    ), "if iterator_types is set, indexing_types must also be set"

    global _g_current_system_desc
    if _g_current_system_desc is None:
        _g_current_system_desc = os.environ.get("SYSTEM_DESC_PATH", None)
    if _g_current_system_desc is None:
        system_desc = runtime.get_current_system_desc()
        _g_current_system_desc = "current.ttsys"
        system_desc.store(_g_current_system_desc)

    if indexing_maps is None:
        indexing_maps = []

    if indexing_maps:
        for indexing_map in indexing_maps:
            num_dims = list(tuple(inspect.signature(indexing_map).parameters))
            if iterator_types is not None:
                assert num_dims == len(iterator_types)
            if block_factors is None:
                block_factors = [1] * len(num_dims)
            assert len(block_factors) == num_dims

    if iterator_types is None:
        iterator_types = []

    def _decorator(f):
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            nonlocal grid
            nonlocal block_factors
            nonlocal indexing_maps
            nonlocal iterator_types
            nonlocal kernel_source_dir

            f_params = inspect.signature(f).parameters

            for param_name, arg in zip(f_params, args):
                arg._global_name = param_name

            if callable(grid):
                grid = grid(*args, **kwargs)

            if callable(block_factors):
                block_factors = block_factors(*args, **kwargs)

            if block_factors is None:
                block_factors = [1] * len(grid)

            inject_kwargs = [
                ("block_factors", block_factors),
                ("grid", grid),
                ("memory_space", memory_space),
                ("tiled", tiled),
            ]
            for injected_kwarg, val in inject_kwargs:
                if injected_kwarg in f_params:
                    kwargs[injected_kwarg] = val

            program = f(*args, **kwargs)
            assert isinstance(program, Program)

            # Inject decorator parameters into program.kwargs so threads receive them
            # Merge with user-provided kwargs (user kwargs take precedence)
            injected_program_kwargs = {
                "grid": grid,
                "memory_space": memory_space,
                "tiled": tiled,
            }
            program.kwargs = {**injected_program_kwargs, **program.kwargs}

            ctx = Context()
            loc = Location.unknown(ctx)
            with ctx, loc:
                compiled_threads = []
                for compile_thread in program.threads:
                    compiled_threads.append(
                        compile_thread(*program.args, **program.kwargs)
                    )

                module = Module.create(loc)

                # Join all compiled_threads' symbol tables into top level.
                module_symbol_table = SymbolTable(module.operation)
                with InsertionPoint.at_block_begin(module.body):
                    _copy_symbol_table_globals(
                        module_symbol_table, compiled_threads, f_params
                    )

                streams = set().union(*[ct.streams for ct in compiled_threads])
                positional_arg_names = list(f_params.keys())[: len(args)]
                stream_func_arg_attrs = [
                    DictAttr.get({"d2m.stream": BoolAttr.get(p in streams)})
                    for p in positional_arg_names
                ]
                assert (
                    positional_arg_names[-num_outs] not in streams
                ), "Output streaming not supported"

                with InsertionPoint(module.body):
                    _create_generic_func(
                        ctx,
                        f.__name__,
                        stream_func_arg_attrs,
                        grid,
                        block_factors,
                        indexing_maps,
                        iterator_types,
                        compiled_threads,
                        num_outs,
                        args,  # Pass original user arguments
                        tiled,
                        memory_space,
                    )

                print(module)
                with open("tmp.mlir", "w") as fd:
                    print(module, file=fd)

                print_ir = True
                device_register_options = f"system-desc-path={_g_current_system_desc}"
                verify = True
                use_tile_matmul = False
                pipeline = f"d2m-generic-replace-globals,ttir-to-ttmetal-pipeline{{use-tile-matmul={1 if use_tile_matmul else 0}}}"

                register_device = "ttcore-register-device"
                if device_register_options:
                    register_device = f"{register_device}{{{device_register_options}}}"

                pipeline_str = (
                    f"builtin.module({','.join([register_device, pipeline])})"
                )
                pm = PassManager.parse(pipeline_str)
                pm.enable_verifier(verify)

                # Enable pass tracking for crash diagnostics
                try:
                    from ttmlir._mlir_libs._ttmlir import enable_pretty_stack_traces

                    enable_pretty_stack_traces(pm._CAPIPtr)
                except Exception as e:
                    print(f"Warning: Could not enable pass tracking: {e}")

                print("Running custom pipeline:", pm)
                if print_ir:
                    print_ir_path = print_ir if isinstance(print_ir, str) else None
                    ctx.enable_multithreading(False)
                    pm.enable_ir_printing(
                        # tree_printing_dir_path=print_ir_path,
                        print_after_all=True,
                        print_before_all=True,
                        print_after_failure=True,
                        enable_debug_info=True,
                    )
                pm.run(module.operation)

                print(module)
                bin = ttmetal_to_flatbuffer_bin(module)

                print("RUNTIME DISABLED")
                return

                if runtime is None or binary is None:
                    print("Warning: runtime not enabled, returning compiled object")
                    return bin

                #
                # Runtime
                #
                fbb = binary.load_binary_from_capsule(bin)
                program_index = 0
                device_options = runtime.MeshDeviceOptions()
                device_options.mesh_shape = fbb.get_program_mesh_shape(program_index)
                runtime.set_compatible_device_runtime(fbb)

                if kernel_source_dir is None:
                    kernel_source_dir = f".pykernel_gen/{f.__name__}/"
                if kernel_source_mode == "store":
                    os.makedirs(kernel_source_dir, exist_ok=True)

                debug_env = runtime.DebugEnv.get(
                    kernel_source_mode == "store",  # dump_kernels_to_disk
                    kernel_source_mode == "load",  # load_kernels_from_disk
                    True,  # use_loc_for_kernel_name
                    kernel_source_dir,
                    False,  # disable_device_address_validation
                    False,  # blocking_cq
                )
                print(f"setting tt runtime debug env={debug_env}")

                inputs = []
                for tensor in args:
                    inputs.append(
                        runtime.create_borrowed_host_tensor(
                            tensor.data_ptr(),
                            list(tensor.shape),
                            list(tensor.stride()),
                            tensor.element_size(),
                            to_data_type(tensor.dtype),
                        )
                    )

                outputs = []
                outputs_torch = args[-num_outs:]
                output_descs = json.loads(
                    fbb.get_program_outputs_as_json(program_index)
                )
                for tensor in outputs_torch:
                    outputs.append(
                        runtime.create_borrowed_host_tensor(
                            tensor.data_ptr(),
                            list(tensor.shape),
                            list(tensor.stride()),
                            tensor.element_size(),
                            to_data_type(tensor.dtype),
                        )
                    )

                device = runtime.open_mesh_device(device_options)
                runtime_outputs = runtime.submit(device, fbb, program_index, inputs)
                runtime.wait(runtime_outputs)
                for i, runtime_output_tensor in enumerate(runtime_outputs):
                    output_host = runtime.to_host(runtime_output_tensor, untilize=True)[
                        0
                    ]
                    runtime.memcpy(outputs[i], output_host)
                    runtime.deallocate_tensor(runtime_output_tensor, force=True)
                runtime.close_mesh_device(device)
                return outputs_torch[i]

        return _wrapper

    return _decorator


matmul_template = {
    "grid": (1, 1),  # | lambda | "auto" | "automatic"
    "block_factors": [1, 1, 1],  # | lambda | "auto" | "automatic"
    "indexing_maps": [
        lambda m, n, k: (m, k),
        lambda m, n, k: (k, n),
        lambda m, n, k: (m, n),
    ],
    "iterator_types": [
        "parallel",
        "parallel",
        "reduction",
    ],
}


def matmul_fused_template(args=3):
    assert args >= 3
    return {
        "grid": (1, 1),  # | lambda | "auto" | "automatic"
        "block_factors": [1, 1, 1],  # | lambda | "auto" | "automatic"
        "indexing_maps": [
            lambda m, n, k: (m, k),
            lambda m, n, k: (k, n),
            lambda m, n, k: (m, n),
        ]
        + [lambda m, n, k: (m, n)] * (args - 3),
        "iterator_types": [
            "parallel",
            "parallel",
            "reduction",
        ],
    }


eltwise_template = {
    "grid": (1, 1),  # | lambda | "auto" | "automatic"
    "block_factors": [1, 1],  # | lambda | "auto" | "automatic"
    "indexing_maps": [
        lambda m, n: (m, n),
        lambda m, n: (m, n),
        lambda m, n: (m, n),
    ],
    "iterator_types": [
        "parallel",
        "parallel",
    ],
}


def eltwise_fused_template(args=1):
    assert args >= 1
    return {
        "grid": (1, 1),  # | lambda | "auto" | "automatic"
        "block_factors": [1, 1],  # | lambda | "auto" | "automatic"
        "indexing_maps": [lambda m, n: (m, n)] * args,
        "iterator_types": [
            "parallel",
            "parallel",
        ],
    }


explicit_template = {
    "grid": (1, 1),  # | lambda | "auto" | "automatic"
    "block_factors": None,
    "indexing_maps": None,
    "iterator_types": None,
}
