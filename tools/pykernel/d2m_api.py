# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import inspect
import functools
import json
import os

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from _ttmlir_runtime import runtime, binary
except (ModuleNotFoundError, ImportError):
    runtime = None
    binary = None

from ttmlir.ir import *
from ttmlir.passmanager import PassManager
from ttmlir.dialects import (
    ttcore,
    d2m,
    func,
    arith,
    tensor,
)
from ttmlir.passes import ttmetal_to_flatbuffer_bin

from ._src.utils import _discover_dialect_ops, _asindex, _cleanup_source_code
from ._src.d2m_ast import D2MGenericCompiler, syntax, Stream, TensorLayout


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
        out = d2m.empty(RankedTensorType.get(out_shape, lhs.type.element_type))
        d2m.tile_matmul_block(lhs, rhs, out)
        return out

    def store(ast_self: TensorBlock, rhs: TensorBlock) -> TensorBlock:
        return d2m.store(ast_self, rhs)


@syntax("remote_load")
def remote_load(
    src, indices, mcast_start_index=None, mcast_shape=None, mcast_dims=None
) -> MemTx:
    if mcast_dims is not None:
        if isinstance(mcast_dims, tuple):
            mcast_dims = list(mcast_dims)
        if not isinstance(mcast_dims, list):
            if isinstance(mcast_dims, int):
                mcast_dims = arith.constant(IndexType.get(src.context), mcast_dims)
            mcast_dims = [mcast_dims]
    dst_type = RankedTensorType.get(
        src.type.shape[len(indices) :], src.type.element_type
    )
    dst = d2m.empty(dst_type)
    return d2m.remote_load(
        dst_type,
        src,
        indices,
        mcast_start_index=mcast_start_index,
        mcast_shape=mcast_shape,
        mcast_dims=mcast_dims,
        local_buffer=dst,
    )


@syntax("remote_store")
def remote_store(dst, indices, src):
    return d2m.remote_store(
        dst.type,
        dst,
        indices,
        local_buffer=src,
    )


@syntax(
    "core_index",
    args_as_attr=[
        lambda node: IntegerAttr.get(IntegerType.get_signless(64), node.value)
    ],
)
def core_index(index):
    return d2m.core_index(index)


# --- Tile Unary Compute Ops ---


@syntax("tile_recip")
def tile_recip(input):
    return d2m.tile_recip(input.type, input)


@syntax("tile_exp")
def tile_exp(input):
    return d2m.tile_exp(input.type, input)


@syntax("tile_log")
def tile_log(input):
    return d2m.tile_log(input.type, input)


@syntax("tile_negative")
def tile_negative(input):
    return d2m.tile_negative(input.type, input)


@syntax("tile_cos")
def tile_cos(input):
    return d2m.tile_cos(input.type, input)


@syntax("tile_acos")
def tile_acos(input):
    return d2m.tile_acos(input.type, input)


@syntax("tile_sin")
def tile_sin(input):
    return d2m.tile_sin(input.type, input)


@syntax("tile_asin")
def tile_asin(input):
    return d2m.tile_asin(input.type, input)


@syntax("tile_tan")
def tile_tan(input):
    return d2m.tile_tan(input.type, input)


@syntax("tile_atan")
def tile_atan(input):
    return d2m.tile_atan(input.type, input)


@syntax("tile_tanh")
def tile_tanh(input):
    return d2m.tile_tanh(input.type, input)


@syntax("tile_sqrt")
def tile_sqrt(input):
    return d2m.tile_sqrt(input.type, input)


@syntax("tile_rsqrt")
def tile_rsqrt(input):
    return d2m.tile_rsqrt(input.type, input)


@syntax("tile_sigmoid")
def tile_sigmoid(input):
    return d2m.tile_sigmoid(input.type, input)


@syntax("tile_hardsigmoid")
def tile_hardsigmoid(input):
    return d2m.tile_hardsigmoid(input.type, input)


@syntax("tile_silu")
def tile_silu(input):
    return d2m.tile_silu(input.type, input)


@syntax("tile_relu")
def tile_relu(input):
    return d2m.tile_relu(input.type, input)


@syntax("tile_gelu")
def tile_gelu(input):
    return d2m.tile_gelu(input.type, input)


@syntax("tile_erf")
def tile_erf(input):
    return d2m.tile_erf(input.type, input)


@syntax("tile_erfc")
def tile_erfc(input):
    return d2m.tile_erfc(input.type, input)


@syntax("tile_sign")
def tile_sign(input):
    return d2m.tile_sign(input.type, input)


@syntax("tile_ceil")
def tile_ceil(input):
    return d2m.tile_ceil(input.type, input)


@syntax("tile_floor")
def tile_floor(input):
    return d2m.tile_floor(input.type, input)


@syntax("tile_abs")
def tile_abs(input):
    return d2m.tile_abs(input.type, input)


@syntax("tile_bitwise_not")
def tile_bitwise_not(input):
    return d2m.tile_bitwise_not(input.type, input)


@syntax("tile_logical_not")
def tile_logical_not(input):
    return d2m.tile_logical_not(input.type, input)


@syntax("tile_eqz")
def tile_eqz(input):
    return d2m.tile_eqz(input.type, input)


@syntax("tile_nez")
def tile_nez(input):
    return d2m.tile_nez(input.type, input)


@syntax("tile_gtz")
def tile_gtz(input):
    return d2m.tile_gtz(input.type, input)


@syntax("tile_gez")
def tile_gez(input):
    return d2m.tile_gez(input.type, input)


@syntax("tile_ltz")
def tile_ltz(input):
    return d2m.tile_ltz(input.type, input)


@syntax("tile_lez")
def tile_lez(input):
    return d2m.tile_lez(input.type, input)


@syntax("tile_typecast")
def tile_typecast(input, result_type):
    return d2m.tile_typecast(result_type, input)


@syntax("tile_transpose")
def tile_transpose(input):
    return d2m.tile_transpose(input.type, input)


# --- Tile Binary Compute Ops ---


@syntax("tile_add")
def tile_add(lhs, rhs):
    return d2m.tile_add(lhs.type, lhs, rhs)


@syntax("tile_sub")
def tile_sub(lhs, rhs):
    return d2m.tile_sub(lhs.type, lhs, rhs)


@syntax("tile_mul")
def tile_mul(lhs, rhs):
    return d2m.tile_mul(lhs.type, lhs, rhs)


@syntax("tile_div")
def tile_div(lhs, rhs):
    return d2m.tile_div(lhs.type, lhs, rhs)


@syntax("tile_pow")
def tile_pow(lhs, rhs):
    return d2m.tile_pow(lhs.type, lhs, rhs)


@syntax("tile_maximum")
def tile_maximum(lhs, rhs):
    return d2m.tile_maximum(lhs.type, lhs, rhs)


@syntax("tile_minimum")
def tile_minimum(lhs, rhs):
    return d2m.tile_minimum(lhs.type, lhs, rhs)


@syntax("tile_bitwise_and")
def tile_bitwise_and(lhs, rhs):
    return d2m.tile_bitwise_and(lhs.type, lhs, rhs)


@syntax("tile_bitwise_or")
def tile_bitwise_or(lhs, rhs):
    return d2m.tile_bitwise_or(lhs.type, lhs, rhs)


@syntax("tile_bitwise_xor")
def tile_bitwise_xor(lhs, rhs):
    return d2m.tile_bitwise_xor(lhs.type, lhs, rhs)


# --- Tile Ternary / Special Compute Ops ---


@syntax("tile_where")
def tile_where(condition, true_value, false_value):
    return d2m.tile_where(true_value.type, condition, true_value, false_value)


@syntax("tile_matmul")
def tile_matmul(a, b, c):
    return d2m.tile_matmul(c.type, a, b, c)


@syntax("tile_clamp_scalar")
def tile_clamp_scalar(input, min, max):
    return d2m.tile_clamp_scalar(input.type, input, min, max)


@syntax("tile_reduce_sum")
def tile_reduce_sum(a, b, c, reduce_dim):
    return d2m.tile_reduce_sum(c.type, a, b, c, reduce_dim)


@syntax("tile_reduce_max")
def tile_reduce_max(a, b, c, reduce_dim):
    return d2m.tile_reduce_max(c.type, a, b, c, reduce_dim)


@syntax("tile_reduce_mean")
def tile_reduce_mean(a, b, c, reduce_dim):
    return d2m.tile_reduce_mean(c.type, a, b, c, reduce_dim)


@syntax("tile_bcast")
def tile_bcast(input, bcast_type):
    return d2m.tile_bcast(input.type, input, bcast_type)


@syntax("fill_tile")
def fill_tile(value, result_type):
    return d2m.fill_tile(result_type, value)


@syntax("tile_tilize_block")
def tile_tilize_block(input, output):
    return d2m.tile_tilize_block(output.type, input, output)


@syntax("tile_untilize_block")
def tile_untilize_block(input, output):
    return d2m.tile_untilize_block(output.type, input, output)


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
            if True or verbose:
                print(b.module)

            b.module.operation.verify()

            return b

        # Make the decorator apply staticmethod for class methods defined using op.py
        _wrapper._decorator_name = kernel_type + "_thread"
        if inspect.ismethod(f):
            return staticmethod(_wrapper)
        return _wrapper

    return _decorator


def unified(verbose: bool = False):
    return _compile(
        kernel_type="unified",
        verbose=verbose,
    )


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
    operands,
    grid,
    block_factors,
    indexing_maps,
    iterator_types,
    compiled_threads,
    num_outs,
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

    tensor_section = True
    ordered_tensor_args = []
    func_entry_types = []
    for i, t in enumerate(compiled_threads[0].func_entry.arguments.types):
        if isinstance(t, RankedTensorType):
            assert tensor_section
            host_type = operands[i].build_host_tensor_type(ctx)
            func_entry_types.append(operands[i].build_host_tensor_type(ctx))
            ordered_tensor_args.append(operands[i])
        else:
            tensor_section = False
            func_entry_types.append(t)
    ret_type = ordered_tensor_args[-1].build_host_tensor_type(ctx)
    func_entry = func.FuncOp(name=name, type=(func_entry_types, [ret_type]))
    func_bb = func_entry.add_entry_block()
    with InsertionPoint(func_bb):
        func_inputs = func_bb.arguments[: len(ordered_tensor_args) - 1]
        inputs = []
        for func_input, operand in zip(func_inputs, operands):
            inputs.append(operand.build_to_device(ctx, func_input))

        output = d2m.empty(ordered_tensor_args[-1].build_device_tensor_type(ctx))
        output = ordered_tensor_args[-1].build_blocked_view(ctx, output)
        output_type = ordered_tensor_args[-1].build_device_tensor_type(
            ctx, blocked=True
        )

        additional_args = func_bb.arguments[len(ordered_tensor_args) :]
        all_operands = inputs + [output] + list(additional_args)
        threads = ArrayAttr.get(
            [
                ct.func_entry.attributes[d2m.ir.ThreadAttr.name]
                for ct in compiled_threads
            ]
        )
        generic = d2m.GenericOp(
            [output_type],
            inputs,
            [output],
            additional_args,
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
            for i, arg in enumerate(generic_region.blocks[0].arguments):
                arg.replace_all_uses_with(all_operands[i])
            for i in range(len(generic_region.blocks[0].arguments)):
                generic_region.blocks[0].erase_argument(0)

        result = ordered_tensor_args[-1].build_device_view(ctx, generic.results[0])
        result = ordered_tensor_args[-1].build_from_device(ctx, result)
        func.ReturnOp([result])


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

_g_supported_arg_types = {TensorLayout, int}


def d2m_jit(
    num_outs=1,
    kernel_source_dir=None,
    kernel_source_mode=None,  # Literal["store", "load"]
):
    assert num_outs == 1

    global _g_current_system_desc
    if _g_current_system_desc is None:
        _g_current_system_desc = os.environ.get("SYSTEM_DESC_PATH", None)
    if _g_current_system_desc is None and runtime is not None:
        system_desc = runtime.get_current_system_desc()
        _g_current_system_desc = "current.ttsys"
        system_desc.store(_g_current_system_desc)

    def _decorator(f):
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            nonlocal kernel_source_dir

            assert (
                "grid" in kwargs
            ), "grid must be specified as a keyword argument to the d2m_jit decorated function"
            grid = kwargs.pop("grid")

            f_params = inspect.signature(f).parameters

            operands = []
            for arg in args:
                if type(arg) not in _g_supported_arg_types:
                    raise TypeError(
                        f"Unsupported argument type {type(arg)} for argument {arg}. Supported types are {_g_supported_arg_types}"
                    )
                operands.append(arg)

            for kwarg in kwargs.values():
                if type(kwarg) not in _g_supported_arg_types:
                    raise TypeError(
                        f"Unsupported argument type {type(kwarg)} for argument {kwarg}. Supported types are {_g_supported_arg_types}"
                    )
                operands.append(kwarg)

            ctx = Context()
            loc = Location.unknown(ctx)
            with ctx, loc:
                compiled_threads = [unified(verbose=False)(f)(*args, **kwargs)]

                module = Module.create(loc)

                with InsertionPoint(module.body):
                    _create_generic_func(
                        ctx,
                        f.__name__,
                        operands,
                        grid,  # grid,
                        [],  # block_factors,
                        [],  # indexing_maps,
                        [],  # iterator_types,
                        compiled_threads,
                        num_outs,
                    )

                print_ir = True
                device_register_options = (
                    f"system-desc-path={_g_current_system_desc}"
                    if _g_current_system_desc is not None
                    else ""
                )
                verify = True
                use_tile_matmul = False
                use_tensor_accessor_dma = True
                pipeline = ",".join(
                    [
                        # FE Passes For prep
                        "convert-elementwise-to-linalg",
                        "arith-to-d2m-tile-ops",
                        "canonicalize",
                        "ttir-to-d2m",  # Only needed for tile_matmul_block promotion
                        "d2m-lower-to-layout",
                        f"ttir-to-ttmetal-me-pipeline{{use-tile-matmul={1 if use_tile_matmul else 0} use-tensor-accessor-dma={1 if use_tensor_accessor_dma else 0}}}",
                        f"ttir-to-ttmetal-be-pipeline{{use-tile-matmul={1 if use_tile_matmul else 0} use-tensor-accessor-dma={1 if use_tensor_accessor_dma else 0}}}",
                        #f"ttir-to-ttmetal-me-pipeline{{use-tile-matmul={1 if use_tile_matmul else 0}}}",
                        #f"ttir-to-ttmetal-be-pipeline{{use-tile-matmul={1 if use_tile_matmul else 0}}}",
                    ]
                )

                register_device = "ttcore-register-device"
                if device_register_options:
                    register_device = f"{register_device}{{{device_register_options}}}"

                pipeline_str = (
                    f"builtin.module({','.join([register_device, pipeline])})"
                )
                pm = PassManager.parse(pipeline_str)
                pm.enable_verifier(verify)
                print("Running custom pipeline:", pm)
                if print_ir:
                    print_ir_path = print_ir if isinstance(print_ir, str) else None
                    ctx.enable_multithreading(False)
                    pm.enable_ir_printing(
                        # tree_printing_dir_path=print_ir_path,
                        print_after_all=True,
                        # print_before_all=True,
                        # print_after_failure=True,
                        enable_debug_info=True,
                    )
                pm.run(module.operation)

                print(module)
                bin = ttmetal_to_flatbuffer_bin(module)

                # print("RUNTIME DISABLED")
                # return

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
                    kernel_source_dir = f".d2m_jit/{f.__name__}/"
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
                output_tensor_idx = 0
                for i, arg in enumerate(args):
                    if isinstance(arg, TensorLayout):
                        tensor = arg.tensor
                        inputs.append(
                            runtime.create_borrowed_host_tensor(
                                tensor.data_ptr(),
                                list(tensor.shape),
                                list(tensor.stride()),
                                tensor.element_size(),
                                to_data_type(tensor.dtype),
                            )
                        )
                        # Output is always last
                        output_tensor_idx = i
                    else:
                        inputs.append(runtime.create_scalar_tensor(arg))

                outputs = []
                outputs_torch = [args[output_tensor_idx]]
                output_descs = json.loads(
                    fbb.get_program_outputs_as_json(program_index)
                )
                for layout in outputs_torch:
                    tensor = layout.tensor
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
