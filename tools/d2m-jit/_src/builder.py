# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Lazy-tensor builder for d2m_jit.

Maintains a process-level singleton that accumulates MLIR ops as the user
calls `to_layout / empty / view_layout`. `to_host(*lts)` closes the open
host function with returns, runs the d2m -> ttmetal pipeline, executes the
resulting binary, copies outputs back into torch tensors, and resets the
builder.

LazyTensor is a thin wrapper around an `ir.Value` plus a `Layout`; it has
no Python-side graph — the MLIR module IS the graph.
"""

import ast as _ast
import functools
import inspect
import os
from typing import Optional

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
from ttmlir.dialects import d2m, func, arith, ttcore
from ttmlir.passes import ttmetal_to_flatbuffer_bin

from .ast import D2MCompiler
from .config import config
from .tensor_layout import Layout
from .utils import _cleanup_source_code


# Reverse of ttcore.DataType for picking output torch dtypes.
_TTCORE_TO_TORCH = None  # lazy-init since torch may be missing


def _ttcore_to_torch_dtype(dt):
    global _TTCORE_TO_TORCH
    if _TTCORE_TO_TORCH is None:
        if torch is None:
            raise RuntimeError("torch not available")
        _TTCORE_TO_TORCH = {
            ttcore.DataType.Float32: torch.float32,
            ttcore.DataType.Float16: torch.float16,
            ttcore.DataType.BFloat16: torch.bfloat16,
        }
    if dt not in _TTCORE_TO_TORCH:
        raise ValueError(f"No torch dtype for ttcore.DataType {dt}")
    return _TTCORE_TO_TORCH[dt]


# --- Runtime dtype mapping ---------------------------------------------------


def _to_runtime_data_type(dtype):
    if torch is None or runtime is None:
        raise RuntimeError("torch/runtime not available")
    mapping = {
        torch.float32: runtime.DataType.Float32,
        torch.float16: runtime.DataType.Float16,
        torch.bfloat16: runtime.DataType.BFloat16,
        torch.uint32: runtime.DataType.UInt32,
        torch.uint16: runtime.DataType.UInt16,
        torch.uint8: runtime.DataType.UInt8,
        torch.int32: runtime.DataType.Int32,
        torch.float64: runtime.DataType.Float64,
        torch.int64: runtime.DataType.Int64,
        torch.int16: runtime.DataType.Int16,
        torch.int8: runtime.DataType.Int8,
        torch.bool: runtime.DataType.Bool,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported torch dtype {dtype}")
    return mapping[dtype]


# --- Builder singleton -------------------------------------------------------


_g_system_desc_path = None


def _get_system_desc_path():
    """Resolve the system descriptor used by ttcore-register-device.

    Cached after the first call. Looks for SYSTEM_DESC_PATH env first, then
    queries the runtime and stores a `current.ttsys` file in the CWD.
    """
    global _g_system_desc_path
    if _g_system_desc_path is not None:
        return _g_system_desc_path
    env = os.environ.get("SYSTEM_DESC_PATH")
    if env:
        _g_system_desc_path = env
        return _g_system_desc_path
    if runtime is not None:
        sd = runtime.get_current_system_desc()
        _g_system_desc_path = "current.ttsys"
        sd.store(_g_system_desc_path)
    return _g_system_desc_path


_PIPELINE = ",".join(
    [
        "convert-elementwise-to-linalg",
        "arith-to-d2m-tile-ops",
        "canonicalize",
        "ttir-to-d2m",
        "d2m-lower-to-layout",
        "canonicalize",
        "ttir-bufferization-pipeline",
        "d2m-insert-scratch-buffers",
        "d2m-generic-apply-interchange",
        "d2m-generate-outer-loops",
        "d2m-allocate",
        "d2m-lower-multicast-loads",
        "d2m-generic-lower-to-explicit-form",
        "canonicalize",
        "d2m-be-pipeline{use-tile-matmul=0}",
        "d2m-to-ttkernel-pipeline",
        "d2m-to-ttmetal-pipeline",
    ]
)


class _Builder:
    """Process-level singleton accumulating MLIR ops for the current lazy graph."""

    _instance: Optional["_Builder"] = None
    _next_generation: int = 1  # monotonic; id() can be reused after GC

    @classmethod
    def get(cls) -> "_Builder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def __init__(self):
        # Unique non-reusable id; LazyTensors compare against this to detect
        # a post-to_host reset.
        self.generation = _Builder._next_generation
        _Builder._next_generation += 1
        self.ctx = Context()
        self.loc = Location.unknown(self.ctx)
        self.module = Module.create(self.loc)
        with self.ctx, self.loc, InsertionPoint(self.module.body):
            self.func_op = func.FuncOp("main", FunctionType.get([], []))
            self.entry_block = self.func_op.add_entry_block()
        # Insertion point at the end of the entry block. Stays here until
        # to_host() emits the terminator.
        self.insert_point = InsertionPoint(self.entry_block)
        # Parallel arrays: MLIR arg types and the torch tensor that backs each.
        self._input_types: list = []
        self._input_tensors: list = []

    def _refresh_function_type(self, results=None):
        with self.ctx, self.loc:
            ft = FunctionType.get(self._input_types, results or [])
            self.func_op.attributes["function_type"] = TypeAttr.get(ft)

    def add_host_input(self, layout: Layout, host_tensor):
        """Append a host-typed func arg and return its BlockArgument."""
        host_ty = layout.build_host_tensor_type(self.ctx)
        bb_arg = self.entry_block.add_argument(host_ty, self.loc)
        self._input_types.append(host_ty)
        self._input_tensors.append(host_tensor)
        self._refresh_function_type()
        return bb_arg

    def add_scalar_input(self, value: int):
        """Append an index-typed func arg backing a Python int and return its
        BlockArgument. Scalars become GenericOp additionalArgs and need to be
        block-arg sourced (not host-scope constants) to satisfy region
        isolation."""
        with self.ctx, self.loc:
            idx_ty = IndexType.get(self.ctx)
        bb_arg = self.entry_block.add_argument(idx_ty, self.loc)
        self._input_types.append(idx_ty)
        self._input_tensors.append(int(value))
        self._refresh_function_type()
        return bb_arg

    @property
    def host_tensors(self):
        return list(self._input_tensors)


# --- LazyTensor --------------------------------------------------------------


class LazyTensor:
    """Host-side handle for a value being built into the lazy graph.

    Holds either:
      - an `ir.Value` at host-func scope (in the current builder generation), or
      - a materialised torch.Tensor (after to_host).
    """

    __slots__ = ("layout", "value", "generation", "materialized")

    def __init__(
        self, layout: Layout, value, generation, materialized=None
    ):
        self.layout = layout
        self.value = value
        self.generation = generation
        self.materialized = materialized

    def to_host(self):
        return to_host(self)[0]

    def _resolve(self) -> "LazyTensor":
        """Return a LazyTensor in the current builder's generation.

        - Same generation: return self.
        - Materialised (different generation): auto-re-enter via to_layout.
        - Stale (different generation, not materialised): raise.
        """
        b = _Builder.get()
        if self.generation == b.generation:
            return self
        if self.materialized is not None:
            return to_layout(self.materialized, layout=self.layout)
        raise RuntimeError(
            "Stale LazyTensor: produced by a prior builder that was reset "
            "by to_host(). Re-materialise its source or include it in the "
            "to_host() call before reset."
        )


# --- Public constructors -----------------------------------------------------


def to_layout(input_, layout: Layout) -> LazyTensor:
    """Convert `input_` to a device tensor at `layout`.

    Polymorphic on the input:
      - host torch.Tensor: appends a host-typed func arg and emits a
        host->device d2m.ToLayoutOp.
      - LazyTensor:        emits a device->device d2m.ToLayoutOp between
        the source's layout and `layout` (different grids/tile-ness/etc).

    Returns a LazyTensor at the layout's *blocked* grid.
    """
    b = _Builder.get()

    if isinstance(input_, LazyTensor):
        src = input_._resolve()
        assert list(src.layout.logical_shape) == list(layout.logical_shape), (
            f"to_layout shape mismatch: src {src.layout.logical_shape} "
            f"vs target {layout.logical_shape}"
        )
        with b.ctx, b.loc, b.insert_point:
            # Step back from src's blocked grid to its unblocked form, then
            # ToLayoutOp into the target's unblocked form, then re-view to
            # the target's blocked grid.
            src_val = src.layout.build_device_view(b.ctx, src.value)
            dst_unblocked_ty = layout.build_device_tensor_type(b.ctx, blocked=False)
            dst_empty = d2m.empty(dst_unblocked_ty)
            converted = d2m.ToLayoutOp(
                [dst_unblocked_ty], src_val, dst_empty
            ).result
            val = layout.build_blocked_view(b.ctx, converted)
        return LazyTensor(layout, val, b.generation)

    if torch is not None and isinstance(input_, torch.Tensor):
        assert list(input_.shape) == list(layout.logical_shape), (
            f"to_layout shape mismatch: tensor {list(input_.shape)} "
            f"vs layout {layout.logical_shape}"
        )
        with b.ctx, b.loc, b.insert_point:
            bb_arg = b.add_host_input(layout, input_)
            dev = layout.build_to_device(b.ctx, bb_arg)
        return LazyTensor(layout, dev, b.generation)

    raise TypeError(
        f"to_layout expected a torch.Tensor or LazyTensor, got {type(input_).__name__}"
    )


def tilize(lt: LazyTensor) -> LazyTensor:
    """Convert a device LazyTensor to a tile-typed (`tiled=True`) layout.

    The target layout is the source's layout with `tiled` set to True;
    all other fields (shape, dtype, block_shape, grid_shape, mem_space,
    collapse) are preserved.
    """
    if not isinstance(lt, LazyTensor):
        raise TypeError(f"tilize expected a LazyTensor, got {type(lt).__name__}")
    return to_layout(lt, lt.layout.replace(tiled=True))


def untilize(lt: LazyTensor) -> LazyTensor:
    """Convert a device LazyTensor to a row-major (`tiled=False`) layout.

    The target layout is the source's layout with `tiled` set to False;
    all other fields are preserved.
    """
    if not isinstance(lt, LazyTensor):
        raise TypeError(f"untilize expected a LazyTensor, got {type(lt).__name__}")
    return to_layout(lt, lt.layout.replace(tiled=False))


def empty(layout: Layout) -> LazyTensor:
    """Allocate an uninitialised device tensor.

    Materialises the buffer at the user's `grid_shape` first, then
    re-views to the blocked grid (which is what kernels operate over).
    This mirrors the old eager flow so d2m-allocate can plan the
    physical placement from the unblocked breadcrumb.
    """
    b = _Builder.get()
    with b.ctx, b.loc, b.insert_point:
        unblocked_ty = layout.build_device_tensor_type(b.ctx, blocked=False)
        raw = d2m.empty(unblocked_ty)
        val = layout.build_blocked_view(b.ctx, raw)
    return LazyTensor(layout, val, b.generation)


def view_layout(lt: LazyTensor, layout: Layout) -> LazyTensor:
    """Re-view a LazyTensor under a different grid_shape with the same
    logical shape/dtype/block_shape/mem_space."""
    lt = lt._resolve()
    assert lt.layout.logical_shape == layout.logical_shape
    assert lt.layout.block_shape == layout.block_shape
    assert lt.layout.dtype == layout.dtype
    assert lt.layout.tiled == layout.tiled

    b = _Builder.get()
    with b.ctx, b.loc, b.insert_point:
        # Source value is at lt.layout's blocked grid. Reblock to target's
        # blocked grid via the existing reblock-map machinery.
        src_shape = lt.layout.get_device_shape(b.ctx, lt.layout.blocked_grid_shape)
        dst_shape = layout.get_device_shape(b.ctx, layout.blocked_grid_shape)
        dst_ty = layout.build_device_tensor_type(b.ctx, blocked=True)
        reblock_map = d2m.ir.calculate_reblock_map(src_shape, dst_shape, b.ctx)
        val = d2m.ViewLayoutOp(dst_ty, lt.value, reblock_map).result
    return LazyTensor(layout, val, b.generation)


# Short alias.
view = view_layout


# --- Materialisation ---------------------------------------------------------


def _emit_returns_and_finalise(b: _Builder, lts):
    """Emit `from_device` for each LazyTensor and a func.ReturnOp, then
    update the func's signature with the new return types."""
    host_values = []
    host_types = []
    with b.ctx, b.loc, b.insert_point:
        for lt in lts:
            dev = lt.layout.build_device_view(b.ctx, lt.value)
            host = lt.layout.build_from_device(b.ctx, dev)
            host_values.append(host)
            host_types.append(lt.layout.build_host_tensor_type(b.ctx))
        func.ReturnOp(host_values)
    b._refresh_function_type(results=host_types)


def _run_pipeline(b: _Builder):
    system_desc = _get_system_desc_path()
    register = "ttcore-register-device"
    if system_desc:
        register += f"{{system-desc-path={system_desc}}}"
    pipeline_str = f"builtin.module({register},{_PIPELINE})"

    if config.print_pipeline:
        print(f"[d2m-jit] pipeline: {pipeline_str}")
    if config.print_ir_before_pipeline:
        print("[d2m-jit] IR before pipeline:")
        print(b.module)

    pm = PassManager.parse(pipeline_str, context=b.ctx)
    pm.enable_verifier(config.verify_passes)
    if config.print_ir_after_each_pass:
        # ir-printing requires single-threaded passes so output is coherent.
        b.ctx.enable_multithreading(False)
        pm.enable_ir_printing(
            print_after_all=True,
            enable_debug_info=config.print_ir_debug_info,
        )
    pm.run(b.module.operation)

    if config.print_ir_after_pipeline:
        print("[d2m-jit] IR after pipeline:")
        print(b.module)


def _execute(b: _Builder, lts):
    """Serialize to flatbuffer, run on a mesh device, return torch tensors."""
    if runtime is None or binary is None:
        raise RuntimeError("ttmlir runtime is not available in this build")
    bin_capsule = ttmetal_to_flatbuffer_bin(b.module)
    fbb = binary.load_binary_from_capsule(bin_capsule)
    if config.save_flatbuffer_path:
        fbb.store(config.save_flatbuffer_path)
        print(f"[d2m-jit] flatbuffer written to {config.save_flatbuffer_path}")
    program_index = 0
    device_options = runtime.MeshDeviceOptions()
    device_options.mesh_shape = fbb.get_program_mesh_shape(program_index)
    runtime.set_compatible_device_runtime(fbb)

    # Marshal inputs from the torch tensors / scalars gathered during graph build.
    rt_inputs = []
    for t in b.host_tensors:
        if isinstance(t, int) and not isinstance(t, bool):
            rt_inputs.append(runtime.create_scalar_tensor(t))
            continue
        rt_inputs.append(
            runtime.create_borrowed_host_tensor(
                t.data_ptr(),
                list(t.shape),
                list(t.stride()),
                t.element_size(),
                _to_runtime_data_type(t.dtype),
            )
        )

    # Allocate output torch tensors and borrowed host wrappers.
    out_torch = []
    rt_outputs = []
    for lt in lts:
        torch_dtype = _ttcore_to_torch_dtype(lt.layout.dtype)
        t_out = torch.empty(list(lt.layout.logical_shape), dtype=torch_dtype)
        out_torch.append(t_out)
        rt_outputs.append(
            runtime.create_borrowed_host_tensor(
                t_out.data_ptr(),
                list(t_out.shape),
                list(t_out.stride()),
                t_out.element_size(),
                _to_runtime_data_type(t_out.dtype),
            )
        )

    device = runtime.open_mesh_device(device_options)
    submitted = runtime.submit(device, fbb, program_index, rt_inputs)
    runtime.wait(submitted)
    for i, rt_out in enumerate(submitted):
        host_view = runtime.to_host(rt_out, untilize=True)[0]
        runtime.memcpy(rt_outputs[i], host_view)
        runtime.deallocate_tensor(rt_out, force=True)
    runtime.close_mesh_device(device)
    return out_torch


def to_host(*lts: LazyTensor):
    """Compile and execute the open graph. Returns a tuple of torch tensors,
    one per LazyTensor. Resets the builder.

    LazyTensors passed in become 'materialised'; their `.value` is dropped
    and `.materialized` is set to the corresponding torch tensor. Any other
    LazyTensors produced by this builder generation become stale and will
    raise on next use unless they were also passed to this to_host call.
    """
    if not lts:
        raise ValueError("to_host requires at least one LazyTensor")

    resolved = [lt._resolve() for lt in lts]
    b = _Builder.get()
    # All resolved tensors must belong to this builder (resolve guarantees that).
    assert all(lt.generation == b.generation for lt in resolved)

    _emit_returns_and_finalise(b, resolved)
    b.module.operation.verify()
    _run_pipeline(b)
    outs = _execute(b, resolved)

    for orig, lt, t in zip(lts, resolved, outs):
        orig.materialized = t
        orig.value = None
        # If the user passed a stale-but-materialised LazyTensor (one that
        # auto-resolved to a fresh `to_layout`), the original still has its
        # earlier materialisation. Update it to the freshly computed value.

    _Builder.reset()
    return tuple(outs)


# --- Kernel emission ---------------------------------------------------------


def _collect_int_captures(fn):
    """Closed-over int free variables, used as immediate captures by D2MCompiler."""
    if fn.__closure__ is None:
        return {}
    out = {}
    for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
        try:
            val = cell.cell_contents
        except ValueError:
            continue
        if isinstance(val, int) and not isinstance(val, bool):
            out[name] = val
    return out


def _affine_map_from_lambda(fn):
    class _Dim:
        def __init__(self, position):
            self.position = position

    dims = tuple(_Dim(i) for i, _ in enumerate(inspect.signature(fn).parameters))
    results = fn(*dims)
    exprs = []
    for r in results:
        if isinstance(r, _Dim):
            exprs.append(AffineDimExpr.get(r.position))
        elif isinstance(r, int):
            assert r == 0, "Only 0 is allowed as an integer constant in indexing_map"
            exprs.append(AffineConstantExpr.get(r))
        else:
            raise TypeError(
                f"Unsupported indexing_map result type {type(r).__name__}: {r}"
            )
    return AffineMap.get(len(dims), 0, exprs)


def _emit_kernel_generic(
    kernel: "CompiledKernel",
    args,
    grid,
    num_outs: int,
    block_factors,
    indexing_maps,
    iterator_types,
):
    """Append a d2m.GenericOp to the open host func that invokes `kernel`."""
    b = _Builder.get()

    # Split args, preserving "all LazyTensors precede all scalars" ordering.
    lazy_args = []
    scalar_args = []
    saw_scalar = False
    for a in args:
        if isinstance(a, LazyTensor):
            if saw_scalar:
                raise TypeError(
                    "LazyTensor arguments must precede scalar arguments in a "
                    "kernel call"
                )
            lazy_args.append(a._resolve())
        elif isinstance(a, int) and not isinstance(a, bool):
            saw_scalar = True
            scalar_args.append(a)
        else:
            raise TypeError(
                f"Unsupported kernel argument type {type(a).__name__}: {a!r}"
            )

    if num_outs < 1:
        raise ValueError("num_outs must be >= 1")
    if len(lazy_args) < num_outs:
        raise ValueError(
            f"kernel call has {len(lazy_args)} tensor args; need at least "
            f"{num_outs} for outputs"
        )
    input_lts = lazy_args[: len(lazy_args) - num_outs]
    output_lts = lazy_args[len(lazy_args) - num_outs :]

    # Compile the kernel body in the current builder's context. D2MCompiler
    # picks up b.ctx via get_default_loc_context.
    with b.ctx, b.loc:
        compiler_args = [lt.layout for lt in lazy_args] + list(scalar_args)
        compiler = D2MCompiler(
            kernel.fn.__name__,
            "unified",
            kernel._captures,
            *compiler_args,
        )
        compiler.visit(kernel._ast)
        compiler.module.operation.verify()

    # Emit the GenericOp + splice the kernel body.
    with b.ctx, b.loc, b.insert_point:
        # Scalars are sourced from func args (not host-scope constants) so the
        # GenericOp's region stays isolated-from-above.
        additional = [b.add_scalar_input(s) for s in scalar_args]
        inputs = [lt.value for lt in input_lts]
        outputs = [lt.value for lt in output_lts]
        output_types = [v.type for v in outputs]

        threads = ArrayAttr.get(
            [compiler.func_entry.attributes[d2m.ir.ThreadAttr.name]]
        )
        grid_attr = ttcore.ir.GridAttr.get(b.ctx, list(grid))

        bf = list(block_factors or [])
        if bf and isinstance(bf[0], tuple):
            bf = [v for tup in bf for v in tup]

        indexing_attrs = [_affine_map_from_lambda(f) for f in (indexing_maps or [])]
        iter_attr = ArrayAttr.get(
            [
                ttcore.ir.IteratorTypeAttr.get(
                    b.ctx, ttcore.IteratorType[i.title()].value
                )
                for i in (iterator_types or [])
            ]
        )

        generic = d2m.GenericOp(
            output_types,
            inputs,
            outputs,
            additional,
            grid_attr,
            bf,
            indexing_attrs,
            iter_attr,
            threads,
            1,  # num_regions
        )

        region = generic.regions[0]
        compiler.func_entry.entry_block.append_to(region)
        block = region.blocks[0]
        if block.operations and block.operations[-1].name == "func.return":
            block.operations[-1].erase()

        all_ops = inputs + outputs + additional
        for orig_arg, op in zip(block.arguments, all_ops):
            orig_arg.replace_all_uses_with(op)
        for _ in range(len(block.arguments)):
            block.erase_argument(0)

    # Rebind output LazyTensors to the GenericOp's results.
    for i, lt in enumerate(output_lts):
        lt.value = generic.results[i]
        lt.generation = b.generation


class CompiledKernel:
    """Wraps a user kernel function. Parses the Python body once; emits a
    `d2m.GenericOp` into the current builder on every call."""

    def __init__(self, fn):
        functools.update_wrapper(self, fn)
        self.fn = fn
        self._source = _cleanup_source_code(fn)
        self._ast = _ast.parse(self._source)
        self._captures = _collect_int_captures(fn)

    def __call__(
        self,
        *args,
        grid,
        num_outs: int = 1,
        block_factors=None,
        indexing_maps=None,
        iterator_types=None,
    ):
        _emit_kernel_generic(
            self,
            args,
            grid=grid,
            num_outs=num_outs,
            block_factors=block_factors,
            indexing_maps=indexing_maps,
            iterator_types=iterator_types,
        )


def kernel(fn):
    """Decorate a user function as a d2m_jit kernel."""
    return CompiledKernel(fn)
