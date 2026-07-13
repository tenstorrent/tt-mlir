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
import contextlib
import functools
import inspect
import os
import threading
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
from ttmlir.dialects import d2m, func, arith, linalg, ttcore
from ttmlir.passes import ttmetal_to_flatbuffer_bin

from .ast import D2MCompiler, SEMAPHORE_ARG
from .config import config
from .errors import D2mJitError
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


# Pre-backend section is a deliberate lean subset (the d2m-jit bypass goal:
# stay out of the TTIR->D2M frontend machinery since we build D2M IR
# directly). Backend and TTKernel/EmitC tail use canonical pipelines so
# they don't drift from createTTIRToTTMetalPipeline in D2MPipelines.cpp.
#
# Interleave note: convert-d2m-to-ttmetal MUST run while the kernel body is
# still in TTKernel form — it walks for `ttkernel.typecast_tile` to choose
# per-thread `UnpackToDestMode` (Fp32 vs Default). If the EmitC tail ran
# first, the typecast would have become `emitc.call_opaque "typecast_tile"`
# and the walk would silently fail to find it → wrong unpack mode → byte
# scramble on f32→bf16 typecast. The pre-emitc / dispatch / hoist-inits /
# emitc-tail split below mirrors what createTTIRToTTMetalPipeline does.
def _pipeline_passes():
    """Build the ordered list of pass names for the d2m -> ttmetal lowering.

    The `d2m-be-pipeline` options are computed here so runtime config (e.g.
    `config.use_split_unified_thread_v2`) can select alternate backend passes
    without forking the whole pipeline string.

    When `config.insert_profiler_traces` is set, the TTKernel
    `insert-device-zone-scopes` pass is spliced in after `ttkernel-hoist-inits`
    and before the EmitC tail — the same slot createTTIRToTTMetalPipeline uses.
    It must run while the kernel body is still in TTKernel form (it walks
    TTKernel ops to wrap them in `DeviceZoneScopedN` scopes) and after
    hoist-inits so the dispatch-level conversion sees the original loop
    structure.
    """
    be_opts = [f"use-tile-matmul={int(config.use_tile_matmul)}"]
    if config.use_split_unified_thread_v2:
        be_opts.append("use-split-unified-thread-v2=1")
    # use-tensor-accessor-dma is consumed by both d2m-be-pipeline (which gates
    # D2MLowerDMAToFullyIndexedForm) and d2m-to-ttkernel-pre-emitc-pipeline
    # (which passes it to convert-d2m-to-ttkernel); set it on both.
    preemitc = "d2m-to-ttkernel-pre-emitc-pipeline"
    if config.use_tensor_accessor_dma:
        be_opts.append("use-tensor-accessor-dma=1")
        preemitc += "{use-tensor-accessor-dma=1}"
    passes = [
        "canonicalize",
        "d2m-lower-to-layout",
        "canonicalize",
        "ttir-bufferization-pipeline",
        "d2m-insert-scratch-buffers",
        "d2m-generic-apply-interchange",
        "d2m-generate-outer-loops",
        "d2m-mark-synchronized-buffers",
        "d2m-allocate",
        "d2m-lower-multicast-loads",
        "d2m-generic-lower-to-explicit-form",
        "canonicalize",
        f"d2m-be-pipeline{{{' '.join(be_opts)}}}",
        preemitc,
        "d2m-to-ttmetal-pipeline",
        "func.func(ttkernel-hoist-inits)",
    ]
    if config.insert_profiler_traces:
        traits = config.profiler_traits.strip() or "device-zone"
        passes.append("func.func(insert-device-zone-scopes{traits=" + traits + "})")
    passes.append("d2m-emitc-pipeline")
    return passes


def _build_pipeline() -> str:
    """Comma-joined pass-pipeline string (compat wrapper over `_pipeline_passes`).

    Kept because tests and ad-hoc callers embed it directly in
    `builtin.module(<register>,<pipeline>)`.
    """
    return ",".join(_pipeline_passes())


# --- Scope abstraction ------------------------------------------------------
#
# A "scope" is the build context that the lazy-emission helpers
# (`to_layout`, `empty`, `view_layout`, `_emit_kernel_generic`, …) target.
# The default scope is `_Builder` — a process-level singleton that owns its
# own `Context`/`Module`/open `func.func` and accumulates MLIR ops there
# until `to_host` runs the pipeline and resets it.
#
# A `RewriteScope` (defined alongside the pattern-rewrite framework) plugs
# in a `PatternRewriter`'s context + insertion point so that calling a
# `@d2m.kernel` from inside a rewrite emits the GenericOp at the matched
# op's site rather than into a fresh module. From the perspective of the
# emission helpers, all scopes quack the same: they expose `ctx`, `loc`,
# `insert_point`, `generation`, `add_host_input`, `add_scalar_input`.
#
# `_get_scope()` returns the top of a thread-local stack, falling back to
# the lazy `_Builder` singleton when nothing is pushed. Push/pop is done
# via the `_push_scope()` context manager — patterns frameworks (and tests)
# use it; user code never does.

_scope_local = threading.local()


def _get_scope():
    """Return the active build scope. Defaults to the lazy `_Builder` singleton."""
    stack = getattr(_scope_local, "stack", None)
    if stack:
        return stack[-1]
    return _Builder.get()


@contextlib.contextmanager
def _push_scope(scope):
    """Push `scope` as the active build scope for the duration of the block."""
    stack = getattr(_scope_local, "stack", None)
    if stack is None:
        stack = []
        _scope_local.stack = stack
    stack.append(scope)
    try:
        yield scope
    finally:
        popped = stack.pop()
        assert popped is scope, "scope stack out of sync"


class _Builder:
    """Process-level singleton accumulating MLIR ops for the current lazy graph.

    This is one concrete `Scope` implementation; see `_get_scope` for the
    abstraction. Owns its own `Context`/`Module`/open `func.func`. Reset by
    `to_host` once the pipeline has run.
    """

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
        # Optional mesh topology (e.g. ("linear", "ring")) for register-device;
        # the mesh *shape* lives in the module's ttcore.meshes attr (set by
        # set_mesh) and is read by ttcore-register-device's determineMeshShape.
        self._mesh_topology = None
        # Name of the declared mesh (None until set_mesh), used to build the
        # `#ttcore.tensor_mesh<...>` encoding that marks mesh_shard boundary
        # tensors as multi-device.
        self._mesh_name = None
        # Set when a kernel is invoked with a `fabric=` config (a CCL kernel).
        # `_execute` must then enable the device fabric (set_fabric_config)
        # before opening the mesh device, or the cross-device fabric ops
        # (device_synchronize / fabric remote_store / fabric semaphore incs)
        # silently no-op and the kernel hangs on its semaphore_wait.
        self._fabric_used = False

    def set_mesh(self, shape, topology=None):
        """Declare the device mesh for this graph.

        Sets the module's `ttcore.meshes` attribute (which ttcore-register-device
        reads to size the device, and which flows to the flatbuffer program mesh
        shape consumed by the runtime). `topology` (e.g. ("linear", "ring")) is
        stashed for register-device's `mesh-topology` option -- needed for CCL
        ops like all_gather that require a ring axis."""
        dims = "x".join(str(int(d)) for d in shape)
        with self.ctx, self.loc:
            attr = Attribute.parse(f'#ttcore.meshes<[<"mesh" = {dims}>]>', self.ctx)
        self.module.operation.attributes["ttcore.meshes"] = attr
        self._mesh_topology = list(topology) if topology is not None else None
        self._mesh_name = "mesh"

    def _refresh_function_type(self, results=None):
        with self.ctx, self.loc:
            ft = FunctionType.get(self._input_types, results or [])
            self.func_op.attributes["function_type"] = TypeAttr.get(ft)

    def add_host_input(self, layout: Layout, host_tensor, host_ty=None):
        """Append a host-typed func arg and return its BlockArgument.

        `host_ty` overrides the type (used for mesh inputs, whose func arg is
        the full un-sharded tensor rather than `layout`'s per-device shard)."""
        if host_ty is None:
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


class RewriteScope:
    """Build-scope view onto an MLIR `PatternRewriter` insertion point.

    Pushed by the d2m-jit rewrite framework so that calling a `@d2m.kernel`
    from inside a pattern body emits the `d2m.GenericOp` (and any supporting
    `to_layout`/`view_layout`/`empty` ops) at the rewriter's IP rather than
    into a fresh host func.

    Quacks like `_Builder` for the emission helpers: exposes `ctx`, `loc`,
    `insert_point`, `generation`. `add_host_input` raises (no host I/O from
    a rewrite). `add_scalar_input` emits an `arith.constant ... : index` at
    the rewriter's IP and returns the resulting Value.

    Has no `module` attribute — the surrounding module is the one being
    mutated, not something this scope owns.
    """

    def __init__(self, rewriter, op, loc=None):
        self.rewriter = rewriter
        # Derive ctx from the matched op (rewriter doesn't bind it directly).
        self.ctx = op.context
        self.loc = loc if loc is not None else op.location
        # InsertionPoint pointing at the rewriter's current insertion point.
        # The PDL driver sets this to "before the matched op" before invoking
        # the native rewrite, which is exactly where we want new IR to land.
        self.insert_point = rewriter.ip
        # Unique non-reusable id, distinct from any _Builder generation.
        self.generation = _Builder._next_generation
        _Builder._next_generation += 1

    def add_host_input(self, layout, host_tensor):
        raise RuntimeError(
            "Cannot lift a host tensor from inside a pattern rewrite. The "
            "graph being built is part of an existing module; use "
            "d2m.from_value(ir.Value) to wrap an SSA value already present in "
            "that module."
        )

    def add_scalar_input(self, value: int):
        # Emit an arith.constant of index type at the rewriter's IP.
        # This is the rewrite-mode analog of _Builder.add_scalar_input, which
        # would have added a func arg in lazy mode.
        with self.ctx, self.loc, self.insert_point:
            idx_ty = IndexType.get(self.ctx)
            return arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, int(value))).result


# --- LazyTensor --------------------------------------------------------------


class LazyTensor:
    """Host-side handle for a value being built into the lazy graph.

    Holds either:
      - an `ir.Value` at host-func scope (in the current builder generation), or
      - a materialised torch.Tensor (after to_host).
    """

    __slots__ = ("layout", "value", "generation", "materialized", "is_view", "mesh")

    def __init__(
        self,
        layout: Layout,
        value,
        generation,
        materialized=None,
        is_view: bool = False,
        mesh=None,
    ):
        self.layout = layout
        self.value = value
        self.generation = generation
        self.materialized = materialized
        # A view is a metadata reinterpretation (d2m.view_layout) of an
        # underlying buffer. to_host on a view is ambiguous -- the buffer
        # data is not in the view's logical form -- so we refuse it and
        # ask the user to materialise via to_layout first.
        self.is_view = is_view
        # Mesh sharding metadata: None for a single-device tensor, else a
        # `MeshShard` describing the full (un-sharded) shape and how this
        # tensor's per-device shard maps onto the mesh. Drives the host-scope
        # mesh_shard(shard_to_full) emitted on the to_host gather path.
        self.mesh = mesh

    def to_host(self):
        return to_host(self)[0]

    def _resolve(self) -> "LazyTensor":
        """Return a LazyTensor in the current builder's generation.

        - Same generation: return self.
        - Materialised (different generation): auto-re-enter via to_layout.
        - Stale (different generation, not materialised): raise.
        """
        b = _get_scope()
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
    b = _get_scope()

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
            converted = d2m.ToLayoutOp([dst_unblocked_ty], src_val, dst_empty).result
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


def tilize(lt: LazyTensor, dtype=None) -> LazyTensor:
    """Convert a device LazyTensor to a tile-typed (`tiled=True`) layout.

    The target layout is the source's layout with `tiled` set to True,
    optionally overriding `dtype` (e.g. f32 -> bf16). All other fields
    (shape, block_shape, grid_shape, mem_space, collapse) are preserved.
    """
    if not isinstance(lt, LazyTensor):
        raise TypeError(f"tilize expected a LazyTensor, got {type(lt).__name__}")
    overrides = {"tiled": True}
    if dtype is not None:
        overrides["dtype"] = dtype
    return to_layout(lt, lt.layout.replace(**overrides))


def untilize(lt: LazyTensor, dtype=None) -> LazyTensor:
    """Convert a device LazyTensor to a row-major (`tiled=False`) layout.

    The target layout is the source's layout with `tiled` set to False,
    optionally overriding `dtype`. All other fields are preserved.
    """
    if not isinstance(lt, LazyTensor):
        raise TypeError(f"untilize expected a LazyTensor, got {type(lt).__name__}")
    overrides = {"tiled": False}
    if dtype is not None:
        overrides["dtype"] = dtype
    return to_layout(lt, lt.layout.replace(**overrides))


def empty(layout: Layout) -> LazyTensor:
    """Allocate an uninitialised device tensor.

    Materialises the buffer at the user's `grid_shape` first, then
    re-views to the blocked grid (which is what kernels operate over).
    This mirrors the old eager flow so d2m-allocate can plan the
    physical placement from the unblocked breadcrumb.
    """
    b = _get_scope()
    with b.ctx, b.loc, b.insert_point:
        unblocked_ty = layout.build_device_tensor_type(b.ctx, blocked=False)
        raw = d2m.empty(unblocked_ty)
        val = layout.build_blocked_view(b.ctx, raw)
    return LazyTensor(layout, val, b.generation)


def full(layout: Layout, value) -> LazyTensor:
    """Allocate a device tensor initialised to `value` (a Python scalar).

    Tiled layouts: emits a `d2m.generic` wrapping
    `linalg.generic { d2m.tile_fill }` + `d2m.remote_store`, mirroring
    `lowerRankedTensorFillViaGeneric` in `TTIRToD2M.cpp`. No host roundtrip.

    Non-tiled layouts: falls back to a host-side `torch.full` + `to_layout`
    copy, since `d2m.tile_fill` is a tile-typed op.

    Note: on Wormhole, `d2m.tile_fill` for f32 routes through the SFPU's
    vFloat (fp19: 1+8+10), so values with non-zero lower-13 mantissa bits
    (e.g. 3.14) are truncated. 0.0, 1.0, and other fp19-exact values are
    bit-perfect; arbitrary f32 values are not.
    """
    if not layout.tiled:
        if torch is None:
            raise RuntimeError(
                "torch is required for d2m_jit.full() on non-tiled layouts"
            )
        torch_dtype = _ttcore_to_torch_dtype(layout.dtype)
        host = torch.full(list(layout.logical_shape), value, dtype=torch_dtype)
        return to_layout(host, layout)

    b = _get_scope()
    with b.ctx, b.loc, b.insert_point:
        # Allocate the output device tensor (mirror empty()).
        unblocked_ty = layout.build_device_tensor_type(b.ctx, blocked=False)
        raw = d2m.empty(unblocked_ty)
        out_blocked = layout.build_blocked_view(b.ctx, raw)
        outer_ty = out_blocked.type
        outer_rt = RankedTensorType(outer_ty)

        # Outer blocked rank is 2N: grid dims (first N) + shard dims (last N).
        # The per-shard tensor type drops the grid dims and the encoding.
        outer_rank = outer_rt.rank
        assert (
            outer_rank % 2 == 0
        ), f"expected blocked tensor rank to be even, got {outer_rank}"
        physical_rank = outer_rank // 2
        tile_ty = outer_rt.element_type
        shard_shape = list(outer_rt.shape)[physical_rank:]
        shard_ty = RankedTensorType.get(shard_shape, tile_ty)

        # Outer d2m.generic attrs: identity affine map and parallel iterators
        # over the user grid; one compute thread.
        identity = AffineMap.get_identity(physical_rank)
        indexing_maps = ArrayAttr.get([AffineMapAttr.get(identity)])
        parallel_iter = ttcore.ir.IteratorTypeAttr.get(
            b.ctx, ttcore.IteratorType.Parallel.value
        )
        iterator_types = ArrayAttr.get([parallel_iter] * physical_rank)
        # Unified, not Compute: remote_store can only live in a
        # datamovement or unified region.
        threads = ArrayAttr.get(
            [d2m.ir.ThreadAttr.get(b.ctx, str(d2m.ThreadType.Unified))]
        )
        grid_attr = ttcore.ir.GridAttr.get(b.ctx, list(layout.grid_shape))

        generic = d2m.GenericOp(
            [outer_ty],
            [],  # inputs
            [out_blocked],  # outputs
            [],  # additionalArgs
            grid_attr,
            [1] * physical_rank,  # block_factors: one block per grid cell
            indexing_maps,
            iterator_types,
            threads,
            1,  # num_regions
        )

        body = Block.create_at_start(generic.regions[0], [], [])
        with InsertionPoint(body):
            # Per-shard buffer that the inner linalg.generic fills.
            shard_buf = d2m.empty(shard_ty)

            # Inner linalg.generic { arith.constant + d2m.tile_fill }.
            linalg_indexing = ArrayAttr.get([AffineMapAttr.get(identity)])
            linalg_parallel = Attribute.parse("#linalg.iterator_type<parallel>")
            linalg_iter = ArrayAttr.get([linalg_parallel] * physical_rank)

            inner_generic = linalg.GenericOp(
                [shard_ty],
                [],  # no inputs
                [shard_buf],
                linalg_indexing,
                linalg_iter,
            )
            inner_body = Block.create_at_start(
                inner_generic.regions[0], [tile_ty], [Location.unknown()]
            )
            with InsertionPoint(inner_body):
                scalar_ty = layout.get_scalar_type(b.ctx)
                if FloatType.isinstance(scalar_ty):
                    scalar_attr = FloatAttr.get(scalar_ty, float(value))
                else:
                    scalar_attr = IntegerAttr.get(scalar_ty, int(value))
                scalar = arith.ConstantOp(scalar_ty, scalar_attr).result
                filled_tile = d2m.TileFillOp(tile_ty, scalar).result
                linalg.yield_([filled_tile])

            # Grid indices for remote_store: d2m.block_index(d) per dim.
            indices = [d2m.block_index(d) for d in range(physical_rank)]
            stored = d2m.remote_store(
                outer_ty,
                out_blocked,
                indices,
                start_device=[],
                device_mcast_shape=[],
                semaphore_indices=[],
                local_buffer=inner_generic.result,
            )
            d2m.yield_([stored])

    return LazyTensor(layout, generic.results[0], b.generation)


def zeros(layout: Layout) -> LazyTensor:
    """`d2m.full(layout, 0)` -- allocate a zero-initialised device tensor."""
    return full(layout, 0)


def arange(layout: Layout, start: int = 0, step: int = 1) -> LazyTensor:
    """Allocate a device tensor filled with arange values.

    Equivalent to `torch.arange(start, start + N*step, step).reshape(shape)`
    where `N = prod(layout.logical_shape)` and `shape = layout.logical_shape`.
    Row-major linear traversal.

    Currently implemented as a host-side `torch.arange` + `to_layout`. This
    matches what TTIR's `arange` ends up costing for a precomputed mask
    (one DRAM transfer), but does **not** exercise the device-side
    `d2m.arange_block` op. A future zero-roundtrip version would emit
    `d2m.GenericOp { d2m.arange_block + remote_store }` (mirroring the C++
    `D2MArangeOpRewriter` in lib/Conversion/TTIRToD2M/TTIRToD2M.cpp).
    """
    if torch is None:
        raise RuntimeError("torch is required for d2m_jit.arange()")
    torch_dtype = _ttcore_to_torch_dtype(layout.dtype)
    numel = 1
    for d in layout.logical_shape:
        numel *= d
    flat = torch.arange(start, start + numel * step, step, dtype=torch_dtype)
    return to_layout(flat.reshape(list(layout.logical_shape)), layout)


# --- Global semaphores -------------------------------------------------------


def _semaphore_backing_type(ctx, grid_shape):
    """Build the device tensor type for a global-semaphore backing buffer.

    `d2m.create_global_semaphore`'s verifier requires the backing tensor to
    have (a) a grid shape equal to the device worker grid, (b) a 1x1 shard,
    and (c) a `ui32` element type. The compute `Layout` helper pads shards to
    the 32x32 tile alignment, so we build the metal_layout directly with
    `dim_alignments = 1x1` (matching `#sem_layout` in
    test/.../generic_global_semaphores.mlir).
    """
    import numpy as np

    rank = len(grid_shape)
    # Identity collapse intervals [[0,1],[1,2],...]: one logical dim per shard
    # dim, no collapsing.
    intervals = np.array([[i, i + 1] for i in range(rank)], dtype=np.int64)
    collapse = DenseIntElementsAttr.get(intervals, context=ctx)
    metal_layout = ttcore.ir.MetalLayoutAttr.get(
        ctx,
        list(grid_shape),
        int(ttcore.MemorySpace.DeviceL1),
        int(ttcore.TensorMemoryLayout.Sharded),
        collapse,
        [1] * rank,  # dim_alignments: 1x1 shard, no tile padding
    )
    device_shape = ttcore.ir.MetalLayoutAttr.maybe_downcast(
        metal_layout
    ).getDeviceShape(list(grid_shape), [])
    ui32 = IntegerType.get_unsigned(32, ctx)
    return RankedTensorType.get(device_shape, ui32, encoding=metal_layout)


class GlobalSemaphore:
    """Host-side handle for a `!d2m.global_semaphore` created via
    `d2m.create_global_semaphore`.

    Passed positionally to a `@d2m.kernel` after the tensor arguments; the
    kernel body sees it as a `!d2m.global_semaphore` parameter usable with
    `semaphore_set` / `semaphore_wait` / `semaphore_inc` / `device_synchronize`
    and the `semaphore=` kwarg of `remote_store`.
    """

    __slots__ = ("value", "generation", "grid_shape", "init")

    def __init__(self, value, generation, grid_shape, init):
        self.value = value
        self.generation = generation
        self.grid_shape = grid_shape
        self.init = init

    def _resolve_value(self):
        b = _get_scope()
        if self.generation != b.generation:
            raise RuntimeError(
                "Stale GlobalSemaphore: produced by a prior builder generation. "
                "Create it in the same generation as the kernel call."
            )
        return self.value


_g_worker_grid = None


def _device_worker_grid():
    """Return the device worker grid as `(rows, cols)` from the system desc.

    `create_global_semaphore`'s verifier requires the semaphore backing tensor
    to span the full device worker grid, which is arch-dependent (8x8 on
    Wormhole, 10x11 on Blackhole p300c). Read it from the system descriptor's
    `grid_size` so `global_semaphore()` is portable across architectures rather
    than hardcoding a single arch's grid. Cached after the first call.
    """
    global _g_worker_grid
    if _g_worker_grid is not None:
        return _g_worker_grid
    import json

    # Use the already-imported `_ttmlir_runtime` bindings, NOT `ttrt.binary`:
    # importing the ttrt wrapper here pulls in a second copy of the runtime
    # extension and double-initialises it, aborting the process.
    sd_path = _get_system_desc_path()
    if binary is not None and sd_path:
        sd = binary.load_system_desc_from_path(sd_path)
    elif runtime is not None:
        sd = runtime.get_current_system_desc()
    else:
        raise RuntimeError(
            "global_semaphore() needs a system descriptor to size the worker "
            "grid; set SYSTEM_DESC_PATH or pass grid_shape explicitly."
        )
    desc = json.loads(sd.as_json())
    root = desc.get("system_desc", desc)
    grid = root["chip_descs"][0]["grid_size"]
    _g_worker_grid = (int(grid["y"]), int(grid["x"]))
    return _g_worker_grid


def global_semaphore(grid_shape=None, init=0) -> GlobalSemaphore:
    """Allocate a global semaphore over the device worker grid.

    `grid_shape` must equal the device worker grid (8x8 on Wormhole, 10x11 on
    Blackhole p300c) — the `create_global_semaphore` verifier checks this. When
    omitted (`None`), it is read from the system descriptor via
    `_device_worker_grid()` so the same call is portable across architectures.
    `init` is the initial value (default 0).

    Emits an uninitialised `ui32` backing buffer plus
    `d2m.create_global_semaphore`, and returns a `GlobalSemaphore` handle to
    pass to a kernel.
    """
    if grid_shape is None:
        grid_shape = _device_worker_grid()
    b = _get_scope()
    with b.ctx, b.loc, b.insert_point:
        backing_ty = _semaphore_backing_type(b.ctx, grid_shape)
        backing = d2m.empty(backing_ty)
        sem_ty = d2m.ir.GlobalSemaphoreType.get(b.ctx)
        # create_global_semaphore does not support type inference; the result
        # type must be given explicitly.
        sem = d2m.create_global_semaphore(backing, value=int(init), results=[sem_ty])
    return GlobalSemaphore(sem, b.generation, tuple(grid_shape), int(init))


def mesh(shape, topology=None):
    """Declare the device mesh for the current graph.

    `shape` is the mesh shape, e.g. `(1, 2)` for a 1x2 mesh; `topology` is an
    optional per-axis topology tuple (e.g. `("linear", "ring")`) needed by CCL
    ops such as all_gather (which require a ring cluster axis). Call before
    building the graph; the mesh persists until the next `to_host` reset.

    Sets the module's `ttcore.meshes` attribute, which ttcore-register-device
    reads to size the device and which flows to the runtime as the program mesh
    shape."""
    b = _get_scope()
    if not isinstance(b, _Builder):
        raise RuntimeError("mesh() requires the lazy builder scope")
    b.set_mesh(shape, topology)


class MeshShard:
    """Mesh-sharding metadata carried by a LazyTensor (see `LazyTensor.mesh`).

    Records the full (un-sharded) logical shape plus how the per-device shard
    maps onto the mesh, so the to_host gather path can emit a
    `mesh_shard(shard_to_full)`."""

    __slots__ = ("full_shape", "shard_dims", "shard_shape")

    def __init__(self, full_shape, shard_dims, shard_shape):
        self.full_shape = list(full_shape)
        self.shard_dims = list(shard_dims)
        self.shard_shape = list(shard_shape)


def _shard_logical_shape(full_shape, shard_dims, shard_shape):
    """Per-device shard shape: `full_shape` divided by `shard_shape` along the
    corresponding `shard_dims`."""
    shard = list(full_shape)
    for i, d in enumerate(shard_dims):
        if shard[d] % shard_shape[i] != 0:
            raise ValueError(
                f"mesh shard: full dim {d} ({shard[d]}) not divisible by shard "
                f"factor {shard_shape[i]}"
            )
        shard[d] //= shard_shape[i]
    return shard


def _emit_mesh_shard(b, value, dst_ty, direction, shard_dims, shard_shape):
    """Emit a `d2m.mesh_shard` (devices) op in the given direction."""
    devices = Attribute.parse("#ttcore.shard_type<devices>", b.ctx)
    dir_attr = Attribute.parse(f"#ttcore.shard_direction<{direction}>", b.ctx)
    return d2m.mesh_shard(
        dst_ty, value, devices, dir_attr, list(shard_shape), list(shard_dims)
    )


def _tensor_mesh_attr(b):
    """The `#ttcore.tensor_mesh<name>` encoding for the declared mesh, or None.

    Tagging the per-device shard at the mesh_shard boundary with this encoding
    is what marks it multi-device: it bufferizes to a `#ttcore.host_layout<...,
    <name>>` so the runtime sizes the distributed host buffer correctly (vs a
    single-device tensor whose size mismatches the mesh device buffer). The
    full-tensor side of mesh_shard stays un-encoded. Mirrors what
    ttir-multi-device-tensor-annotation does for `ttir.mesh_shard`."""
    name = getattr(b, "_mesh_name", None)
    if name is None:
        return None
    with b.ctx, b.loc:
        return Attribute.parse(f'#ttcore.tensor_mesh<"{name}">', b.ctx)


def mesh_shard(input_, layout: Layout, shard_dims, shard_shape) -> LazyTensor:
    """Distribute a full host tensor across the mesh, one shard per device
    (`full_to_shard`).

    `input_` is the full `torch.Tensor`; `layout` is the **per-device shard**
    layout, whose `logical_shape` must equal `input_`'s shape divided by
    `shard_shape` along `shard_dims`. `shard_shape` is the mesh shard factor
    per axis (e.g. `[1, 2]` for a 1x2 mesh) and `shard_dims` maps tensor dims
    to mesh axes (e.g. `[0, 1]`).

    Returns a device LazyTensor carrying `MeshShard` metadata so `to_host`
    gathers it back (`shard_to_full`)."""
    if torch is None or not isinstance(input_, torch.Tensor):
        raise TypeError("mesh_shard expects a torch.Tensor (the full tensor)")
    b = _get_scope()
    full_shape = list(input_.shape)
    expected = _shard_logical_shape(full_shape, shard_dims, shard_shape)
    if list(layout.logical_shape) != expected:
        raise ValueError(
            f"mesh_shard: layout shard shape {list(layout.logical_shape)} does "
            f"not match {expected} (full {full_shape} / {list(shard_shape)} along "
            f"dims {list(shard_dims)})"
        )
    with b.ctx, b.loc, b.insert_point:
        elem = layout.get_host_elem_type(b.ctx)
        full_ty = RankedTensorType.get(full_shape, elem)
        # The per-device shard carries the tensor_mesh encoding (multi-device);
        # the full tensor (func arg) stays plain.
        tensor_mesh = _tensor_mesh_attr(b)
        shard_ty = RankedTensorType.get(expected, elem, encoding=tensor_mesh)
        bb_arg = b.add_host_input(layout, input_, host_ty=full_ty)
        sharded = _emit_mesh_shard(
            b, bb_arg, shard_ty, "full_to_shard", shard_dims, shard_shape
        )
        dev = layout.build_to_device(b.ctx, sharded)
    return LazyTensor(
        layout, dev, b.generation, mesh=MeshShard(full_shape, shard_dims, shard_shape)
    )


def mesh_gather(lt: LazyTensor, shard_dims=None, shard_shape=None) -> LazyTensor:
    """Mark a per-device shard LazyTensor to be gathered to its full tensor on
    `to_host` (`shard_to_full`).

    A tensor produced by `mesh_shard` already carries the metadata, so this is
    a no-op for it. For a tensor produced some other way (e.g. a kernel output),
    pass `shard_dims`/`shard_shape`; the full shape is derived from the layout's
    per-device shard shape."""
    lt = lt._resolve()
    if lt.mesh is None:
        if shard_dims is None or shard_shape is None:
            raise ValueError(
                "mesh_gather needs shard_dims/shard_shape for a tensor not "
                "produced by mesh_shard"
            )
        full = list(lt.layout.logical_shape)
        for i, d in enumerate(shard_dims):
            full[d] *= shard_shape[i]
        lt.mesh = MeshShard(full, shard_dims, shard_shape)
    return lt


def reblock(lt: LazyTensor, grid) -> LazyTensor:
    """Reblock a device tensor onto a different worker-core `grid` (a metadata
    `d2m.view_layout`, no data movement).

    Used to build the CCL "stream" operands: spread a shard's work across cores
    and (for an all_gather output) span the mesh on the gather dim. The new
    layout's `block_shape` is chosen so its blocked grid equals `grid` (the
    layout is self-consistent), so a later device-view / to_host on the result
    is a no-op rather than an implicit re-reblock."""
    lt = lt._resolve()
    layout = lt.layout
    b = _get_scope()
    grid = list(grid)
    logical = list(layout.logical_shape)
    tiles = [(s + 31) // 32 for s in logical] if layout.tiled else logical
    if len(grid) != len(tiles):
        raise ValueError(f"reblock: grid {grid} rank != logical rank {len(tiles)}")
    new_block = []
    for t, g in zip(tiles, grid):
        if g <= 0 or t % g != 0:
            raise ValueError(
                f"reblock: {t} tiles along a dim not divisible by grid {g}"
            )
        new_block.append(t // g)
    new_layout = layout.replace(grid_shape=grid, block_shape=new_block)
    with b.ctx, b.loc, b.insert_point:
        old_shape = list(lt.value.type.shape)
        new_ty = new_layout.build_device_tensor_type(b.ctx, blocked=False)
        reblock_map = d2m.ir.calculate_reblock_map(old_shape, list(new_ty.shape), b.ctx)
        val = d2m.ViewLayoutOp(new_ty, lt.value, reblock_map).result
    return LazyTensor(new_layout, val, b.generation, is_view=True, mesh=lt.mesh)


def reshape(lt: LazyTensor, *shape) -> LazyTensor:
    """torch.reshape-style logical-shape change.

    Total element count must match. Currently implemented via a host
    roundtrip (`to_host` -> `torch.reshape` -> `to_layout`), so it pays a
    DRAM transfer and re-tilises the data. Use it for shape changes that
    don't cleanly map to a `view` -- e.g. coalescing two non-adjacent dims
    or splitting one dim into many.

    Distinct from `view` / `view_layout` / `permute`, which are metadata
    reinterpretations of the buffer (no data movement, but require the
    new logical layout to be expressible as a permutation of the source's
    grid/tile dims).

    The destination layout reuses the source layout's `dtype`, `mem_space`,
    `tiled` setting, and either:
      - keeps the source's `block_shape` / `grid_shape` if they fit the
        new shape divisibility-wise, or
      - falls back to `block_shape=[1]*rank`, `grid_shape=[1]*rank`.
    Use `to_layout(reshaped, target_layout)` to land it in a specific
    layout afterwards.
    """
    if not isinstance(lt, LazyTensor):
        raise TypeError(f"reshape expected a LazyTensor, got {type(lt).__name__}")
    if torch is None:
        raise RuntimeError("torch is required for d2m_jit.reshape()")

    # Accept reshape(lt, 1, 2, 256, 64) and reshape(lt, [1, 2, 256, 64]).
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        new_shape = tuple(shape[0])
    else:
        new_shape = tuple(shape)

    src_numel = 1
    for d in lt.layout.logical_shape:
        src_numel *= d
    dst_numel = 1
    for d in new_shape:
        dst_numel *= d
    if src_numel != dst_numel:
        raise ValueError(
            f"reshape: total element count must match: "
            f"src {tuple(lt.layout.logical_shape)} ({src_numel}) "
            f"!= dst {new_shape} ({dst_numel})"
        )

    # Pick a destination layout: keep src's block/grid if compatible,
    # otherwise fall back to a trivial single-block single-grid layout
    # (the user can to_layout to something denser if perf matters).
    rank = len(new_shape)
    src_block = list(lt.layout.block_shape)
    src_grid = list(lt.layout.grid_shape)
    if (
        len(src_block) == rank
        and len(src_grid) == rank
        and all(
            d % (b * g * (32 if lt.layout.tiled else 1)) == 0
            for d, b, g in zip(new_shape, src_block, src_grid)
        )
    ):
        block_shape = src_block
        grid_shape = src_grid
    else:
        block_shape = [1] * rank
        grid_shape = [1] * rank

    dst_layout = lt.layout.replace(
        shape=new_shape,
        block_shape=block_shape,
        grid_shape=grid_shape,
    )

    host = lt.to_host().reshape(new_shape)
    return to_layout(host, dst_layout)


def reduction_layout(layout: Layout, dim, allow_cross_tile: bool = False) -> Layout:
    """Return the output layout for a keepdim per-tile reduction.

    The DSL's float reductions can reduce across all tiles contained on one
    core. Reductions spanning multiple cores in the reduced dimension need a
    core gather/redistribute op to collect partials and place reduced values on
    the output-owning cores.
    """
    rank = len(layout.logical_shape)
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        raise ValueError(
            f"reduce dim must be in range [-{rank}, {rank - 1}], got {dim}"
        )
    if layout.grid_shape[dim] > 1 and not allow_cross_tile:
        raise ValueError(
            "collapsed reductions only support a reduced logical dimension "
            "that fits on one core; got "
            f"{layout.grid_shape[dim]} cores along dimension {dim}. "
            "Pass allow_cross_tile=True only when the kernel has an explicit "
            "cross-core gather/redistribute strategy for the reduced dimension."
        )

    shape = list(layout.logical_shape)
    block_shape = list(layout.block_shape)
    grid_shape = list(layout.grid_shape)
    shape[dim] = 1
    block_shape[dim] = 1
    grid_shape[dim] = 1
    return layout.replace(shape=shape, block_shape=block_shape, grid_shape=grid_shape)


def arange(layout: Layout, start: int = 0, step: int = 1) -> LazyTensor:
    """Allocate a device tensor filled with arange values.

    Equivalent to `torch.arange(start, start + N*step, step).reshape(shape)`
    where `N = prod(layout.logical_shape)` and `shape = layout.logical_shape`.
    Row-major linear traversal.

    Currently implemented as a host-side `torch.arange` + `to_layout`. This
    matches what TTIR's `arange` ends up costing for a precomputed mask
    (one DRAM transfer), but does **not** exercise the device-side
    `d2m.arange_block` op. A future zero-roundtrip version would emit
    `d2m.GenericOp { d2m.arange_block + remote_store }` (mirroring the C++
    `D2MArangeOpRewriter` in lib/Conversion/TTIRToD2M/TTIRToD2M.cpp).
    """
    if torch is None:
        raise RuntimeError("torch is required for d2m_jit.arange()")
    torch_dtype = _ttcore_to_torch_dtype(layout.dtype)
    numel = 1
    for d in layout.logical_shape:
        numel *= d
    flat = torch.arange(start, start + numel * step, step, dtype=torch_dtype)
    return to_layout(flat.reshape(list(layout.logical_shape)), layout)


def reshape(lt: LazyTensor, *shape) -> LazyTensor:
    """torch.reshape-style logical-shape change.

    Total element count must match. A single dimension may be given as
    `-1`, in which case its size is inferred from the remaining dims
    (e.g. `reshape(lt, -1)` flattens, `reshape(lt, 2, -1)` infers the
    last dim). Currently implemented via a host
    roundtrip (`to_host` -> `torch.reshape` -> `to_layout`), so it pays a
    DRAM transfer and re-tilises the data. Use it for shape changes that
    don't cleanly map to a `view` -- e.g. coalescing two non-adjacent dims
    or splitting one dim into many.

    Distinct from `view` / `view_layout` / `permute`, which are metadata
    reinterpretations of the buffer (no data movement, but require the
    new logical layout to be expressible as a permutation of the source's
    grid/tile dims).

    The destination layout reuses the source layout's `dtype`, `mem_space`,
    `tiled` setting, and either:
      - keeps the source's `block_shape` / `grid_shape` if they fit the
        new shape divisibility-wise, or
      - falls back to `block_shape=[1]*rank`, `grid_shape=[1]*rank`.
    Use `to_layout(reshaped, target_layout)` to land it in a specific
    layout afterwards.
    """
    if not isinstance(lt, LazyTensor):
        raise TypeError(f"reshape expected a LazyTensor, got {type(lt).__name__}")
    if torch is None:
        raise RuntimeError("torch is required for d2m_jit.reshape()")

    # Accept reshape(lt, 1, 2, 256, 64) and reshape(lt, [1, 2, 256, 64]).
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        new_shape = tuple(shape[0])
    else:
        new_shape = tuple(shape)

    src_numel = 1
    for d in lt.layout.logical_shape:
        src_numel *= d

    # Support the torch idiom of a single `-1` dim whose size is inferred
    # from the remaining dims (e.g. reshape(lt, -1) flattens; reshape(lt,
    # 2, -1) infers the second dim).
    neg_axes = [i for i, d in enumerate(new_shape) if d == -1]
    if len(neg_axes) > 1:
        raise ValueError(
            f"reshape: only one dimension may be inferred (-1), " f"got {new_shape}"
        )
    if any(d < -1 for d in new_shape):
        raise ValueError(f"reshape: dimensions must be >= -1, got {new_shape}")
    if neg_axes:
        known = 1
        for d in new_shape:
            if d != -1:
                known *= d
        if known == 0 or src_numel % known != 0:
            raise ValueError(
                f"reshape: cannot infer -1 dimension: src has {src_numel} "
                f"elements which is not divisible by the product of the "
                f"known dims {known} (from {new_shape})"
            )
        inferred = src_numel // known
        new_shape = tuple(inferred if d == -1 else d for d in new_shape)

    dst_numel = 1
    for d in new_shape:
        dst_numel *= d
    if src_numel != dst_numel:
        raise ValueError(
            f"reshape: total element count must match: "
            f"src {tuple(lt.layout.logical_shape)} ({src_numel}) "
            f"!= dst {new_shape} ({dst_numel})"
        )

    # Pick a destination layout: keep src's block/grid if compatible,
    # otherwise fall back to a trivial single-block single-grid layout
    # (the user can to_layout to something denser if perf matters).
    rank = len(new_shape)
    src_block = list(lt.layout.block_shape)
    src_grid = list(lt.layout.grid_shape)
    if (
        len(src_block) == rank
        and len(src_grid) == rank
        and all(
            d % (b * g * (32 if lt.layout.tiled else 1)) == 0
            for d, b, g in zip(new_shape, src_block, src_grid)
        )
    ):
        block_shape = src_block
        grid_shape = src_grid
    else:
        block_shape = [1] * rank
        grid_shape = [1] * rank

    dst_layout = lt.layout.replace(
        shape=new_shape,
        block_shape=block_shape,
        grid_shape=grid_shape,
    )

    host = lt.to_host().reshape(new_shape)
    return to_layout(host, dst_layout)


def _derive_perm_layout(src_layout: Layout, spec):
    """If `spec` (from _affine_map_from_lambda) describes a clean permutation
    of paired (grid, tile) dims, return a Layout with logical_shape/
    block_shape/grid_shape permuted accordingly. Otherwise return None."""
    n_logical = len(src_layout.logical_shape)
    expected = 2 * n_logical
    if len(spec) != expected:
        return None
    # The lifted blocked-rank perm has the form
    #   [p0, p1, ..., p_{N-1}, p0+N, p1+N, ..., p_{N-1}+N]
    # where (p0..p_{N-1}) is a permutation of (0..N-1).
    head = spec[:n_logical]
    tail = spec[n_logical:]
    perm = []
    for tag, val in head:
        if tag != "dim" or val >= n_logical:
            return None
        perm.append(val)
    # Verify tail mirrors head with +N offset.
    for i, (tag, val) in enumerate(tail):
        if tag != "dim" or val != perm[i] + n_logical:
            return None
    if sorted(perm) != list(range(n_logical)):
        return None
    return src_layout.replace(
        shape=[src_layout.logical_shape[p] for p in perm],
        block_shape=[src_layout.block_shape[p] for p in perm],
        grid_shape=[src_layout.grid_shape[p] for p in perm],
    )


def _emit_view_layout(lt: LazyTensor, affine_map, spec) -> LazyTensor:
    """Lower form: take an already-built AffineMap + spec and emit
    `d2m.view_layout`. Used by both view_layout (with a user lambda)
    and view (with a lifted blocked-rank spec built in Python)."""
    b = _get_scope()
    with b.ctx, b.loc, b.insert_point:
        src_type = lt.value.type
        src_shape = list(src_type.shape)
        if affine_map.n_dims != len(src_shape):
            raise ValueError(
                f"view_layout: lambda takes {affine_map.n_dims} args but "
                f"source MLIR rank is {len(src_shape)}"
            )
        simple_dim_or_const = all(tag in {"dim", "const"} for tag, _ in spec)
        if simple_dim_or_const:
            dst_shape = []
            for tag, val in spec:
                dst_shape.append(src_shape[val] if tag == "dim" else 1)
        else:
            # For now, affine-arithmetic view_layout lambdas are remappings over
            # the same physical shape. If future users need arithmetic maps that
            # also change shape/rank, add an explicit shape= parameter rather
            # than trying to infer bounds from arbitrary affine expressions.
            if len(spec) != len(src_shape):
                raise ValueError(
                    "view_layout: affine-arithmetic remappings currently "
                    "preserve source rank and shape; got "
                    f"{len(spec)} results for source rank {len(src_shape)}"
                )
            dst_shape = src_shape
        dst_ty = RankedTensorType.get(
            dst_shape, src_type.element_type, encoding=src_type.encoding
        )
        val = d2m.ViewLayoutOp(dst_ty, lt.value, affine_map).result
    new_layout = _derive_perm_layout(lt.layout, spec) or lt.layout
    return LazyTensor(new_layout, val, b.generation, is_view=True)


def view_layout(lt: LazyTensor, remapping_fn) -> LazyTensor:
    """Emit a `d2m.view_layout` with a user-supplied affine remapping.

    `remapping_fn` is a Python lambda whose parameter count matches the
    source value's MLIR rank (typically 2N for an N-dim logical tiled
    tensor: the first N dims are grid, the trailing N are per-grid tile
    indices). Each result expression may reference a parameter (perm /
    passthrough), be the literal 0 (broadcast-to-1), or use affine arithmetic
    with integer constants (`+`, `-`, `*`, `//`, `%`).

    The result LazyTensor's Layout is derived from the source by
    permuting logical_shape/block_shape/grid_shape if the lambda is a
    paired (grid, tile) permutation. Arithmetic remappings preserve the source
    physical shape and inherit the source Layout unchanged.
    """
    lt = lt._resolve()
    b = _get_scope()
    with b.ctx, b.loc:
        affine_map, spec = _affine_map_from_lambda(remapping_fn)
    return _emit_view_layout(lt, affine_map, spec)


def _emit_perm_view(lt: LazyTensor, perm) -> LazyTensor:
    """Lift a logical-rank permutation to blocked rank and emit a view.

    `perm` is a list of logical dim indices forming a permutation. The
    blocked map applies the same permutation independently to the grid
    half and the tile half of the source's MLIR shape.
    """
    n_logical = len(lt.layout.logical_shape)
    if sorted(perm) != list(range(n_logical)):
        raise ValueError(
            f"permutation {list(perm)} is not a rearrangement of (0..{n_logical-1})"
        )
    lifted_perm = list(perm) + [p + n_logical for p in perm]
    lifted_spec = [("dim", p) for p in lifted_perm]
    b = _get_scope()
    with b.ctx, b.loc:
        lifted_map = AffineMap.get(
            2 * n_logical, 0, [AffineDimExpr.get(p) for p in lifted_perm]
        )
    return _emit_view_layout(lt, lifted_map, lifted_spec)


def view(lt: LazyTensor, remapping_fn) -> LazyTensor:
    """Logical-rank view. `remapping_fn`'s parameter count matches the
    source's *logical* rank (e.g. 2 for a 512x512 tensor).

    Lifts the logical permutation to the blocked MLIR rank by applying
    the same permutation independently to the grid dims and the per-grid
    tile dims, then delegates to `view_layout`'s emit body. Only true
    permutations (no constants) are supported here -- use `view_layout`
    for richer remappings.
    """
    lt = lt._resolve()
    b = _get_scope()
    with b.ctx, b.loc:
        _, logical_spec = _affine_map_from_lambda(remapping_fn)
    n_logical = len(lt.layout.logical_shape)
    if len(logical_spec) != n_logical or any(tag != "dim" for tag, _ in logical_spec):
        raise ValueError(
            "view: lambda must be a permutation of logical dims (no constants); "
            "use view_layout for richer remappings"
        )
    return _emit_perm_view(lt, [val for _, val in logical_spec])


def permute(lt: LazyTensor, *dims) -> LazyTensor:
    """torch.permute-style logical-dim permutation.

    `dims` is a positional list of logical dim indices in the new order:

      d2m.permute(lt, 1, 0)       # 2D transpose
      d2m.permute(lt, 0, 2, 1)    # swap last two of a 3D logical tensor

    Returns a view; subsequent `to_host` requires a materialising
    `to_layout` (same rule as for any d2m view).
    """
    if not isinstance(lt, LazyTensor):
        raise TypeError(f"permute expected a LazyTensor, got {type(lt).__name__}")
    lt = lt._resolve()
    n_logical = len(lt.layout.logical_shape)
    if len(dims) != n_logical:
        raise ValueError(
            f"permute: expected {n_logical} dim indices for logical rank "
            f"{n_logical}, got {len(dims)}: {dims}"
        )
    return _emit_perm_view(lt, list(dims))


# --- Materialisation ---------------------------------------------------------


def _emit_returns_and_finalise(b: _Builder, lts):
    """Emit `from_device` for each LazyTensor and a func.ReturnOp, then
    update the func's signature with the new return types."""
    host_values = []
    host_types = []
    with b.ctx, b.loc, b.insert_point:
        for lt in lts:
            dev = lt.layout.build_device_view(b.ctx, lt.value)
            if lt.mesh is not None:
                # Bring the device shard back to a tensor_mesh-encoded host
                # shard (multi-device), then gather to the full tensor. The
                # encoding is what lets the runtime size the distributed host
                # buffer to match the mesh device buffer.
                elem = lt.layout.get_host_elem_type(b.ctx)
                tensor_mesh = _tensor_mesh_attr(b)
                host_shard_ty = RankedTensorType.get(
                    lt.layout.logical_shape, elem, encoding=tensor_mesh
                )
                out_empty = d2m.empty(host_shard_ty)
                host_shard = d2m.ToLayoutOp([host_shard_ty], dev, out_empty).result
                full_ty = RankedTensorType.get(lt.mesh.full_shape, elem)
                host = _emit_mesh_shard(
                    b,
                    host_shard,
                    full_ty,
                    "shard_to_full",
                    lt.mesh.shard_dims,
                    lt.mesh.shard_shape,
                )
                host_ty = full_ty
            else:
                host = lt.layout.build_from_device(b.ctx, dev)
                host_ty = lt.layout.build_host_tensor_type(b.ctx)
            host_values.append(host)
            host_types.append(host_ty)
        func.ReturnOp(host_values)
    b._refresh_function_type(results=host_types)


def _register_device(b: _Builder):
    """Run `ttcore-register-device` on the module.

    Done as its own pass (separate from `_run_pipeline`) and *before* the
    pre-pipeline `verify()` in `to_host`: some op verifiers (e.g.
    `create_global_semaphore`, which sizes its backing buffer to the device
    worker grid) call `ttcore::lookupDevice`, which asserts when no device op
    is present. Registering first makes that verify well-defined.
    """
    system_desc = _get_system_desc_path()
    opts = []
    if system_desc:
        opts.append(f"system-desc-path={system_desc}")
    # Mesh shape comes from the module's ttcore.meshes attr (determineMeshShape
    # reads it); only the topology needs to be passed as a pass option. Passing
    # mesh-shape here too would risk a conflict-error against the attr.
    topology = getattr(b, "_mesh_topology", None)
    if topology:
        opts.append("mesh-topology=" + ",".join(topology))
    register = "ttcore-register-device"
    if opts:
        register += "{" + " ".join(opts) + "}"
    if config.print_pipeline:
        print(f"[d2m-jit] register: {register}")
    pm = PassManager.parse(f"builtin.module({register})", context=b.ctx)
    pm.enable_verifier(config.verify_passes)
    pm.run(b.module.operation)


def _run_pipeline(b: _Builder):
    # The device is already registered (see `_register_device`, run before the
    # pre-pipeline verify in `to_host`), so this is the compute pipeline only.
    pipeline_str = f"builtin.module({','.join(_pipeline_passes())})"

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


# --- Mesh-device cache --------------------------------------------------------
#
# Opening + closing a mesh device on every execution repeatedly inits and tears
# down the UMD cluster: slow, and on n300 it flakes the ARC startup. Mirroring
# test/python/golden/conftest.py's `_get_device_for_target`, we keep one device
# open and reuse it across executions, only closing + reopening when the mesh
# shape or fabric setting changes. The d2m-jit conftest closes the cached device
# at session end and after a failing test (the hardware may be in an undefined
# state). A test that opens its own mesh device must call `_close_cached_device`
# first so two meshes are never open at once.
_cached_device = None
_cached_mesh_shape = None
_cached_fabric_used = None


def _get_cached_device(mesh_shape, fabric_used):
    """Return a mesh device for (`mesh_shape`, `fabric_used`), reusing the cached
    one on a match and otherwise closing it and opening a fresh one."""
    global _cached_device, _cached_mesh_shape, _cached_fabric_used
    mesh_shape = tuple(mesh_shape)
    if _cached_device is not None:
        if _cached_mesh_shape == mesh_shape and _cached_fabric_used == fabric_used:
            return _cached_device
        # Mesh / fabric mismatch: tear down before opening the new device.
        _close_cached_device()
    # CCL kernels need the device fabric enabled *before* the mesh is opened
    # (matching the golden harness). Without it the program's fabric ops
    # (device_synchronize / fabric remote_store / fabric semaphore incs) silently
    # no-op and the kernel hangs on semaphore_wait.
    if fabric_used:
        runtime.set_fabric_config(runtime.FabricConfig.FABRIC_1D_RING)
    device_options = runtime.MeshDeviceOptions()
    device_options.mesh_shape = list(mesh_shape)
    _cached_device = runtime.open_mesh_device(device_options)
    _cached_mesh_shape = mesh_shape
    _cached_fabric_used = fabric_used
    return _cached_device


def _close_cached_device():
    """Close and forget the cached mesh device (disabling fabric if it was on).
    Safe to call when nothing is cached."""
    global _cached_device, _cached_mesh_shape, _cached_fabric_used
    if _cached_device is None:
        return
    fabric_used = _cached_fabric_used
    try:
        runtime.close_mesh_device(_cached_device)
    finally:
        if fabric_used:
            runtime.set_fabric_config(runtime.FabricConfig.DISABLED)
        _cached_device = None
        _cached_mesh_shape = None
        _cached_fabric_used = None


_g_perf_trace_enabled = False


def _maybe_enable_perf_trace():
    """Flip the perf::Env singleton so the ttmetal executor dumps device
    profiler results after each workload. Must run before the first submit in
    the process (the singleton is seeded on first access). Idempotent.

    Device-side capture is controlled by tt-metal env vars that must be present
    *before* the device is opened (and DISPATCH must be 0 or the profiler read
    hangs on dispatch-core data). We do not mutate them here -- tt-metal reads
    them too early for that to be reliable -- but we warn if they are missing so
    the user sets them on the command line:
        TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_DISPATCH=0
    """
    global _g_perf_trace_enabled
    if not config.enable_perf_trace or _g_perf_trace_enabled:
        return
    if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        print(
            "[d2m-jit] WARNING: D2M_JIT_ENABLE_PERF_TRACE is set but "
            "TT_METAL_DEVICE_PROFILER=1 is not in the environment; no device "
            "profiler csv will be produced. Re-run with "
            "TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_DISPATCH=0 set."
        )
    runtime.PerfEnv.get(enable_perf_trace=True)
    _g_perf_trace_enabled = True
    print(
        "[d2m-jit] perf trace enabled; device profiler csv -> "
        "$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv"
    )


def _execute(b: _Builder, lts):
    """Serialize to flatbuffer, run on a mesh device, return torch tensors."""
    if runtime is None or binary is None:
        raise RuntimeError("ttmlir runtime is not available in this build")
    _maybe_enable_perf_trace()
    bin_capsule = ttmetal_to_flatbuffer_bin(b.module)
    fbb = binary.load_binary_from_capsule(bin_capsule)
    if config.save_flatbuffer_path:
        fbb.store(config.save_flatbuffer_path)
        print(f"[d2m-jit] flatbuffer written to {config.save_flatbuffer_path}")
    program_index = 0
    mesh_shape = fbb.get_program_mesh_shape(program_index)
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
        out_shape = (
            lt.mesh.full_shape if lt.mesh is not None else lt.layout.logical_shape
        )
        t_out = torch.empty(list(out_shape), dtype=torch_dtype)
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

    # Reuse a cached mesh device when possible (see _get_cached_device); CCL
    # kernels need the device fabric enabled before the mesh is opened, which
    # the cache handles based on `fabric_used`.
    fabric_used = getattr(b, "_fabric_used", False)
    device = _get_cached_device(mesh_shape, fabric_used)
    try:
        submitted = runtime.submit(device, fbb, program_index, rt_inputs)
        runtime.wait(submitted)
        for i, rt_out in enumerate(submitted):
            host_view = runtime.to_host(rt_out, untilize=True)[0]
            runtime.memcpy(rt_outputs[i], host_view)
            runtime.deallocate_tensor(rt_out, force=True)
    except Exception:
        # After a failed execution the device may be in an undefined state;
        # drop it so the next run opens a clean one.
        _close_cached_device()
        raise
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

    b = _get_scope()
    if not isinstance(b, _Builder):
        raise RuntimeError(
            "to_host() cannot be called from inside a non-lazy scope (e.g. a "
            "pattern-rewrite scope). The graph being built is part of the host "
            "module; its pipeline/execution is the host compiler's job, not the "
            "rewrite's."
        )

    resolved = [lt._resolve() for lt in lts]
    for i, lt in enumerate(resolved):
        if lt.is_view:
            raise ValueError(
                f"to_host: argument {i} is a view (created via "
                f"view/view_layout). Views are metadata reinterpretations "
                f"of an underlying buffer and cannot be materialised "
                f"directly. Convert to a concrete layout first, e.g. "
                f"to_layout(v, v.layout)."
            )
    # All resolved tensors must belong to this builder (resolve guarantees that).
    assert all(lt.generation == b.generation for lt in resolved)

    _emit_returns_and_finalise(b, resolved)
    # Register the device before verifying: device-dependent verifiers (e.g.
    # create_global_semaphore) assert without a registered device.
    _register_device(b)
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
    """Build an MLIR AffineMap by running `fn` with sentinel dim objects.

    Returns `(AffineMap, spec)` where `spec` is a list of one tag per
    result expression: `("dim", i)` for a bare input dim, `("const", 0)` for
    literal zero, or `("expr", None)` for affine arithmetic.
    """

    class _AffineExprProxy:
        def __init__(self, expr, spec=("expr", None)):
            self.expr = expr
            self.spec = spec

        @staticmethod
        def _constant(value):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    "view_layout affine expressions only support integer constants"
                )
            return AffineConstantExpr.get(value)

        @classmethod
        def _expr(cls, value):
            if isinstance(value, _AffineExprProxy):
                return value.expr
            return cls._constant(value)

        def _new(self, expr):
            return _AffineExprProxy(expr)

        def __add__(self, rhs):
            return self._new(self.expr + self._expr(rhs))

        def __radd__(self, lhs):
            return self._new(self._expr(lhs) + self.expr)

        def __sub__(self, rhs):
            return self._new(self.expr - self._expr(rhs))

        def __rsub__(self, lhs):
            return self._new(self._expr(lhs) - self.expr)

        def __mul__(self, rhs):
            self._constant(rhs)
            return self._new(self.expr * rhs)

        def __rmul__(self, lhs):
            self._constant(lhs)
            return self._new(self.expr * lhs)

        def __floordiv__(self, rhs):
            return self._new(AffineFloorDivExpr.get(self.expr, self._constant(rhs)))

        def __mod__(self, rhs):
            return self._new(AffineModExpr.get(self.expr, self._constant(rhs)))

        def __rfloordiv__(self, lhs):
            raise TypeError("view_layout does not support int // affine_expr")

        def __rmod__(self, lhs):
            raise TypeError("view_layout does not support int % affine_expr")

    class _Dim(_AffineExprProxy):
        def __init__(self, position):
            super().__init__(AffineDimExpr.get(position), ("dim", position))

    dims = tuple(_Dim(i) for i, _ in enumerate(inspect.signature(fn).parameters))
    results = fn(*dims)
    exprs = []
    spec = []
    for r in results:
        if isinstance(r, _AffineExprProxy):
            exprs.append(r.expr)
            spec.append(r.spec)
        elif isinstance(r, int):
            assert r == 0, "Only 0 is allowed as an integer constant in indexing_map"
            exprs.append(AffineConstantExpr.get(r))
            spec.append(("const", r))
        else:
            raise TypeError(
                f"Unsupported indexing_map result type {type(r).__name__}: {r}"
            )
    return AffineMap.get(len(dims), 0, exprs), spec


def _to_dram_kernel_arg(lt: LazyTensor) -> LazyTensor:
    if lt.layout.mem_space == ttcore.MemorySpace.DeviceDRAM:
        return lt
    return to_layout(lt, lt.layout.replace(mem_space=ttcore.MemorySpace.DeviceDRAM))


def _emit_kernel_generic(
    kernel: "CompiledKernel",
    args,
    grid,
    num_outs: int,
    block_factors,
    indexing_maps,
    iterator_types,
    fabric=None,
    kernel_io_in_dram=None,
):
    """Append a d2m.GenericOp to the open host func that invokes `kernel`."""
    b = _get_scope()

    # A `fabric=` kernel needs the device fabric enabled at open time; flag it so
    # `_execute` calls set_fabric_config before open_mesh_device.
    if fabric is not None and isinstance(b, _Builder):
        b._fabric_used = True

    def _call_error(msg, hint=None, cause=None):
        # Pin call-site errors to the kernel's `def` line. The user's actual
        # call site is already visible in the traceback, so the def-line
        # pointer at least tells them *which* kernel rejected the call.
        return D2mJitError(
            msg=msg,
            file=kernel._source_file,
            line=(
                kernel._source_firstlineno + (kernel._ast.body[0].lineno - 1)
                if kernel._ast.body
                else kernel._source_firstlineno
            ),
            col=None,
            source_lines=kernel._source_lines,
            snippet_line=(kernel._ast.body[0].lineno if kernel._ast.body else None),
            hint=hint,
            cause=cause,
        )

    # Split args, preserving "all LazyTensors precede all non-tensor args"
    # ordering. Non-tensor args (`extras`) are scalars (int) and global
    # semaphores, kept in their original order so they map 1:1 onto the
    # kernel's trailing parameters and the GenericOp's additionalArgs.
    lazy_args = []
    extras = []  # list of ("scalar", int) | ("sem", GlobalSemaphore)
    saw_extra = False
    for i, a in enumerate(args):
        if isinstance(a, LazyTensor):
            if saw_extra:
                raise _call_error(
                    f"argument {i} to kernel '{kernel.fn.__name__}' is a "
                    f"LazyTensor but a scalar/semaphore was already seen; "
                    f"tensor arguments must precede scalars and semaphores",
                    cause=TypeError(),
                )
            lazy_args.append(a._resolve())
        elif isinstance(a, GlobalSemaphore):
            saw_extra = True
            extras.append(("sem", a))
        elif isinstance(a, int) and not isinstance(a, bool):
            saw_extra = True
            extras.append(("scalar", a))
        else:
            raise _call_error(
                f"argument {i} to kernel '{kernel.fn.__name__}' has "
                f"unsupported type {type(a).__name__}: {a!r}",
                hint=(
                    "kernel arguments must be d2m_jit.LazyTensor, int, or "
                    "d2m_jit.GlobalSemaphore. Use d2m.to_layout(t, L) to lift "
                    "a torch tensor."
                ),
                cause=TypeError(),
            )

    if num_outs < 1:
        raise _call_error(f"num_outs must be >= 1 (got {num_outs})", cause=ValueError())
    if len(lazy_args) < num_outs:
        raise _call_error(
            f"kernel call has {len(lazy_args)} tensor args; need at least "
            f"{num_outs} for outputs",
            cause=ValueError(),
        )
    input_lts = lazy_args[: len(lazy_args) - num_outs]
    output_lts = lazy_args[len(lazy_args) - num_outs :]
    user_output_lts = output_lts

    if kernel_io_in_dram is None:
        kernel_io_in_dram = config.kernel_io_in_dram
    elif not isinstance(kernel_io_in_dram, bool):
        raise _call_error(
            f"kernel_io_in_dram must be a bool, got {type(kernel_io_in_dram).__name__}",
            cause=TypeError(),
        )

    if kernel_io_in_dram:
        dram_arg_cache = {}

        def to_dram(lt):
            key = id(lt)
            if key not in dram_arg_cache:
                dram_arg_cache[key] = _to_dram_kernel_arg(lt)
            return dram_arg_cache[key]

        input_lts = [to_dram(lt) for lt in input_lts]
        output_lts = [to_dram(lt) for lt in output_lts]
        lazy_args = input_lts + output_lts

    # In a non-lazy (rewrite) scope the surrounding module is not ours to add
    # function params to, so runtime scalars would lower to host-scope
    # `arith.constant` index values fed into the generic's additionalArgs --
    # which the ttmetal flatbuffer translator cannot serialize (it only
    # resolves scalar kernel args that are program inputs, so an inline
    # constant hits a missing-BufferRef assertion). Since rewrite-scope scalars
    # are always Python int constants, bake them into the kernel body as
    # captures (in-region constants) and emit no additionalArgs for them. The
    # lazy `_Builder` keeps the runtime-arg form: scalars stay index func args
    # (see add_scalar_input) so the binary remains parameterised. Semaphores are
    # runtime handles and can never be baked, so they always stay in `extras`.
    bake_scalars = not isinstance(b, _Builder)
    scalar_args = [val for kind, val in extras if kind == "scalar"]
    if bake_scalars and scalar_args:
        formal_names = [a.arg for a in kernel._ast.body[0].args.args]
        # The trailing formals map 1:1 onto `extras` in order; pair each
        # "scalar" extra with its formal name so it becomes an in-region capture.
        extra_names = formal_names[len(lazy_args) : len(lazy_args) + len(extras)]
        scalar_names = [
            name for (kind, _), name in zip(extras, extra_names) if kind == "scalar"
        ]
        effective_captures = dict(kernel._captures)
        effective_captures.update(
            {n: int(v) for n, v in zip(scalar_names, scalar_args)}
        )
        # Baked scalars leave `extras`; only semaphores remain as runtime
        # additionalArgs / func params.
        extras = [(kind, val) for kind, val in extras if kind != "scalar"]
    else:
        effective_captures = kernel._captures

    # Compile the kernel body in the current builder's context. D2MCompiler
    # picks up b.ctx via get_default_loc_context.
    with b.ctx, b.loc:
        # Per-extra D2MCompiler arg: scalars pass their int (-> index func
        # arg); semaphores pass the SEMAPHORE_ARG sentinel (-> global_semaphore
        # func arg). Baked scalars have been removed from `extras` above and
        # instead live in `effective_captures`. Order matches the kernel's
        # trailing parameters.
        extra_compiler_args = [
            val if kind == "scalar" else SEMAPHORE_ARG for kind, val in extras
        ]
        compiler_args = [lt.layout for lt in lazy_args] + extra_compiler_args
        compiler = D2MCompiler(
            kernel.fn.__name__,
            kernel.thread_type,
            effective_captures,
            *compiler_args,
            source_file=kernel._source_file,
            source_firstlineno=kernel._source_firstlineno,
            source_lines=kernel._source_lines,
        )
        compiler.visit(kernel._ast)
        compiler.module.operation.verify()

    # Emit the GenericOp + splice the kernel body.
    with b.ctx, b.loc, b.insert_point:
        # additionalArgs operands, in the kernel's trailing-parameter order:
        #  - scalars are sourced from func args (not host-scope constants) so
        #    the GenericOp's region stays isolated-from-above (in a rewrite
        #    scope the scalars were baked into captures above and dropped from
        #    `extras`, so none are emitted here),
        #  - semaphores reuse the host-scope `create_global_semaphore` result.
        additional = [
            b.add_scalar_input(val) if kind == "scalar" else val._resolve_value()
            for kind, val in extras
        ]
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

        indexing_attrs = [_affine_map_from_lambda(f)[0] for f in (indexing_maps or [])]
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
            fabricConnectionConfig=fabric,
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

        # Reset each semaphore after the generic that consumed it, so its
        # backing buffer becomes dead and is deallocated (matching the
        # reset_global_semaphore + dealloc pattern in
        # test/.../generic_global_semaphores.mlir). NOTE: this assumes a
        # semaphore is used by a single kernel call; multi-kernel reuse would
        # need an explicit host-side reset instead.
        for kind, val in extras:
            if kind == "sem":
                d2m.reset_global_semaphore(val._resolve_value(), 0)

    # Rebind output LazyTensors to the GenericOp's results.
    for i, lt in enumerate(output_lts):
        lt.value = generic.results[i]
        lt.generation = b.generation
        # The output may have been a reblocked stream view (is_view=True); after
        # the kernel writes through it, it holds a real result and is
        # materialisable (the result aliases the underlying buffer).
        lt.is_view = False
    if kernel_io_in_dram:
        for i, (user_lt, kernel_lt) in enumerate(zip(user_output_lts, output_lts)):
            user_lt.layout = kernel_lt.layout
            user_lt.value = generic.results[i]
            user_lt.generation = b.generation
            user_lt.materialized = None
            user_lt.is_view = kernel_lt.is_view


# All kernels are authored as the "unified" implicit-blocked compute+
# datamovement form; the backend (split-unified-thread-v2 -> schedule-dma)
# splits it into per-thread regions. CCL kernels (device_synchronize,
# cross-device remote_store, semaphores) are authored unified too -- the split
# pins the device_synchronize barrier to a single datamovement thread (see
# ScheduleDMA). Explicit semaphore_set/inc are permitted but not yet split
# safely (see checkForIllegalSemaphoreOps's TODO in DMAUtils.cpp).


class CompiledKernel:
    """Wraps a user kernel function. Parses the Python body once; emits a
    `d2m.GenericOp` (unified thread) into the current builder on every call."""

    def __init__(self, fn):
        functools.update_wrapper(self, fn)
        self.fn = fn
        self.thread_type = "unified"
        (
            self._source,
            self._source_firstlineno,
            self._source_file,
            self._source_lines,
        ) = _cleanup_source_code(fn)
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
        fabric=None,
        kernel_io_in_dram=None,
    ):
        _emit_kernel_generic(
            self,
            args,
            grid=grid,
            num_outs=num_outs,
            block_factors=block_factors,
            indexing_maps=indexing_maps,
            iterator_types=iterator_types,
            fabric=fabric,
            kernel_io_in_dram=kernel_io_in_dram,
        )


def fabric_config(
    cluster_axis,
    topology="ring",
    num_links=1,
    noc="noc0",
    routing="unidir_ring_torus",
    router_cores=None,
):
    """Build a `#ttcore.fabric_connection_config` attribute for a CCL kernel.

    Pass the result as the `fabric=` argument of a `@d2m.kernel` call; it
    configures the cross-device fabric routing for the GenericOp (required for
    all_gather and other collectives). Defaults match the all_gather lowering
    (`noc0`, ring topology, unidirectional ring/torus routing).

    `router_cores` optionally restricts the fabric to a subset of the generic's
    grid: a list of `(y, x)` grid coordinates, one per `(link, direction)` slot
    (slot `i` -> routing plane `i // cores_per_link`, direction
    `i % cores_per_link`; `cores_per_link` is 1 for `bidir_line_mesh`, 2 for
    `unidir_ring_torus`). Omitted == the whole grid (legacy behavior). The kernel
    queries membership via `is_router_core()` / `router_direction()`. See
    tools/d2m-jit/fabric_router_cores_design.md."""
    b = _get_scope()
    routers = ""
    if router_cores:
        flat = ", ".join(str(int(v)) for yx in router_cores for v in yx)
        routers = f", router_cores = [{flat}]"
    with b.ctx, b.loc:
        return Attribute.parse(
            f"#ttcore.fabric_connection_config<noc_index = {noc}, "
            f"topology = {topology}, cluster_axis = {int(cluster_axis)}, "
            f"routing_mode = {routing}, num_links = {int(num_links)}{routers}>",
            b.ctx,
        )


def kernel(fn):
    """Decorate a user function as a d2m_jit kernel.

    All kernels are authored as the `unified` compute+datamovement form; the
    backend splits it into per-thread regions. CCL kernels (device_synchronize,
    cross-device remote_store, semaphores) are authored the same way -- the
    split pins the device_synchronize barrier to a single datamovement thread.
    """
    return CompiledKernel(fn)
