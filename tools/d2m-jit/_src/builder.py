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
from collections import Counter
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
from .layout_math import (
    reduction_layout,
    resolve_reshape,
    MeshShard,
    validate_mesh_mapping as _validate_mesh_mapping,
    shard_logical_shape as _shard_logical_shape,
    set_current_mesh as _set_current_mesh,
    clear_current_mesh as _clear_current_mesh,
)
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
            ttcore.DataType.UInt32: torch.uint32,
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


class FabricConfig:
    """Host-side description of a generic's 1D fabric connection."""

    __slots__ = ("cluster_axis", "topology", "routing", "noc", "num_links")

    _ROUTING_BY_TOPOLOGY = {
        "linear": "bidir_line_mesh",
        "ring": "unidir_ring_torus",
    }

    def __init__(
        self,
        cluster_axis,
        topology="ring",
        routing=None,
        noc="noc0",
        num_links=1,
    ):
        if (
            not isinstance(cluster_axis, int)
            or isinstance(cluster_axis, bool)
            or cluster_axis < 0
        ):
            raise ValueError(
                f"fabric cluster_axis must be a non-negative integer, got {cluster_axis!r}"
            )

        topology = str(topology).lower()
        if topology not in self._ROUTING_BY_TOPOLOGY:
            raise ValueError(
                "fabric topology must be 'linear' or 'ring', " f"got {topology!r}"
            )

        expected_routing = self._ROUTING_BY_TOPOLOGY[topology]
        routing = expected_routing if routing is None else str(routing).lower()
        if routing != expected_routing:
            raise ValueError(
                f"fabric topology {topology!r} requires routing "
                f"{expected_routing!r}, got {routing!r}"
            )

        noc = str(noc).lower()
        if noc not in {"noc0", "noc1"}:
            raise ValueError(f"fabric noc must be 'noc0' or 'noc1', got {noc!r}")
        if (
            not isinstance(num_links, int)
            or isinstance(num_links, bool)
            or num_links <= 0
        ):
            raise ValueError(
                f"fabric num_links must be a positive integer, got {num_links!r}"
            )

        self.cluster_axis = cluster_axis
        self.topology = topology
        self.routing = routing
        self.noc = noc
        self.num_links = num_links

    @property
    def runtime_mode(self):
        return "FABRIC_1D_RING" if self.topology == "ring" else "FABRIC_1D"

    def build_attr(self, ctx):
        return Attribute.parse(
            "#ttcore.fabric_connection_config<"
            f"noc_index = {self.noc}, "
            f"topology = {self.topology}, "
            f"cluster_axis = {self.cluster_axis}, "
            f"routing_mode = {self.routing}, "
            f"num_links = {self.num_links}>",
            ctx,
        )


def fabric_config(
    cluster_axis,
    topology="ring",
    routing=None,
    noc="noc0",
    num_links=1,
):
    """Create a validated 1D fabric configuration for a kernel call."""
    return FabricConfig(cluster_axis, topology, routing, noc, num_links)


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

    When `config.insert_profiler_traces` is set, the TTKernel
    `insert-device-zone-scopes` pass is spliced in after `ttkernel-hoist-inits`
    and before the EmitC tail — the same slot createTTIRToTTMetalPipeline uses.
    It must run while the kernel body is still in TTKernel form (it walks
    TTKernel ops to wrap them in `DeviceZoneScopedN` scopes) and after
    hoist-inits so the dispatch-level conversion sees the original loop
    structure.
    """
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
        f"d2m-be-pipeline{{use-tile-matmul={int(config.use_tile_matmul)}}}",
        "d2m-to-ttkernel-pre-emitc-pipeline",
        "d2m-to-ttmetal-pipeline",
        "func.func(ttkernel-hoist-inits)",
    ]
    if config.insert_profiler_traces:
        traits = config.profiler_traits.strip() or "device-zone"
        passes.append("func.func(insert-device-zone-scopes{traits=" + traits + "})")
    passes.append("d2m-emitc-pipeline")
    return passes


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
# op's site rather than into a fresh module.
#
# `_SpatialRegionScope` is pushed for each `d2m.spatial` region so nested
# kernel calls emit into that region's block (and apply spatial-only policy
# such as grid offset / VGM remap). `d2m.spatial()` itself requires the
# lazy `_Builder` scope; nested spatial is not supported.
#
# From the perspective of the emission helpers, all scopes quack the same:
# they expose `ctx`, `loc`, `insert_point`, `generation`, `add_host_input`,
# `add_scalar_input`.
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
        # The shape is represented by the module's ttcore.meshes attribute.
        # Topology is a register-device pass option.
        self._mesh_shape = None
        self._mesh_topology = None
        self._mesh_name = None
        self._fabric_runtime_mode = None
        # Reset the MLIR-free mesh mirror the simulator reads (a fresh graph
        # declares its own mesh).
        _clear_current_mesh()

    def set_mesh(self, shape, topology=None):
        """Declare the device mesh used by this graph."""
        shape = tuple(shape)
        if not shape or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
            for dim in shape
        ):
            raise ValueError(f"mesh shape must contain positive integers, got {shape}")
        if len(shape) != 2:
            raise ValueError(f"TTMetal requires a rank-2 mesh, got shape {shape}")
        if topology is not None:
            topology = tuple(topology)
            if len(topology) != len(shape):
                raise ValueError(
                    "mesh topology must have one entry per mesh dimension, "
                    f"got shape {shape} and topology {topology}"
                )
            invalid = [
                value
                for value in topology
                if value not in {"disabled", "linear", "ring"}
            ]
            if invalid:
                raise ValueError(
                    "mesh topology entries must be 'disabled', 'linear', or "
                    f"'ring', got {invalid}"
                )
        requested_topology = list(topology) if topology is not None else None
        if self._mesh_shape is not None:
            if (
                self._mesh_shape == list(shape)
                and self._mesh_topology == requested_topology
            ):
                return
            raise RuntimeError(
                "the current graph already declares mesh "
                f"{tuple(self._mesh_shape)} with topology "
                f"{self._mesh_topology}; it cannot be redefined"
            )

        dims = "x".join(str(dim) for dim in shape)
        with self.ctx, self.loc:
            meshes = Attribute.parse(f'#ttcore.meshes<[<"mesh" = {dims}>]>', self.ctx)
        self.module.operation.attributes["ttcore.meshes"] = meshes
        self._mesh_shape = list(shape)
        self._mesh_topology = requested_topology
        self._mesh_name = "mesh"
        _set_current_mesh(self._mesh_shape, self._mesh_topology, self._mesh_name)

    def enable_fabric(self, fabric):
        """Validate a kernel fabric config and record its runtime mode."""
        if not isinstance(fabric, FabricConfig):
            raise TypeError(
                "fabric must be created with d2m.fabric_config(), "
                f"got {type(fabric).__name__}"
            )
        if self._mesh_shape is None:
            raise RuntimeError(
                "fabric kernels require a device mesh; call d2m.mesh() first"
            )
        if fabric.cluster_axis >= len(self._mesh_shape):
            raise ValueError(
                f"fabric cluster_axis {fabric.cluster_axis} is out of range for "
                f"mesh shape {tuple(self._mesh_shape)}"
            )
        if self._mesh_shape[fabric.cluster_axis] < 2:
            raise ValueError(
                f"fabric cluster_axis {fabric.cluster_axis} has size "
                f"{self._mesh_shape[fabric.cluster_axis]}; expected at least 2"
            )
        if self._mesh_topology is None:
            raise RuntimeError(
                "fabric kernels require an explicit mesh topology; pass "
                "topology= to d2m.mesh()"
            )
        mesh_topology = self._mesh_topology[fabric.cluster_axis]
        if mesh_topology != fabric.topology:
            raise ValueError(
                f"fabric topology {fabric.topology!r} does not match mesh "
                f"topology {mesh_topology!r} on axis {fabric.cluster_axis}"
            )
        if (
            self._fabric_runtime_mode is not None
            and self._fabric_runtime_mode != fabric.runtime_mode
        ):
            raise ValueError(
                "all fabric kernels in one graph must use the same runtime "
                f"mode; already using {self._fabric_runtime_mode}, got "
                f"{fabric.runtime_mode}"
            )
        self._fabric_runtime_mode = fabric.runtime_mode

    def _refresh_function_type(self, results=None):
        with self.ctx, self.loc:
            ft = FunctionType.get(self._input_types, results or [])
            self.func_op.attributes["function_type"] = TypeAttr.get(ft)

    def add_host_input(self, layout: Layout, host_tensor, host_ty=None):
        """Append a host-typed func arg and return its BlockArgument."""
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
        # Rewrite mode has no host func to add args to, so materialize the host
        # tensor as an `arith.constant dense<...>` at the rewriter IP and return
        # its host-typed SSA value. This lets rewrites introduce small constant
        # operands (e.g. identity / eps tiles a fused kernel needs) that are not
        # present in the matched graph. `to_layout` then emits the to_device.
        import numpy as np

        host_ty = layout.build_host_tensor_type(self.ctx)
        cpu = host_tensor.detach().cpu().contiguous()
        # PyTorch bfloat16 cannot convert to NumPy directly; pack as uint16
        # bits and attach the host tensor type explicitly (same as TTIRBuilder).
        if cpu.dtype == torch.bfloat16:
            arr = cpu.view(torch.int16).numpy().astype(np.uint16)
        else:
            arr = np.ascontiguousarray(cpu.numpy())
        with self.ctx, self.loc, self.insert_point:
            attr = DenseElementsAttr.get(arr, type=host_ty)
            return arith.ConstantOp(host_ty, attr).result

    def add_scalar_input(self, value: int):
        # Emit an arith.constant of index type at the rewriter's IP.
        # This is the rewrite-mode analog of _Builder.add_scalar_input, which
        # would have added a func arg in lazy mode.
        with self.ctx, self.loc, self.insert_point:
            idx_ty = IndexType.get(self.ctx)
            return arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, int(value))).result


class _SpatialRegionScope:
    """Build scope for one `d2m.spatial` region (exactly one nested generic)."""

    def __init__(self, parent, insert_point, spatial_op, grid_shape, offset):
        self.parent = parent
        self.ctx = parent.ctx
        self.loc = parent.loc
        self.insert_point = insert_point
        self.generation = parent.generation
        self.spatial_op = spatial_op
        self.grid_shape = list(grid_shape)
        self.offset = list(offset)
        if len(self.offset) != 2:
            raise ValueError(f"d2m.spatial grid offset must be 2D, got {self.offset}")
        self._emitted_generic = None
        self._finished = False

    def add_host_input(self, layout, host_tensor):
        raise RuntimeError(
            "Cannot lift a host tensor from inside d2m.spatial. Spatial "
            "regions operate on device tensors only; prepare inputs/outputs "
            "outside the spatial op and pass them via inputs=/outputs=."
        )

    def add_scalar_input(self, value: int):
        return self.parent.add_scalar_input(value)

    def finish(self):
        """Emit `d2m.spatial_yield` of the nested generic results; return its
        output operands for matching against SpatialOp outs=."""
        if self._finished:
            raise RuntimeError("d2m.spatial region already finished")
        if self._emitted_generic is None:
            raise ValueError(
                "each d2m.spatial region must emit exactly one @d2m.kernel call"
            )
        # TODO (hkwon): Allow yielding values other than the nested generic results.
        with self.ctx, self.loc, self.insert_point:
            d2m.spatial_yield(list(self._emitted_generic.results))
        self._finished = True
        return list(self._emitted_generic.outputs)

    def _validate_grid(self, grid):
        """Validate nested generic grid; return normalized 2D list."""
        grid = list(grid)
        if len(grid) != 2:
            raise ValueError(
                f"d2m.spatial only supports 2D nested generic grids, got {grid}"
            )
        if grid[0] > self.grid_shape[0] or grid[1] > self.grid_shape[1]:
            raise ValueError(
                f"nested generic grid {grid} exceeds spatial region shape "
                f"{self.grid_shape}"
            )
        return grid

    def _make_offset_vgm_maps(self):
        oy, ox = self.offset
        d0 = AffineDimExpr.get(0)
        d1 = AffineDimExpr.get(1)
        zero = AffineExpr.get_constant(0, self.ctx)

        def add_const(expr, value):
            if value == 0:
                return expr
            return AffineExpr.get_add(expr, AffineExpr.get_constant(value, self.ctx))

        inverse = AffineMap.get(2, 0, [zero, add_const(d0, -oy), add_const(d1, -ox)])
        dims = [AffineDimExpr.get(i) for i in range(4)]
        forward = AffineMap.get(
            4,
            0,
            [
                add_const(dims[0], oy),
                add_const(dims[1], ox),
                dims[2],
                dims[3],
            ],
        )
        return inverse, forward

    def _remap_spatial_output(self, lt: "LazyTensor"):
        # Replace the matching SpatialOp out with a VGM empty so L1 lands on
        # this region's physical cores.
        output_idx = next(
            (i for i, v in enumerate(self.spatial_op.outputs) if v == lt.value), None
        )
        if output_idx is None:
            raise ValueError("kernel output not listed in d2m.spatial(outputs=...)")

        inverse, forward = self._make_offset_vgm_maps()
        with self.ctx, self.loc, InsertionPoint(self.spatial_op.operation):
            remapped_value = d2m.empty(
                lt.value.type,
                virtual_grid_inverse_mapping=inverse,
                virtual_grid_forward_mapping=forward,
            )
        self.spatial_op.operation.operands[
            len(self.spatial_op.inputs) + output_idx
        ] = remapped_value
        lt.value = remapped_value
        return lt

    def _remap_output_args(self, args, num_outs):
        """Remap trailing num_outs LazyTensors; return prepared args."""
        args = list(args)
        num_tensors = next(
            (i for i, a in enumerate(args) if not isinstance(a, LazyTensor)),
            len(args),
        )
        if num_outs < 1 or num_tensors < num_outs:
            raise ValueError(
                f"need at least {num_outs} tensor args for outputs, got {num_tensors}"
            )
        tensors, rest = args[:num_tensors], args[num_tensors:]
        inputs = tensors[:-num_outs]
        outputs = [lt._resolve() for lt in tensors[-num_outs:]]
        remapped = [self._remap_spatial_output(lt) for lt in outputs]
        return tuple(inputs + remapped + rest)

    def _emit_kernel_for_spatial(
        self,
        kernel: "CompiledKernel",
        args,
        grid,
        num_outs: int,
        block_factors,
        indexing_maps,
        iterator_types,
        kernel_io_in_dram=None,
    ):
        """Emit one nested generic for this spatial region."""
        if self._emitted_generic is not None:
            raise ValueError(
                "each d2m.spatial region must emit exactly one @d2m.kernel call"
            )

        grid = self._validate_grid(grid)

        # TODO (hkwon): Support kernel_io_in_dram inside d2m.spatial by
        # updating SpatialOp outs/result types to the DRAM destination,
        # similar to _remap_spatial_output's VGM operand rewrite.
        resolved_dram = (
            config.kernel_io_in_dram if kernel_io_in_dram is None else kernel_io_in_dram
        )
        if resolved_dram is True:
            raise ValueError(
                "kernel_io_in_dram is not currently supported inside "
                "d2m.spatial; pass kernel_io_in_dram=False or disable "
                "d2m.config.kernel_io_in_dram"
            )

        if self.offset[0] != 0 or self.offset[1] != 0:
            args = self._remap_output_args(args, num_outs)

        self._emitted_generic = _emit_kernel_generic(
            kernel,
            args,
            grid=grid,
            num_outs=num_outs,
            block_factors=block_factors,
            indexing_maps=indexing_maps,
            iterator_types=iterator_types,
            kernel_io_in_dram=kernel_io_in_dram,
            grid_offset=self.offset,
        )


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
        # Optional metadata describing this tensor's per-device mesh shard.
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
            if self.mesh is not None:
                return mesh_shard(
                    self.materialized,
                    self.layout,
                    self.mesh.shard_dims,
                    self.mesh.shard_shape,
                )
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
        return LazyTensor(layout, val, b.generation, mesh=src.mesh)

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
    """Build a 1x1-sharded ui32 backing tensor over the worker grid."""
    rank = len(grid_shape)
    i64_type = IntegerType.get_signless(64, ctx)
    interval_type = RankedTensorType.get([rank, 2], i64_type)
    interval_values = [
        IntegerAttr.get(i64_type, endpoint)
        for i in range(rank)
        for endpoint in (i, i + 1)
    ]
    collapse = DenseIntElementsAttr.get(interval_values, type=interval_type)
    metal_layout = ttcore.ir.MetalLayoutAttr.get(
        ctx,
        list(grid_shape),
        int(ttcore.MemorySpace.DeviceL1),
        int(ttcore.TensorMemoryLayout.Sharded),
        collapse,
        [1] * rank,
    )
    device_shape = ttcore.ir.MetalLayoutAttr.maybe_downcast(
        metal_layout
    ).getDeviceShape(list(grid_shape), [])
    return RankedTensorType.get(
        device_shape, IntegerType.get_unsigned(32, ctx), encoding=metal_layout
    )


class GlobalSemaphore:
    """Host handle for a `!d2m.global_semaphore` kernel argument."""

    __slots__ = ("value", "generation", "grid_shape", "init", "_consumed")

    def __init__(self, value, generation, grid_shape, init):
        self.value = value
        self.generation = generation
        self.grid_shape = tuple(grid_shape)
        self.init = init
        self._consumed = False

    def _resolve_value(self):
        b = _get_scope()
        if self.generation != b.generation:
            raise RuntimeError(
                "Stale GlobalSemaphore: create it in the same builder "
                "generation as the kernel call."
            )
        if self._consumed:
            raise RuntimeError(
                "GlobalSemaphore has already been consumed by a kernel call. "
                "Create a separate semaphore for each call."
            )
        return self.value


_g_worker_grid = None
_g_worker_grid_source = None


def _device_worker_grid():
    """Return the worker grid as `(rows, cols)` from the system descriptor."""
    import json
    import re

    global _g_worker_grid, _g_worker_grid_source
    system_desc_path = _get_system_desc_path()
    source = system_desc_path or "<runtime>"
    if _g_worker_grid is not None and _g_worker_grid_source == source:
        return _g_worker_grid

    if binary is not None and system_desc_path:
        system_desc = binary.load_system_desc_from_path(system_desc_path)
    elif runtime is not None:
        system_desc = runtime.get_current_system_desc()
    else:
        raise RuntimeError(
            "global_semaphore() needs a system descriptor to size the worker "
            "grid; set SYSTEM_DESC_PATH or pass grid_shape explicitly."
        )

    desc_json = re.sub(r"\bnan\b", "NaN", system_desc.as_json())
    desc_json = re.sub(r"\binf\b", "Infinity", desc_json)
    desc = json.loads(desc_json)
    root = desc.get("system_desc", desc)
    grid = root["chip_descs"][0]["grid_size"]
    _g_worker_grid = (int(grid["y"]), int(grid["x"]))
    _g_worker_grid_source = source
    return _g_worker_grid


def global_semaphore(grid_shape=None, init=0) -> GlobalSemaphore:
    """Create a global semaphore backed by the device worker grid."""
    b = _get_scope()
    if not isinstance(b, _Builder):
        raise RuntimeError("global_semaphore() requires the lazy builder scope")
    if grid_shape is None:
        grid_shape = _device_worker_grid()
    grid_shape = tuple(grid_shape)
    if len(grid_shape) != 2 or any(
        not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
        for dim in grid_shape
    ):
        raise ValueError(
            "global semaphore grid_shape must contain two positive integers, "
            f"got {grid_shape}"
        )
    if not isinstance(init, int) or isinstance(init, bool) or not 0 <= init < 2**32:
        raise ValueError(f"global semaphore init must be a ui32 value, got {init!r}")

    with b.ctx, b.loc, b.insert_point:
        backing_ty = _semaphore_backing_type(b.ctx, grid_shape)
        backing = d2m.empty(backing_ty)
        sem_ty = d2m.ir.GlobalSemaphoreType.get(b.ctx)
        sem = d2m.create_global_semaphore(backing, value=init, results=[sem_ty])
    return GlobalSemaphore(sem, b.generation, grid_shape, init)


# --- Device meshes -----------------------------------------------------------


def mesh(shape, topology=None):
    """Declare the device mesh used by the current lazy graph."""
    b = _get_scope()
    if not isinstance(b, _Builder):
        raise RuntimeError("mesh() requires the lazy builder scope")
    b.set_mesh(shape, topology)


def _emit_mesh_shard(b, value, dst_ty, direction, shard_dims, shard_shape):
    shard_type = Attribute.parse("#ttcore.shard_type<devices>", b.ctx)
    shard_direction = Attribute.parse(f"#ttcore.shard_direction<{direction}>", b.ctx)
    return d2m.mesh_shard(
        dst_ty,
        value,
        shard_type,
        shard_direction,
        list(shard_shape),
        list(shard_dims),
    )


def _tensor_mesh_attr(b):
    if b._mesh_name is None:
        raise RuntimeError("mesh_shard() requires a preceding mesh() declaration")
    with b.ctx, b.loc:
        return Attribute.parse(f'#ttcore.tensor_mesh<"{b._mesh_name}">', b.ctx)


def mesh_shard(input_, layout: Layout, shard_dims, shard_shape) -> LazyTensor:
    """Distribute a full host tensor into one `layout` shard per device.

    `shard_dims` maps each mesh axis to a tensor dimension (`-1` replicates
    that mesh axis). `shard_shape` has tensor rank and records the resulting
    factor for each tensor dimension.
    """
    if torch is None or not isinstance(input_, torch.Tensor):
        raise TypeError("mesh_shard expects a torch.Tensor containing the full tensor")
    b = _get_scope()
    if not isinstance(b, _Builder):
        raise RuntimeError("mesh_shard() requires the lazy builder scope")
    if b._mesh_name is None:
        raise RuntimeError("mesh_shard() requires a preceding mesh() declaration")

    shard_dims = list(shard_dims)
    shard_shape = list(shard_shape)
    full_shape = list(input_.shape)
    expected_shape = _shard_logical_shape(
        b._mesh_shape, full_shape, shard_dims, shard_shape
    )
    if list(layout.logical_shape) != expected_shape:
        raise ValueError(
            f"mesh_shard layout shape {list(layout.logical_shape)} does not "
            f"match expected per-device shape {expected_shape}"
        )

    with b.ctx, b.loc, b.insert_point:
        elem_type = layout.get_host_elem_type(b.ctx)
        full_ty = RankedTensorType.get(full_shape, elem_type)
        shard_ty = RankedTensorType.get(
            expected_shape, elem_type, encoding=_tensor_mesh_attr(b)
        )
        host = b.add_host_input(layout, input_, host_ty=full_ty)
        shard = _emit_mesh_shard(
            b, host, shard_ty, "full_to_shard", shard_dims, shard_shape
        )
        device = layout.build_to_device(b.ctx, shard)
    return LazyTensor(
        layout,
        device,
        b.generation,
        mesh=MeshShard(full_shape, shard_dims, shard_shape),
    )


def mesh_gather(lt: LazyTensor, shard_dims=None, shard_shape=None) -> LazyTensor:
    """Mark a per-device tensor for a `shard_to_full` gather in `to_host`.

    Tensors returned by `mesh_shard` already carry the mapping. Other tensors,
    such as kernel outputs, require `shard_dims` and `shard_shape`.
    """
    if not isinstance(lt, LazyTensor):
        raise TypeError(f"mesh_gather expected a LazyTensor, got {type(lt).__name__}")
    b = _get_scope()
    if not isinstance(b, _Builder) or b._mesh_name is None:
        raise RuntimeError("mesh_gather() requires a preceding mesh() declaration")

    if lt.mesh is not None:
        if shard_dims is not None and list(shard_dims) != lt.mesh.shard_dims:
            raise ValueError("mesh_gather shard_dims do not match existing metadata")
        if shard_shape is not None and list(shard_shape) != lt.mesh.shard_shape:
            raise ValueError("mesh_gather shard_shape does not match existing metadata")
        return lt._resolve()
    if shard_dims is None or shard_shape is None:
        raise ValueError(
            "mesh_gather needs shard_dims and shard_shape for a tensor not "
            "produced by mesh_shard"
        )

    shard_dims = list(shard_dims)
    shard_shape = list(shard_shape)
    _validate_mesh_mapping(
        b._mesh_shape, len(lt.layout.logical_shape), shard_dims, shard_shape
    )
    lt = lt._resolve()
    full_shape = [
        dim * factor for dim, factor in zip(lt.layout.logical_shape, shard_shape)
    ]
    lt.mesh = MeshShard(full_shape, shard_dims, shard_shape)
    return lt


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
    if lt.mesh is not None:
        raise ValueError(
            "reshape does not yet define how to remap mesh sharding; gather "
            "and create a new mesh_shard explicitly"
        )
    if torch is None:
        raise RuntimeError("torch is required for d2m_jit.reshape()")

    new_shape, dst_layout = resolve_reshape(lt.layout, shape)
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
    if lt.mesh is not None:
        raise ValueError(
            "view operations do not yet remap mesh sharding; materialize a "
            "gathered tensor and create a new mesh_shard explicitly"
        )
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
    if not isinstance(lt, LazyTensor):
        raise TypeError(f"view_layout expected a LazyTensor, got {type(lt).__name__}")
    if lt.mesh is not None:
        raise ValueError(
            "view_layout does not yet remap mesh sharding; materialize a "
            "gathered tensor and create a new mesh_shard explicitly"
        )
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
    if not isinstance(lt, LazyTensor):
        raise TypeError(f"view expected a LazyTensor, got {type(lt).__name__}")
    if lt.mesh is not None:
        raise ValueError(
            "view does not yet remap mesh sharding; materialize a gathered "
            "tensor and create a new mesh_shard explicitly"
        )
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
    if lt.mesh is not None:
        raise ValueError(
            "permute does not yet remap mesh sharding; materialize a gathered "
            "tensor and create a new mesh_shard explicitly"
        )
    lt = lt._resolve()
    n_logical = len(lt.layout.logical_shape)
    if len(dims) != n_logical:
        raise ValueError(
            f"permute: expected {n_logical} dim indices for logical rank "
            f"{n_logical}, got {len(dims)}: {dims}"
        )
    return _emit_perm_view(lt, list(dims))


def _parse_grid_ranges(grid_ranges):
    """Parse grid_ranges into inclusive ((start_y, start_x), (end_y, end_x))."""
    parsed = []
    for i, grid_range in enumerate(grid_ranges):
        try:
            start_yx, end_yx = grid_range
            sy, sx = start_yx
            ey, ex = end_yx
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "grid_ranges entries must be ((start_y, start_x), "
                f"(end_y, end_x)) pairs, got {grid_range!r} at index {i}"
            ) from exc
        if ey < sy or ex < sx:
            raise ValueError(
                f"grid_ranges[{i}] end ({ey}, {ex}) must be greater than "
                f"or equal to start ({sy}, {sx})"
            )
        parsed.append(((sy, sx), (ey, ex)))
    return parsed


def spatial(inputs, outputs, grid_ranges, region_builders):
    """Emit a `d2m.spatial` op around one kernel call per region.

    Only supported under the lazy `_Builder` scope (not RewriteScope).

    Args:
      inputs: Device tensors referenced by the nested kernels.
      outputs: Device tensors written by the nested kernels, in spatial
        result order.
      grid_ranges: One inclusive `((start_y, start_x), (end_y, end_x))` core
        range per region.
      region_builders: One zero-argument callable per region. Each must emit
        exactly one `@d2m.kernel` call on device LazyTensors from
        inputs=/outputs= (no host tensor lifts).

    Returns a tuple of the output LazyTensors.
    """
    b = _get_scope()
    if isinstance(b, _SpatialRegionScope):
        raise RuntimeError("nested d2m.spatial is not supported")
    if not isinstance(b, _Builder):
        raise RuntimeError("d2m.spatial() requires the lazy builder scope")

    builders = list(region_builders)
    ranges = list(grid_ranges)
    if len(ranges) != len(builders):
        raise ValueError(
            f"grid_ranges has {len(ranges)} entries but region_builders has "
            f"{len(builders)}"
        )
    if not builders:
        raise ValueError("d2m.spatial requires at least one region")
    for i, builder_fn in enumerate(builders):
        if not callable(builder_fn):
            raise TypeError(
                f"region_builders[{i}] must be callable, got "
                f"{type(builder_fn).__name__}"
            )

    input_lts = [v._resolve() for v in inputs]
    output_lts = [v._resolve() for v in outputs]
    if not output_lts:
        raise ValueError("d2m.spatial outputs= must be non-empty")

    emitted_output_values = []
    with b.ctx, b.loc, b.insert_point:
        parsed_ranges = _parse_grid_ranges(ranges)
        grid_ranges_attr = ArrayAttr.get(
            [
                ttcore.ir.CoreRangeAttr.get(
                    b.ctx,
                    ttcore.ir.CoreCoordAttr.get(b.ctx, sy, sx),
                    ttcore.ir.CoreCoordAttr.get(b.ctx, ey, ex),
                )
                for (sy, sx), (ey, ex) in parsed_ranges
            ]
        )
        result_types = [lt.value.type for lt in output_lts]
        spatial_op = d2m.SpatialOp(
            result_types,
            [lt.value for lt in input_lts],
            [lt.value for lt in output_lts],
            grid_ranges_attr,
            len(builders),
        )

        for region_idx, build_region in enumerate(builders):
            block = Block.create_at_start(spatial_op.regions[region_idx], [], [])
            (sy, sx), (ey, ex) = parsed_ranges[region_idx]
            region_scope = _SpatialRegionScope(
                b,
                InsertionPoint(block),
                spatial_op,
                grid_shape=[ey - sy + 1, ex - sx + 1],
                offset=[sy, sx],
            )
            with _push_scope(region_scope):
                build_region()
            emitted_output_values.extend(region_scope.finish())

    # Counter compares as a multiset: same Values/counts, order ignored.
    # Check that region-kernel dst operands match SpatialOp outs=.
    if Counter(emitted_output_values) != Counter(spatial_op.outputs):
        raise ValueError(
            "d2m.spatial outputs= must list the same tensors written by the "
            "region kernels"
        )

    for i, lt in enumerate(output_lts):
        lt.value = spatial_op.results[i]
        lt.generation = b.generation
        lt.materialized = None
        lt.is_view = False
    return tuple(output_lts)


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
                elem_type = lt.layout.get_host_elem_type(b.ctx)
                host_shard_ty = RankedTensorType.get(
                    lt.layout.logical_shape,
                    elem_type,
                    encoding=_tensor_mesh_attr(b),
                )
                host_shard = d2m.ToLayoutOp(
                    [host_shard_ty], dev, d2m.empty(host_shard_ty)
                ).result
                host_ty = RankedTensorType.get(lt.mesh.full_shape, elem_type)
                host = _emit_mesh_shard(
                    b,
                    host_shard,
                    host_ty,
                    "shard_to_full",
                    lt.mesh.shard_dims,
                    lt.mesh.shard_shape,
                )
            else:
                host = lt.layout.build_from_device(b.ctx, dev)
                host_ty = lt.layout.build_host_tensor_type(b.ctx)
            host_values.append(host)
            host_types.append(host_ty)
        func.ReturnOp(host_values)
    b._refresh_function_type(results=host_types)


def _run_pipeline(b: _Builder):
    system_desc = _get_system_desc_path()
    register_options = []
    if system_desc:
        register_options.append(f"system-desc-path={system_desc}")
    if b._mesh_topology:
        register_options.append("mesh-topology=" + ",".join(b._mesh_topology))
    register = "ttcore-register-device"
    if register_options:
        register += "{" + " ".join(register_options) + "}"
    pipeline_str = f"builtin.module({register},{','.join(_pipeline_passes())})"

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
        output_shape = (
            lt.mesh.full_shape if lt.mesh is not None else lt.layout.logical_shape
        )
        t_out = torch.empty(list(output_shape), dtype=torch_dtype)
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

    device = None
    fabric_enabled = False
    try:
        if b._fabric_runtime_mode is not None:
            runtime.set_fabric_config(
                getattr(runtime.FabricConfig, b._fabric_runtime_mode)
            )
            fabric_enabled = True

        device = runtime.open_mesh_device(device_options)
        submitted = runtime.submit(device, fbb, program_index, rt_inputs)
        runtime.wait(submitted)
        for i, rt_out in enumerate(submitted):
            host_view = runtime.to_host(rt_out, untilize=True)[0]
            runtime.memcpy(rt_outputs[i], host_view)
            runtime.deallocate_tensor(rt_out, force=True)
        return out_torch
    finally:
        try:
            if device is not None:
                runtime.close_mesh_device(device)
        finally:
            if fabric_enabled:
                runtime.set_fabric_config(runtime.FabricConfig.DISABLED)


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


def _make_grid_attr(ctx, grid, grid_offset=(0, 0)):
    """Build a GridAttr from grid shape and an optional virtual-grid offset.

    Zero offset yields a plain shape-only GridAttr. Nonzero offset attaches
    virt_to_physical / physical_to_virt maps (2D, leading zero pad) so the
    nested generic's virtual cores land on the region's physical cores.
    """
    grid = list(grid)
    offset = list(grid_offset)
    if len(offset) != 2:
        raise ValueError(f"grid_offset must be 2D, got {offset}")
    if offset[0] == 0 and offset[1] == 0:
        return ttcore.ir.GridAttr.get(ctx, grid)

    oy, ox = offset

    def add_const(expr, value):
        if value == 0:
            return expr
        return AffineExpr.get_add(expr, AffineExpr.get_constant(value, ctx))

    d0 = AffineDimExpr.get(0)
    d1 = AffineDimExpr.get(1)
    zero = AffineExpr.get_constant(0, ctx)
    # 2D GridAttr maps: leading zero pad, then y/x with +/- offset.
    grid_forward = AffineMap.get(2, 0, [zero, add_const(d0, oy), add_const(d1, ox)])
    offset_inverse = AffineMap.get(2, 0, [zero, add_const(d0, -oy), add_const(d1, -ox)])
    return ttcore.ir.GridAttr.get(ctx, grid, grid_forward, offset_inverse)


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
    grid_offset=(0, 0),
):
    """Append a d2m.GenericOp to the open host func that invokes `kernel`."""
    b = _get_scope()

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

    fabric_attr = None
    if fabric is not None:
        if not isinstance(b, _Builder):
            raise _call_error(
                "fabric kernels require the top-level lazy builder and are not "
                "supported inside a pattern rewrite or d2m.spatial",
                cause=TypeError(),
            )
        b.enable_fabric(fabric)
        fabric_attr = fabric.build_attr(b.ctx)

    # Split args while preserving the GenericOp contract: all tensors precede
    # all additional args (runtime scalars and global semaphores).
    lazy_args = []
    extras = []
    seen_semaphores = set()
    saw_extra = False
    for i, a in enumerate(args):
        if isinstance(a, LazyTensor):
            if saw_extra:
                raise _call_error(
                    f"argument {i} to kernel '{kernel.fn.__name__}' is a "
                    f"LazyTensor but an additional argument was already seen; "
                    f"tensor arguments must precede scalars and semaphores",
                    cause=TypeError(),
                )
            lazy_args.append(a._resolve())
        elif isinstance(a, GlobalSemaphore):
            if not isinstance(b, _Builder):
                raise _call_error(
                    "global semaphore arguments require the top-level lazy "
                    "builder and are not supported inside a pattern rewrite "
                    "or d2m.spatial",
                    cause=TypeError(),
                )
            if id(a) in seen_semaphores:
                raise _call_error(
                    "the same GlobalSemaphore cannot be passed more than once "
                    "to a kernel call",
                    cause=ValueError(),
                )
            a._resolve_value()
            seen_semaphores.add(id(a))
            saw_extra = True
            extras.append(("semaphore", a))
        elif isinstance(a, int) and not isinstance(a, bool):
            saw_extra = True
            extras.append(("scalar", a))
        else:
            raise _call_error(
                f"argument {i} to kernel '{kernel.fn.__name__}' has "
                f"unsupported type {type(a).__name__}: {a!r}",
                hint=(
                    "kernel arguments must be d2m_jit.LazyTensor, int, or "
                    "d2m_jit.GlobalSemaphore. Use d2m.to_layout(t, L) to "
                    "lift a torch tensor."
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
    # (see add_scalar_input) so the binary remains parameterised.
    bake_scalars = not isinstance(b, (_Builder, _SpatialRegionScope))
    if bake_scalars and extras:
        formal_names = [a.arg for a in kernel._ast.body[0].args.args]
        scalar_names = formal_names[len(lazy_args) : len(lazy_args) + len(extras)]
        effective_captures = dict(kernel._captures)
        effective_captures.update(
            {name: int(value) for name, (_, value) in zip(scalar_names, extras)}
        )
        runtime_extras = []
    else:
        effective_captures = kernel._captures
        runtime_extras = list(extras)

    # Compile the kernel body in the current builder's context. D2MCompiler
    # picks up b.ctx via get_default_loc_context.
    with b.ctx, b.loc:
        extra_compiler_args = [
            value if kind == "scalar" else SEMAPHORE_ARG
            for kind, value in runtime_extras
        ]
        compiler_args = [lt.layout for lt in lazy_args] + extra_compiler_args
        compiler = D2MCompiler(
            kernel.fn.__name__,
            "unified",
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
        # Scalars come from func args; semaphores reuse their host-scope
        # create_global_semaphore results.
        additional = [
            (b.add_scalar_input(value) if kind == "scalar" else value._resolve_value())
            for kind, value in runtime_extras
        ]
        inputs = [lt.value for lt in input_lts]
        outputs = [lt.value for lt in output_lts]
        output_types = [v.type for v in outputs]

        threads = ArrayAttr.get(
            [compiler.func_entry.attributes[d2m.ir.ThreadAttr.name]]
        )
        grid_attr = _make_grid_attr(b.ctx, grid, grid_offset)

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
            fabricConnectionConfig=fabric_attr,
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

        # Reset closes the semaphore's lifetime and allows its backing buffer
        # to be deallocated. A handle is intentionally single-use.
        for kind, value in runtime_extras:
            if kind == "semaphore":
                d2m.reset_global_semaphore(value._resolve_value(), value.init)
                value._consumed = True

    # Rebind output LazyTensors to the GenericOp's results.
    for i, lt in enumerate(output_lts):
        lt.value = generic.results[i]
        lt.generation = b.generation
    if kernel_io_in_dram:
        for i, (user_lt, kernel_lt) in enumerate(zip(user_output_lts, output_lts)):
            user_lt.layout = kernel_lt.layout
            user_lt.value = generic.results[i]
            user_lt.generation = b.generation
            user_lt.materialized = None
            user_lt.is_view = kernel_lt.is_view
    return generic


class CompiledKernel:
    """Wraps a user kernel function. Parses the Python body once; emits a
    `d2m.GenericOp` into the current builder on every call."""

    def __init__(self, fn):
        functools.update_wrapper(self, fn)
        self.fn = fn
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
        b = _get_scope()
        if not isinstance(b, _SpatialRegionScope):
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
        else:
            if fabric is not None:
                raise ValueError(
                    "fabric kernels are not currently supported inside d2m.spatial"
                )
            b._emit_kernel_for_spatial(
                self,
                args,
                grid=grid,
                num_outs=num_outs,
                block_factors=block_factors,
                indexing_maps=indexing_maps,
                iterator_types=iterator_types,
                kernel_io_in_dram=kernel_io_in_dram,
            )


def kernel(fn):
    """Decorate a user function as a d2m_jit kernel."""
    return CompiledKernel(fn)
