# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-device mesh tests (workstream C-a / C-b).

C-a (plumbing): `test_mesh_sets_module_attr` is a no-device unit test for the
`d2m.mesh(...)` config; `test_mesh_shard_roundtrip_1x2` is a device-gated
end-to-end check that the d2m-jit pipeline + runtime can open a 1x2 mesh and
execute a hand-built `d2m.mesh_shard` full->shard->full identity (targets the
runtime/pipeline plumbing).

C-b (DSL surface): `test_mesh_shard_dsl_emits_full_and_shard` checks the
`d2m.mesh_shard` / `to_host` op emission (no device);
`test_mesh_shard_dsl_roundtrip_1x2` runs the same identity through the real
`mesh_shard(...).to_host()` DSL flow; `test_mesh_compute_roundtrip_1x2` adds a
compute generic between shard and gather.

C5 (CCL): `test_all_gather_1x2_lowers` compiles a full 1x2 all_gather kernel
(reblock streams + fabric config + device_synchronize / cross-device
remote_store / semaphore_wait) to a ttmetal flatbuffer. On-device execution +
PCC is a follow-up (needs a healthy mesh). See CCL_SPEC.md section 7.
"""

import functools
import json
import re

import pytest

from utils import assert_pcc

import d2m_jit as d2m
from d2m_jit._src.builder import (
    _Builder,
    _build_pipeline,
    _close_cached_device,
    _get_system_desc_path,
    _to_runtime_data_type,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from _ttmlir_runtime import runtime, binary
except (ModuleNotFoundError, ImportError):
    runtime = None
    binary = None

from ttmlir import ir
from ttmlir.passmanager import PassManager

try:
    from ttmlir.passes import ttmetal_to_flatbuffer_bin
except ImportError:
    ttmetal_to_flatbuffer_bin = None


@functools.lru_cache(maxsize=1)
def _num_devices():
    """Number of physical chips, read *statically* from the system descriptor.

    Avoids `runtime.get_num_available_devices()`, which opens+destroys the UMD
    cluster on every call. This is evaluated once per skipif decorator at import
    time (~7x), and that repeated cluster init flakes the n300 ARC startup; worse,
    a single transient flake cached here would poison the whole session with a
    bogus 0. The system descriptor's `chip_desc_indices` is the authoritative,
    hardware-free chip count (mirrors golden conftest's get_board_id), so the
    cache is safe."""
    if binary is None:
        return 0
    sd = _get_system_desc_path()
    if not sd:
        return 0
    try:
        js = binary.load_system_desc_from_path(sd).as_json()
        js = re.sub(r"\bnan\b", "NaN", js)
        js = re.sub(r"\binf\b", "Infinity", js)
        desc = json.loads(js)["system_desc"]
        return len(desc["chip_desc_indices"])
    except Exception:
        return 0


def test_mesh_sets_module_attr():
    """d2m.mesh(...) sets the module's ttcore.meshes attr and stores topology."""
    d2m.mesh((1, 2), topology=("linear", "ring"))
    b = _Builder.get()
    assert '#ttcore.meshes<[<"mesh" = 1x2>]>' in str(b.module.operation)
    assert b._mesh_topology == ["linear", "ring"]


_ROUNDTRIP_IR = """
module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x2>]>} {
  func.func @main(%arg0: tensor<512x1024xf32>) -> tensor<512x1024xf32> {
    %0 = "d2m.mesh_shard"(%arg0) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<512x1024xf32>) -> tensor<512x512xf32>
    %1 = "d2m.mesh_shard"(%0) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<512x512xf32>) -> tensor<512x1024xf32>
    return %1 : tensor<512x1024xf32>
  }
}
"""


@pytest.mark.skipif(
    torch is None or runtime is None or ttmetal_to_flatbuffer_bin is None,
    reason="requires torch + ttmlir runtime",
)
@pytest.mark.skipif(_num_devices() < 2, reason="requires a >=2-device mesh")
def test_mesh_shard_roundtrip_1x2():
    """A 1x2 mesh_shard full->shard->full identity executes and round-trips.

    Validates: module ttcore.meshes attr -> register-device -> flatbuffer mesh
    shape -> runtime opens 1x2 mesh -> full_to_shard distributes / shard_to_full
    gathers correctly."""
    ctx = ir.Context()
    ctx.load_all_available_dialects()
    mod = ir.Module.parse(_ROUNDTRIP_IR, ctx)

    sd = _get_system_desc_path()
    register = "ttcore-register-device" + (f"{{system-desc-path={sd}}}" if sd else "")
    PassManager.parse(f"builtin.module({register})", context=ctx).run(mod.operation)
    PassManager.parse(f"builtin.module({_build_pipeline()})", context=ctx).run(
        mod.operation
    )

    fbb = binary.load_binary_from_capsule(ttmetal_to_flatbuffer_bin(mod))
    mesh_shape = fbb.get_program_mesh_shape(0)
    assert tuple(mesh_shape) == (1, 2), mesh_shape

    runtime.set_compatible_device_runtime(fbb)
    t = torch.randn(512, 1024, dtype=torch.float32)
    rin = runtime.create_borrowed_host_tensor(
        t.data_ptr(),
        list(t.shape),
        list(t.stride()),
        t.element_size(),
        _to_runtime_data_type(t.dtype),
    )
    # This test opens its own mesh device, so drop any device the cached-device
    # path (builder._execute) may still hold open -- two open meshes conflict.
    _close_cached_device()
    opts = runtime.MeshDeviceOptions()
    opts.mesh_shape = mesh_shape
    dev = runtime.open_mesh_device(opts)
    try:
        submitted = runtime.submit(dev, fbb, 0, [rin])
        runtime.wait(submitted)
        out = torch.empty(512, 1024, dtype=torch.float32)
        rout = runtime.create_borrowed_host_tensor(
            out.data_ptr(),
            list(out.shape),
            list(out.stride()),
            out.element_size(),
            _to_runtime_data_type(out.dtype),
        )
        host_view = runtime.to_host(submitted[0], untilize=True)[0]
        runtime.memcpy(rout, host_view)
    finally:
        runtime.close_mesh_device(dev)

    assert torch.allclose(out, t, atol=1e-2), "mesh_shard round-trip mismatch"


def test_mesh_shard_dsl_emits_full_and_shard(monkeypatch):
    """d2m.mesh_shard + to_host emit full_to_shard and shard_to_full ops
    bracketing the graph, with the full tensor as the func arg (no device)."""
    from d2m_jit._src import builder as _b

    captured = {}

    def _fake_pipeline_and_execute(b, resolved):
        captured["ir"] = str(b.module)
        # Return zero tensors of the right (full) shape so to_host completes
        # without touching a device.
        return [
            torch.zeros(
                lt.mesh.full_shape if lt.mesh else list(lt.layout.logical_shape)
            )
            for lt in resolved
        ]

    # Stub out the device path: run returns/finalise (so mesh ops are emitted),
    # capture IR, skip register/pipeline/execute.
    orig_finalise = _b._emit_returns_and_finalise

    def _patched_to_host(*lts):
        b = _b._get_scope()
        resolved = [lt._resolve() for lt in lts]
        orig_finalise(b, resolved)
        outs = _fake_pipeline_and_execute(b, resolved)
        for orig, lt, t in zip(lts, resolved, outs):
            orig.materialized = t
            orig.value = None
        return tuple(outs)

    monkeypatch.setattr(_b, "to_host", _patched_to_host)
    monkeypatch.setattr(d2m, "to_host", _patched_to_host)

    d2m.mesh((1, 2), topology=("linear", "ring"))
    L = d2m.Layout(
        shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 2]
    )
    full = torch.randn(64, 128, dtype=torch.float32)
    lt = d2m.mesh_shard(full, L, shard_dims=[0, 1], shard_shape=[1, 2])
    _patched_to_host(lt)

    ir_text = captured["ir"]
    assert "shard_direction = #ttcore.shard_direction<full_to_shard>" in ir_text
    assert "shard_direction = #ttcore.shard_direction<shard_to_full>" in ir_text
    # Func arg is the full (plain) tensor; the per-device shard carries the
    # tensor_mesh encoding marking it multi-device.
    assert "tensor<64x128xf32>" in ir_text
    assert 'tensor<64x64xf32, #ttcore.tensor_mesh<"mesh">>' in ir_text


@pytest.mark.skipif(
    torch is None or runtime is None,
    reason="requires torch + ttmlir runtime",
)
@pytest.mark.skipif(_num_devices() < 2, reason="requires a >=2-device mesh")
def test_mesh_shard_dsl_roundtrip_1x2():
    """End-to-end through the DSL: mesh_shard(full) -> to_host gathers back the
    full tensor (full_to_shard then shard_to_full is identity)."""
    d2m.mesh((1, 2), topology=("linear", "ring"))
    L = d2m.Layout(
        shape=(512, 512), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 2]
    )
    full = torch.randn(512, 1024, dtype=torch.float32)
    lt = d2m.mesh_shard(full, L, shard_dims=[0, 1], shard_shape=[1, 2])
    out = lt.to_host()
    assert tuple(out.shape) == (512, 1024)
    assert torch.allclose(out, full, atol=1e-2), "DSL mesh_shard round-trip mismatch"


@pytest.mark.skipif(
    torch is None or runtime is None,
    reason="requires torch + ttmlir runtime",
)
@pytest.mark.skipif(_num_devices() < 2, reason="requires a >=2-device mesh")
def test_mesh_compute_roundtrip_1x2():
    """Compute on a mesh shard: full -> shard -> eltwise kernel -> gather -> full.

    Exercises a compute generic between full_to_shard and shard_to_full, which
    requires the tensor_mesh encoding on the shard boundary (without it the
    distributed host buffer mismatches the mesh device buffer at runtime)."""

    @d2m.kernel
    def sig_kernel(in_t, out_t, m_blocks, n_blocks):
        m_off = core_index(0) * m_blocks
        n_off = core_index(1) * n_blocks
        for m in range(m_blocks):
            for n in range(n_blocks):
                shard = remote_load(in_t, [m_off + m, n_off + n])
                remote_store(out_t, [m_off + m, n_off + n], sigmoid(shard))

    d2m.mesh((1, 2), topology=("linear", "ring"))
    L = d2m.Layout(
        shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 2]
    )
    full = torch.randn(64, 128, dtype=torch.float32) * 0.5
    in_d = d2m.mesh_shard(full, L, shard_dims=[0, 1], shard_shape=[1, 2])
    out_d = d2m.empty(L)
    sig_kernel(in_d, out_d, 1, 1, grid=(2, 2))
    out_d = d2m.mesh_gather(out_d, shard_dims=[0, 1], shard_shape=[1, 2])
    result = out_d.to_host()
    expected = torch.sigmoid(full)
    diff = (expected - result).abs().max().item()
    assert tuple(result.shape) == (64, 128)
    assert diff < 0.05, f"mesh compute round-trip: max abs diff {diff}"


@pytest.mark.skipif(
    torch is None or runtime is None or ttmetal_to_flatbuffer_bin is None,
    reason="requires torch + ttmlir runtime",
)
@pytest.mark.skipif(_num_devices() < 2, reason="requires a >=2-device mesh")
def test_all_gather_1x2_lowers():
    """A 1x2 all_gather kernel (datamovement form via unified + split-v2)
    compiles end-to-end to a ttmetal flatbuffer.

    Mirrors D2MAllGatherRewriter: mesh_shard(full_to_shard) -> reblock to the
    stream grids -> all_gather generic (device_synchronize + cross-device
    remote_store + semaphore_wait, with a fabric connection config) -> reblock
    output -> mesh_gather. Validates lowering (incl. the fabric connection
    manager setup, which needs the in-kernel scratch buffer to be tensor.empty
    and the fabric chain on one datamovement thread). On-device execution +
    PCC is a follow-up (needs a healthy mesh)."""
    from d2m_jit._src.builder import _build_pipeline, _emit_returns_and_finalise

    @d2m.kernel
    def all_gather(in0, out0, start_sem, end_sem):
        dy = mesh_position(0)
        cy = core_index(0)
        cx = core_index(1)
        device_synchronize(
            start_sem,
            start_device=[dy, 0],
            mcast_shape=[1, 2],
            num_receivers=1,
            core_indices=[cy, cx],
        )
        buf = empty([1, 2])
        remote_load(buf, in0, [cy, 0])
        dx = mesh_position(1)
        remote_store(
            out0,
            [dx * 2 + cy, 0],
            buf,
            start_device=[dy, 0],
            device_mcast_shape=[1, 2],
            semaphore=end_sem,
            semaphore_indices=[cy, 0],
        )
        semaphore_wait(end_sem, 1)

    sd = _get_system_desc_path()
    if not sd:
        pytest.skip("no system descriptor available")

    d2m.mesh((1, 2), topology=("linear", "ring"))
    L_in = d2m.Layout(
        shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 2]
    )
    L_out = d2m.Layout(
        shape=(128, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[4, 2]
    )
    in_s = d2m.reblock(
        d2m.mesh_shard(
            torch.randn(64, 128), L_in, shard_dims=[0, 1], shard_shape=[1, 2]
        ),
        [2, 1],
    )
    out_s = d2m.reblock(d2m.empty(L_out), [4, 1])
    ss = d2m.global_semaphore(grid_shape=(8, 8))
    es = d2m.global_semaphore(grid_shape=(8, 8))
    all_gather(
        in_s, out_s, ss, es, grid=(2, 1), fabric=d2m.fabric_config(cluster_axis=1)
    )
    out = d2m.mesh_gather(out_s, shard_dims=[0, 1], shard_shape=[1, 2])

    # Build + lower to a ttmetal flatbuffer (no execution).
    b = _Builder.get()
    _emit_returns_and_finalise(b, [out._resolve()])
    PassManager.parse(
        f"builtin.module(ttcore-register-device{{system-desc-path={sd} "
        f"mesh-shape=1,2 mesh-topology=linear,ring}})",
        context=b.ctx,
    ).run(b.module.operation)
    b.module.operation.verify()
    PassManager.parse(f"builtin.module({_build_pipeline()})", context=b.ctx).run(
        b.module.operation
    )
    fbb = binary.load_binary_from_capsule(ttmetal_to_flatbuffer_bin(b.module))
    assert tuple(fbb.get_program_mesh_shape(0)) == (1, 2)


@pytest.mark.skipif(
    torch is None or runtime is None,
    reason="requires torch + ttmlir runtime",
)
@pytest.mark.skipif(_num_devices() < 2, reason="requires a >=2-device mesh")
def test_all_gather_1x2_roundtrip():
    """A 1x2 line-config all_gather, the d2m-jit DSL mirror of
    `test/python/golden/d2m/test_allgather.py` (which drives
    D2MAllGatherRewriter). Hand-builds the same generic on a bidirectional
    2-device line: device_synchronize barrier -> cross-device mcast
    remote_store -> semaphore_wait, with a linear / bidir_line_mesh fabric
    config (a 2-device ring has no backward link).

    Authored as a `unified` kernel (the only form). With no compute between the
    load and store, the backend splits it into a single datamovement thread, so
    the remote_load/remote_store run in implicit (local-buffer) form, which
    D2MLowerLoadStoreOpsToDMA lowers straight to dma_read/dma_write.

    Line-config specifics vs the ring algorithm in CCL_SPEC.md section 7:
    - topology "linear" + routing "bidir_line_mesh" (not ring / unidir).
    - end-wait target num_devices (2), not num_devices - 1: remote_store
      increments endSemaphore on every device in the mcast range *including
      this one* (a local self-increment for the shard written locally), so each
      device sees num_devices increments; an exact-equality wait on
      num_devices - 1 overshoots by one and deadlocks (CCL_SPEC.md section 7).
    - one core per device (num_cores = num_links*1 = 1, vs ring's *2): a
      single-core (grid 1x1) kernel, whole per-device shard on one core.

    The `fabric=` config also makes `_execute` enable the device fabric
    (`set_fabric_config`) before opening the mesh device; without that the
    cross-device fabric semaphore incs silently no-op and the kernel deadlocks.

    Data flow (full (64,128), shard dim 1 by 2, gather dim 0, cluster axis 1):
    device d's shard is full[:, 64*d:64*(d+1)]; after the gather every device
    holds vstack(shard0, shard1) (128,64); shard_to_full column-concats the two
    identical halves -> (128,128)."""

    @d2m.kernel
    def all_gather(in0, out0, start_sem, end_sem):
        dy = mesh_position(0)
        cy = core_index(0)
        cx = core_index(1)
        device_synchronize(
            start_sem,
            start_device=[dy, 0],
            mcast_shape=[1, 2],
            num_receivers=1,
            core_indices=[cy, cx],
        )
        buf = empty([2, 2])  # whole 64x64 shard (2x2 tiles) on one core
        # Rebind buf to the load result so the store depends on the load (the
        # rewriter threads remote_load's result into remote_store; passing the
        # pre-load buffer drops the dependency and lets the DMA scheduler hoist
        # the store's fabric mcast ahead of the load).
        buf = remote_load(buf, in0, [0, 0])
        dx = mesh_position(1)
        # shardOffset = 1: device d writes its shard to output block [d, 0].
        remote_store(
            out0,
            [dx, 0],
            buf,
            start_device=[dy, 0],
            device_mcast_shape=[1, 2],
            semaphore=end_sem,
            semaphore_indices=[cy, 0],
        )
        # num_devices (1 remote + 1 local self-inc), not num_devices - 1.
        semaphore_wait(end_sem, 2)

    d2m.mesh((1, 2), topology=("linear", "linear"))
    L_in = d2m.Layout(
        shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 2]
    )
    L_out = d2m.Layout(
        shape=(128, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[4, 2]
    )
    full = torch.randn(64, 128, dtype=torch.float32)
    # Single core: whole shard on one block; output spans the gather dim
    # (2 device slots) over one core.
    in_s = d2m.reblock(
        d2m.mesh_shard(full, L_in, shard_dims=[0, 1], shard_shape=[1, 2]),
        [1, 1],
    )
    out_s = d2m.reblock(d2m.empty(L_out), [2, 1])
    ss = d2m.global_semaphore(grid_shape=(8, 8))
    es = d2m.global_semaphore(grid_shape=(8, 8))
    all_gather(
        in_s,
        out_s,
        ss,
        es,
        grid=(1, 1),
        fabric=d2m.fabric_config(
            cluster_axis=1, topology="linear", routing="bidir_line_mesh"
        ),
    )
    out_s = d2m.mesh_gather(out_s, shard_dims=[0, 1], shard_shape=[1, 2])
    result = out_s.to_host()

    shard0 = full[:, :64]
    shard1 = full[:, 64:]
    vstack = torch.cat([shard0, shard1], dim=0)  # (128, 64)
    expected = torch.cat([vstack, vstack], dim=1)  # (128, 128)
    assert tuple(result.shape) == (128, 128), result.shape
    diff = (expected - result).abs().max().item()
    assert diff < 0.05, f"1x2 all_gather max abs diff {diff}"


@pytest.mark.skipif(
    torch is None or runtime is None,
    reason="requires torch + ttmlir runtime",
)
@pytest.mark.skipif(_num_devices() < 2, reason="requires a >=2-device mesh")
def test_matmul_all_gather_fused_1x2_roundtrip():
    """A *fused* distributed-matmul + all_gather in a single `unified` kernel.

    Each device d holds one 32x32 shard of lhs and rhs (the column shards
    A_d / B_d). The kernel computes the local product C_d = A_d @ B_d on the
    compute thread, then all-gathers C_d across the 2-device line so every
    device ends holding vstack(C_0, C_1) (64,32); shard_to_full column-concats
    the two identical halves -> (64,64).

    Fusion is what makes this interesting: the matmul (compute) and the CCL
    data movement (device_synchronize / cross-device remote_store /
    semaphore_wait) live in ONE generic. `split-unified-thread-v2` separates
    them into a compute thread + a datamovement thread, handshaking the matmul
    output CB across the boundary (CBComputeInfo: compute *produces* it, the DM
    `remote_store` consumes it). The fabric remote_store's source is therefore
    a compute-produced CB rather than a remote_load result -- the case the
    split pass's `produced`/`dmStore` partner logic exists for.

    Because the matmul forces the datamovement ops to split across two DM
    processor threads (a reader feeding compute + a writer draining it), this
    is also the case that exercises ScheduleDMA pinning the device_synchronize
    barrier to a single DM thread: replicating it across both DM threads makes
    each core emit two fabric barrier increments and deadlocks (the single-
    thread datamovement form in test_all_gather_1x2_roundtrip never split, so
    it never hit this).

    CCL specifics mirror test_all_gather_1x2_roundtrip (linear topology /
    bidir_line_mesh routing, end-wait num_devices=2 because remote_store also
    self-increments the local shard, single core per device). The `fabric=`
    config also makes `_execute` enable the device fabric (set_fabric_config).
    """

    @d2m.kernel
    def matmul_all_gather(lhs, rhs, out, start_sem, end_sem):
        dy = mesh_position(0)
        cy = core_index(0)
        cx = core_index(1)
        device_synchronize(
            start_sem,
            start_device=[dy, 0],
            mcast_shape=[1, 2],
            num_receivers=1,
            core_indices=[cy, cx],
        )
        # Local per-device matmul: a single 32x32 tile each, so `a @ b` is a
        # one-tile matmul (M=K=N=1) -- the well-supported single-tile form.
        a = remote_load(lhs, [0, 0])
        b = remote_load(rhs, [0, 0])
        c = a @ b
        dx = mesh_position(1)
        # all_gather: device d mcasts its product C_d into output block [d, 0]
        # on every device, incrementing end_sem on each receiver.
        remote_store(
            out,
            [dx, 0],
            c,
            start_device=[dy, 0],
            device_mcast_shape=[1, 2],
            semaphore=end_sem,
            semaphore_indices=[cy, 0],
        )
        # num_devices (1 remote + 1 local self-inc), not num_devices - 1.
        semaphore_wait(end_sem, 2)

    d2m.mesh((1, 2), topology=("linear", "linear"))
    # Per-device shard is a single 32x32 tile.
    L_in = d2m.Layout(
        shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )
    # Output spans the gather dim (2 device slots stacked) over one core.
    L_out = d2m.Layout(
        shape=(64, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 1]
    )
    full_a = torch.randn(32, 64, dtype=torch.float32)
    full_b = torch.randn(32, 64, dtype=torch.float32)
    a_s = d2m.reblock(
        d2m.mesh_shard(full_a, L_in, shard_dims=[0, 1], shard_shape=[1, 2]),
        [1, 1],
    )
    b_s = d2m.reblock(
        d2m.mesh_shard(full_b, L_in, shard_dims=[0, 1], shard_shape=[1, 2]),
        [1, 1],
    )
    out_s = d2m.reblock(d2m.empty(L_out), [2, 1])
    ss = d2m.global_semaphore(grid_shape=(8, 8))
    es = d2m.global_semaphore(grid_shape=(8, 8))
    matmul_all_gather(
        a_s,
        b_s,
        out_s,
        ss,
        es,
        grid=(1, 1),
        fabric=d2m.fabric_config(
            cluster_axis=1, topology="linear", routing="bidir_line_mesh"
        ),
    )
    out_s = d2m.mesh_gather(out_s, shard_dims=[0, 1], shard_shape=[1, 2])
    result = out_s.to_host()

    a0, a1 = full_a[:, :32], full_a[:, 32:]
    b0, b1 = full_b[:, :32], full_b[:, 32:]
    c0 = a0 @ b0  # device 0's product (32, 32)
    c1 = a1 @ b1  # device 1's product (32, 32)
    vstack = torch.cat([c0, c1], dim=0)  # (64, 32)
    expected = torch.cat([vstack, vstack], dim=1)  # (64, 64)
    assert tuple(result.shape) == (64, 64), result.shape
    assert_pcc(expected, result)
