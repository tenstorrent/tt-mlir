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
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
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
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
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
@pytest.mark.skipif(_num_devices() < 4, reason="requires a >=4-device mesh")
def test_all_gather_1x4_roundtrip():
    """A 1x4 line-config all_gather -- the multi-hop generalization of
    `test_all_gather_1x2_roundtrip`.

    Distinct from 1x2 because the cross-device mcast must distribute payload
    across >=2 hops in both directions. It is a regression guard for the
    TensorAccessor-DMA write path silently swallowing the fabric multicast:
    when `use-tensor-accessor-dma` is on (the default), a cross-device
    `remote_store` must still lower to a fabric mcast write, not a local-only
    NOC write -- otherwise every device keeps only its own shard (a
    block-diagonal result, PCC ~0.5) instead of the full gather. See
    D2MLowerDMAToFullyIndexedForm (the `getStartDevice().empty()` guard).

    1x2 cannot run on a 4-chip ring box anyway: opening a 2-chip sub-mesh leaves
    the out-of-mesh ethernet cores unhandshaked and fabric bringup times out, so
    the full mesh (here 1x4) is required.

    Data flow mirrors the 1x2 case scaled to 4 devices: full (64,256), shard
    dim 1 by 4, gather dim 0; device d's shard is full[:, 64*d:64*(d+1)]; after
    the gather every device holds vstack(shard0..shard3) (256,64); shard_to_full
    column-concats the four identical quarters -> (256,256)."""

    @d2m.kernel
    def all_gather(in0, out0, start_sem, end_sem):
        dy = mesh_position(0)
        cy = core_index(0)
        cx = core_index(1)
        device_synchronize(
            start_sem,
            start_device=[dy, 0],
            mcast_shape=[1, 4],
            num_receivers=3,
            core_indices=[cy, cx],
        )
        buf = empty([2, 2])  # whole 64x64 shard (2x2 tiles) on one core
        buf = remote_load(buf, in0, [0, 0])
        dx = mesh_position(1)
        remote_store(
            out0,
            [dx, 0],
            buf,
            start_device=[dy, 0],
            device_mcast_shape=[1, 4],
            semaphore=end_sem,
            semaphore_indices=[cy, 0],
        )
        # num_devices (3 remote + 1 local self-inc), not num_devices - 1.
        semaphore_wait(end_sem, 4)

    d2m.mesh((1, 4), topology=("linear", "linear"))
    L_in = d2m.Layout(
        shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 2]
    )
    L_out = d2m.Layout(
        shape=(256, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[8, 2]
    )
    full = torch.randn(64, 256, dtype=torch.float32)
    in_s = d2m.reblock(
        d2m.mesh_shard(full, L_in, shard_dims=[0, 1], shard_shape=[1, 4]),
        [1, 1],
    )
    out_s = d2m.reblock(d2m.empty(L_out), [4, 1])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
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
    out_s = d2m.mesh_gather(out_s, shard_dims=[0, 1], shard_shape=[1, 4])
    result = out_s.to_host()

    shards = [full[:, i * 64 : (i + 1) * 64] for i in range(4)]
    vstack = torch.cat(shards, dim=0)  # (256, 64)
    expected = torch.cat([vstack] * 4, dim=1)  # (256, 256)
    assert tuple(result.shape) == (256, 256), result.shape
    diff = (expected - result).abs().max().item()
    assert diff < 0.05, f"1x4 all_gather max abs diff {diff}"


@pytest.mark.skipif(
    torch is None or runtime is None,
    reason="requires torch + ttmlir runtime",
)
@pytest.mark.skipif(_num_devices() < 4, reason="requires a >=4-device mesh")
def test_all_gather_1x4_dram_roundtrip():
    """`test_all_gather_1x4_roundtrip` with the gathered output staged in DRAM.

    Regression guard for the fabric remote_store->DRAM deadlock: a fabric
    datamovement thread that lands on NoC1 cannot talk to the EDM and blocks at
    its first send. ScheduleDMA's single-DM-thread path used `writesDRAM ? 0 : 1`
    (processorIndex 0 -> NoC1), so a fabric store to DRAM deadlocked while the
    same store to L1 (NoC0) worked. The fix pins cross-device (fabric) stores to
    NoC0. Identical to the L1 1x4 all_gather except the output Layout is
    `mem_space="dram"`, exercising the fabric mcast write to a DRAM destination."""

    @d2m.kernel
    def all_gather(in0, out0, start_sem, end_sem):
        dy = mesh_position(0)
        cy = core_index(0)
        cx = core_index(1)
        device_synchronize(
            start_sem,
            start_device=[dy, 0],
            mcast_shape=[1, 4],
            num_receivers=3,
            core_indices=[cy, cx],
        )
        buf = empty([2, 2])
        buf = remote_load(buf, in0, [0, 0])
        dx = mesh_position(1)
        remote_store(
            out0,
            [dx, 0],
            buf,
            start_device=[dy, 0],
            device_mcast_shape=[1, 4],
            semaphore=end_sem,
            semaphore_indices=[cy, 0],
        )
        semaphore_wait(end_sem, 4)

    d2m.mesh((1, 4), topology=("linear", "linear"))
    L_in = d2m.Layout(
        shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 2]
    )
    L_out = d2m.Layout(
        shape=(256, 64),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[8, 2],
        mem_space="dram",
    )
    full = torch.randn(64, 256, dtype=torch.float32)
    in_s = d2m.reblock(
        d2m.mesh_shard(full, L_in, shard_dims=[0, 1], shard_shape=[1, 4]),
        [1, 1],
    )
    out_s = d2m.reblock(d2m.empty(L_out), [4, 1])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
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
    out_s = d2m.mesh_gather(out_s, shard_dims=[0, 1], shard_shape=[1, 4])
    result = out_s.to_host()

    shards = [full[:, i * 64 : (i + 1) * 64] for i in range(4)]
    vstack = torch.cat(shards, dim=0)  # (256, 64)
    expected = torch.cat([vstack] * 4, dim=1)  # (256, 256)
    assert tuple(result.shape) == (256, 256), result.shape
    diff = (expected - result).abs().max().item()
    assert diff < 0.05, f"1x4 all_gather->DRAM max abs diff {diff}"


@pytest.mark.skipif(
    torch is None or runtime is None,
    reason="requires torch + ttmlir runtime",
)
@pytest.mark.skipif(_num_devices() < 4, reason="requires a >=4-device mesh")
def test_all_gather_1x4_large_block_roundtrip():
    """Gather-geometry scaling: a much larger per-device shard (16 row-tiles)
    gathered via the [num_devices, 1] *block* layout.

    The naive streaming gather lays the gathered output on a [num_devices*tiles,
    1] grid -- one worker core per tile -- which exceeds the worker grid's row
    count (~10) once tiles_per_device grows. Instead, block-pack each device's
    whole shard onto ONE core: input on a [1,1] grid with block_shape=[N,1] and
    output on a [num_devices,1] grid with block_shape=[N,1]. The grid stays tiny
    (1 / num_devices cores) while block_shape carries the N tiles, so the gather
    scales to large shards (bounded by per-core L1, not the worker-grid rows).
    Here N=16 -> per-device 512x32, gathered 2048x32 (128 tiles). Each device
    mcasts its whole 16-tile shard to output slot [dx,0]."""
    N = 16
    d2m.mesh((1, 4), topology=("linear", "linear"))
    L_in = d2m.Layout(
        shape=(N * 32, 32), dtype=d2m.float32, block_shape=[N, 1], grid_shape=[1, 1]
    )
    L_out = d2m.Layout(
        shape=(4 * N * 32, 32),
        dtype=d2m.float32,
        block_shape=[N, 1],
        grid_shape=[4, 1],
    )

    @d2m.kernel
    def all_gather(in0, out0, start_sem, end_sem):
        dy = mesh_position(0)
        cy = core_index(0)
        cx = core_index(1)
        device_synchronize(
            start_sem,
            start_device=[dy, 0],
            mcast_shape=[1, 4],
            num_receivers=3,
            core_indices=[cy, cx],
        )
        buf = empty([16, 1])  # whole 16-tile shard on one core
        buf = remote_load(buf, in0, [0, 0])
        dx = mesh_position(1)
        remote_store(
            out0,
            [dx, 0],
            buf,
            start_device=[dy, 0],
            device_mcast_shape=[1, 4],
            semaphore=end_sem,
            semaphore_indices=[cy, 0],
        )
        semaphore_wait(end_sem, 4)

    full = torch.randn(N * 32, 128, dtype=torch.float32)
    in_s = d2m.reblock(
        d2m.mesh_shard(full, L_in, shard_dims=[0, 1], shard_shape=[1, 4]), [1, 1]
    )
    out_s = d2m.reblock(d2m.empty(L_out), [4, 1])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
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
    out_s = d2m.mesh_gather(out_s, shard_dims=[0, 1], shard_shape=[1, 4])
    result = out_s.to_host()

    shards = [full[:, i * 32 : (i + 1) * 32] for i in range(4)]  # each (512, 32)
    vstack = torch.cat(shards, dim=0)  # (2048, 32)
    expected = torch.cat([vstack] * 4, dim=1)  # (2048, 128)
    assert tuple(result.shape) == (2048, 128), result.shape
    assert_pcc(expected, result)


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
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
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


@pytest.mark.skipif(
    torch is None or runtime is None,
    reason="requires torch + ttmlir runtime",
)
@pytest.mark.skipif(_num_devices() < 4, reason="requires a >=4-device mesh")
def test_matmul_all_gather_fused_1x4_roundtrip():
    """The fused distributed-matmul + all_gather of
    `test_matmul_all_gather_fused_1x2_roundtrip` scaled to the full 1x4 mesh.

    Each device d computes its local product C_d = A_d @ B_d (one 32x32 tile),
    then all-gathers C_d across the 4-device line so every device ends holding
    vstack(C_0..C_3) (128,32); shard_to_full column-concats the four identical
    quarters -> (128,128).

    Beyond the 1x2 case this exercises the multi-hop fabric mcast of a
    compute-produced CB on the full mesh -- i.e. the fused-matmul source feeding
    the cross-device write that D2MLowerDMAToFullyIndexedForm must keep as a
    fabric mcast (not defer to the local accessor write). 1x2 cannot run on a
    4-chip ring box (sub-mesh fabric bringup times out), so the full mesh is
    required to validate the fused path on this hardware at all."""

    @d2m.kernel
    def matmul_all_gather(lhs, rhs, out, start_sem, end_sem):
        dy = mesh_position(0)
        cy = core_index(0)
        cx = core_index(1)
        device_synchronize(
            start_sem,
            start_device=[dy, 0],
            mcast_shape=[1, 4],
            num_receivers=3,
            core_indices=[cy, cx],
        )
        a = remote_load(lhs, [0, 0])
        b = remote_load(rhs, [0, 0])
        c = a @ b
        dx = mesh_position(1)
        remote_store(
            out,
            [dx, 0],
            c,
            start_device=[dy, 0],
            device_mcast_shape=[1, 4],
            semaphore=end_sem,
            semaphore_indices=[cy, 0],
        )
        # num_devices (3 remote + 1 local self-inc), not num_devices - 1.
        semaphore_wait(end_sem, 4)

    d2m.mesh((1, 4), topology=("linear", "linear"))
    L_in = d2m.Layout(
        shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )
    L_out = d2m.Layout(
        shape=(128, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[4, 1]
    )
    full_a = torch.randn(32, 128, dtype=torch.float32)
    full_b = torch.randn(32, 128, dtype=torch.float32)
    a_s = d2m.reblock(
        d2m.mesh_shard(full_a, L_in, shard_dims=[0, 1], shard_shape=[1, 4]),
        [1, 1],
    )
    b_s = d2m.reblock(
        d2m.mesh_shard(full_b, L_in, shard_dims=[0, 1], shard_shape=[1, 4]),
        [1, 1],
    )
    out_s = d2m.reblock(d2m.empty(L_out), [4, 1])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
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
    out_s = d2m.mesh_gather(out_s, shard_dims=[0, 1], shard_shape=[1, 4])
    result = out_s.to_host()

    cs = [full_a[:, i * 32 : (i + 1) * 32] @ full_b[:, i * 32 : (i + 1) * 32]
          for i in range(4)]
    vstack = torch.cat(cs, dim=0)  # (128, 32)
    expected = torch.cat([vstack] * 4, dim=1)  # (128, 128)
    assert tuple(result.shape) == (128, 128), result.shape
    assert_pcc(expected, result)


@pytest.mark.skipif(
    torch is None or runtime is None,
    reason="requires torch + ttmlir runtime",
)
@pytest.mark.skipif(_num_devices() < 4, reason="requires a >=4-device mesh")
def test_matmul_all_gather_fused_1x4_large_shards_roundtrip():
    """`test_matmul_all_gather_fused_1x4_roundtrip` with larger per-device
    shards: each device's matmul is a 2x2-tile output with a K=2 reduction
    (A_d, B_d, C_d are each 64x64), instead of a single 32x32 tile.

    This scales the fused path on two axes at once: the local matmul is now a
    genuine multi-tile M=N=K=2 product (the K=2 reduction accumulates in DST and
    routes through the SFPU, hence PCC -- not tight abs-diff -- per the f32
    K-reduction precision note in fused_matmul_allgather_8x8_design.md), and the
    all_gather payload grows to a 2x2 tile block mcast across the line. Each
    device ends holding vstack(C_0..C_3) (256,64); shard_to_full column-concats
    the four identical quarters -> (256,256). (Verified to scale further to
    4x4-tile / K=4 shards -> 512x512; 2x2 is kept here as a fast regression.)"""

    @d2m.kernel
    def matmul_all_gather(lhs, rhs, out, start_sem, end_sem):
        dy = mesh_position(0)
        cy = core_index(0)
        cx = core_index(1)
        device_synchronize(
            start_sem,
            start_device=[dy, 0],
            mcast_shape=[1, 4],
            num_receivers=3,
            core_indices=[cy, cx],
        )
        a = remote_load(lhs, [0, 0])  # (64,64) = 2x2 tiles (M=2, K=2)
        b = remote_load(rhs, [0, 0])  # (64,64) = 2x2 tiles (K=2, N=2)
        c = a @ b  # (64,64), reduces over K=2
        dx = mesh_position(1)
        remote_store(
            out,
            [dx, 0],
            c,
            start_device=[dy, 0],
            device_mcast_shape=[1, 4],
            semaphore=end_sem,
            semaphore_indices=[cy, 0],
        )
        semaphore_wait(end_sem, 4)

    d2m.mesh((1, 4), topology=("linear", "linear"))
    # Per-device A_d/B_d/C_d are 64x64 (2x2 tiles).
    L_in = d2m.Layout(
        shape=(64, 64), dtype=d2m.float32, block_shape=[2, 2], grid_shape=[1, 1]
    )
    # Output: 4 device slots stacked on the gather dim -> 256x64, each a 2x2 block.
    L_out = d2m.Layout(
        shape=(256, 64), dtype=d2m.float32, block_shape=[2, 2], grid_shape=[4, 1]
    )
    full_a = torch.randn(64, 256, dtype=torch.float32)
    full_b = torch.randn(64, 256, dtype=torch.float32)
    a_s = d2m.reblock(
        d2m.mesh_shard(full_a, L_in, shard_dims=[0, 1], shard_shape=[1, 4]),
        [1, 1],
    )
    b_s = d2m.reblock(
        d2m.mesh_shard(full_b, L_in, shard_dims=[0, 1], shard_shape=[1, 4]),
        [1, 1],
    )
    out_s = d2m.reblock(d2m.empty(L_out), [4, 1])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
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
    out_s = d2m.mesh_gather(out_s, shard_dims=[0, 1], shard_shape=[1, 4])
    result = out_s.to_host()

    cs = [full_a[:, i * 64 : (i + 1) * 64] @ full_b[:, i * 64 : (i + 1) * 64]
          for i in range(4)]  # each (64, 64)
    vstack = torch.cat(cs, dim=0)  # (256, 64)
    expected = torch.cat([vstack] * 4, dim=1)  # (256, 256)
    assert tuple(result.shape) == (256, 256), result.shape
    assert_pcc(expected, result)


@d2m.kernel
def _mm_shard(lhs, rhs, c):
    cy = core_index(0)
    cx = core_index(1)
    a = remote_load(lhs, [cy, cx])
    b = remote_load(rhs, [cy, cx])
    cc = a @ b
    remote_store(c, [cy, cx], cc)


@d2m.kernel
def _gather4(c, out):
    # grid (1,1): one core reads all four shards of C and stores them.
    g00 = empty([1, 1])
    g00 = remote_load(g00, c, [0, 0])
    remote_store(out, [0, 0], g00)
    g01 = empty([1, 1])
    g01 = remote_load(g01, c, [0, 1])
    remote_store(out, [0, 1], g01)
    g10 = empty([1, 1])
    g10 = remote_load(g10, c, [1, 0])
    remote_store(out, [1, 0], g10)
    g11 = empty([1, 1])
    g11 = remote_load(g11, c, [1, 1])
    remote_store(out, [1, 1], g11)


@pytest.mark.skipif(
    torch is None or runtime is None, reason="requires torch + ttmlir runtime"
)
@pytest.mark.skipif(_num_devices() < 1, reason="requires a device")
def test_two_generic_matmul_gather():
    """Two-generic structure (single device, no fabric) -- the building block for
    a full-grid matmul + all_gather, where the fabric's single-core constraint
    forces the matmul (grid N) and the gather/send (grid 1) into separate
    generics. Here: a grid-(2,2) matmul writes a 4-shard C, then a grid-(1,1)
    gather generic reads all four shards of C and stores them. Validates (a) two
    chained generics with the gather correctly ordered after the matmul, (b) a
    grid-1 generic reading a sharded input, and (c) the aliased-store fix -- the
    four gather stores must not alias the gather core's single local output
    shard (Allocate.cpp materializeAliasedLoadStore)."""
    L = d2m.Layout(
        shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 2]
    )
    a_full = torch.randn(64, 64, dtype=torch.float32)
    b_full = torch.randn(64, 64, dtype=torch.float32)
    c_d = d2m.empty(L)
    out_d = d2m.empty(L)
    _mm_shard(d2m.to_layout(a_full, L), d2m.to_layout(b_full, L), c_d, grid=(2, 2))
    _gather4(c_d, out_d, grid=(1, 1))
    result = out_d.to_host()

    # Each shard [cy,cx] is an independent per-tile matmul a[cy,cx] @ b[cy,cx].
    def _shard(t, cy, cx):
        return t[cy * 32 : (cy + 1) * 32, cx * 32 : (cx + 1) * 32]

    expected = torch.zeros(64, 64, dtype=torch.float32)
    for cy in range(2):
        for cx in range(2):
            expected[cy * 32 : (cy + 1) * 32, cx * 32 : (cx + 1) * 32] = _shard(
                a_full, cy, cx
            ) @ _shard(b_full, cy, cx)
    assert tuple(result.shape) == (64, 64), result.shape
    assert_pcc(expected, result)


@d2m.kernel
def _mm_col(lhs, rhs, c):
    # Multicore matmul: each core [cy,cx] computes an independent output tile
    # c[cy,cx] = lhs[cy,cx] @ rhs[cy,cx]. On grid (2,1) this is a 2-core matmul
    # producing a 2-row-tile per-device shard C_d.
    cy = core_index(0)
    cx = core_index(1)
    a = remote_load(lhs, [cy, cx])
    b = remote_load(rhs, [cy, cx])
    cc = a @ b
    remote_store(c, [cy, cx], cc)


@d2m.kernel
def _all_gather_mc(in0, out0, start_sem, end_sem):
    # all_gather of a multi-tile per-device shard. The grid-1 link core reads the
    # whole (reblocked) per-device shard and fabric-mcasts it to every device.
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
    buf = empty([2, 1])  # two row-tiles (the whole per-device shard) on one core
    buf = remote_load(buf, in0, [0, 0])
    dx = mesh_position(1)
    remote_store(
        out0,
        [dx, 0],
        buf,
        start_device=[dy, 0],
        device_mcast_shape=[1, 2],
        semaphore=end_sem,
        semaphore_indices=[cy, 0],
    )
    semaphore_wait(end_sem, 2)


@pytest.mark.skipif(
    torch is None or runtime is None, reason="requires torch + ttmlir runtime"
)
@pytest.mark.skipif(_num_devices() < 1, reason="requires a device")
def test_multicore_matmul_all_gather():
    """Multicore matmul + all_gather on a 1x2 mesh -- the full-grid target shape,
    built as the two-generic structure the fabric single-core constraint forces.

    Per device d (1x2 column shard of a (64,64) operand pair): a grid-(2,1)
    matmul generic computes C_d -- core [cy,0] does the independent tile product
    A_d[cy] @ B_d[cy], giving a 2-row-tile shard C_d = (64,32). reblock(C_d, 1x1)
    gathers both tiles onto the link core. A grid-(1,1) all_gather generic then
    reads the whole reblocked C_d and fabric-mcasts it across the line, so every
    device ends with vstack(C_0, C_1) = (128,32); mesh_gather column-concats the
    two identical device copies -> (128,64).

    This composes the pieces validated on the way here:
    - the matmul runs on a *multi-core* grid (vs the grid-1 fused proof), so the
      matmul and the single-core fabric send cannot share a generic;
    - they are therefore *two chained generics* (matmul -> all_gather), with the
      gather correctly ordered after the matmul (test_two_generic_matmul_gather);
    - reblock(matmul_output) feeds the gather to the link core (design sketch
      fused_matmul_allgather_8x8_design.md, step 1).
    """
    d2m.mesh((1, 2), topology=("linear", "linear"))
    L_in = d2m.Layout(
        shape=(64, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 1]
    )
    L_out = d2m.Layout(
        shape=(128, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 1]
    )
    full_a = torch.randn(64, 64, dtype=torch.float32)
    full_b = torch.randn(64, 64, dtype=torch.float32)
    a_s = d2m.reblock(
        d2m.mesh_shard(full_a, L_in, shard_dims=[0, 1], shard_shape=[1, 2]), [2, 1]
    )
    b_s = d2m.reblock(
        d2m.mesh_shard(full_b, L_in, shard_dims=[0, 1], shard_shape=[1, 2]), [2, 1]
    )
    c_d = d2m.empty(L_in)
    out_s = d2m.reblock(d2m.empty(L_out), [2, 1])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
    _mm_col(a_s, b_s, c_d, grid=(2, 1))
    c_rb = d2m.reblock(c_d, [1, 1])
    _all_gather_mc(
        c_rb,
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

    per_device = []
    for d in range(2):
        a_d = full_a[:, 32 * d : 32 * (d + 1)]
        b_d = full_b[:, 32 * d : 32 * (d + 1)]
        cd = torch.zeros(64, 32, dtype=torch.float32)
        for cy in range(2):
            cd[32 * cy : 32 * (cy + 1), :] = (
                a_d[32 * cy : 32 * (cy + 1), :] @ b_d[32 * cy : 32 * (cy + 1), :]
            )
        per_device.append(cd)
    stacked = torch.cat([per_device[0], per_device[1]], dim=0)  # (128, 32)
    expected = torch.cat([stacked, stacked], dim=1)  # (128, 64)
    assert tuple(result.shape) == (128, 64), result.shape
    assert_pcc(expected, result)


@d2m.kernel
def _all_gather_stream(in0, out0, start_sem, end_sem):
    # Streaming all_gather: the link core reads ONE tile of the per-device shard
    # at a time and fabric-mcasts it immediately, reusing a single 1-tile buffer.
    # Unlike _all_gather_mc (which reblocks the whole shard onto the link core
    # and sends it in one shot), this never materialises the whole shard -> L1
    # use is bounded by one tile regardless of grid, the mechanism that scales to
    # 8x8. Multiple fabric sends in one generic require ScheduleDMA to keep the
    # kernel on a single datamovement thread (one fabric_connection_manager);
    # otherwise the sends split across NOC threads and the two managers deadlock.
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
    dx = mesh_position(1)
    b0 = empty([1, 1])
    b0 = remote_load(b0, in0, [0, 0])
    remote_store(
        out0,
        [dx * 2 + 0, 0],
        b0,
        start_device=[dy, 0],
        device_mcast_shape=[1, 2],
        semaphore=end_sem,
        semaphore_indices=[cy, 0],
    )
    b1 = empty([1, 1])
    b1 = remote_load(b1, in0, [1, 0])
    remote_store(
        out0,
        [dx * 2 + 1, 0],
        b1,
        start_device=[dy, 0],
        device_mcast_shape=[1, 2],
        semaphore=end_sem,
        semaphore_indices=[cy, 0],
    )
    semaphore_wait(end_sem, 4)  # 2 sends x (1 remote + 1 self-inc)


@pytest.mark.skipif(
    torch is None or runtime is None, reason="requires torch + ttmlir runtime"
)
@pytest.mark.skipif(_num_devices() < 1, reason="requires a device")
def test_streaming_matmul_all_gather():
    """Streaming multicore matmul + all_gather on a 1x2 mesh -- the L1-scalable
    form of test_multicore_matmul_all_gather. Same grid-(2,1) matmul producing a
    2-row-tile per-device shard C_d, but the all_gather generic *streams*: it
    reads one tile of C_d and fabric-mcasts it, reusing a single 1-tile buffer,
    rather than reblocking the whole shard onto the link core and sending it in
    one shot. So the link core never holds more than one tile -> bounded L1
    regardless of grid (the path to 8x8, where an atomic gather of 64 tiles would
    pressure L1).

    The two fabric sends in one generic exercise ScheduleDMA keeping a fabric
    kernel on a single datamovement thread (one fabric_connection_manager);
    distributing the sends across NOC threads deadlocks two managers on one core.
    """
    d2m.mesh((1, 2), topology=("linear", "linear"))
    L_in = d2m.Layout(
        shape=(64, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 1]
    )
    L_out = d2m.Layout(
        shape=(128, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[4, 1]
    )
    full_a = torch.randn(64, 64, dtype=torch.float32)
    full_b = torch.randn(64, 64, dtype=torch.float32)
    a_s = d2m.reblock(
        d2m.mesh_shard(full_a, L_in, shard_dims=[0, 1], shard_shape=[1, 2]), [2, 1]
    )
    b_s = d2m.reblock(
        d2m.mesh_shard(full_b, L_in, shard_dims=[0, 1], shard_shape=[1, 2]), [2, 1]
    )
    c_d = d2m.empty(L_in)
    out_s = d2m.reblock(d2m.empty(L_out), [4, 1])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
    _mm_col(a_s, b_s, c_d, grid=(2, 1))
    _all_gather_stream(
        c_d,
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

    per_device = []
    for d in range(2):
        a_d = full_a[:, 32 * d : 32 * (d + 1)]
        b_d = full_b[:, 32 * d : 32 * (d + 1)]
        cd = torch.zeros(64, 32, dtype=torch.float32)
        for cy in range(2):
            cd[32 * cy : 32 * (cy + 1), :] = (
                a_d[32 * cy : 32 * (cy + 1), :] @ b_d[32 * cy : 32 * (cy + 1), :]
            )
        per_device.append(cd)
    stacked = torch.cat([per_device[0], per_device[1]], dim=0)
    expected = torch.cat([stacked, stacked], dim=1)
    assert tuple(result.shape) == (128, 64), result.shape
    assert_pcc(expected, result)


@d2m.kernel
def _all_gather_stream_1x4(in0, out0, start_sem, end_sem, n_tiles):
    # Streaming all_gather on the full 1x4 mesh: the link core reads ONE tile of
    # the per-device shard at a time and fabric-mcasts it across the 4-device
    # line immediately, reusing a single 1-tile buffer (bounded L1 regardless of
    # n_tiles). The Python `for` unrolls at trace time (n_tiles is a kernel arg).
    dy = mesh_position(0)
    cy = core_index(0)
    cx = core_index(1)
    device_synchronize(
        start_sem,
        start_device=[dy, 0],
        mcast_shape=[1, 4],
        num_receivers=3,
        core_indices=[cy, cx],
    )
    dx = mesh_position(1)
    for t in range(n_tiles):
        bt = empty([1, 1])
        bt = remote_load(bt, in0, [t, 0])
        remote_store(
            out0,
            [dx * n_tiles + t, 0],
            bt,
            start_device=[dy, 0],
            device_mcast_shape=[1, 4],
            semaphore=end_sem,
            semaphore_indices=[cy, 0],
        )
    semaphore_wait(end_sem, 4 * n_tiles)  # n_tiles sends x (3 remote + 1 self)


@pytest.mark.skipif(
    torch is None or runtime is None, reason="requires torch + ttmlir runtime"
)
@pytest.mark.skipif(_num_devices() < 4, reason="requires a >=4-device mesh")
def test_streaming_matmul_all_gather_1x4():
    """Streaming multicore matmul + all_gather on the full 1x4 mesh -- the
    full-mesh generalization of `test_streaming_matmul_all_gather`.

    A grid-(N,1) matmul produces an N-row-tile per-device shard C_d; the
    all_gather generic then *streams* it, reading one tile of C_d and
    fabric-mcasting it across the 4-device line, reusing a single 1-tile buffer.
    The link core therefore never holds more than one tile -- the L1-bounded
    mechanism that lets the per-device matmul grow without an atomic N-tile
    gather pressuring the link core's L1.

    N=2 here keeps the *gathered* output (4 devices x N tiles, stacked in a
    column) within the worker-grid height. Scaling the gathered output beyond
    the worker grid needs a DRAM-staged output (Option 2); that path currently
    deadlocks in the fabric remote_store->DRAM and is the next blocker (see
    STATUS_fused_matmul_allgather.md)."""
    N = 2
    d2m.mesh((1, 4), topology=("linear", "linear"))
    L_in = d2m.Layout(
        shape=(N * 32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[N, 1]
    )
    L_out = d2m.Layout(
        shape=(4 * N * 32, 32),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[4 * N, 1],
    )
    full_a = torch.randn(N * 32, 128, dtype=torch.float32)
    full_b = torch.randn(N * 32, 128, dtype=torch.float32)
    a_s = d2m.reblock(
        d2m.mesh_shard(full_a, L_in, shard_dims=[0, 1], shard_shape=[1, 4]), [N, 1]
    )
    b_s = d2m.reblock(
        d2m.mesh_shard(full_b, L_in, shard_dims=[0, 1], shard_shape=[1, 4]), [N, 1]
    )
    c_d = d2m.empty(L_in)
    out_s = d2m.reblock(d2m.empty(L_out), [4 * N, 1])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
    _mm_col(a_s, b_s, c_d, grid=(N, 1))
    _all_gather_stream_1x4(
        c_d,
        out_s,
        ss,
        es,
        N,
        grid=(1, 1),
        fabric=d2m.fabric_config(
            cluster_axis=1, topology="linear", routing="bidir_line_mesh"
        ),
    )
    out_s = d2m.mesh_gather(out_s, shard_dims=[0, 1], shard_shape=[1, 4])
    result = out_s.to_host()

    per_device = []
    for d in range(4):
        a_d = full_a[:, 32 * d : 32 * (d + 1)]
        b_d = full_b[:, 32 * d : 32 * (d + 1)]
        cd = torch.zeros(N * 32, 32, dtype=torch.float32)
        for cy in range(N):
            cd[32 * cy : 32 * (cy + 1), :] = (
                a_d[32 * cy : 32 * (cy + 1), :] @ b_d[32 * cy : 32 * (cy + 1), :]
            )
        per_device.append(cd)
    stacked = torch.cat(per_device, dim=0)  # (4*N*32, 32)
    expected = torch.cat([stacked] * 4, dim=1)  # (4*N*32, 128)
    assert tuple(result.shape) == (4 * N * 32, 128), result.shape
    assert_pcc(expected, result)


@d2m.kernel
def _fused_chunked_matmul_all_gather_1x4(lhs, rhs, out, start_sem, end_sem, n_tiles):
    # Fused single-generic matmul + all_gather with the matmul BLOCKED into
    # n_tiles chunks, each chunk streamed across the 1x4 line the moment it is
    # produced -- compute and communication interleave on the single link core.
    #
    # Contrast with test_streaming_matmul_all_gather_1x4: there a grid-(N,1)
    # matmul generic computes the whole C_d first, and a *separate* grid-1
    # all_gather generic streams the precomputed shard -- two chained generics,
    # so compute fully precedes comm. Here the matmul of chunk t and the fabric
    # mcast of chunk t live in ONE generic: split-unified-thread-v2 hands each
    # chunk's matmul-output CB from the compute thread to the DM thread, so the
    # compute thread can produce chunk t+1 while the DM thread still has chunk t
    # in flight on the fabric. That per-block compute/comm overlap is the d2m
    # analog of tt-metal's MatmulFusedOpSignaler (fused_matmul_allgather_8x8_
    # design.md): the matmul is chunked and the collective streams each chunk
    # interleaved, rather than a barrier between the two phases.
    #
    # The matmul runs only on the single grid-(1,1) core (which is also the link
    # core), so unlike _fused_matmul_all_gather there is no cross-core gather and
    # no producer-done `ready` fence -- the structure is
    # test_matmul_all_gather_fused_1x4_roundtrip's grid-1 fused body wrapped in
    # the per-tile streaming loop of _all_gather_stream_1x4.
    dy = mesh_position(0)
    cy = core_index(0)
    cx = core_index(1)
    device_synchronize(
        start_sem,
        start_device=[dy, 0],
        mcast_shape=[1, 4],
        num_receivers=3,
        core_indices=[cy, cx],
    )
    dx = mesh_position(1)
    for t in range(n_tiles):
        a = remote_load(lhs, [t, 0])  # chunk t of A_d (one row-tile)
        b = remote_load(rhs, [t, 0])  # chunk t of B_d
        c = a @ b  # matmul this chunk
        remote_store(  # stream it immediately, before computing chunk t+1
            out,
            [dx * n_tiles + t, 0],
            c,
            start_device=[dy, 0],
            device_mcast_shape=[1, 4],
            semaphore=end_sem,
            semaphore_indices=[cy, 0],
        )
    semaphore_wait(end_sem, 4 * n_tiles)  # n_tiles sends x (3 remote + 1 self)


@pytest.mark.skipif(
    torch is None or runtime is None, reason="requires torch + ttmlir runtime"
)
@pytest.mark.skipif(_num_devices() < 4, reason="requires a >=4-device mesh")
def test_fused_chunked_matmul_all_gather_1x4():
    """Fused *single-generic* chunked matmul + all_gather on the full 1x4 mesh --
    the compute/comm-overlap form of test_streaming_matmul_all_gather_1x4.

    The per-device matmul C_d (an N-row-tile shard) is blocked into N chunks;
    the single link core computes chunk t (C_d[t] = A_d[t] @ B_d[t]) and
    fabric-mcasts it across the 4-device line *immediately*, then moves on to
    chunk t+1, reusing one tile of L1. The intent is that, because the matmul
    and the fabric send are fused into one generic, split-unified-thread-v2's
    compute->DM CB handoff lets chunk t+1 compute while chunk t is still
    streaming -- the block-granular overlap the 8x8 design targets (the
    MatmulFusedOpSignaler analog), vs the two-generic streaming test where the
    matmul fully precedes the gather.

    Every device ends with vstack(C_0..C_3) (4*N row-tiles); mesh_gather
    column-concats the four identical device copies. Result is identical to
    test_streaming_matmul_all_gather_1x4 -- only the fusion/interleaving differs.

    Regression guard for the matmul-in-a-streaming-loop accumulate bug: the
    matmul L1-accumulate guard in D2MInsertDstRegisterAccess used to pick the
    enclosing per-chunk loop as the matmul's reduction loop, so chunk t>0
    accumulated onto stale output (every chunk after the first was garbage).
    Fixed by excluding loops that index the generic's output store
    (genericOutputStoreDependsOnIV in InsertDstRegisterAccess/Shared.cpp) from
    the accumulate-guard / L1-acc trigger selection.

    N=2 keeps the gathered output (4*N row-tiles) within the worker-grid height;
    larger N needs the block-packed gather output (see
    test_all_gather_1x4_large_block_roundtrip) or a DRAM-staged output."""
    N = 2
    d2m.mesh((1, 4), topology=("linear", "linear"))
    L_in = d2m.Layout(
        shape=(N * 32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[N, 1]
    )
    L_out = d2m.Layout(
        shape=(4 * N * 32, 32),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[4 * N, 1],
    )
    full_a = torch.randn(N * 32, 128, dtype=torch.float32)
    full_b = torch.randn(N * 32, 128, dtype=torch.float32)
    a_s = d2m.reblock(
        d2m.mesh_shard(full_a, L_in, shard_dims=[0, 1], shard_shape=[1, 4]), [N, 1]
    )
    b_s = d2m.reblock(
        d2m.mesh_shard(full_b, L_in, shard_dims=[0, 1], shard_shape=[1, 4]), [N, 1]
    )
    out_s = d2m.reblock(d2m.empty(L_out), [4 * N, 1])
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
    _fused_chunked_matmul_all_gather_1x4(
        a_s,
        b_s,
        out_s,
        ss,
        es,
        N,
        grid=(1, 1),
        fabric=d2m.fabric_config(
            cluster_axis=1, topology="linear", routing="bidir_line_mesh"
        ),
    )
    out_s = d2m.mesh_gather(out_s, shard_dims=[0, 1], shard_shape=[1, 4])
    result = out_s.to_host()

    per_device = []
    for d in range(4):
        a_d = full_a[:, 32 * d : 32 * (d + 1)]
        b_d = full_b[:, 32 * d : 32 * (d + 1)]
        cd = torch.zeros(N * 32, 32, dtype=torch.float32)
        for cy in range(N):
            cd[32 * cy : 32 * (cy + 1), :] = (
                a_d[32 * cy : 32 * (cy + 1), :] @ b_d[32 * cy : 32 * (cy + 1), :]
            )
        per_device.append(cd)
    stacked = torch.cat(per_device, dim=0)  # (4*N*32, 32)
    expected = torch.cat([stacked] * 4, dim=1)  # (4*N*32, 128)
    assert tuple(result.shape) == (4 * N * 32, 128), result.shape
    assert_pcc(expected, result)


# ---------------------------------------------------------------------------
# Compute-isolation references for the fused matmul + all_gather
# ---------------------------------------------------------------------------
#
# These two kernels are the *fused single-generic* all_gather (producer-done
# `ready` fence -> router core_read-gathers both row tiles -> fabric-mcasts
# each across the 1x2 line) with the matmul swapped for a trivial compute.
# They were the bisection that root-caused the fused-matmul all_gather race:
#
#   zeros   -> reads NO input, trivial compute      -> all-gathers correctly
#   eltwise -> reads ONE input, trivial compute (abs)-> all-gathers correctly
#   matmul  -> reads TWO inputs via tile_matmul_block-> RACED (garbage/inf)
#
# So the core_read + fabric path and the input-streaming handshake are sound;
# the race was specific to tile_matmul_block, which reads whole CB blocks
# directly and (unlike the eltwise ops, whose per-tile memref loads carry the
# consume handshake) was missing the input cb_wait_front/cb_pop_front in the
# V2 split's compute thread -- it read its inputs before the input DMA landed.
# Keeping these as regression refs: if the fused matmul ever races again, run
# these first to confirm the gather/fabric path is still innocent.


@d2m.kernel
def _fused_zeros_all_gather(lhs, out, start_sem, ready, end_sem):
    # `lhs` is unused on purpose: the compute fills zeros and reads no input,
    # so a correct all-gather here is independent of inputs AND of the compute.
    cy = core_index(0)
    cx = core_index(1)
    dy = mesh_position(0)
    if is_router_core():
        device_synchronize(
            start_sem,
            start_device=[dy, 0],
            mcast_shape=[1, 2],
            num_receivers=1,
            core_indices=[cy, cx],
        )
    c = zeros([1, 1])
    semaphore_inc(ready, 1, core=[0, 0], compute=True)  # producer-done fence
    if is_router_core():
        semaphore_wait(ready, 2)  # both cores' compute is done
        dx = mesh_position(1)
        g0 = empty([1, 1])
        g0 = core_read(g0, c, core=[0, 0])
        g1 = empty([1, 1])
        g1 = core_read(g1, c, core=[1, 0])
        remote_store(
            out,
            [dx * 2 + 0, 0],
            g0,
            start_device=[dy, 0],
            device_mcast_shape=[1, 2],
            semaphore=end_sem,
            semaphore_indices=[cy, 0],
        )
        remote_store(
            out,
            [dx * 2 + 1, 0],
            g1,
            start_device=[dy, 0],
            device_mcast_shape=[1, 2],
            semaphore=end_sem,
            semaphore_indices=[cy, 0],
        )
        semaphore_wait(end_sem, 4)  # 2 sends x (1 remote + 1 self-inc)


@pytest.mark.skipif(
    torch is None or runtime is None, reason="requires torch + ttmlir runtime"
)
@pytest.mark.skipif(_num_devices() < 2, reason="requires a >=2-device mesh")
def test_fused_zeros_all_gather():
    """Fused single-generic all_gather whose compute is a trivial `zeros` fill
    (no input). Proves the producer-done fence + cross-core core_read gather +
    fabric mcast all-gather correctly on a 1x2 mesh, independent of any input
    or compute -- the compute-isolation baseline for the fused matmul path."""
    d2m.mesh((1, 2), topology=("linear", "linear"))
    L_in = d2m.Layout(
        shape=(64, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 1]
    )
    L_out = d2m.Layout(
        shape=(128, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[4, 1]
    )
    full_a = torch.randn(64, 64, dtype=torch.float32)
    a_s = d2m.reblock(
        d2m.mesh_shard(full_a, L_in, shard_dims=[0, 1], shard_shape=[1, 2]), [2, 1]
    )
    out_s = d2m.reblock(d2m.empty(L_out), [4, 1])
    ss = d2m.global_semaphore()
    ready = d2m.global_semaphore(init=0)
    es = d2m.global_semaphore()
    _fused_zeros_all_gather(
        a_s,
        out_s,
        ss,
        ready,
        es,
        grid=(2, 1),
        fabric=d2m.fabric_config(
            cluster_axis=1,
            topology="linear",
            routing="bidir_line_mesh",
            router_cores=[(0, 0)],
        ),
    )
    out_s = d2m.mesh_gather(out_s, shard_dims=[0, 1], shard_shape=[1, 2])
    result = out_s.to_host()
    # All output tiles are the zeros the compute produced. (corrcoef of two
    # constant tensors is nan, so check the value directly rather than pcc.)
    assert tuple(result.shape) == (128, 64), result.shape
    assert torch.isfinite(result).all().item(), result
    assert result.abs().max().item() < 1e-3, result.abs().max().item()


@d2m.kernel
def _fused_eltwise_all_gather(lhs, out, start_sem, ready, end_sem):
    cy = core_index(0)
    cx = core_index(1)
    dy = mesh_position(0)
    if is_router_core():
        device_synchronize(
            start_sem,
            start_device=[dy, 0],
            mcast_shape=[1, 2],
            num_receivers=1,
            core_indices=[cy, cx],
        )
    a = remote_load(lhs, [cy, cx])
    c = abs(a)  # read one input + trivial elementwise compute
    semaphore_inc(ready, 1, core=[0, 0], compute=True)  # producer-done fence
    if is_router_core():
        semaphore_wait(ready, 2)
        dx = mesh_position(1)
        g0 = empty([1, 1])
        g0 = core_read(g0, c, core=[0, 0])
        g1 = empty([1, 1])
        g1 = core_read(g1, c, core=[1, 0])
        remote_store(
            out,
            [dx * 2 + 0, 0],
            g0,
            start_device=[dy, 0],
            device_mcast_shape=[1, 2],
            semaphore=end_sem,
            semaphore_indices=[cy, 0],
        )
        remote_store(
            out,
            [dx * 2 + 1, 0],
            g1,
            start_device=[dy, 0],
            device_mcast_shape=[1, 2],
            semaphore=end_sem,
            semaphore_indices=[cy, 0],
        )
        semaphore_wait(end_sem, 4)


@pytest.mark.skipif(
    torch is None or runtime is None, reason="requires torch + ttmlir runtime"
)
@pytest.mark.skipif(_num_devices() < 2, reason="requires a >=2-device mesh")
def test_fused_eltwise_all_gather():
    """Fused single-generic all_gather whose compute reads ONE input and applies
    a trivial elementwise op (abs), then core_read-gathers both row tiles and
    fabric-mcasts them across the 1x2 mesh. Same structure as the fused matmul
    all_gather but with the matmul swapped for abs -- isolates the input-stream
    handshake + gather/fabric path from tile_matmul_block."""
    d2m.mesh((1, 2), topology=("linear", "linear"))
    L_in = d2m.Layout(
        shape=(64, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 1]
    )
    L_out = d2m.Layout(
        shape=(128, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[4, 1]
    )
    full_a = torch.randn(64, 64, dtype=torch.float32)
    a_s = d2m.reblock(
        d2m.mesh_shard(full_a, L_in, shard_dims=[0, 1], shard_shape=[1, 2]), [2, 1]
    )
    out_s = d2m.reblock(d2m.empty(L_out), [4, 1])
    ss = d2m.global_semaphore()
    ready = d2m.global_semaphore(init=0)
    es = d2m.global_semaphore()
    _fused_eltwise_all_gather(
        a_s,
        out_s,
        ss,
        ready,
        es,
        grid=(2, 1),
        fabric=d2m.fabric_config(
            cluster_axis=1,
            topology="linear",
            routing="bidir_line_mesh",
            router_cores=[(0, 0)],
        ),
    )
    out_s = d2m.mesh_gather(out_s, shard_dims=[0, 1], shard_shape=[1, 2])
    result = out_s.to_host()

    # Each device d computes abs of its (64,32) column shard, tiled into the two
    # row tiles g0=[0,0], g1=[1,0]; the gather stacks them and mcasts so every
    # device holds vstack(abs(A_0), abs(A_1)) (128,32); mesh_gather column-concats
    # the two identical device copies -> (128,64).
    ca0 = full_a[:, 0:32].abs()
    ca1 = full_a[:, 32:64].abs()
    stacked = torch.cat([ca0, ca1], dim=0)  # (128, 32)
    expected = torch.cat([stacked, stacked], dim=1)  # (128, 64)
    assert tuple(result.shape) == (128, 64), result.shape
    assert_pcc(expected, result)


@d2m.kernel
def _fused_matmul_all_gather(lhs, rhs, out, start_sem, ready, end_sem):
    # The target fused single-generic matmul + all_gather: each core does its
    # independent row-tile product C[cy] = A[cy] @ B[cy] (compute thread), a
    # producer-done `ready` fence orders the matmul before the gather, then the
    # router core core_read-gathers both row tiles and fabric-mcasts each across
    # the 1x2 line. The compute-isolation siblings _fused_zeros_all_gather /
    # _fused_eltwise_all_gather are the same kernel with the matmul swapped out;
    # this one regressed (garbage/inf) until tile_matmul_block got its input CB
    # wait/pop handshake in SplitUnifiedThreadV2.
    cy = core_index(0)
    cx = core_index(1)
    dy = mesh_position(0)
    if is_router_core():
        device_synchronize(
            start_sem,
            start_device=[dy, 0],
            mcast_shape=[1, 2],
            num_receivers=1,
            core_indices=[cy, cx],
        )
    a = remote_load(lhs, [cy, cx])
    b = remote_load(rhs, [cy, cx])
    c = a @ b
    semaphore_inc(ready, 1, core=[0, 0], compute=True)  # producer-done fence
    if is_router_core():
        semaphore_wait(ready, 2)  # both cores' matmul is done
        dx = mesh_position(1)
        g0 = empty([1, 1])
        g0 = core_read(g0, c, core=[0, 0])
        g1 = empty([1, 1])
        g1 = core_read(g1, c, core=[1, 0])
        remote_store(
            out,
            [dx * 2 + 0, 0],
            g0,
            start_device=[dy, 0],
            device_mcast_shape=[1, 2],
            semaphore=end_sem,
            semaphore_indices=[cy, 0],
        )
        remote_store(
            out,
            [dx * 2 + 1, 0],
            g1,
            start_device=[dy, 0],
            device_mcast_shape=[1, 2],
            semaphore=end_sem,
            semaphore_indices=[cy, 0],
        )
        semaphore_wait(end_sem, 4)  # 2 sends x (1 remote + 1 self-inc)


@pytest.mark.skipif(
    torch is None or runtime is None, reason="requires torch + ttmlir runtime"
)
@pytest.mark.skipif(_num_devices() < 2, reason="requires a >=2-device mesh")
def test_fused_matmul_all_gather():
    """Fused *single-generic* matmul + all_gather on a 1x2 mesh -- the target the
    zeros/eltwise references isolate. A grid-(2,1) matmul produces each device's
    2-row-tile shard C_d; a producer-done semaphore fence orders it before the
    router core's cross-core core_read gather of both tiles, which are then
    fabric-mcast across the line so every device holds vstack(C_0, C_1) (128,32);
    mesh_gather column-concats the two identical device copies -> (128,64).

    This is the structural analog of test_streaming_matmul_all_gather collapsed
    into ONE generic (core_read gather instead of a second remote_load generic),
    and it exercises the tile_matmul_block input CB handshake that the V2 split
    pass was missing (the race this test family root-caused)."""
    d2m.mesh((1, 2), topology=("linear", "linear"))
    L_in = d2m.Layout(
        shape=(64, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 1]
    )
    L_out = d2m.Layout(
        shape=(128, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[4, 1]
    )
    full_a = torch.randn(64, 64, dtype=torch.float32)
    full_b = torch.randn(64, 64, dtype=torch.float32)
    a_s = d2m.reblock(
        d2m.mesh_shard(full_a, L_in, shard_dims=[0, 1], shard_shape=[1, 2]), [2, 1]
    )
    b_s = d2m.reblock(
        d2m.mesh_shard(full_b, L_in, shard_dims=[0, 1], shard_shape=[1, 2]), [2, 1]
    )
    out_s = d2m.reblock(d2m.empty(L_out), [4, 1])
    ss = d2m.global_semaphore()
    ready = d2m.global_semaphore(init=0)
    es = d2m.global_semaphore()
    _fused_matmul_all_gather(
        a_s,
        b_s,
        out_s,
        ss,
        ready,
        es,
        grid=(2, 1),
        fabric=d2m.fabric_config(
            cluster_axis=1,
            topology="linear",
            routing="bidir_line_mesh",
            router_cores=[(0, 0)],
        ),
    )
    out_s = d2m.mesh_gather(out_s, shard_dims=[0, 1], shard_shape=[1, 2])
    result = out_s.to_host()

    per_device = []
    for d in range(2):
        a_d = full_a[:, 32 * d : 32 * (d + 1)]
        b_d = full_b[:, 32 * d : 32 * (d + 1)]
        cd = torch.zeros(64, 32, dtype=torch.float32)
        for cy in range(2):
            cd[32 * cy : 32 * (cy + 1), :] = (
                a_d[32 * cy : 32 * (cy + 1), :] @ b_d[32 * cy : 32 * (cy + 1), :]
            )
        per_device.append(cd)
    stacked = torch.cat([per_device[0], per_device[1]], dim=0)  # (128, 32)
    expected = torch.cat([stacked, stacked], dim=1)  # (128, 64)
    assert tuple(result.shape) == (128, 64), result.shape
    assert_pcc(expected, result)


@d2m.kernel
def _fused_matmul_all_gather_grid(lhs, rhs, out, start_sem, ready, end_sem, gy, gx):
    # Grid-(gy,gx) generalization of _fused_matmul_all_gather: the router core
    # core_read-gathers all gy*gx output tiles in a nested loop (vs the 2 hard-
    # coded reads) and fabric-mcasts each across the 1x2 line. This is the path
    # toward the 8x8 target -- it scales the gather fan-in to gy*gx->1. The
    # one-tile-per-core output grid is [2*gy, gx], so gy*gx is bounded by
    # 2*gy*gx <= physical cores (64): 4x4 (16->1) fits exactly at 32 output
    # tiles; 8x8 (128 output tiles) needs a coarser output / multi-tile cores.
    cy = core_index(0)
    cx = core_index(1)
    dy = mesh_position(0)
    if is_router_core():
        device_synchronize(
            start_sem,
            start_device=[dy, 0],
            mcast_shape=[1, 2],
            num_receivers=1,
            core_indices=[cy, cx],
        )
    a = remote_load(lhs, [cy, cx])
    b = remote_load(rhs, [cy, cx])
    c = a @ b
    semaphore_inc(ready, 1, core=[0, 0], compute=True)  # producer-done fence
    if is_router_core():
        semaphore_wait(ready, gy * gx)  # every core's matmul is done
        dx = mesh_position(1)
        for ty in range(gy):
            for tx in range(gx):
                g = empty([1, 1])
                g = core_read(g, c, core=[ty, tx])
                remote_store(
                    out,
                    [dx * gy + ty, tx],
                    g,
                    start_device=[dy, 0],
                    device_mcast_shape=[1, 2],
                    semaphore=end_sem,
                    semaphore_indices=[cy, 0],
                )
        semaphore_wait(end_sem, 2 * gy * gx)  # each send: 1 remote + 1 self-inc


@pytest.mark.skipif(
    torch is None or runtime is None, reason="requires torch + ttmlir runtime"
)
@pytest.mark.skipif(_num_devices() < 2, reason="requires a >=2-device mesh")
def test_fused_matmul_all_gather_grid_4x4():
    """Fused single-generic matmul + all_gather scaled to a 4x4 grid: a 16->1
    cross-core core_read gather on the router core, fabric-mcast across the 1x2
    mesh. Block-diagonal matmul (core [cy,cx] does the independent tile product
    A[cy,cx] @ B[cy,cx]); device d's shard C_d is (128,128), the two devices'
    shards stack to (256,128) per device and mesh_gather column-concats the
    identical copies -> (256,256). Largest gather fan-in that fits one tile per
    core (output grid [8,4] = 32 <= 64); 8x8 needs a coarser output layout."""
    GY, GX = 4, 4
    rows, cols = 32 * GY, 32 * GX
    d2m.mesh((1, 2), topology=("linear", "linear"))
    L_in = d2m.Layout(
        shape=(rows, cols), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[GY, GX]
    )
    L_out = d2m.Layout(
        shape=(2 * rows, cols),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[2 * GY, GX],
    )
    full_a = torch.randn(rows, 2 * cols, dtype=torch.float32)
    full_b = torch.randn(rows, 2 * cols, dtype=torch.float32)
    a_s = d2m.reblock(
        d2m.mesh_shard(full_a, L_in, shard_dims=[0, 1], shard_shape=[1, 2]), [GY, GX]
    )
    b_s = d2m.reblock(
        d2m.mesh_shard(full_b, L_in, shard_dims=[0, 1], shard_shape=[1, 2]), [GY, GX]
    )
    out_s = d2m.reblock(d2m.empty(L_out), [2 * GY, GX])
    ss = d2m.global_semaphore()
    ready = d2m.global_semaphore(init=0)
    es = d2m.global_semaphore()
    _fused_matmul_all_gather_grid(
        a_s,
        b_s,
        out_s,
        ss,
        ready,
        es,
        GY,
        GX,
        grid=(GY, GX),
        fabric=d2m.fabric_config(
            cluster_axis=1,
            topology="linear",
            routing="bidir_line_mesh",
            router_cores=[(0, 0)],
        ),
    )
    out_s = d2m.mesh_gather(out_s, shard_dims=[0, 1], shard_shape=[1, 2])
    result = out_s.to_host()

    per_device = []
    for d in range(2):
        a_d = full_a[:, d * cols : (d + 1) * cols]
        b_d = full_b[:, d * cols : (d + 1) * cols]
        cd = torch.zeros(rows, cols, dtype=torch.float32)
        for ty in range(GY):
            for tx in range(GX):
                cd[32 * ty : 32 * (ty + 1), 32 * tx : 32 * (tx + 1)] = (
                    a_d[32 * ty : 32 * (ty + 1), 32 * tx : 32 * (tx + 1)]
                    @ b_d[32 * ty : 32 * (ty + 1), 32 * tx : 32 * (tx + 1)]
                )
        per_device.append(cd)
    stacked = torch.cat([per_device[0], per_device[1]], dim=0)  # (256, 128)
    expected = torch.cat([stacked, stacked], dim=1)  # (256, 256)
    assert tuple(result.shape) == (256, 256), result.shape
    assert_pcc(expected, result)
