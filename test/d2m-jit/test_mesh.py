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
`mesh_shard(...).to_host()` DSL flow. (Compute *between* the shard and gather
is blocked on TensorMeshAttr annotation -- see CCL_SPEC.md section 7.)
"""

import pytest

import d2m_jit as d2m
from d2m_jit._src.builder import (
    _Builder,
    _build_pipeline,
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


def _num_devices():
    if runtime is None:
        return 0
    try:
        return runtime.get_num_available_devices()
    except Exception:
        return 0


def test_mesh_sets_module_attr():
    """d2m.mesh(...) sets the module's ttcore.meshes attr and stores topology."""
    try:
        d2m.mesh((1, 2), topology=("linear", "ring"))
        b = _Builder.get()
        assert '#ttcore.meshes<[<"mesh" = 1x2>]>' in str(b.module.operation)
        assert b._mesh_topology == ["linear", "ring"]
    finally:
        _Builder.reset()


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
        _b._Builder.reset()
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
    # Func arg is the full tensor; mesh_shard produces the per-device shard.
    assert "tensor<64x128xf32>" in ir_text and "tensor<64x64xf32>" in ir_text


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
