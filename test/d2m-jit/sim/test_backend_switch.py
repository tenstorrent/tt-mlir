# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""`config.backend` switch on the canonical `import d2m_jit` surface.

The device half of the switch needs silicon to run `to_host`, so here we only
exercise the device-free parts: that the sim backend dispatches and computes
correctly, that the device backend still builds the lazy graph, and that an
invalid backend is rejected. The autouse fixture restores the process-global
`config.backend` so these tests cannot leak into the device tests.
"""

import pytest
import torch

import d2m_jit as d2m
from d2m_jit._src.sim import SimTensor
from d2m_jit._src.builder import LazyTensor


@pytest.fixture(autouse=True)
def _restore_backend():
    saved = d2m.config.backend
    yield
    d2m.config.backend = saved


@d2m.kernel
def add(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], a + b)


def _layout():
    return d2m.Layout(
        shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 2]
    )


def test_sim_backend_dispatch_and_run():
    d2m.config.backend = "sim"
    layout = _layout()
    lhs, rhs = torch.randn(64, 64), torch.randn(64, 64)
    a = d2m.to_layout(lhs, layout)
    assert isinstance(a, SimTensor)
    out = d2m.empty(layout)
    add(a, d2m.to_layout(rhs, layout), out, 1, 1, grid=(2, 2))
    assert torch.allclose(lhs + rhs, out.to_host(), atol=1e-5)


def test_device_backend_builds_lazy_graph():
    d2m.config.backend = "device"
    lt = d2m.to_layout(torch.randn(64, 64), _layout())
    assert isinstance(lt, LazyTensor)  # no to_host (needs a device)


def test_invalid_backend_raises():
    d2m.config.backend = "nonsense"
    with pytest.raises(ValueError, match="backend"):
        d2m.empty(_layout())


def test_every_device_syntax_name_has_a_sim_backing():
    """Mechanical audit: every in-kernel name the device DSL registers resolves in
    the sim registries too, so a kernel body resolves identically under either
    backend. This catches "new device op landed with no sim backing" at the
    source, rather than waiting for some kernel to use it -- the gap that left
    the comparisons and in-kernel `zeros` unbacked.

    Lives here rather than in test_sim.py because reading the device registry
    needs the MLIR bindings, which that file deliberately does without."""
    import d2m_jit.api  # noqa: F401 -- the @syntax decorators populate _syntax
    from d2m_jit._src.ast import D2MCompiler
    from d2m_jit._src.sim.ops import SIM_METHODS, SIM_OPS
    from d2m_jit._src.sim.tensors import SimBlock

    registry = D2MCompiler._syntax
    # Guard against a vacuous pass: an empty registry would satisfy the subset
    # assertion below trivially (it is only populated by importing api).
    assert len(registry) > 100, f"device syntax registry looks unpopulated: {registry}"

    missing = []
    for qualified in registry:
        name = qualified.rsplit(".", 1)[-1]
        if qualified.startswith("!d2m.semaphore."):
            # Backed as methods on the Semaphore class injected into SIM_OPS.
            if not hasattr(SIM_OPS["Semaphore"], name):
                missing.append(qualified)
        elif name.startswith("__") and name.endswith("__"):
            # Operator forms are implemented directly on SimBlock.
            if not hasattr(SimBlock, name):
                missing.append(qualified)
        elif name not in SIM_OPS and name not in SIM_METHODS:
            missing.append(qualified)

    # `!tensor.store` belongs to the declarative generic form, and
    # `__matmul_acc__` is supplied by native Python `+=` on a SimBlock; both are
    # v2 items tracked in SIMULATOR_SPEC.md §12.
    known_gaps = {"!tensor.store", "__matmul_acc__"}
    assert (
        set(missing) <= known_gaps
    ), f"device syntax names with no sim backing: {sorted(set(missing) - known_gaps)}"


def test_every_dispatched_host_op_is_backed_and_exported_by_sim():
    """Host-op analog of the `@syntax` audit above.

    Every host op the canonical `import d2m_jit` surface dispatches (each tagged
    `_d2m_dispatch_name` by `api._dispatch`) must (a) resolve in the sim package,
    (b) be listed in `_src/sim/__init__.__all__`, and (c) be listed in the
    `d2m_jit.sim` shadow surface's `__all__`. Without this, a host op added to
    the device builder + `_dispatch` but forgotten in `_src/sim/host.py` only
    fails at call time under sim, and only if some test happens to exercise it --
    the gap that left `reduction_layout` running the device impl under
    `backend="sim"`."""
    import d2m_jit.api as _api
    import d2m_jit._src.sim as _sim_pkg
    import d2m_jit.sim as _sim_shadow

    dispatched = {
        v._d2m_dispatch_name
        for v in vars(_api).values()
        if getattr(v, "_d2m_dispatch_name", None) is not None
    }
    # Guard against a vacuous pass if the tagging ever silently breaks (an empty
    # set would satisfy every subset assertion below trivially).
    assert len(dispatched) >= 12, f"dispatched host-op set looks wrong: {dispatched}"

    not_backed = sorted(n for n in dispatched if not hasattr(_sim_pkg, n))
    assert not not_backed, f"dispatched host ops with no sim backing: {not_backed}"

    missing_pkg = sorted(n for n in dispatched if n not in _sim_pkg.__all__)
    assert (
        not missing_pkg
    ), f"dispatched host ops missing from _src/sim __all__: {missing_pkg}"

    missing_shadow = sorted(n for n in dispatched if n not in _sim_shadow.__all__)
    assert (
        not missing_shadow
    ), f"dispatched host ops missing from d2m_jit.sim shadow __all__: {missing_shadow}"


def test_kernel_forwards_attributes_to_backend_kernel():
    """Before the switch, `@d2m.kernel` *was* the concrete kernel, so callers
    read attributes straight off it -- `CompiledKernel._captures` is checked by
    test/d2m-jit/lit/captures.py. The dispatch wrapper has to forward those."""
    d2m.config.backend = "device"
    LIMIT = 7

    @d2m.kernel
    def captures_limit(in_t, out_t):
        for _ in range(LIMIT):
            remote_store(out_t, [0, 0], remote_load(in_t, [0, 0]))

    assert captures_limit._captures.get("LIMIT") == LIMIT
    # Still the wrapper's own name, not the impl's.
    assert captures_limit.__name__ == "captures_limit"
    with pytest.raises(AttributeError):
        captures_limit.not_a_real_attribute


def test_kernel_decorator_picks_backend_per_call():
    # The same @d2m.kernel object dispatches by backend at call time.
    layout = _layout()
    lhs, rhs = torch.randn(64, 64), torch.randn(64, 64)

    d2m.config.backend = "sim"
    out = d2m.empty(layout)
    add(d2m.to_layout(lhs, layout), d2m.to_layout(rhs, layout), out, 1, 1, grid=(2, 2))
    assert isinstance(out, SimTensor)
    assert torch.allclose(lhs + rhs, out.to_host(), atol=1e-5)
