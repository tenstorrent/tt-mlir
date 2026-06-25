# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import math


def assert_pcc(golden, actual, threshold=0.99):
    combined = torch.stack([golden.flatten(), actual.flatten()])
    pcc = torch.corrcoef(combined)[0, 1].item()
    assert (
        pcc >= threshold
    ), f"Expected pcc {pcc} >= {threshold}\ngolden:\n{golden}\nactual:\n{actual}"


def device_runtime_available():
    """True when the d2m-jit device runtime extension is importable (a proxy
    for 'can run the device backend'). Used to skip sim-vs-device parity tests
    in no-device environments."""
    try:
        from _ttmlir_runtime import runtime
    except (ModuleNotFoundError, ImportError):
        return False
    return runtime is not None


def _run_on_backend(build_fn, backend, seed):
    """Run `build_fn` (which constructs a graph and returns its to_host
    output(s)) under `config.backend == backend`, reseeding torch first so the
    two backends see identical random inputs. Restores the prior backend and
    resets the lazy builder around the run."""
    import d2m_jit as d2m
    from d2m_jit._src.builder import _Builder

    saved = d2m.config.backend
    d2m.config.backend = backend
    _Builder.reset()
    try:
        torch.manual_seed(seed)
        result = build_fn()
    finally:
        d2m.config.backend = saved
        _Builder.reset()
    return result if isinstance(result, tuple) else (result,)


def assert_parity(build_fn, threshold=0.99, seed=0):
    """Run the same kernel graph on the device and the simulator and assert the
    outputs agree by PCC.

    `build_fn` must be a zero-arg callable that builds the graph fresh on each
    call (creating its inputs with torch RNG) and returns the materialised
    output -- a `torch.Tensor` or a tuple of them. The simulator is the
    intended-semantics oracle, so this catches device lowering regressions.

    Only use it for kernels where device and sim are expected to agree (most
    eltwise / reduction / softmax / matmul-with-zeros-prefill). Cases where the
    device deliberately diverges (raw `empty` contents, matmul into `empty`,
    multicast) are not parity-testable -- see SIMULATOR_SPEC.md §9.
    """
    sim_out = _run_on_backend(build_fn, "sim", seed)
    device_out = _run_on_backend(build_fn, "device", seed)
    assert len(sim_out) == len(
        device_out
    ), f"output count mismatch: sim {len(sim_out)} vs device {len(device_out)}"
    for i, (sim_t, dev_t) in enumerate(zip(sim_out, device_out)):
        assert tuple(sim_t.shape) == tuple(dev_t.shape), (
            f"output {i} shape mismatch: sim {tuple(sim_t.shape)} vs "
            f"device {tuple(dev_t.shape)}"
        )
        assert_pcc(sim_t.float(), dev_t.float(), threshold)


def arange_tile(*shape, tile_size=[32, 32], dtype=None):
    assert len(shape) >= 2
    assert shape[-2] % tile_size[-2] == 0
    assert shape[-1] % tile_size[-1] == 0
    tiled_shape = list(shape)
    tiled_shape[-2] //= tile_size[-2]
    tiled_shape[-1] //= tile_size[-1]
    tensor = torch.arange(math.prod(tiled_shape), dtype=dtype).reshape(tiled_shape)
    tensor = tensor.unsqueeze(-1).unsqueeze(-1)
    tensor = tensor.repeat([1] * len(tiled_shape) + tile_size)
    return tensor.transpose(-2, -3).reshape(shape)
