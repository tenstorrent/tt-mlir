# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import functools
import json
import os
import re

import pytest
import torch


# CI machine types (the RUNS_ON env var) a test runs on when it carries no
# `machines` marker: the single-chip lanes. Tests that need other hardware opt
# in with e.g. @pytest.mark.machines("n300", "llmbox").
_DEFAULT_MACHINES = frozenset({"n150", "p150"})

# Devices each CI machine type provides. A test's device requirement is the
# smallest count among the machines it opted into; local runs (RUNS_ON unset)
# use it to skip tests this system doesn't have enough chips for.
_MACHINE_NUM_DEVICES = {"n150": 1, "p150": 1, "n300": 2, "llmbox": 8}


@functools.lru_cache(maxsize=1)
def _num_devices():
    """Chip count read from the builder's resolved system descriptor
    (SYSTEM_DESC_PATH if set, otherwise queried from the runtime).

    Unknown (no system desc, no runtime bindings) counts as a single-chip box:
    single-device tests still run and multi-chip tests skip."""
    try:
        # Imported lazily: the simulator suite runs with no MLIR bindings.
        from _ttmlir_runtime import binary
        from d2m_jit._src.builder import _get_system_desc_path

        system_desc = _get_system_desc_path()
        if not system_desc:
            return 1
        desc = binary.load_system_desc_from_path(system_desc).as_json()
        desc = re.sub(r"\bnan\b", "NaN", desc)
        desc = re.sub(r"\binf\b", "Infinity", desc)
        return len(json.loads(desc)["system_desc"]["chip_desc_indices"])
    except Exception:
        return 1


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "device_only(reason): asserts device-specific behavior -- an intended "
        "simulator divergence (SIMULATOR_SPEC.md §9), a host API the backend "
        "switch does not dispatch, or a device-specific error type (§8). Skipped "
        "when the suite is re-run with D2M_JIT_BACKEND=sim. Always pass "
        "`reason=`: it is what shows up in the skip report, and it should name "
        "the root cause, not the symptom.",
    )
    config.addinivalue_line(
        "markers",
        "machines(*names): CI machine types (RUNS_ON values) this test runs "
        "on. Unmarked tests default to the single-chip lanes (n150/p150); "
        "multi-chip tests opt into n300/llmbox. In CI (RUNS_ON set) a test "
        "runs only on its listed machines; locally it runs whenever this "
        "system has at least as many devices as the smallest machine listed.",
    )


def _sim_backend_requested():
    return os.environ.get("D2M_JIT_BACKEND", "device") == "sim"


def pytest_collection_modifyitems(config, items):
    """Skip tests that don't apply to this backend or CI machine.

    Two filters, both skip-by-marker rather than deselect-by-path so the
    exclusions stay visible in the junit report instead of silently narrowing
    what a lane covers:

    - `device_only` tests are skipped when the whole suite is re-run with
      D2M_JIT_BACKEND=sim (see .github/test_scripts/d2m_jit.sh).
    - When RUNS_ON is set (CI), each test runs only on its `machines` marker's
      machine types, defaulting to the single-chip lanes (_DEFAULT_MACHINES).
      This is what keeps the n300/llmbox lanes down to the multi-chip tests.
    - Locally (RUNS_ON unset), a test is skipped when this system has fewer
      devices than the least-demanding machine the test opted into, so a
      single-chip box runs everything except the multi-chip tests.
    """
    sim = _sim_backend_requested()
    runs_on = os.environ.get("RUNS_ON")
    for item in items:
        if sim:
            marker = item.get_closest_marker("device_only")
            if marker is not None:
                reason = marker.kwargs.get("reason") or (
                    marker.args[0]
                    if marker.args
                    else "NO REASON GIVEN -- please add one"
                )
                item.add_marker(pytest.mark.skip(reason=f"device_only: {reason}"))
        marker = item.get_closest_marker("machines")
        if marker is not None and not marker.args:
            raise pytest.UsageError(
                "machines marker requires at least one machine name, e.g. "
                '@pytest.mark.machines("n300")'
            )
        allowed = frozenset(marker.args) if marker else _DEFAULT_MACHINES
        unknown = sorted(m for m in allowed if m not in _MACHINE_NUM_DEVICES)
        if unknown:
            raise pytest.UsageError(f"Unknown machines marker values: {unknown}")
        required = min(_MACHINE_NUM_DEVICES[m] for m in allowed)
        if runs_on:
            if runs_on not in allowed:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"machines: runs on {sorted(allowed)}, not {runs_on}"
                    )
                )
        elif _num_devices() < required:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"machines: needs >= {required} devices, "
                    f"this system has {_num_devices()}"
                )
            )


@pytest.fixture(scope="function", autouse=True)
def _set_seed():
    """Deterministic torch RNG per-test for reproducibility."""
    torch.manual_seed(0)


@pytest.fixture(scope="function", autouse=True)
def _reset_builder():
    """Drop the process-level builder singleton between tests so a failed
    compile (negative tests) doesn't leak MLIR state into the next test."""
    yield
    # Imported here, not at module scope: the simulator suite runs with no MLIR
    # bindings at all, and there is no builder state to drop in that case.
    try:
        from d2m_jit._src.builder import _Builder
    except ImportError:
        return
    _Builder.reset()


@pytest.fixture(scope="function", autouse=True)
def _clear_mesh_mirror():
    """Clear the process-global mesh mirror between tests. The device builder
    clears it whenever a fresh graph starts, but the sim has no graph lifecycle
    (SIMULATOR_SPEC.md §14.2), so a `d2m.mesh(...)` declared by one test would
    otherwise leak per-device allocation into every test that runs after it."""
    yield
    # `layout_math` is deliberately MLIR-free, so this import is safe in the
    # binding-free simulator lane.
    from d2m_jit._src.layout_math import clear_current_mesh

    clear_current_mesh()


def pytest_generate_tests(metafunc):
    """Parametrize the generic pattern tests over every spec declared in the
    bundled pattern files (test/d2m-jit/patterns/*.py). Adding a pattern file with
    PATTERN_TESTS / KERNEL_BENCHES is picked up here with no harness edits."""
    # `runner` imports the MLIR bindings, so only reach for it when a test
    # actually asks for one of the parametrized fixtures -- otherwise collecting
    # the device-free simulator suite would require a tt-metal build.
    if not {"pattern_test", "kernel_bench", "e2e_spec"} & set(metafunc.fixturenames):
        return

    from runner import discover

    pattern_tests, kernel_benches = discover()
    if "pattern_test" in metafunc.fixturenames:
        metafunc.parametrize(
            "pattern_test", pattern_tests, ids=[t.name for t in pattern_tests]
        )
    if "kernel_bench" in metafunc.fixturenames:
        metafunc.parametrize(
            "kernel_bench", kernel_benches, ids=[b.name for b in kernel_benches]
        )
    if "e2e_spec" in metafunc.fixturenames:
        # `golden` is optional: a spec with no golden cross-checks against the
        # ttnn device baseline of its original TTIR (see run_e2e).
        e2e = [t for t in pattern_tests if t.e2e]
        metafunc.parametrize("e2e_spec", e2e, ids=[t.name for t in e2e])


@pytest.fixture(scope="function")
def e2e_device():
    """An in-process mesh-device handle for one e2e test, opened lazily on first
    use and closed afterwards. Function-scoped so at most one device is open at
    a time (the in-process builder device tests open/close their own per call),
    avoiding any cross-test device contention — no subprocess, no marker split.

    For large-scale CI, prefer a single batch driver that opens one device and
    loops over all specs in-process, rather than one pytest case per pattern."""
    from runner import E2EDevice

    holder = E2EDevice()
    yield holder
    holder.close()
