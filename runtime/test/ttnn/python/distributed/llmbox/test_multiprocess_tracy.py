# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#

import glob
import os

import pytest
import ttrt
import ttrt.runtime

from ...utils import DeviceContext, TT_METAL_RUNTIME_ROOT_EXTERNAL, TT_MLIR_HOME

TRACY_OUTPUT_DIR = "/tmp/tt_mlir_tracy_smoke_test"

RANK_BINDING_PATH = (
    f"{TT_METAL_RUNTIME_ROOT_EXTERNAL}"
    "/tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml"
)


def _launch_distributed_runtime_with_tracy(tracy_args):
    """Launch the distributed runtime in MultiProcess mode with --tracy enabled."""
    assert os.path.exists(
        RANK_BINDING_PATH
    ), f"Rank binding path not found: {RANK_BINDING_PATH}"

    ttrt.runtime.set_mlir_home(TT_MLIR_HOME)
    ttrt.runtime.set_metal_home(TT_METAL_RUNTIME_ROOT_EXTERNAL)

    mp_args = ttrt.runtime.MultiProcessArgs.create(RANK_BINDING_PATH)
    mp_args.with_allow_run_as_root(True)
    mp_args.with_tracy(True)
    if tracy_args:
        mp_args.with_tracy_args(tracy_args)

    distributed_options = ttrt.runtime.DistributedOptions()
    distributed_options.mode = ttrt.runtime.DistributedMode.MultiProcess
    distributed_options.multi_process_args = mp_args
    ttrt.runtime.set_current_host_runtime(ttrt.runtime.HostRuntime.Distributed)
    ttrt.runtime.launch_distributed_runtime(distributed_options)


def _shutdown_distributed_runtime():
    ttrt.runtime.shutdown_distributed_runtime()
    ttrt.runtime.set_current_host_runtime(ttrt.runtime.HostRuntime.Local)


def _find_per_rank(output_root, pattern):
    """Recursive glob under each rank<N>/ for files matching `pattern`."""
    matches = set()
    matches.update(
        glob.glob(os.path.join(output_root, "rank*", "**", pattern), recursive=True)
    )
    matches.update(glob.glob(os.path.join(output_root, f"rank*{pattern}")))
    return sorted(matches)


def _find_tracy_files(output_root):
    return _find_per_rank(output_root, "*.tracy")


def _find_perf_csv_files(output_root):
    """`ops_perf_results_*.csv` is the processed op-level report. Glob is loose
    to also accept legacy `ops_perf.csv` / similar names."""
    return _find_per_rank(output_root, "*ops_perf*.csv")


def test_tracy_multi_host_smoke():
    """End-to-end check that --tracy plumbing produces per-rank Tracy captures.

    Asserts:
      - Distributed runtime launches successfully with tracy enabled.
      - A trivial runtime call (get_num_available_devices) sees the full mesh.
      - After shutdown, per-rank .tracy files exist on disk and are non-empty.
    """
    os.makedirs(TRACY_OUTPUT_DIR, exist_ok=True)

    _launch_distributed_runtime_with_tracy(["--output-folder", TRACY_OUTPUT_DIR])
    try:
        # Workload: confirm the mesh is visible, then open a mesh device so
        # tracy captures real device-init activity on the workers (mirrors
        # what test_get_mesh_shape does in the sibling test_multiprocess.py).
        num_devices = ttrt.runtime.get_num_available_devices()
        assert num_devices == 8, f"expected 8 devices on a llmbox, got {num_devices}"
        with DeviceContext(mesh_shape=[2, 4]) as device:
            assert device.get_mesh_shape() == [2, 4]
    finally:
        _shutdown_distributed_runtime()

    tracy_files = _find_tracy_files(TRACY_OUTPUT_DIR)
    assert len(tracy_files) >= 2, (
        f"expected >=2 per-rank .tracy files under {TRACY_OUTPUT_DIR}, "
        f"found {tracy_files}"
    )
    for f in tracy_files:
        assert os.path.getsize(f) > 0, f"tracy file is empty: {f}"

    perf_csv_files = _find_perf_csv_files(TRACY_OUTPUT_DIR)
    assert len(perf_csv_files) >= 2, (
        f"expected >=2 per-rank ops_perf CSV files under {TRACY_OUTPUT_DIR}, "
        f"found {perf_csv_files}"
    )
    for f in perf_csv_files:
        assert os.path.getsize(f) > 0, f"perf CSV is empty: {f}"

    print(
        f"\nTracy captures written to: {TRACY_OUTPUT_DIR}\n"
        f"  .tracy files: {tracy_files}\n"
        f"  ops_perf CSVs: {perf_csv_files}\n"
        f"Open any .tracy file in the Tracy GUI to inspect."
    )
