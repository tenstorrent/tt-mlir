# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#

import glob
import os

import ttrt
import ttrt.runtime
from ttrt.common.util import Binary, FileManager, Logger

from ...utils import (
    DeviceContext,
    ProgramTestConfig,
    ProgramTestRunner,
    TT_METAL_RUNTIME_ROOT_EXTERNAL,
    TT_MLIR_HOME,
)

FLATBUFFER_BASE_PATH = (
    f"{TT_MLIR_HOME}/build/test/ttmlir/Runtime/TTNN/llmbox/binary/Output"
)

TRACY_OUTPUT_DIR = "/tmp/tt_mlir_tracy_smoke_test"

RANK_BINDING_PATH = (
    f"{TT_METAL_RUNTIME_ROOT_EXTERNAL}"
    "/tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml"
)

TRACY_TOOL_NAMES = ("capture-release", "csvexport-release")


def _resolve_tracy_tools_dir():
    """Find the dir holding the tracy capture binaries."""
    candidates = []
    install_dir = os.environ.get("INSTALL_DIR")
    if install_dir:
        candidates.append(os.path.join(install_dir, "bin"))
    candidates.append(os.path.join(TT_MLIR_HOME, "install", "bin"))
    candidates.append(os.path.join(os.path.dirname(ttrt.__file__), "runtime"))
    candidates.append(
        os.path.join(
            TT_METAL_RUNTIME_ROOT_EXTERNAL, "build", "tools", "profiler", "bin"
        )
    )

    for candidate in candidates:
        if all(os.path.exists(os.path.join(candidate, t)) for t in TRACY_TOOL_NAMES):
            return candidate

    raise AssertionError(
        "Tracy tools (%s) not found. Searched:\n%s\n"
        "Build tt-mlir with -DTT_RUNTIME_ENABLE_PERF_TRACE=ON."
        % (", ".join(TRACY_TOOL_NAMES), "\n".join(candidates))
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
    return sorted(
        glob.glob(
            os.path.join(output_root, "rank*", "**", pattern),
            recursive=True,
            include_hidden=True,
        )
    )


def _find_tracy_files(output_root):
    return _find_per_rank(output_root, "tracy_profile_log_host.tracy")


def _find_perf_csv_files(output_root):
    return _find_per_rank(output_root, "ops_perf_results*.csv")


def test_tracy_multiprocess_smoke():
    """Smoke test for Tracy profiling under the MultiProcess distributed runtime.

    Launches a distributed runtime on llmbox with --tracy enabled, runs
    simple_add a few times, then asserts that every rank produced a non-empty
    .tracy capture and a non-empty ops_perf CSV.
    """
    os.makedirs(TRACY_OUTPUT_DIR, exist_ok=True)

    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "simple_add_2x4.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), (
        f"Binary not found: {binary_path}\n"
        "Generate it with: llvm-lit "
        "test/ttmlir/Runtime/TTNN/llmbox/binary/simple_add_2x4.mlir"
    )

    test_config = ProgramTestConfig(
        name="simple_add_tracy",
        expected_num_inputs=2,
        compute_golden=lambda inputs: (inputs[0] + inputs[1]),
        description="Simple add under tracy profiling",
    )
    logger = Logger()
    file_manager = FileManager(logger)
    binary = Binary(logger, file_manager, binary_path)

    test_runner = ProgramTestRunner(test_config, binary, 0)

    # Resolve (and fail fast if missing) the dir holding the tracy binaries.
    tracy_tools_dir = _resolve_tracy_tools_dir()

    # -r enables the per-rank capture + csvexport report pipeline.
    TRACY_ARGS = [
        "--output-folder",
        TRACY_OUTPUT_DIR,
        "--tracy-tools-folder",
        tracy_tools_dir,
        "-r",
    ]

    _launch_distributed_runtime_with_tracy(TRACY_ARGS)
    try:
        num_devices = ttrt.runtime.get_num_available_devices()
        assert num_devices == 8, f"expected 8 devices on a llmbox, got {num_devices}"
        with DeviceContext(mesh_shape=[2, 4]) as device:
            assert device.get_mesh_shape() == [2, 4]
            inputs_runtime_with_layout, golden, _ = test_runner.get_inputs_and_golden(
                device, borrow=False
            )
            for _ in range(4):
                test_runner.run_program_and_compare_golden(
                    device, inputs_runtime_with_layout, golden
                )
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
    )
