# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#

import glob
import importlib.metadata
import importlib.util
import os
import shutil

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

# Per-rank artifact names emitted by `python -m tracy -r`.
TRACY_CAPTURE_GLOB = "tracy_profile_log_host.tracy"
PERF_CSV_GLOB = "ops_perf_results*.csv"

# The 2x4 multiprocess rank binding maps to 2 host ranks (4 devices each).
EXPECTED_NUM_RANKS = 2

RANK_BINDING_PATH = (
    f"{TT_METAL_RUNTIME_ROOT_EXTERNAL}"
    "/tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml"
)


# Directly receive the name of tracy tools found in import
# Default to a common known name
def _tracy_tool_names():
    try:
        from tracy.common import TRACY_CAPTURE_TOOL, TRACY_CSVEXPROT_TOOL

        return (TRACY_CAPTURE_TOOL, TRACY_CSVEXPROT_TOOL)
    except Exception:
        return ("tracy-capture", "tracy-csvexport")


TRACY_TOOL_NAMES = _tracy_tool_names()

# Directly find the tracy tools directory
def _resolve_tracy_tools_dir():
    """Find the dir containing the tracy capture binaries"""
    # Primary: the installed ttrt wheel's bundled copies (site-packages).
    try:
        for f in importlib.metadata.files("ttrt") or []:
            if os.path.basename(str(f)) in TRACY_TOOL_NAMES:
                tools_dir = os.path.dirname(str(f.locate()))
                if all(
                    os.path.exists(os.path.join(tools_dir, t)) for t in TRACY_TOOL_NAMES
                ):
                    return tools_dir
    except Exception:
        pass

    # Fallbacks: env-driven build / install layouts.
    candidates = []

    def add(path):
        if path and path not in candidates:
            candidates.append(path)

    add(os.path.join(os.environ.get("TT_METAL_HOME", ""), "build/tools/profiler/bin"))
    build_home = os.environ.get("TT_METAL_BUILD_HOME")
    if build_home:
        add(os.path.join(build_home, "tools/profiler/bin"))

    for var in ("TT_METAL_RUNTIME_ROOT", "TT_METAL_RUNTIME_ROOT_EXTERNAL"):
        root = os.environ.get(var)
        if root:
            add(root)
            add(os.path.join(root, "build/tools/profiler/bin"))

    for base in list(getattr(ttrt, "__path__", [])):
        add(os.path.join(base, "runtime"))

    install_dir = os.environ.get("INSTALL_DIR")
    if install_dir:
        add(os.path.join(install_dir, "bin"))
    add(os.path.join(TT_MLIR_HOME, "install", "bin"))

    for candidate in candidates:
        if all(os.path.exists(os.path.join(candidate, t)) for t in TRACY_TOOL_NAMES):
            return candidate

    raise AssertionError(
        "Tracy tools (%s) not found. Searched (after importlib.metadata):\n%s\n"
        "Build tt-mlir with -DTT_RUNTIME_ENABLE_PERF_TRACE=ON."
        % (", ".join(TRACY_TOOL_NAMES), "\n".join("  " + c for c in candidates))
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


def _find_in_dir(root, pattern):
    """Recursive glob under `root` (incl. hidden dirs) for files matching `pattern`."""
    return sorted(
        glob.glob(
            os.path.join(root, "**", pattern),
            recursive=True,
            include_hidden=True,
        )
    )


def test_tracy_multiprocess_smoke():
    """Smoke test for Tracy profiling under the MultiProcess distributed runtime.

    Launches a distributed runtime on llmbox with --tracy enabled, runs
    simple_add a few times, then asserts that every rank produced a non-empty
    .tracy capture and a non-empty ops_perf CSV.
    """
    # Start from a clean output dir so the assertions below can only pass on
    # artifacts produced by this run, disregarding any possible stale captures.
    shutil.rmtree(TRACY_OUTPUT_DIR, ignore_errors=True)
    os.makedirs(TRACY_OUTPUT_DIR, exist_ok=True)

    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "simple_add_2x4.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), (
        f"Binary not found: {binary_path}\n"
        "Expected to be produced by the lit test "
        "test/ttmlir/Runtime/TTNN/llmbox/binary/simple_add_2x4.mlir."
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

    # The workers run `python -m tracy`, so the tracy module must be importable
    # in this environment (it ships in the ttrt wheel). Fail clearly here rather
    # than with an opaque "No module named tracy" crash inside the workers.
    assert importlib.util.find_spec("tracy") is not None, (
        "tracy module is not importable, so the workers cannot run "
        "`python -m tracy`. Ensure the ttrt wheel (which bundles the tracy "
        "package) is installed in this environment."
    )

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

    # Verify per rank produced a non-empty .tracy capture and ops_perf CSV
    rank_dirs = sorted(glob.glob(os.path.join(TRACY_OUTPUT_DIR, "rank*")))
    assert len(rank_dirs) >= EXPECTED_NUM_RANKS, (
        f"expected output for >={EXPECTED_NUM_RANKS} ranks under {TRACY_OUTPUT_DIR}, "
        f"found rank dirs: {rank_dirs}"
    )

    for rank_dir in rank_dirs:
        rank = os.path.basename(rank_dir)

        tracy_files = _find_in_dir(rank_dir, TRACY_CAPTURE_GLOB)
        assert tracy_files, f"{rank}: no .tracy capture under {rank_dir}"
        for f in tracy_files:
            assert os.path.getsize(f) > 0, f"{rank}: empty .tracy capture: {f}"

        perf_csv_files = _find_in_dir(rank_dir, PERF_CSV_GLOB)
        assert perf_csv_files, f"{rank}: no ops_perf CSV under {rank_dir}"
        for f in perf_csv_files:
            assert os.path.getsize(f) > 0, f"{rank}: empty perf CSV: {f}"

        print(
            f"{rank}: {len(tracy_files)} .tracy capture(s), "
            f"{len(perf_csv_files)} ops_perf CSV(s)"
        )
