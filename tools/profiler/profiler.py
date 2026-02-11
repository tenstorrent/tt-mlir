# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
import os
import sys
import signal
import subprocess
import shutil

from tracy.process_ops_logs import process_ops


@contextmanager
def trace(log_dir: str, port: int, host_only: bool = False):
    os.makedirs(log_dir, exist_ok=True)

    TT_METAL_RUNTIME_ROOT = os.environ.get(
        "TT_METAL_RUNTIME_ROOT", "third_party/tt-metal/src/tt-metal"
    )
    tracy_capture_tool_path = os.path.join(TT_METAL_RUNTIME_ROOT, "capture-release")
    tracy_csvexport_tool_path = os.path.join(TT_METAL_RUNTIME_ROOT, "csvexport-release")
    tracy_file_path = log_dir + "/tracy_profile_log_host.tracy"
    tracy_ops_times_file_path = log_dir + "/tracy_ops_times.csv"
    tracy_ops_data_file_path = log_dir + "/tracy_ops_data.csv"
    profiler_logs_dir = TT_METAL_RUNTIME_ROOT + "/generated/profiler/.logs/"
    profiler_csv_file_path = (
        TT_METAL_RUNTIME_ROOT + "/generated/profiler/reports/ops_perf_results.csv"
    )
    os.environ["TRACY_PORT"] = str(port)
    if not host_only:
        os.environ["TT_METAL_CLEAR_L1"] = "1"
        os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
        os.environ["TTNN_OP_PROFILER"] = "1"
        os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "0"
        os.environ["TT_METAL_PROFILER_CPP_POST_PROCESS"] = "1"

    if os.path.exists(profiler_logs_dir):
        shutil.rmtree(profiler_logs_dir)
    os.makedirs(profiler_logs_dir)

    tracy_capture_tool_command = (
        f"{tracy_capture_tool_path} -o {tracy_file_path} -f -p {port}"
    )
    tracy_capture_tool_process = subprocess.Popen(
        tracy_capture_tool_command, shell=True, start_new_session=True
    )

    def signal_handler(sig, frame):
        os.killpg(os.getpgid(testProcess.pid), signal.SIGTERM)
        tracy_capture_tool_process.terminate()
        tracy_capture_tool_process.communicate()
        sys.exit(3)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        yield
    finally:
        # Need to send kill signal to tracy server to flush the .tracy file.
        os.killpg(tracy_capture_tool_process.pid, signal.SIGINT)
        tracy_capture_tool_process.communicate(timeout=15)

        with open(tracy_ops_times_file_path, "w") as csv_file:
            child_calls = ["CompileProgram", "HWCommandQueue_write_buffer"]
            child_calls_str = f"-x {','.join(child_calls)}"
            subprocess.run(
                f"{tracy_csvexport_tool_path} -u -p TT_DNN {child_calls_str} {tracy_file_path}",
                shell=True,
                check=True,
                stdout=csv_file,
                stderr=subprocess.DEVNULL,
            )

        with open(tracy_ops_data_file_path, "w") as csv_file:
            subprocess.run(
                f'{tracy_csvexport_tool_path} -m -s ";" {tracy_file_path}',
                shell=True,
                check=True,
                stdout=csv_file,
                stderr=subprocess.DEVNULL,
            )

        shutil.copy(tracy_file_path, profiler_logs_dir)
        shutil.copy(tracy_ops_times_file_path, profiler_logs_dir)
        shutil.copy(tracy_ops_data_file_path, profiler_logs_dir)
        process_ops(None, None, False)

        if os.path.exists(profiler_csv_file_path):
            shutil.copy(profiler_csv_file_path, log_dir)
