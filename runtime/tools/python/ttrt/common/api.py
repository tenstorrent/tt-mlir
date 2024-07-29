# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttrt.binary
import os
import json
import importlib.machinery
import sys
import signal
import os
import io
import subprocess
import time
import socket
from pkg_resources import get_distribution
import sys
import shutil

from ttrt.common.util import *

try:
    from ttrt.common.perf_trace import *

    perf_trace = True
except:
    perf_trace = False

#######################################################################################
########################################**API**########################################
#######################################################################################
"""
API: read
  - read contents of flatbuffer file
"""


def read(args):
    check_file_exists(args.binary)
    copy_file_into_ttrt_artifact(args.binary)
    fbb = ttrt.binary.load_from_path(args.binary)
    check_version(fbb.version)
    read_actions[args.section](fbb)


"""
API: run
  - run flatbuffer on device
"""


def run(args):
    import ttrt.runtime

    try:
        import torch
    except ModuleNotFoundError:
        raise ImportError(
            "Error: torch required for offline run, please `pip install torch`"
        )

    def toDataType(dtype):
        if dtype == torch.float32:
            return ttrt.runtime.DataType.Float32
        if dtype == torch.float16:
            return ttrt.runtime.DataType.Float16
        if dtype == torch.bfloat16:
            return ttrt.runtime.DataType.BFloat16
        if dtype == torch.uint32:
            return ttrt.runtime.DataType.UInt32
        if dtype == torch.uint16:
            return ttrt.runtime.DataType.UInt16
        if dtype == torch.uint8:
            return ttrt.runtime.DataType.UInt8
        raise ValueError(f"unsupported dtype: {dtype}")

    def fromDataType(dtype):
        if dtype == "Float32":
            return torch.float32
        if dtype == "Float16":
            return torch.float16
        if dtype == "BFloat16":
            return torch.bfloat16
        if dtype == "UInt32":
            return torch.uint32
        if dtype == "UInt16":
            return torch.uint16
        if dtype == "UInt8":
            return torch.uint8
        raise ValueError(f"unsupported dtype: {dtype}")

    check_file_exists(args.binary)
    copy_file_into_ttrt_artifact(args.binary)
    fbb = ttrt.binary.load_binary_from_path(args.binary)
    check_version(fbb.version)
    assert fbb.file_identifier == "TTNN", "Only TTNN binaries are supported"
    d = ttrt.binary.as_dict(fbb)

    program_index = int(args.program_index)
    assert program_index <= len(d["programs"]), "args.program_index out of range"
    program = d["programs"][program_index]
    print(f"running program[{program_index}]:", program["name"])

    torch_inputs = []
    torch_outputs = []
    for i in program["inputs"]:
        torch_inputs.append(
            torch.randn(
                i["desc"]["shape"],
                dtype=fromDataType(i["desc"]["layout"]["memory_desc"]["data_type"]),
            )
        )
    for i in program["outputs"]:
        torch_outputs.append(
            torch.zeros(
                i["desc"]["shape"],
                dtype=fromDataType(i["desc"]["layout"]["memory_desc"]["data_type"]),
            )
        )

    print("inputs:\n", torch_inputs)

    inputs = []
    outputs = []
    for i in torch_inputs:
        inputs.append(
            ttrt.runtime.create_tensor(
                i.data_ptr(),
                list(i.shape),
                list(i.stride()),
                i.element_size(),
                toDataType(i.dtype),
            )
        )

    for i in torch_outputs:
        outputs.append(
            ttrt.runtime.create_tensor(
                i.data_ptr(),
                list(i.shape),
                list(i.stride()),
                i.element_size(),
                toDataType(i.dtype),
            )
        )

    system_desc, device_ids = ttrt.runtime.get_current_system_desc()
    device = ttrt.runtime.open_device(device_ids)
    ttrt.runtime.submit(device, fbb, 0, inputs, outputs)
    print("outputs:\n", torch_outputs)
    ttrt.runtime.close_device(device)


"""
API: query
  - query device for system descriptor in the form of a flatbuffer
"""


def query(args):
    import ttrt.runtime

    if args.system_desc or args.system_desc_as_json:
        print(ttrt.runtime.get_current_system_desc()[0].as_json())
    if args.system_desc_as_dict:
        print(system_desc_as_dict(ttrt.runtime.get_current_system_desc()[0]))
    if args.save_system_desc:
        desc = ttrt.runtime.get_current_system_desc()[0]
        if args.save_system_desc:
            file_name = args.save_system_desc
        else:
            d = system_desc_as_dict(desc)
            file_name = d["product_identifier"] + ".ttsys"
        desc.store(file_name)
        print("system desc saved to:", file_name)
        copy_file_into_ttrt_artifact(file_name)


"""
API: perf
  - run flatbuffer on device in performance mode
"""


def perf(args):
    if args.generate_params:
        # generate consolidated model report
        check_file_exists(args.perf_csv)
        generate_params_dict(args.perf_csv)
    else:
        if not perf_trace:
            print("Perf mode is not enabled. Please rebuild tt-mlir with perf mode")
            sys.exit(1)

        # get available port for tracy client and server to communicate on
        port = get_available_port()

        if not port:
            print("No available port found")
            sys.exit(1)
        print(f"Using port {port}")

        # setup environment flags
        envVars = dict(os.environ)
        envVars["TRACY_PORT"] = port
        envVars["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "0"

        if args.device:
            envVars["TT_METAL_DEVICE_PROFILER"] = "1"
        else:
            if "TT_METAL_DEVICE_PROFILER" in envVars.keys():
                del envVars["TT_METAL_DEVICE_PROFILER"]

        # run perf artifact setup
        captureProcess = run_perf_artifact_setup(port)

        # generate test command to execute
        check_file_exists(args.binary)
        copy_file_into_ttrt_artifact(args.binary)
        testCommand = f"python -m tracy -p {TTMLIR_VENV_DIR}/bin/ttrt run {args.binary}"
        testProcess = subprocess.Popen(
            [testCommand], shell=True, env=envVars, preexec_fn=os.setsid
        )
        print(f"Test process started")

        # setup multiprocess signal handler
        def signal_handler(sig, frame):
            os.killpg(os.getpgid(testProcess.pid), signal.SIGTERM)
            captureProcess.terminate()
            captureProcess.communicate()
            sys.exit(3)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        testProcess.communicate()

        try:
            captureProcess.communicate(timeout=15)
            generate_report()
        except subprocess.TimeoutExpired as e:
            captureProcess.terminate()
            captureProcess.communicate()
            print(
                f"No profiling data could be captured. Please make sure you are on the correct build. Use scripts/build_scripts/build_with_profiler_opt.sh to build if you are not sure."
            )
            sys.exit(1)
