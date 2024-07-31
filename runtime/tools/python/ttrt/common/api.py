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
import atexit

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
    # initialization
    binaries = []
    fbb_list = []

    # acquire parameters
    arg_binary = args.binary
    arg_clean_artifacts = args.clean_artifacts

    # preprocessing
    if os.path.isdir(arg_binary):
        binaries = find_ttnn_files(arg_binary)
    else:
        binaries.append(arg_binary)

    if arg_clean_artifacts:
        clean_artifacts()

    setup_artifacts(binaries)

    # constraint checking
    for binary in binaries:
        check_file_exists(binary)
        fbb = ttrt.binary.load_from_path(binary)
        check_version(fbb.version)
        fbb_list.append(fbb)

    # execution
    for fbb in fbb_list:
        read_actions[args.section](fbb)

    # save artifacts
    for binary in binaries:
        copy_ttnn_binary_into_artifact(binary)


"""
API: run
  - run flatbuffer on device
"""


def run(args):
    import ttrt.runtime

    # initialization
    binaries = []
    fbb_list = []
    torch_inputs = {}
    torch_outputs = {}
    system_desc = None

    # acquire parameters
    arg_binary = args.binary
    arg_program_index = args.program_index
    arg_clean_artifacts = args.clean_artifacts
    arg_loops = int(args.loops)

    # preprocessing
    if os.path.isdir(arg_binary):
        binaries = find_ttnn_files(arg_binary)
    else:
        binaries.append(arg_binary)

    if arg_clean_artifacts:
        clean_artifacts()

    setup_artifacts(binaries)

    # constraint checking
    for binary in binaries:
        check_file_exists(binary)
        fbb = ttrt.binary.load_binary_from_path(binary)
        check_version(fbb.version)
        assert fbb.file_identifier == "TTNN", "Only TTNN binaries are supported"
        fbb_dict = ttrt.binary.as_dict(fbb)
        fbb_list.append((os.path.splitext(os.path.basename(binary))[0], fbb, fbb_dict))
        program_index = int(arg_program_index)
        assert program_index <= len(
            fbb_dict["programs"]
        ), "args.program_index out of range"

    # execution
    for (binary_name, fbb, fbb_dict) in fbb_list:
        torch_inputs[binary_name] = []
        torch_outputs[binary_name] = []
        program = fbb_dict["programs"][program_index]
        print(f"running program[{program_index}]:", program["name"])

        for i in program["inputs"]:
            torch_tensor = torch.randn(
                i["desc"]["shape"],
                dtype=fromDataType(i["desc"]["layout"]["memory_desc"]["data_type"]),
            )
            torch_inputs[binary_name].append(torch_tensor)
        for i in program["outputs"]:
            torch_tensor = torch.zeros(
                i["desc"]["shape"],
                dtype=fromDataType(i["desc"]["layout"]["memory_desc"]["data_type"]),
            )
            torch_outputs[binary_name].append(torch_tensor)

        print("inputs:\n", torch_inputs)

        total_inputs = []
        total_outputs = []
        for loop in range(arg_loops):
            inputs = []
            outputs = []
            for i in torch_inputs[binary_name]:
                inputs.append(
                    ttrt.runtime.create_tensor(
                        i.data_ptr(),
                        list(i.shape),
                        list(i.stride()),
                        i.element_size(),
                        toDataType(i.dtype),
                    )
                )

            for i in torch_outputs[binary_name]:
                outputs.append(
                    ttrt.runtime.create_tensor(
                        i.data_ptr(),
                        list(i.shape),
                        list(i.stride()),
                        i.element_size(),
                        toDataType(i.dtype),
                    )
                )

            total_inputs.append(inputs)
            total_outputs.append(outputs)

        system_desc, device_ids = ttrt.runtime.get_current_system_desc()
        device = ttrt.runtime.open_device(device_ids)
        atexit.register(lambda: ttrt.runtime.close_device(device))
        for loop in range(arg_loops):
            ttrt.runtime.submit(device, fbb, 0, total_inputs[loop], total_outputs[loop])
            print(f"finished loop={loop}")
        print("outputs:\n", torch_outputs)

    # save artifacts
    for binary in binaries:
        copy_ttnn_binary_into_artifact(binary)
        binary_name = os.path.splitext(os.path.basename(binary))[0]
        torch_input_tensors = torch_inputs[binary_name]
        torch_output_tensors = torch_outputs[binary_name]

        for i, input in enumerate(torch_input_tensors):
            save_torch_tensor_into_ttrt_artifacts(input, f"{binary_name}/input_{i}.pt")

        for i, output in enumerate(torch_output_tensors):
            save_torch_tensor_into_ttrt_artifacts(
                output, f"{binary_name}/output_{i}.pt"
            )

        save_system_desc_into_ttrt_artifacts(
            system_desc, f"{binary_name}/system_desc.ttsys"
        )


"""
API: query
  - query device for system descriptor in the form of a flatbuffer
"""


def query(args):
    import ttrt.runtime

    # initialization
    system_desc = None

    # acquire parameters
    arg_system_desc = args.system_desc
    arg_system_desc_as_json = args.system_desc_as_json
    arg_system_desc_as_dict = args.system_desc_as_dict
    args_save_system_desc = args.save_system_desc
    arg_clean_artifacts = args.clean_artifacts

    # preprocessing
    if arg_clean_artifacts:
        clean_artifacts()

    setup_artifacts()

    # execution
    system_desc = ttrt.runtime.get_current_system_desc()[0]

    if arg_system_desc or arg_system_desc_as_json:
        print(system_desc.as_json())
    if arg_system_desc_as_dict:
        print(system_desc_as_dict(system_desc))

    # save artifacts
    save_system_desc_into_ttrt_artifacts(system_desc, args_save_system_desc)


"""
API: perf
  - run flatbuffer on device in performance mode
"""


def perf(args):
    # initialization
    binaries = []

    # acquire parameters
    arg_binary = args.binary
    arg_perf_csv = args.perf_csv
    arg_clean_artifacts = args.clean_artifacts
    arg_loops = int(args.loops)

    # preprocessing
    if os.path.isdir(arg_binary):
        binaries = find_ttnn_files(arg_binary)
    else:
        binaries.append(arg_binary)

    if arg_clean_artifacts:
        clean_artifacts()

    setup_artifacts(binaries)

    # constraint checking
    if not perf_trace:
        print("Perf mode is not enabled. Please rebuild tt-mlir with perf mode")
        sys.exit(1)

    if args.generate_params:
        check_file_exists(arg_perf_csv)

    for binary in binaries:
        check_file_exists(binary)

    # execution
    if args.generate_params:
        generate_params_dict(arg_perf_csv)
    else:
        for binary in binaries:
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
            testCommand = f"python -m tracy -p {TTMLIR_VENV_DIR}/bin/ttrt run {binary} --loops {arg_loops}"
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
