# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttrt.binary
import os
import json


def system_desc_as_dict(desc):
    return json.loads(desc.as_json())


if "LOGGER_LEVEL" not in os.environ:
    os.environ["LOGGER_LEVEL"] = "FATAL"
if "TT_METAL_LOGGER_LEVEL" not in os.environ:
    os.environ["TT_METAL_LOGGER_LEVEL"] = "FATAL"


def mlir_sections(fbb):
    d = ttrt.binary.as_dict(fbb)
    for i, program in enumerate(d["programs"]):
        if "debug_info" not in program:
            print("// no debug info found for program:", program["name"])
            continue
        print(
            f"// program[{i}]:",
            program["name"],
            "-",
            program["debug_info"]["mlir"]["name"],
        )
        print(program["debug_info"]["mlir"]["source"], end="")


def cpp_sections(fbb):
    d = ttrt.binary.as_dict(fbb)
    for i, program in enumerate(d["programs"]):
        if "debug_info" not in program:
            print("// no debug info found for program:", program["name"])
            continue
        print(f"// program[{i}]:", program["name"])
        print(program["debug_info"]["cpp"], end="")


def program_inputs(fbb):
    d = ttrt.binary.as_dict(fbb)
    for program in d["programs"]:
        print("program:", program["name"])
        print(json.dumps(program["inputs"], indent=2))


def program_outputs(fbb):
    d = ttrt.binary.as_dict(fbb)
    for program in d["programs"]:
        print("program:", program["name"])
        print(json.dumps(program["outputs"], indent=2))


read_actions = {
    "all": lambda fbb: print(fbb.as_json()),
    "version": lambda fbb: print(
        f"Version: {fbb.version}\ntt-mlir git hash: {fbb.ttmlir_git_hash}"
    ),
    "system-desc": lambda fbb: print(
        json.dumps(ttrt.binary.as_dict(fbb)["system_desc"], indent=2)
    ),
    "mlir": mlir_sections,
    "cpp": cpp_sections,
    "inputs": program_inputs,
    "outputs": program_outputs,
}


def read(args):
    fbb = ttrt.binary.load_from_path(args.binary)
    read_actions[args.section](fbb)


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

    fbb = ttrt.binary.load_from_path(args.binary)
    assert fbb.file_identifier == "TTNN", "Only TTNN binaries are supported"
    d = ttrt.binary.as_dict(fbb)
    assert args.program_index < len(d["programs"]), "args.program_index out of range"
    program = d["programs"][args.program_index]
    print(f"running program[{args.program_index}]:", program["name"])
    torch_inputs = []
    torch_outputs = []
    for i in program["inputs"]:
        torch_inputs.append(torch.randn(i["desc"]["shape"]))
    for i in program["outputs"]:
        torch_outputs.append(torch.zeros(i["desc"]["shape"]))

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

    device = ttrt.runtime.open_device()
    ttrt.runtime.submit(device, fbb, 0, inputs, outputs)
    print("oututs:\n", torch_outputs)
    ttrt.runtime.close_device(device)


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
        ttrt.binary.store(desc, file_name)
        print("system desc saved to:", file_name)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ttrt: a runtime tool for parsing and executing flatbuffer binaries"
    )
    subparsers = parser.add_subparsers(required=True)

    read_parser = subparsers.add_parser(
        "read", help="read information from flatbuffer binary"
    )
    read_parser.add_argument(
        "-s",
        "--section",
        default="all",
        choices=sorted(list(read_actions.keys())),
        help="output sections of the fb",
    )
    read_parser.add_argument("binary", help="flatbuffer binary file")
    read_parser.set_defaults(func=read)

    run_parser = subparsers.add_parser("run", help="run a flatbuffer binary")
    run_parser.add_argument(
        "-p",
        "--program-index",
        default=0,
        help="the program inside the fbb to run",
    )
    run_parser.add_argument("binary", help="flatbuffer binary file")
    run_parser.set_defaults(func=run)

    query_parser = subparsers.add_parser(
        "query", help="query information about the current system"
    )
    query_parser.add_argument(
        "--system-desc",
        action="store_true",
        help="serialize a system desc for the current system to a file",
    )
    query_parser.add_argument(
        "--system-desc-as-json",
        action="store_true",
        help="print the system desc as json",
    )
    query_parser.add_argument(
        "--system-desc-as-dict",
        action="store_true",
        help="print the system desc as python dict",
    )
    query_parser.add_argument(
        "--save-system-desc",
        nargs="?",
        default="",
        help="serialize a system desc for the current system to a file",
    )
    query_parser.set_defaults(func=query)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        return
    args.func(args)
