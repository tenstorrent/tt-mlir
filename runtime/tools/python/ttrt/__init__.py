import ttrt.binary
import os
import json


def system_desc_as_dict(desc):
    return json.loads(desc.as_json())


if "LOGGER_LEVEL" not in os.environ:
    os.environ["LOGGER_LEVEL"] = "FATAL"
if "TT_METAL_LOGGER_LEVEL" not in os.environ:
    os.environ["TT_METAL_LOGGER_LEVEL"] = "FATAL"


def read(args):
    fbb = ttrt.binary.load_from_path(args.binary)
    if args.version:
        print("Version:", fbb.version)
        print("tt-mlir git hash:", fbb.ttmlir_git_hash)
    if args.json:
        print(fbb.as_json())
    if args.dict:
        print(ttrt.binary.as_dict(fbb))
    if args.system_desc:
        print(ttrt.binary.as_dict(fbb)["system_desc"])
    if args.mlir:
        d = ttrt.binary.as_dict(fbb)
        for program in d["programs"]:
            if "debug_info" not in program:
                print("// no debug info found for program:", program["name"])
                continue
            print(
                "// program:",
                program["name"],
                "-",
                program["debug_info"]["mlir"]["name"],
            )
            print(program["debug_info"]["mlir"]["source"], end="")
    if args.cpp:
        d = ttrt.binary.as_dict(fbb)
        for program in d["programs"]:
            if "debug_info" not in program:
                print("// no debug info found for program:", program["name"])
                continue
            print("// program:", program["name"])
            print(program["debug_info"]["cpp"], end="")


def run(args):
    import ttrt.runtime

    raise NotImplementedError("run not implemented")


def query(args):
    import ttrt.runtime

    if args.system_desc or args.system_desc_as_json:
        print(ttrt.runtime.get_current_system_desc().as_json())
    if args.system_desc_as_dict:
        print(system_desc_as_dict(ttrt.runtime.get_current_system_desc()))
    if args.save_system_desc:
        desc = ttrt.runtime.get_current_system_desc()
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
    read_parser.add_argument("--json", action="store_true", help="output as json")
    read_parser.add_argument(
        "--dict", action="store_true", help="output as python dict"
    )
    read_parser.add_argument(
        "--version", action="store_true", help="output version information"
    )
    read_parser.add_argument(
        "--system-desc", action="store_true", help="output embedded system desc"
    )
    read_parser.add_argument(
        "--mlir", action="store_true", help="extract mlir from binary (if available)"
    )
    read_parser.add_argument(
        "--cpp", action="store_true", help="extract cpp from binary (if available)"
    )
    read_parser.add_argument("binary", help="flatbuffer binary file")
    read_parser.set_defaults(func=read)

    run_parser = subparsers.add_parser("run", help="run a flatbuffer binary")
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
