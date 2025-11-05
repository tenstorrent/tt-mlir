# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# NOTE: it is _VERY_ important that this import & setup call is _BEFORE_ any
# other `ttrt` imports and _AFTER_ all system imports to ensure a well ordered
# setup of the pybound `.so`. Otherwise, undefined behaviour ensues related to
# the timing of when `TTMETAL_HOME` environment variable is set. DO NOT MOVE
# w.r.t. other imports. This is a temporary workaround until `TT_METAL_RUNTIME_ROOT` is
# not used anymore in TTMetal
import ttrt.library_tweaks

ttrt.library_tweaks.set_tt_metal_home()

import ttrt.binary
import sys
import os
from ttrt.common.api import API
from importlib.metadata import version


def relaunch_as_gdb(args):
    args = [arg for arg in args if arg != "--gdb"]
    os.execvp("gdb", ["gdb", "-ex", "run", "--args", sys.executable] + args)


def main():
    import argparse

    if "--gdb" in sys.argv:
        relaunch_as_gdb(sys.argv)

    parser = argparse.ArgumentParser(
        description="ttrt: a runtime tool for parsing and executing flatbuffer binaries"
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"ttrt {version('ttrt')}"
    )
    parser.add_argument("--gdb", action="store_true", help="launch ttrt with gdb")
    subparsers = parser.add_subparsers(required=True)

    API.initialize_apis()
    for api_name, api_class in API.registered_apis.items():
        api_class.generate_subparser(subparsers)

    try:
        args = parser.parse_args()
    except SystemExit:
        return 1

    request_api = args.api(args)
    result_code, results = request_api()

    return result_code


if __name__ == "__main__":
    main()
