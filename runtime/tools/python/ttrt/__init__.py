# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import importlib.machinery
import importlib.util
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


def get_ttrt_metal_home_path():
    package_name = "ttrt"
    spec = importlib.util.find_spec(package_name)
    package_path = os.path.dirname(spec.origin)
    tt_metal_home = f"{package_path}/runtime"
    return tt_metal_home


os.environ["TT_METAL_HOME"] = get_ttrt_metal_home_path()

new_linker_path = f"{get_ttrt_metal_home_path()}/tests"
current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
if current_ld_library_path:
    updated_ld_library_path = f"{new_linker_path}:{current_ld_library_path}"
else:
    updated_ld_library_path = new_linker_path
os.environ["LD_LIBRARY_PATH"] = updated_ld_library_path

import ttrt.binary
from ttrt.common.api import API


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ttrt: a runtime tool for parsing and executing flatbuffer binaries"
    )
    subparsers = parser.add_subparsers(required=True)

    API.initialize_apis()
    for api_name, api_class in API.registered_apis.items():
        api_class.generate_subparser(subparsers)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        return 1

    request_api = args.api(args)
    result_code, results = request_api()

    return result_code


if __name__ == "__main__":
    main()
