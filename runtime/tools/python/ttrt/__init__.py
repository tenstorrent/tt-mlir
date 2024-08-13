# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

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

import ttrt.binary
from ttrt.common.api import API

#######################################################################################
#######################################**MAIN**########################################
#######################################################################################
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
        return

    # run command
    args.func(args)

if __name__ == "__main__":
    main()
