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
from ttrt.common.api import read, run, query, perf
from ttrt.common.util import read_actions

#######################################################################################
#######################################**MAIN**########################################
#######################################################################################
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ttrt: a runtime tool for parsing and executing flatbuffer binaries"
    )
    subparsers = parser.add_subparsers(required=True)

    """
    API: read
    """
    read_parser = subparsers.add_parser(
        "read", help="read information from flatbuffer binary"
    )
    read_parser.add_argument(
        "--section",
        default="all",
        choices=sorted(list(read_actions.keys())),
        help="output sections of the fb",
    )
    read_parser.add_argument(
        "--clean-artifacts",
        action="store_true",
        help="clean all artifacts from previous runs",
    )
    read_parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="save all artifacts during run",
    )
    read_parser.add_argument("binary", help="flatbuffer binary file")
    read_parser.set_defaults(func=read)

    """
    API: run
    """
    run_parser = subparsers.add_parser("run", help="run a flatbuffer binary")
    run_parser.add_argument(
        "--program-index",
        default="all",
        help="the program inside the fbb to run",
    )
    run_parser.add_argument(
        "--clean-artifacts",
        action="store_true",
        help="clean all artifacts from previous runs",
    )
    run_parser.add_argument(
        "--loops",
        default=1,
        help="number of loops",
    )
    run_parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="save all artifacts during run",
    )
    run_parser.add_argument(
        "--seed",
        default=0,
        help="Seed for random number generator",
    )
    run_parser.add_argument("binary", help="flatbuffer binary file")
    run_parser.set_defaults(func=run)

    """
    API: query
    """
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
        "--clean-artifacts",
        action="store_true",
        help="clean all artifacts from previous runs",
    )
    query_parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="save all artifacts during run",
    )
    query_parser.set_defaults(func=query)

    """
    API: perf
    """
    perf_parser = subparsers.add_parser(
        "perf", help="run performance trace and collect performance data"
    )
    perf_parser.add_argument(
        "--program-index",
        default="all",
        help="the program inside the fbb to run",
    )
    perf_parser.add_argument(
        "--device",
        action="store_true",
        help="collect performance trace on both host and device",
    )
    perf_parser.add_argument(
        "--generate-params",
        action="store_true",
        help="generate json file of model parameters based off of perf csv file",
    )
    perf_parser.add_argument(
        "--perf-csv",
        default="",
        help="perf csv file generated from performance run",
    )
    perf_parser.add_argument(
        "--clean-artifacts",
        action="store_true",
        help="clean all artifacts from previous runs",
    )
    perf_parser.add_argument(
        "--loops",
        default=1,
        help="number of loops",
    )
    perf_parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="save all artifacts during run",
    )
    perf_parser.add_argument("binary", help="flatbuffer binary file")
    perf_parser.set_defaults(func=perf)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        return

    # run command
    args.func(args)


if __name__ == "__main__":
    main()
