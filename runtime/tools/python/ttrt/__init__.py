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
from ttrt.common.api import version, read, run, query, perf
from ttrt.common.util import read_actions
from ttrt.remote.api import remote_set, remote_get, remote_run, remote_query, remote_perf, remote_download, remote_upload, remote_create

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
    API: version
    """
    version_parser = subparsers.add_parser(
        "version", help="get version of ttrt"
    )
    version_parser.set_defaults(func=version)

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
        default=0,
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
        default=0,
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

    """
    API: remote
    """
    remote_parser = subparsers.add_parser(
        "remote", help="run ttrt commands on a remote machine"
    )
    remote_subparsers = remote_parser.add_subparsers()

    """
    API: remote set
    """
    remote_upload_parser = remote_subparsers.add_parser(
        "set", help="set ttrt remote configs"
    )

    remote_upload_parser.add_argument("config", help="config file for ttrt")
    remote_upload_parser.set_defaults(func=remote_set)

    """
    API: remote get
    """
    remote_upload_parser = remote_subparsers.add_parser(
        "get", help="get ttrt remote configs"
    )
    remote_upload_parser.set_defaults(func=remote_get)

    """
    API: remote run
    """
    remote_run_parser = remote_subparsers.add_parser("run", help="flatbuffer binary file")
    remote_run_parser.add_argument(
        "--program-index",
        default=0,
        help="the program inside the fbb to run",
    )
    remote_run_parser.add_argument(
        "--clean-artifacts",
        action="store_true",
        help="clean all artifacts from previous runs",
    )
    remote_run_parser.add_argument(
        "--loops",
        default=1,
        help="number of loops",
    )
    remote_run_parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="save all artifacts during run",
    )
    remote_run_parser.add_argument("binary", help="flatbuffer binary file")
    remote_run_parser.set_defaults(func=remote_run)

    """
    API: remote query
    """
    remote_query_parser = remote_subparsers.add_parser(
        "query", help="query information about the current system"
    )
    remote_query_parser.add_argument(
        "--system-desc",
        action="store_true",
        help="serialize a system desc for the current system to a file",
    )
    remote_query_parser.add_argument(
        "--system-desc-as-json",
        action="store_true",
        help="print the system desc as json",
    )
    remote_query_parser.add_argument(
        "--system-desc-as-dict",
        action="store_true",
        help="print the system desc as python dict",
    )
    remote_query_parser.add_argument(
        "--clean-artifacts",
        action="store_true",
        help="clean all artifacts from previous runs",
    )
    remote_query_parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="save all artifacts during run",
    )
    remote_query_parser.set_defaults(func=remote_query)

    """
    API: remote perf
    """
    remote_perf_parser = remote_subparsers.add_parser(
        "perf", help="run performance trace and collect performance data"
    )
    remote_perf_parser.add_argument(
        "--program-index",
        default=0,
        help="the program inside the fbb to run",
    )
    remote_perf_parser.add_argument(
        "--device",
        action="store_true",
        help="collect performance trace on both host and device",
    )
    remote_perf_parser.add_argument(
        "--generate-params",
        action="store_true",
        help="generate json file of model parameters based off of perf csv file",
    )
    remote_perf_parser.add_argument(
        "--perf-csv",
        default="",
        help="perf csv file generated from performance run",
    )
    remote_perf_parser.add_argument(
        "--clean-artifacts",
        action="store_true",
        help="clean all artifacts from previous runs",
    )
    remote_perf_parser.add_argument(
        "--loops",
        default=1,
        help="number of loops",
    )
    remote_perf_parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="save all artifacts during run",
    )
    remote_perf_parser.add_argument("binary", help="flatbuffer binary file")
    remote_perf_parser.set_defaults(func=remote_perf)

    """
    API: remote download
    """
    remote_download_parser = remote_subparsers.add_parser(
        "download", help="download flatbuffer"
    )
    remote_download_parser.add_argument(
        "--issue",
        help="download flatbuffer file from issue",
    )
    remote_download_parser.add_argument(
        "url",
        help="the url of the flatbuffer to download",
    )
    remote_download_parser.add_argument(
        "-n",
        help="name of downloaded flatbuffer file",
    )
    remote_download_parser.set_defaults(func=remote_download)

    """
    API: remote upload
    """
    remote_upload_parser = remote_subparsers.add_parser(
        "upload", help="upload flatbuffer"
    )
    remote_upload_parser.add_argument(
        "--issue",
        help="upload flatbuffer file to issue",
    )
    remote_upload_parser.add_argument("binary", help="flatbuffer binary file to upload")
    remote_upload_parser.set_defaults(func=remote_upload)

    """
    API: remote create
    """
    remote_create_parser = remote_subparsers.add_parser(
        "create", help="create flatbuffer related issue with file"
    )
    remote_create_parser.add_argument(
        "--issue",
        help="name of issue",
    )
    remote_create_parser.add_argument("binary", help="flatbuffer binary file")
    remote_create_parser.set_defaults(func=remote_create)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        return

    # run command
    args.func(args)


if __name__ == "__main__":
    main()
