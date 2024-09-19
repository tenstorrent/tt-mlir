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
import shutil
import atexit

from ttrt.common.util import *
from ttrt.common.read import Read
from ttrt.common.query import Query
from ttrt.common.check import Check
from ttrt.common.run import Run
from ttrt.common.perf import Perf


class API:
    registered_apis = {}
    registered_functions = [
        "initialize_api",
        "__init__",
        "preprocess",
        "check_constraints",
        "execute",
        "postprocess",
        "__getitem__",
        "__setitem__",
        "__call__",
        "register_arg",
        "generate_subparser",
    ]

    @staticmethod
    def initialize_apis():
        API.register_api(Query)
        API.register_api(Read)
        API.register_api(Run)
        API.register_api(Perf)
        API.register_api(Check)

        API.Query = Query
        API.Read = Read
        API.Run = Run
        API.Perf = Perf
        API.Check = Check

    @staticmethod
    def register_api(api_class):
        missing_methods = [
            func for func in API.registered_functions if not hasattr(api_class, func)
        ]

        if missing_methods:
            raise TypeError(
                f"API class={api_class.__name__} is missing methods={missing_methods}"
            )

        api_class.initialize_api()
        API.registered_apis[api_class.__name__] = api_class
