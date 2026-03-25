# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys


class Perf:
    registered_args = {}

    @staticmethod
    def initialize_api():
        pass

    def __init__(self, args={}, logger=None, artifacts=None):
        pass

    def preprocess(self):
        pass

    def check_constraints(self):
        pass

    def execute(self):
        pass

    def postprocess(self):
        pass

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __call__(self):
        deprecation_message = (
            "\n"
            "DeprecationWarning: `ttrt perf` is deprecated.\n"
            "Use the following command instead:\n"
            "\n"
            "  python -m tracy -r -v --output-folder prof -m ttrt run ...\n"
            "\n"
            "Note: the --output-folder flag is required.\n"
        )
        print(deprecation_message, file=sys.stderr)
        return 1, []

    @staticmethod
    def register_arg(name, type, default, choices, help):
        Perf.registered_args[name] = {
            "type": type,
            "default": default,
            "choices": choices,
            "help": help,
        }

    @staticmethod
    def generate_subparser(subparsers):
        perf_parser = subparsers.add_parser(
            "perf",
            help="[deprecated] use `python -m tracy -r -v --output-folder prof -m ttrt run ...` instead",
        )
        perf_parser.set_defaults(api=Perf)
        return perf_parser
