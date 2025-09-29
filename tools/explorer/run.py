#!/usr/bin/env python
#
# # SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import model_explorer
import argparse
from tt_adapter import ttrt_loader

parser = argparse.ArgumentParser(prog="tt-explorer")
parser.add_argument(
    "-p",
    "--port",
    help="Port that model-explorer server will be exposed to",
    type=int,
    default=8080,
)
parser.add_argument(
    "-u", "--url", help="Host URL Address for server", default="localhost"
)
parser.add_argument(
    "-q",
    "--no-browser",
    help="Create server without opening browser tab",
    action="store_true",
)
parser.add_argument(
    "-x",
    "--no-model-execution",
    help="Disable execution of models from the UI",
    action="store_true",
)
parser.add_argument(
    "-c", "--cors_host", help="The Host to allow for CORS requests", default=None
)

args = parser.parse_args()

ttrt_loader.set_is_execution_enabled(not args.no_model_execution)

# TODO(odjuricic): Hack to make our extension default for .mlir files.
# This can be handled better when we switch to our model-explorer fork.
model_explorer.extension_manager.ExtensionManager.BUILTIN_ADAPTER_MODULES = []
model_explorer.visualize_from_config(
    extensions=["tt_adapter"],
    no_open_in_browser=args.no_browser,
    port=args.port,
    host=args.url,
    enable_execution=ttrt_loader.get_is_ttrt_available() and not args.no_model_execution,
    cors_host=args.cors_host,
)
