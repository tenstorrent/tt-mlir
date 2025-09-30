# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import logging
import importlib

# TTRT module reference. Dynamically imported to enable or disable module execution.
ttrt = None

is_execution_enabled = True


def get_is_ttrt_available():
    if not is_execution_enabled:
        return False

    # Checks for a specific submodule,
    # so it doesn't error out when the folder for `ttrt` exists but is empty
    is_ttrt_available = importlib.util.find_spec("ttrt.API") is not None

    return is_ttrt_available


def load_ttrt():
    global ttrt

    if not get_is_ttrt_available():
        return None

    if ttrt is None:
        try:
            # Attempt to import the module dynamically
            ttrt = importlib.import_module("ttrt")
        except (ModuleNotFoundError, ImportError):
            logging.info("TTRT not available. Models will not be compiled.")

    return ttrt


def set_is_execution_enabled(is_enabled: bool):
    global is_execution_enabled

    is_execution_enabled = is_enabled
