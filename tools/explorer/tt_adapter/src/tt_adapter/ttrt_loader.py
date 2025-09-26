# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import logging
import importlib

# TTRT module reference. Dynamically imported to enable or disable module execution.
ttrt = None

def get_is_ttrt_available():
  is_ttrt_available = importlib.util.find_spec("ttrt") is not None
  print(f"TTRT status: {is_ttrt_available}")

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
