# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# -*- Python -*-

import os
import sys
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool


# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "TTMLIR"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# Stablehlo tests can be optionally enabled.
if config.enable_stablehlo:
    config.available_features.add("stablehlo")

# Pykernel tests are optionally enabled.
if config.enable_pykernel:
    config.available_features.add("pykernel")

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.ttmlir_obj_root, "test")

# system_desc_path: The system desc that is to be used to generate the binary files.
config.system_desc_path = os.getenv("SYSTEM_DESC_PATH", "")

# set features based on system
system_desc = None
if config.system_desc_path:
    try:
        import ttrt

        system_desc = ttrt.binary.as_dict(
            ttrt.binary.load_system_desc_from_path(config.system_desc_path)
        )["system_desc"]
        config.available_features.add(system_desc["chip_descs"][0]["arch"])
    except ImportError:
        print(
            "ttrt not found. Please install ttrt to use have system desc driven test requirements set.",
            file=sys.stderr,
        )


# set targets based on the system (default = n150)
"""
available_targets:
- n150
- n300
- llmbox
- tg
"""
config.targets = {"n150"}

if system_desc != None:
    if len(system_desc["chip_desc_indices"]) == 1:
        config.targets = {"n150"}
    elif len(system_desc["chip_desc_indices"]) == 2:
        config.targets = {"n300"}
    elif len(system_desc["chip_desc_indices"]) == 8:
        config.targets = {"llmbox"}
    elif len(system_desc["chip_desc_indices"]) == 32:
        config.targets = {"tg"}

for target in config.targets:
    config.available_features.add(target)

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%system_desc_path%", config.system_desc_path))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs", "Examples", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.ttmlir_obj_root, "test")
config.ttmlir_tools_dir = os.path.join(config.ttmlir_obj_root, "bin")
config.ttmlir_libs_dir = os.path.join(config.ttmlir_obj_root, "lib")

config.substitutions.append(("%ttmlir_libs", config.ttmlir_libs_dir))

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.ttmlir_tools_dir, config.llvm_tools_dir]
tools = ["mlir-opt", "ttmlir-opt", "ttmlir-translate"]

llvm_config.add_tool_substitutions(tools, tool_dirs)

llvm_config.with_environment(
    "PYTHONPATH",
    [
        os.path.join(config.mlir_obj_dir, "python_packages"),
    ],
    append_path=True,
)

# Add `TT_MLIR_HOME` to lit environment.
if "TT_MLIR_HOME" in os.environ:
    llvm_config.with_environment("TT_MLIR_HOME", os.environ["TT_MLIR_HOME"])
else:
    raise OSError("Error: TT_MLIR_HOME not set")

# Add `TT_METAL_HOME` to lit environment.
if "TT_METAL_HOME" in os.environ:
    llvm_config.with_environment("TT_METAL_HOME", os.environ["TT_METAL_HOME"])
else:
    raise OSError("Error: TT_METAL_HOME not set")

# Add `TT_METAL_BUILD_HOME` to lit environment.
if "TT_METAL_BUILD_HOME" in os.environ:
    llvm_config.with_environment(
        "TT_METAL_BUILD_HOME", os.environ["TT_METAL_BUILD_HOME"]
    )
else:
    raise OSError("Error: TT_METAL_BUILD_HOME not set")

# Add `ARCH_NAME` to lit environment.
if "ARCH_NAME" in os.environ:
    llvm_config.with_environment("ARCH_NAME", os.environ["ARCH_NAME"])
else:
    raise OSError("Error: ARCH_NAME not set.")
