# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN-JIT internal utilities and implementation."""

import importlib.util

# Dev layout (python_packages/ on PYTHONPATH): the top-level _ttmlir_runtime
# extension is already loaded by the rest of the stack; importing the bundled
# copy under the alias ttnn_jit._ttmlir_runtime re-runs its nanobind module
# init in the same domain and aborts ("refusing to add duplicate key").
# Only fall back to the bundled copy for the wheel layout, which carries its
# own runtime and has no top-level one.
if importlib.util.find_spec("_ttmlir_runtime") is not None:
    _runtime_mod = "_ttmlir_runtime"
else:
    _runtime_mod = "ttnn_jit._ttmlir_runtime"

from importlib import import_module

runtime = import_module(f"{_runtime_mod}.runtime")
binary = import_module(f"{_runtime_mod}.binary")
utils = import_module(f"{_runtime_mod}.utils")

submit = runtime.submit
set_compatible_device_runtime = runtime.set_compatible_device_runtime
get_current_device_runtime = runtime.get_current_device_runtime
get_current_system_desc = runtime.get_current_system_desc
load_binary_from_path = binary.load_binary_from_path
load_binary_from_capsule = binary.load_binary_from_capsule
create_runtime_device_from_ttnn = utils.create_runtime_device_from_ttnn
create_runtime_tensor_from_ttnn = utils.create_runtime_tensor_from_ttnn
get_ttnn_tensor_from_runtime_tensor = utils.get_ttnn_tensor_from_runtime_tensor
allocate_l1_buffer = utils.allocate_l1_buffer
allocate_dram_buffer = utils.allocate_dram_buffer
MeshBuffer = utils.MeshBuffer
get_l1_base_allocator_addr = utils.get_l1_base_allocator_addr
get_lowest_occupied_compute_l1_address = utils.get_lowest_occupied_compute_l1_address
get_l1_size_per_core = utils.get_l1_size_per_core
from ttnn_jit._ttnn_jit import JitCache
