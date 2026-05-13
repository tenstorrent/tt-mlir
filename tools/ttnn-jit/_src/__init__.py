# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN-JIT internal utilities and implementation."""

from ttnn_jit._ttmlir_runtime.runtime import (
    submit,
    set_compatible_device_runtime,
    get_current_device_runtime,
    get_current_system_desc,
)
from ttnn_jit._ttmlir_runtime.binary import (
    load_binary_from_path,
    load_binary_from_capsule,
)
from ttnn_jit._ttmlir_runtime.utils import (
    create_runtime_device_from_ttnn,
    create_runtime_tensor_from_ttnn,
    get_ttnn_tensor_from_runtime_tensor,
    allocate_l1_buffer,
    allocate_dram_buffer,
    MeshBuffer,
    get_l1_base_allocator_addr,
    get_lowest_occupied_compute_l1_address,
    get_l1_size_per_core,
)
from ttnn_jit._ttnn_jit import JitCache
