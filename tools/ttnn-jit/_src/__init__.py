# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN-JIT internal utilities and implementation."""

from ttnn_jit._ttmlir_runtime.runtime import (
    submit,
    set_compatible_device_runtime,
    get_current_device_runtime,
    get_current_system_desc,
    DispatchCoreType,
)
from ttnn_jit._ttmlir_runtime.binary import (
    load_binary_from_path,
    load_binary_from_capsule,
)
from ttnn_jit._ttmlir_runtime.utils import (
    create_runtime_device_from_ttnn,
    create_runtime_tensor_from_ttnn,
    get_ttnn_tensor_from_runtime_tensor,
)
from ttnn_jit._ttnn_jit import JitCache
