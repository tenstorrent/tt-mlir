# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

try:
    from ._C import (
        Device,
        Event,
        Tensor,
        DataType,
        DeviceRuntime,
        DebugEnv,
        DebugHooks,
        get_current_runtime,
        set_compatible_runtime,
        get_current_system_desc,
        open_device,
        close_device,
        submit,
        create_tensor,
        create_multi_device_tensor,
        wait,
        get_op_output_tensor,
        get_op_debug_str,
        WorkaroundEnv,
        get_op_loc_info,
    )
except ModuleNotFoundError:
    raise ImportError(
        "Error: Project was not built with runtime enabled, rebuild with: -DTTMLIR_ENABLE_RUNTIME=ON"
    )
