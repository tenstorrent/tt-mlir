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
        get_current_runtime,
        set_compatible_runtime,
        get_current_system_desc,
        open_device,
        close_device,
        submit,
        create_tensor,
        wait,
        WorkaroundEnv,
        open_so,
        run_so_program,
    )
except ModuleNotFoundError:
    raise ImportError(
        "Error: Project was not built with runtime enabled, rebuild with: -DTTMLIR_ENABLE_RUNTIME=ON"
    )
