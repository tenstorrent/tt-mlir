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
        set_current_runtime,
        set_compatible_runtime,
        get_current_system_desc,
        open_device,
        close_device,
        submit,
        create_tensor,
        create_empty_tensor,
        create_multi_device_tensor,
        wait,
        to_host,
        to_layout,
        get_layout,
        get_op_output_tensor,
        get_op_debug_str,
        memcpy,
        deallocate_tensor,
        open_so,
        run_so_program,
        compare_outs,
        WorkaroundEnv,
    )
except ModuleNotFoundError:
    raise ImportError(
        "Error: Project was not built with runtime enabled, rebuild with: -DTTMLIR_ENABLE_RUNTIME=ON"
    )

try:
    from ._C import testing
except ImportError:
    print(
        "Warning: not importing testing submodule since project was not built with runtime testing enabled. To enable, rebuild with: -DTTMLIR_ENABLE_RUNTIME_TESTS=ON"
    )
