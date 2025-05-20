# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

try:
    from ._C import (
        Device,
        Event,
        Tensor,
        TensorDesc,
        MemoryBufferType,
        DataType,
        DeviceRuntime,
        DispatchCoreType,
        DebugEnv,
        DebugPerfEnv,
        DebugHooks,
        MeshDeviceOptions,
        get_current_runtime,
        set_current_runtime,
        set_compatible_runtime,
        get_current_system_desc,
        get_num_available_devices,
        open_mesh_device,
        close_mesh_device,
        create_sub_mesh_device,
        release_sub_mesh_device,
        reshape_mesh_device,
        submit,
        create_tensor,
        create_owned_tensor,
        create_empty_tensor,
        create_multi_device_tensor,
        wait,
        to_host,
        to_layout,
        get_layout,
        get_op_output_tensor,
        get_op_debug_str,
        get_debug_info_golden,
        memcpy,
        deallocate_tensor,
        WorkaroundEnv,
        get_op_loc_info,
        unregister_hooks,
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
