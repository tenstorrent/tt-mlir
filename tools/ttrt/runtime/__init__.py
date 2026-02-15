# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

try:
    from ttrt.runtime._ttmlir_runtime.runtime import (
        Device,
        Event,
        Tensor,
        TensorRef,
        TensorDesc,
        MemoryBufferType,
        CallbackContext,
        OpContext,
        DataType,
        DeviceRuntime,
        HostRuntime,
        DispatchCoreType,
        MemoryLogLevel,
        DebugEnv,
        PerfEnv,
        DebugHooks,
        DebugStats,
        MeshDeviceOptions,
        MultiProcessArgs,
        DistributedOptions,
        DistributedMode,
        set_mlir_home,
        set_metal_home,
        set_memory_log_level,
        get_current_device_runtime,
        set_current_device_runtime,
        set_compatible_device_runtime,
        get_current_host_runtime,
        set_current_host_runtime,
        get_available_host_runtimes,
        get_current_system_desc,
        launch_distributed_runtime,
        shutdown_distributed_runtime,
        get_num_available_devices,
        open_mesh_device,
        close_mesh_device,
        create_sub_mesh_device,
        release_sub_mesh_device,
        reshape_mesh_device,
        submit,
        create_borrowed_host_tensor,
        create_owned_host_tensor,
        create_empty_tensor,
        create_multi_device_host_tensor,
        create_multi_device_borrowed_host_tensor,
        set_fabric_config,
        wait,
        to_host,
        get_device_tensors,
        to_layout,
        get_layout,
        get_op_output_tensor,
        get_op_output_ref,
        get_op_input_refs,
        retrieve_tensor_from_pool,
        update_tensor_in_pool,
        get_op_debug_str,
        memcpy,
        deallocate_tensor,
        WorkaroundEnv,
        get_op_loc_info,
        unregister_hooks,
        FabricConfig,
    )
except ModuleNotFoundError:
    raise ImportError(
        "Error: Project was not built with runtime enabled, rebuild with: -DTTMLIR_ENABLE_RUNTIME=ON"
    )

try:
    from ttrt.runtime._ttmlir_runtime.runtime import test
except ImportError:
    print(
        "Warning: not importing testing submodule since project was not built with runtime testing enabled. To enable, rebuild with: -DTTMLIR_ENABLE_RUNTIME_TESTS=ON"
    )

def bind_callbacks():
    from ttrt.runtime import (
        get_op_input_refs,
        get_op_output_ref,
        get_op_debug_str,
        get_op_loc_info,
        retrieve_tensor_from_pool,
        CallbackContext,
        OpContext,
        DebugHooks,
    )
    from ttrt.binary import Binary

    def pre_op_callback(binary: Binary, program_ctx: CallbackContext, op_ctx: OpContext):
        """Called before each TTNN operation.

        Args:
            binary: Binary object
            program_ctx: CallbackContext reference for tensor pool access
            op_ctx: OpContext reference with operation metadata
        """
        # Get operation info
        debug_str = get_op_debug_str(op_ctx)
        loc_info = get_op_loc_info(op_ctx)

        print(f"✓ PRE: {debug_str}")
        print(f"  Location: {loc_info}")

        # Access input tensors
        input_refs = get_op_input_refs(op_ctx, program_ctx)
        print(f"  Inputs: {len(input_refs)} tensors")

    def post_op_callback(binary: Binary, program_ctx: CallbackContext, op_ctx: OpContext):
        """Called after each TTNN operation.

        Args:
            binary: Binary object
            program_ctx: CallbackContext reference
            op_ctx: OpContext reference
        """
        debug_str = get_op_debug_str(op_ctx)

        # Get output tensor info
        output_ref = get_op_output_ref(op_ctx, program_ctx)

        print(f"✓ POST: {debug_str}")
        print(f"  Has output: {output_ref is not None}")
    
    DebugHooks.get(pre_op_callback, post_op_callback)


import atexit

def cleanup():
    unregister_hooks()

atexit.register(cleanup)