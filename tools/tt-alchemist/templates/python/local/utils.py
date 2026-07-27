# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import math


class DeviceGetter:
    _instance = None
    _mesh_shape = None
    _fabric_config = None
    l1_small_size = 1 << 16  # 64kB

    def __init__(self):
        raise RuntimeError("This is Singleton, invoke get_device() instead.")

    def __del__(self):
        if self._instance is not None:
            ttnn.close_mesh_device(self._instance)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    @classmethod
    def get_device(cls, mesh_shape, fabric_config=None):
        if cls._instance is None:
            if (
                not isinstance(mesh_shape, (list, tuple))
                or len(mesh_shape) == 0
                or not all(isinstance(x, int) and x > 0 for x in mesh_shape)
            ):
                raise ValueError(
                    f"mesh_shape must be a non-empty list or tuple of positive integers, got {mesh_shape}"
                )
            cls._mesh_shape = mesh_shape

            # If the caller doesn't specify a fabric config, fallback to
            # FABRIC_1D for multi-device meshes.
            if fabric_config is None:
                fabric_config = (
                    ttnn.FabricConfig.FABRIC_1D
                    if math.prod(mesh_shape) >= 2
                    else ttnn.FabricConfig.DISABLED
                )
            cls._fabric_config = fabric_config

            ttnn.set_fabric_config(fabric_config)
            cls._instance = ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(mesh_shape),
                l1_small_size=cls.l1_small_size,
            )
            print(f"Device: {cls._instance}")

        # Compare requested mesh_shape with _mesh_shape used to initialize the device
        if tuple(cls._mesh_shape) != tuple(mesh_shape):
            raise ValueError(
                f"Device already initialized with mesh_shape={cls._mesh_shape}, but got mesh_shape={mesh_shape}"
            )

        # Same for fabric_config: if the caller explicitly requests one, it
        # must match the config the singleton was initialized with.
        if fabric_config is not None and fabric_config != cls._fabric_config:
            raise ValueError(
                f"Device already initialized with fabric_config={cls._fabric_config}, "
                f"but got fabric_config={fabric_config}"
            )

        return cls._instance


def get_scalar_from_tensor(tensor: ttnn.Tensor) -> int:
    assert tensor.logical_volume() == 1, "expected scalar tensor"
    assert tensor.dtype == ttnn.DataType.UINT32, "expected uint32 tensor"

    host_tensor = ttnn.from_device(tensor)
    return host_tensor.item()


def load_tensor(file_path: str, layout, dtype, device, memory_config) -> ttnn.Tensor:
    loaded_tensor = ttnn.load_tensor(file_path)

    assert loaded_tensor.device() is None, "loaded tensor must be on host"

    if loaded_tensor.layout != layout:
        loaded_tensor = ttnn.to_layout(loaded_tensor, layout)
    if loaded_tensor.dtype != dtype:
        loaded_tensor = ttnn.to_dtype(loaded_tensor, dtype)
    if device is not None:
        loaded_tensor = ttnn.to_device(loaded_tensor, device, memory_config)

    return loaded_tensor


# Heavy-lifting helper for CPU-hoisted functions. Mirrors the runtime logic in
# runtime/lib/ttnn/operations/cpu/cpu.cpp (runSingleChip / runMultiChip):
# CPU-hoisted segments are barrier-free local compute, so each device's shard is
# computed independently on the host and the per-shard results are reassembled
# into a multi-device tensor.
def execute_cpu_hoisted_function(inputs, function):
    """Run a pure-torch CPU-hoisted body shard-by-shard over a mesh.

    inputs:   list of ttnn.Tensor operands (device-resident, possibly sharded).
    function: pure-torch callable mapping torch tensors -> torch tensor(s).
    Returns a single ttnn.Tensor, or a tuple of them for multi-output bodies.
    """

    def _wrap_outputs(result):
        return result if isinstance(result, (list, tuple)) else (result,)

    # The mesh device comes from the program context (the DeviceGetter
    # singleton), not from the inputs: CPU-hoisted inputs have already been
    # brought to host by the device program, so their .device() is None. This
    # mirrors how the runtime obtains the mesh device from the ProgramContext.
    mesh_device = DeviceGetter._instance

    # No mesh context: run the body once on the host and return host tensor(s).
    if mesh_device is None:
        torch_inputs = [ttnn.to_torch(tensor) for tensor in inputs]
        outputs = _wrap_outputs(function(*torch_inputs))
        host_outputs = [ttnn.from_torch(out) for out in outputs]
        return host_outputs[0] if len(host_outputs) == 1 else tuple(host_outputs)

    mesh_shape = mesh_device.shape
    num_shards = mesh_device.get_num_devices()

    # Split each input into per-device torch shards. get_device_tensors returns
    # one shard per device for a sharded tensor, or a single shard for an
    # unsharded (replicated) tensor, which is then reused across devices.
    input_shards = []
    for tensor in inputs:
        if tensor.device() is not None:
            tensor = ttnn.from_device(tensor)
        shards = ttnn.get_device_tensors(tensor)
        input_shards.append([ttnn.to_torch(shard) for shard in shards])

    # Run the body once per device shard.
    output_shards = []
    for shard_idx in range(num_shards):
        args = [
            shards[shard_idx] if len(shards) > 1 else shards[0]
            for shards in input_shards
        ]
        output_shards.append(_wrap_outputs(function(*args)))

    # Reassemble each output across shards into a multi-device host tensor.
    num_outputs = len(output_shards[0])
    results = []
    for out_idx in range(num_outputs):
        torch_shards = [output_shards[s][out_idx] for s in range(num_shards)]
        ttnn_shards = [ttnn.from_torch(shard) for shard in torch_shards]
        results.append(ttnn.from_host_shards(ttnn_shards, mesh_shape))
    return results[0] if num_outputs == 1 else tuple(results)


# Helpers for distributed RMS norm EmitPy support.
# These mirror the runtime logic in
# runtime/lib/ttnn/operations/normalization/distributed_rms_norm.cpp
# TODO(jserbedzija): Remove this once the following issue if fixed in tt-metal: https://github.com/tenstorrent/tt-metal/issues/37746
def create_global_semaphore(input_tensor):
    """Create a global semaphore from the input tensor's device and shard grid."""
    mesh_device = input_tensor.device()
    shard_spec = input_tensor.memory_config().shard_spec
    return ttnn.create_global_semaphore(mesh_device, shard_spec.grid, 0)
