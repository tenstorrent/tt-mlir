# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from loguru import logger


@pytest.fixture(scope="function")
def device():
    # Only care about single device, multi-device will use mesh_device fixture
    if ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.P150:
        dispatch_core_type = ttnn.DispatchCoreType.WORKER
    else:
        dispatch_core_type = ttnn.DispatchCoreType.ETH
    d = ttnn.open_device(
        device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig(dispatch_core_type)
    )
    d.disable_and_clear_program_cache()
    yield d
    ttnn.close_device(d)


# Reset fabric config to DISABLED if not None, and do nothing otherwise
# Temporarily require previous state to be passed in as even setting it to DISABLED might be unstable
# This is to ensure that we don't propagate the instability to the rest of CI
def reset_fabric(fabric_config):
    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# Set fabric config to passed in value
# Do nothing if not set
# Must be called before creating the mesh device
def set_fabric(fabric_config, reliability_mode=None, fabric_tensix_config=None):
    # If fabric_config is not None, set it to fabric_config
    if fabric_config:
        if reliability_mode is None:
            reliability_mode = ttnn.FabricReliabilityMode.STRICT_INIT

        # Apply default logic for fabric_tensix_config,
        # fabric_tensix_config is used for enabling tensix extensions for the fabric router,
        # some sender channels in the fabric router are moved to the fabric tensix extension
        # (currently the extension is mux kernel, can have other kernels in future as well).
        if fabric_tensix_config is None:
            fabric_tensix_config = get_default_fabric_tensix_config()

        ttnn.set_fabric_config(
            fabric_config, reliability_mode, None, fabric_tensix_config
        )  # num_planes


def get_default_fabric_tensix_config():
    # Default to MUX for Blackhole when fabric is enabled, DISABLED otherwise
    if ttnn.device.is_blackhole():
        return ttnn.FabricTensixConfig.MUX
    else:
        return ttnn.FabricTensixConfig.DISABLED


def get_updated_device_params(device_params):
    new_device_params = device_params.copy()

    dispatch_core_axis = new_device_params.pop("dispatch_core_axis", None)
    dispatch_core_type = new_device_params.pop("dispatch_core_type", None)
    fabric_tensix_config = new_device_params.get("fabric_tensix_config", None)

    if ttnn.device.is_blackhole():
        # If fabric_tensix_config is not specified but fabric_config is specified on Blackhole,
        # default to MUX mode
        fabric_config = new_device_params.get("fabric_config", None)
        if fabric_config and not fabric_tensix_config:
            fabric_tensix_config = ttnn.FabricTensixConfig.MUX
            dispatch_core_axis = ttnn.DispatchCoreAxis.ROW
            new_device_params["fabric_tensix_config"] = fabric_tensix_config
            logger.warning(
                "Blackhole with fabric enabled, defaulting to fabric_tensix_config=MUX and use DispatchCoreAxis.ROW"
            )
        elif not fabric_config and not fabric_tensix_config:
            if dispatch_core_axis == ttnn.DispatchCoreAxis.ROW:
                logger.warning(
                    "when fabric_tensix_config disabled, blackhole arch does not support DispatchCoreAxis.ROW, using DispatchCoreAxis.COL instead."
                )
                dispatch_core_axis = ttnn.DispatchCoreAxis.COL

    dispatch_core_config = ttnn.DispatchCoreConfig(
        dispatch_core_type, dispatch_core_axis, fabric_tensix_config
    )
    new_device_params["dispatch_core_config"] = dispatch_core_config

    return new_device_params


@pytest.fixture(scope="function")
def mesh_device(request, device_params):
    """
    Pytest fixture to set up a device mesh for tests.

    If `request.param` is an integer, it specifies the number of devices to use (up to available devices).
    If `request.param` is a tuple, it defines the 2D grid dimensions (rows, columns) for TG, e.g., (8, 4) creates
    a device mesh grid of 8 rows and 4 columns, totaling 32 devices. The total number of devices should not exceed available devices.

    Args:
        request: Pytest request object.
        silicon_arch_name: Name of the silicon architecture.
        device_params: Additional device configuration parameters.

    Yields:
        mesh_device: Initialized device mesh object.
    """

    request.node.pci_ids = ttnn.get_pcie_device_ids()

    try:
        param = request.param
    except (ValueError, AttributeError):
        # Get number of devices from the system mesh descriptor.
        param = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size()

    if isinstance(param, tuple):
        grid_dims = param
        assert (
            len(grid_dims) == 2
        ), "Device mesh grid shape should have exactly two elements."
        num_devices_requested = grid_dims[0] * grid_dims[1]
        if (
            not ttnn.using_distributed_env()
            and num_devices_requested > ttnn.get_num_devices()
        ):
            pytest.skip(
                "Requested more devices than available. Test not applicable for machine"
            )
        mesh_shape = ttnn.MeshShape(*grid_dims)
    else:
        if not ttnn.using_distributed_env() and param > ttnn.get_num_devices():
            pytest.skip(
                "Requested more devices than available. Test not applicable for machine"
            )
        mesh_shape = ttnn.MeshShape(1, param)

    updated_device_params = get_updated_device_params(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    fabric_tensix_config = updated_device_params.pop("fabric_tensix_config", None)
    reliability_mode = updated_device_params.pop("reliability_mode", None)
    set_fabric(fabric_config, reliability_mode, fabric_tensix_config)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

    logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created")
    yield mesh_device

    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)

    ttnn.close_mesh_device(mesh_device)
    reset_fabric(fabric_config)
    del mesh_device


@pytest.fixture(scope="function")
def device_params(request):
    return getattr(request, "param", {})


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(0)
