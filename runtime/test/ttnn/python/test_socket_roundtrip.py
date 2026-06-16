# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Pipeline-parallel basic: Python-binding smoke for the device-to-device socket
# API. Carves two 1x1 submeshes from a [2,2] mesh over FABRIC_2D and round-trips
# a tensor A -> B over a fabric socket.
# See plans/pipeline-parallel-basic/implementation-plan.md (Phase 1, Task 1.6).
#
# Runs on the single-host Blackhole [2,2] quietbox. Requires:
#   ttrt query --save-artifacts
#   export SYSTEM_DESC_PATH=$(pwd)/ttrt-artifacts/system_desc.ttsys

import pytest
import torch
import ttrt
import ttrt.runtime

from .utils import get_runtime_tensor_from_torch, Storage


def _device_tile_layout(device, runtime_dtype):
    # NOTE (on-box wiring): obtaining a standalone DRAM/tile device Layout without
    # a compiled flatbuffer is the one piece to finalize here. Options:
    #   (a) bind a small "dram interleaved tile layout" helper to python (the C++
    #       gtest uses tt::runtime::test::ttnn::getDramInterleavedTileLayout), or
    #   (b) reuse a layout from a tiny compiled binary via get_to_layout_inputs.
    # The C++ gtest (test_mesh_socket.cpp) already proves the full round-trip; this
    # python test validates the bindings end-to-end once the layout helper exists.
    raise NotImplementedError("wire device tile-layout helper on-box (see note)")


def test_socket_roundtrip():
    num_devices = ttrt.runtime.get_num_available_devices()
    assert num_devices >= 2, f"need >=2 devices, got {num_devices}"

    ttrt.runtime.set_current_device_runtime(ttrt.runtime.DeviceRuntime.TTNN)
    # Fabric MUST be set before opening the mesh.
    ttrt.runtime.set_fabric_config(ttrt.runtime.FabricConfig.FABRIC_2D)

    options = ttrt.runtime.MeshDeviceOptions()
    options.mesh_shape = [2, 2]
    parent = ttrt.runtime.open_mesh_device(options)
    try:
        m_a = ttrt.runtime.create_sub_mesh_device(parent, [1, 1], [0, 0])
        m_b = ttrt.runtime.create_sub_mesh_device(parent, [1, 1], [0, 1])

        x = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
        host = get_runtime_tensor_from_torch(x, storage=Storage.Owned)
        runtime_dtype = ttrt.binary.Binary.Program.to_data_type(x.dtype)
        layout = _device_tile_layout(m_a, runtime_dtype)

        src_a = ttrt.runtime.to_layout(host, m_a, layout, True)
        dst_b = ttrt.runtime.to_layout(host, m_b, layout, True)

        send_sock, recv_sock = ttrt.runtime.create_socket_pair(
            m_a, m_b, [0, 0], [0, 1], 16 * 1024
        )
        ttrt.runtime.socket_send(send_sock, src_a)   # enqueue on m_a
        ttrt.runtime.socket_recv(recv_sock, dst_b)   # enqueue on m_b
        ttrt.runtime.synchronize_device(m_a)         # fence both, after issuing both
        ttrt.runtime.synchronize_device(m_b)

        src_host = ttrt.runtime.to_host(src_a, untilize=True, blocking=True)[0]
        dst_host = ttrt.runtime.to_host(dst_b, untilize=True, blocking=True)[0]
        assert ttrt.runtime.get_tensor_data_buffer(
            src_host
        ) == ttrt.runtime.get_tensor_data_buffer(dst_host)

        ttrt.runtime.close_socket(send_sock)
        ttrt.runtime.close_socket(recv_sock)
        ttrt.runtime.release_sub_mesh_device(m_a)
        ttrt.runtime.release_sub_mesh_device(m_b)
    finally:
        ttrt.runtime.close_mesh_device(parent)
