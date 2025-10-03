# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn

try:
    device = ttnn.open_device(device_id=0)
    arg0 = ttnn.full(
        shape=[10, 10],
        fill_value=3.0,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn.dump_tensor("arg0.tensorbin", arg0)
    arg1 = ttnn.full(
        shape=[10, 10],
        fill_value=5.0,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn.dump_tensor("arg1.tensorbin", arg1)
finally:
    ttnn.close_device(device)
