# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn

device_id = 0
device = ttnn.open_device(device_id=device_id)

ttnn.SetDefaultDevice(device)

try:
    tensor = ttnn.zeros((128, 32))
    tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
    tensor = ttnn.to_device(
        tensor,
        device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM
        ),
    )

    out_mem_config = ttnn.create_sharded_memory_config(
        shape=(128, 32),
        core_grid=ttnn.CoreGrid(x=2, y=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    print("Desired grid:", out_mem_config.shard_spec.grid)
    # Desired grid: {[(x=0,y=0) - (x=1,y=0)]}
    out = ttnn.add(tensor, tensor, memory_config=out_mem_config)

    print("Output grid:", out.memory_config().shard_spec.grid)
    # Output grid: {[(x=0,y=0) - (x=3,y=0)]}
finally:
    ttnn.close_device(device)
