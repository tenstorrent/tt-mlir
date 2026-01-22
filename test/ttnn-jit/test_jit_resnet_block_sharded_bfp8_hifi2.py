# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import time
import torch
import ttnn
import ttnn_jit

from utils import get_block_sharding_grid

import os

# Get parameters from environment variables with defaults
LAYER_SIZE = int(os.getenv("LAYER_SIZE", "1024"))
DEPTH = int(os.getenv("DEPTH", "4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
TRACE_REGION_SIZE = int(os.getenv("TRACE_REGION_SIZE", "773120"))


@ttnn_jit.jit(
    debug=True,
    graph_capture=False,
    enable_cache=True,
    math_fidelity=ttnn.MathFidelity.HiFi2,
)
def jit_call(x, w1, b1, w2, b2):

    out = ttnn.matmul(x, w1)
    out = ttnn.add(out, b1)
    out = ttnn.relu(out)

    out = ttnn.matmul(out, w2)
    out = ttnn.add(out, b2)

    out = ttnn.add(out, x)
    return out


class LinearResNetBlock:
    """
    Implements the ResNet-like building block:
    x -> [Linear -> Activation -> Linear] + x -> output
    """

    def __init__(self, device, size, batch, activation="relu"):
        self.device = device

        # Initialize weights
        # We use identically-sized linear layers (size x size)
        torch_w1 = torch.randn((size, size))
        torch_b1 = torch.zeros((batch, size))

        torch_w2 = torch.randn((size, size))
        torch_b2 = torch.zeros((batch, size))

        # Create L1 block sharded memory config
        # For weights (size x size)
        weight_grid = get_block_sharding_grid((size, size))
        weight_memory_config = ttnn.create_sharded_memory_config(
            shape=(size, size),
            core_grid=ttnn.CoreGrid(x=weight_grid[0] + 1, y=weight_grid[1] + 1),
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=False,
        )

        # For biases (batch x size)
        bias_grid = get_block_sharding_grid((batch, size))
        bias_memory_config = ttnn.create_sharded_memory_config(
            shape=(batch, size),
            core_grid=ttnn.CoreGrid(x=bias_grid[0] + 1, y=bias_grid[1] + 1),
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=False,
        )

        # Use L1 block sharded memory for weights and biases
        self.w1 = ttnn.from_torch(
            torch_w1,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=weight_memory_config,
        )
        self.b1 = ttnn.from_torch(
            torch_b1,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=bias_memory_config,
        )
        self.w2 = ttnn.from_torch(
            torch_w2,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=weight_memory_config,
        )
        self.b2 = ttnn.from_torch(
            torch_b2,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=bias_memory_config,
        )

        self.activation_name = activation

    def __call__(self, x):

        out = jit_call(x, self.w1, self.b1, self.w2, self.b2)
        # out = jit_call_1(x, self.w1, self.b1)
        # out = jit_call_2(out, self.w2, self.b2)

        """ out = jit_mmul(x, self.w1)
        out = jit_add(out, self.b1)
        out = jit_relu(out)

        out = jit_mmul(out, self.w2)
        out = jit_add(out, self.b2)
        out = jit_add(out, x) """

        """ identity = x
        # Layer 1: Linear + Fused Activation
        out = ttnn.linear(x, self.w1, bias=self.b1, activation=self.activation_name)

        # Layer 2: Linear
        out = ttnn.linear(out, self.w2, bias=self.b2)

        # Residual connection
        out = jit_add(out, identity) """

        return out


class TenstorrentResNetModel:
    def __init__(self, device, depth, layer_size, batch_size, activation="relu"):
        self.device = device
        self.blocks = []
        print(
            f"Initializing Model | Depth: {depth} | Layer Size: {layer_size} | Activation: {activation}"
        )

        for i in range(depth):
            block = LinearResNetBlock(device, layer_size, batch_size, activation)
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def profile_with_trace(model, batch_size, layer_size, device, num_iter=10):
    """
    Profiles performance using TTNN Trace and Replay.
    """
    print("\n" + "=" * 50)
    print(f"Profiling with TRACE | Batch: {batch_size} | Size: {layer_size}")
    print("=" * 50)

    # 1. Setup Input Tensor
    # Pre-allocate tensor on device with L1 block sharded memory
    torch_input = torch.randn((batch_size, layer_size))
    input_grid = get_block_sharding_grid((batch_size, layer_size))

    # Create L1 block sharded memory config for input
    input_memory_config = ttnn.create_sharded_memory_config(
        shape=(batch_size, layer_size),
        core_grid=ttnn.CoreGrid(x=input_grid[0] + 1, y=input_grid[1] + 1),
        strategy=ttnn.ShardStrategy.BLOCK,
        use_height_and_width_as_shard_shape=False,
    )

    # Convert host tensor with sharded memory config
    input_a = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_memory_config,
    )

    # Pre-allocate tensor on device with sharded memory
    # Extract shard_spec from the memory_config
    tt_input = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(
            (batch_size, layer_size),
            dtype=ttnn.DataType.BFLOAT8_B,
            layout=ttnn.TILE_LAYOUT,
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            shard_spec=input_memory_config.shard_spec,
            buffer_type=ttnn.BufferType.L1,
        ),
        device,
    )

    # 2. Compile & Warmup (Crucial step)
    # Run the model once normally to compile all kernels and populate the cache.
    # If we trace before compiling, the trace will capture compilation steps which is wrong.
    print("1. Warmup / Compilation run...")
    ttnn.copy_host_to_device_tensor(input_a, tt_input)
    _ = model.forward(tt_input)
    ttnn.synchronize_device(device)

    # 3. Capture Trace
    # We record the command queue operations.
    print("2. Capturing Trace...")
    ttnn.copy_host_to_device_tensor(input_a, tt_input)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    output_tensor = model.forward(tt_input)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    ttnn.synchronize_device(device)
    print("   Trace captured.")

    # 4. Replay Loop
    print(f"3. Running Replay ({num_iter} iterations)...")
    ttnn.copy_host_to_device_tensor(input_a, tt_input)

    # Sync before starting timer
    ttnn.synchronize_device(device)
    start_time = time.time()

    for _ in range(num_iter):
        # Execute the recorded commands
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)

    # Sync after loop to ensure all device execution is finished
    ttnn.synchronize_device(device)
    end_time = time.time()

    # 5. Metrics
    total_time = end_time - start_time
    avg_time = total_time / num_iter
    samples_per_sec = batch_size / avg_time

    print("-" * 50)
    print(f"Total Time:       {total_time:.4f} s")
    print(f"Avg Latency:      {avg_time*1000000:.6f} us")
    print(f"Throughput:       {samples_per_sec:.2f} samples/sec")
    print("=" * 50)

    # 6. Cleanup
    ttnn.release_trace(device, trace_id)


def test_main(
    layer_size=LAYER_SIZE,
    depth=DEPTH,
    batch_size=BATCH_SIZE,
    trace_region_size=TRACE_REGION_SIZE,
):
    # User Configuration

    ACTIVATION = "relu"

    device_id = 0
    device = ttnn.open_device(device_id=device_id, trace_region_size=trace_region_size)

    try:
        model = TenstorrentResNetModel(
            device, depth, layer_size, batch_size, ACTIVATION
        )

        profile_with_trace(
            model=model, batch_size=batch_size, layer_size=layer_size, device=device
        )

    finally:
        ttnn.close_device(device)
        print("Device closed.")
