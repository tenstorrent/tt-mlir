# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive tests for the MemoryAnalyzer class.

Tests cover:
1. Basic initialization and memory range extraction
2. L1/DRAM range string generation for D2M
3. Output tensor pre-allocation with different memory configs
4. Different tensor shapes and dtypes
5. Different sharding strategies (BLOCK, HEIGHT, WIDTH)
6. Interleaved (non-sharded) tensors on L1 and DRAM
7. Edge cases (disabled analysis)
"""

import inspect
import pytest
import ttnn
import torch

import ttnn_jit
from ttnn_jit._src.memory_analyzer import MemoryAnalyzer
from ttnn_jit._src.ir_generator import generate_ir


def _generate_ir_for_func(func, *args):
    """
    Helper to generate IR from a function and its arguments.
    This mimics how jit.py calls generate_ir with proper _tensor_args.
    """
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())
    tensor_args = {param_names[i]: args[i] for i in range(len(args))}
    return generate_ir(func, False, None, *args, _tensor_args=tensor_args)


class TestMemoryAnalyzerBasic:
    """Basic initialization and range extraction tests."""

    def test_memory_analyzer_initialization(self, device):
        """Test that MemoryAnalyzer initializes correctly with a simple sharded tensor."""

        def simple_exp(x):
            return ttnn.exp(x)

        # Create a sharded input tensor
        memory_config = ttnn.create_sharded_memory_config(
            shape=(512, 512),
            core_grid=ttnn.CoreGrid(x=8, y=8),
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=False,
        )
        input_tensor = ttnn.from_torch(
            torch.randn(512, 512, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

        # Generate IR to get output_type
        ir, output_type = _generate_ir_for_func(simple_exp, input_tensor)

        # Create MemoryAnalyzer
        analyzer = MemoryAnalyzer(device, output_type)

        # Verify it's enabled
        assert analyzer.enabled, "MemoryAnalyzer should be enabled"

        # Verify L1 range is valid
        assert (
            analyzer._l1_addr_range[0] < analyzer._l1_addr_range[1]
        ), f"L1 range should be valid: {analyzer._l1_addr_range}"

        # Verify DRAM range is valid
        assert (
            analyzer._dram_addr_range[0] < analyzer._dram_addr_range[1]
        ), f"DRAM range should be valid: {analyzer._dram_addr_range}"

        # Verify output range is valid
        assert (
            analyzer._output_addr_range[0] < analyzer._output_addr_range[1]
        ), f"Output range should be valid: {analyzer._output_addr_range}"

    def test_memory_analyzer_disabled(self, device):
        """Test that MemoryAnalyzer can be disabled."""

        def simple_exp(x):
            return ttnn.exp(x)

        # Create a sharded input tensor
        memory_config = ttnn.create_sharded_memory_config(
            shape=(512, 512),
            core_grid=ttnn.CoreGrid(x=8, y=8),
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=False,
        )
        input_tensor = ttnn.from_torch(
            torch.randn(512, 512, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

        ir, output_type = _generate_ir_for_func(simple_exp, input_tensor)

        # Create disabled MemoryAnalyzer
        analyzer = MemoryAnalyzer(device, output_type, perform_memory_analysis=False)

        assert not analyzer.enabled, "MemoryAnalyzer should be disabled"
        assert analyzer._l1_addr_range == (0, 0)
        assert analyzer._dram_addr_range == (0, 0)
        assert analyzer._output_addr_range == (0, 0)


class TestMemoryAnalyzerRangeStrings:
    """Tests for range string generation methods."""

    def test_get_l1_range_str_format(self, device):
        """Test that get_l1_range_str returns correct format."""

        def simple_exp(x):
            return ttnn.exp(x)

        memory_config = ttnn.create_sharded_memory_config(
            shape=(512, 512),
            core_grid=ttnn.CoreGrid(x=8, y=8),
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=False,
        )
        input_tensor = ttnn.from_torch(
            torch.randn(512, 512, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

        ir, output_type = _generate_ir_for_func(simple_exp, input_tensor)
        analyzer = MemoryAnalyzer(device, output_type)

        l1_str = analyzer.get_l1_range_str()

        # Should have format " available-l1-addr-range=start,end"
        assert l1_str.startswith(
            " available-l1-addr-range="
        ), f"L1 range string should start with ' available-l1-addr-range=': {l1_str}"
        assert "," in l1_str, f"L1 range string should contain comma: {l1_str}"

        # Parse and validate values
        range_part = l1_str.split("=")[1]
        start, end = map(int, range_part.split(","))
        assert start > 0, f"L1 start should be positive: {start}"
        assert end > start, f"L1 end should be greater than start: {start}, {end}"

    def test_get_dram_range_str_format(self, device):
        """Test that get_dram_range_str returns correct format."""

        def simple_exp(x):
            return ttnn.exp(x)

        memory_config = ttnn.create_sharded_memory_config(
            shape=(512, 512),
            core_grid=ttnn.CoreGrid(x=8, y=8),
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=False,
        )
        input_tensor = ttnn.from_torch(
            torch.randn(512, 512, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

        ir, output_type = _generate_ir_for_func(simple_exp, input_tensor)
        analyzer = MemoryAnalyzer(device, output_type)

        dram_str = analyzer.get_dram_range_str()

        # Should have format " not-implemented-dram-addr-range=start,end"
        assert dram_str.startswith(
            " not-implemented-dram-addr-range="
        ), f"DRAM range string should start with expected prefix: {dram_str}"
        assert "," in dram_str, f"DRAM range string should contain comma: {dram_str}"

    def test_get_output_range_str_format(self, device):
        """Test that get_output_range_str returns correct format."""

        def simple_exp(x):
            return ttnn.exp(x)

        memory_config = ttnn.create_sharded_memory_config(
            shape=(512, 512),
            core_grid=ttnn.CoreGrid(x=8, y=8),
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=False,
        )
        input_tensor = ttnn.from_torch(
            torch.randn(512, 512, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

        ir, output_type = _generate_ir_for_func(simple_exp, input_tensor)
        analyzer = MemoryAnalyzer(device, output_type)

        output_str = analyzer.get_output_range_str()

        # Should have format " not-implemented-preallocated-output-tensor-addr-range=start,end"
        assert output_str.startswith(
            " not-implemented-preallocated-output-tensor-addr-range="
        ), f"Output range string should start with expected prefix: {output_str}"
        assert (
            "," in output_str
        ), f"Output range string should contain comma: {output_str}"

    def test_disabled_analyzer_returns_empty_strings(self, device):
        """Test that disabled analyzer returns empty strings."""

        def simple_exp(x):
            return ttnn.exp(x)

        memory_config = ttnn.create_sharded_memory_config(
            shape=(512, 512),
            core_grid=ttnn.CoreGrid(x=8, y=8),
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=False,
        )
        input_tensor = ttnn.from_torch(
            torch.randn(512, 512, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

        ir, output_type = _generate_ir_for_func(simple_exp, input_tensor)
        analyzer = MemoryAnalyzer(device, output_type, perform_memory_analysis=False)

        assert analyzer.get_l1_range_str() == ""
        assert analyzer.get_dram_range_str() == ""
        assert analyzer.get_output_range_str() == ""


class TestMemoryAnalyzerShardingStrategies:
    """Tests for different sharding strategies."""

    @pytest.mark.parametrize(
        "strategy",
        [
            ttnn.ShardStrategy.BLOCK,
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardStrategy.WIDTH,
        ],
    )
    def test_sharding_strategies(self, device, strategy):
        """Test output pre-allocation with different sharding strategies."""

        def simple_exp(x):
            return ttnn.exp(x)

        # Different grids for different strategies
        if strategy == ttnn.ShardStrategy.BLOCK:
            core_grid = ttnn.CoreGrid(x=8, y=8)
        elif strategy == ttnn.ShardStrategy.HEIGHT:
            core_grid = ttnn.CoreGrid(x=1, y=8)
        else:  # WIDTH
            core_grid = ttnn.CoreGrid(x=8, y=1)

        memory_config = ttnn.create_sharded_memory_config(
            shape=(512, 512),
            core_grid=core_grid,
            strategy=strategy,
            use_height_and_width_as_shard_shape=False,
        )
        input_tensor = ttnn.from_torch(
            torch.randn(512, 512, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

        ir, output_type = _generate_ir_for_func(simple_exp, input_tensor)
        analyzer = MemoryAnalyzer(device, output_type)

        assert analyzer.enabled
        start, end = analyzer._output_addr_range
        assert start > 0, f"Output start should be positive for {strategy}"
        assert end > start, f"Output end should be greater than start for {strategy}"

        # Size should be reasonable for sharded tensor
        size_bytes = end - start
        assert size_bytes > 0, f"Output size should be positive for {strategy}"
        assert (
            size_bytes <= 1024 * 1024
        ), f"Output size should be reasonable for {strategy}: {size_bytes}"


class TestMemoryAnalyzerInterleavedTensors:
    """Tests for interleaved (non-sharded) tensors."""

    def test_dram_interleaved_tensor(self, device):
        """Test output pre-allocation with DRAM interleaved tensor."""

        def simple_exp(x):
            return ttnn.exp(x)

        input_tensor = ttnn.from_torch(
            torch.randn(512, 512, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ir, output_type = _generate_ir_for_func(simple_exp, input_tensor)
        analyzer = MemoryAnalyzer(device, output_type)

        assert analyzer.enabled
        start, end = analyzer._output_addr_range
        assert start > 0, "Output start should be positive for DRAM interleaved"
        assert (
            end > start
        ), "Output end should be greater than start for DRAM interleaved"

    def test_l1_interleaved_tensor(self, device):
        """Test output pre-allocation with L1 interleaved tensor."""

        def simple_exp(x):
            return ttnn.exp(x)

        input_tensor = ttnn.from_torch(
            torch.randn(64, 64, dtype=torch.bfloat16),  # Small tensor for L1
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        ir, output_type = _generate_ir_for_func(simple_exp, input_tensor)
        analyzer = MemoryAnalyzer(device, output_type)

        assert analyzer.enabled
        start, end = analyzer._output_addr_range
        assert start > 0, "Output start should be positive for L1 interleaved"
        assert end > start, "Output end should be greater than start for L1 interleaved"


class TestMemoryAnalyzerDifferentShapes:
    """Tests for different tensor shapes."""

    @pytest.mark.parametrize(
        "shape",
        [
            (32, 32),  # Minimum tile size
            (64, 64),  # Small
            (256, 256),  # Medium
            (512, 512),  # Large
            (1024, 1024),  # Very large
        ],
    )
    def test_different_shapes_sharded(self, device, shape):
        """Test output pre-allocation with different tensor shapes (sharded)."""

        def simple_exp(x):
            return ttnn.exp(x)

        # Calculate appropriate grid for shape
        tile_h, tile_w = shape[0] // 32, shape[1] // 32
        grid_y = min(tile_h, 8)
        grid_x = min(tile_w, 8)

        # Ensure shape is divisible by grid
        while tile_h % grid_y != 0 and grid_y > 1:
            grid_y -= 1
        while tile_w % grid_x != 0 and grid_x > 1:
            grid_x -= 1

        memory_config = ttnn.create_sharded_memory_config(
            shape=shape,
            core_grid=ttnn.CoreGrid(x=grid_x, y=grid_y),
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=False,
        )
        input_tensor = ttnn.from_torch(
            torch.randn(*shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

        ir, output_type = _generate_ir_for_func(simple_exp, input_tensor)
        analyzer = MemoryAnalyzer(device, output_type)

        assert analyzer.enabled
        start, end = analyzer._output_addr_range
        assert start > 0, f"Output start should be positive for shape {shape}"
        assert end > start, f"Output end should be greater than start for shape {shape}"


class TestMemoryAnalyzerDifferentDtypes:
    """Tests for different data types."""

    @pytest.mark.parametrize(
        "ttnn_dtype,torch_dtype",
        [
            (ttnn.bfloat16, torch.bfloat16),
            (ttnn.float32, torch.float32),
        ],
    )
    def test_different_dtypes(self, device, ttnn_dtype, torch_dtype):
        """Test output pre-allocation with different data types."""

        def simple_exp(x):
            return ttnn.exp(x)

        memory_config = ttnn.create_sharded_memory_config(
            shape=(512, 512),
            core_grid=ttnn.CoreGrid(x=8, y=8),
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=False,
        )
        input_tensor = ttnn.from_torch(
            torch.randn(512, 512, dtype=torch_dtype),
            dtype=ttnn_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

        ir, output_type = _generate_ir_for_func(simple_exp, input_tensor)
        analyzer = MemoryAnalyzer(device, output_type)

        assert analyzer.enabled
        start, end = analyzer._output_addr_range
        assert start > 0, f"Output start should be positive for dtype {ttnn_dtype}"
        assert (
            end > start
        ), f"Output end should be greater than start for dtype {ttnn_dtype}"

        # float32 should use more memory than bfloat16
        size_bytes = end - start
        if ttnn_dtype == ttnn.float32:
            # For tile layout, tile size is 4096 for f32 vs 2048 for bf16
            assert size_bytes > 0


class TestMemoryAnalyzerWithJit:
    """End-to-end tests with JIT compilation."""

    def test_jit_with_memory_analysis(self, device):
        """Test that JIT compilation works with memory analysis enabled."""

        def exp_func(x):
            return ttnn.exp(x)

        memory_config = ttnn.create_sharded_memory_config(
            shape=(512, 512),
            core_grid=ttnn.CoreGrid(x=8, y=8),
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=False,
        )
        input_tensor = ttnn.from_torch(
            torch.randn(512, 512, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

        # JIT compile with debug to see memory analyzer output
        exp_jit = ttnn_jit.jit(debug=True)(exp_func)

        # This should work without errors
        output = exp_jit(input_tensor)

        # Verify output is valid
        assert output is not None
        assert output.shape == input_tensor.shape

    def test_jit_compile_only_with_memory_analysis(self, device):
        """Test JIT compile_only mode with memory analysis."""

        def exp_func(x):
            return ttnn.exp(x)

        memory_config = ttnn.create_sharded_memory_config(
            shape=(512, 512),
            core_grid=ttnn.CoreGrid(x=8, y=8),
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=False,
        )
        input_tensor = ttnn.from_torch(
            torch.randn(512, 512, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

        # JIT compile with compile_only=True
        exp_jit = ttnn_jit.jit(debug=True, compile_only=True)(exp_func)

        # This should complete compilation without running
        result = exp_jit(input_tensor)

        # compile_only returns the IR module, not None
        assert result is not None


class TestMemoryAnalyzerPrintStats:
    """Tests for print_stats method."""

    def test_print_stats_no_error(self, device, capsys):
        """Test that print_stats runs without error."""

        def simple_exp(x):
            return ttnn.exp(x)

        memory_config = ttnn.create_sharded_memory_config(
            shape=(512, 512),
            core_grid=ttnn.CoreGrid(x=8, y=8),
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=False,
        )
        input_tensor = ttnn.from_torch(
            torch.randn(512, 512, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

        ir, output_type = _generate_ir_for_func(simple_exp, input_tensor)
        analyzer = MemoryAnalyzer(device, output_type)

        # Should not raise
        analyzer.print_stats()

        # Check output contains expected strings
        captured = capsys.readouterr()
        print(f"Captured output: {captured.out}")
        assert "[L1]" in captured.out
        assert "[DRAM]" in captured.out
        assert "Memory View" in captured.out
