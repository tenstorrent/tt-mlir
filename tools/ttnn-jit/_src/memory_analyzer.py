# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import ttnn.device

from ttmlir.dialects import ttnn as ttnn_dialect
from ttnn_jit._src import (
    allocate_dram_buffer,
    get_l1_base_allocator_addr,
    get_lowest_occupied_compute_l1_address,
    get_l1_size_per_core,
)
from ttnn_jit._src.conversions import ttnn_dtype_from_mlir_element_type
from ttnn_jit._src.tensor_translator import TILE_WIDTH, TILE_HEIGHT


class MemoryAnalyzer:
    """
    Extracts memory addresses that can be used by D2M allocator by taking into
    account the already-in-use memory blocks prior to jit.

    This class queries the device's allocator directly to determine available
    memory ranges.
    This class acts in the following way:
    1. L1:
        1.1 Uses lowest_occupied_compute_l1_address() to get the end of free region
        1.2 Uses get_l1_base_allocator_addr() to get the start of free region
        1.3 Stores the [start, end) address range
    2. DRAM:
        2.1 It allocates the biggest possible DRAM temporary buffer using Metal's allocator,
        2.2 It stores the [start,end) address range of the allocated buffer,
        2.3 It deallocates the allocated buffer,
    3. Output tensor:
        3.1 It pre-allocates the output tensor on either L1 or DRAM, depending on the memory config of the output tensor. This is done via creating a ttnn.empty tensor that can hold a tensor as big as the output tensor.
        3.2 It stores the [start,end) address range of the output tensor.
        3.3 It deallocates the output tensor.

    * Note that the range in 3.2 is not necessarily within the ranges in 1.2 and 2.2, since the memory might be fragmented.
    * Note that the numbers in 2.2 and 3.2 are not yet supported by D2M, so we output them as "not-implemented-...".
    """

    # Alignment requirement for addresses
    # It was suggested by @audrey_kertesz to always round down the addresses to the nearest ALIGNMENT.
    ALIGNMENT_L1 = 32
    ALIGNMENT_DRAM = 64

    def __init__(self, device, output_type, perform_memory_analysis: bool = True):
        self.device = device
        self.enabled = perform_memory_analysis

        if not self.enabled:
            self._memory_view_l1 = None
            self._memory_view_dram = None
            self._l1_addr_range = (0, 0)
            self._dram_addr_range = (0, 0)
            self._output_addr_range = (0, 0)
            return

        # Use get_memory_view API to query memory state
        self._memory_view_l1 = ttnn.device.get_memory_view(device, ttnn.BufferType.L1)
        self._memory_view_dram = ttnn.device.get_memory_view(
            device, ttnn.BufferType.DRAM
        )

        # Disable if we don't have valid memory info
        if self._memory_view_l1 is None or self._memory_view_dram is None:
            print("[MemoryAnalyzer] Warning: Failed to get memory view. Disabling.")
            self.enabled = False
            self._l1_addr_range = (0, 0)
            self._dram_addr_range = (0, 0)
            self._output_addr_range = (0, 0)
            return

        # Get largest free block sizes from memory view
        l1_buffer_size = self._memory_view_l1.largest_contiguous_bytes_free_per_bank
        dram_buffer_size = self._memory_view_dram.largest_contiguous_bytes_free_per_bank

        if l1_buffer_size <= 0 or dram_buffer_size <= 0:
            print(
                "[MemoryAnalyzer] Warning: L1 or DRAM largest_contiguous_bytes_free_per_bank "
                "is not positive. Disabling memory analysis."
            )
            self.enabled = False
            self._l1_addr_range = (0, 0)
            self._dram_addr_range = (0, 0)
            self._output_addr_range = (0, 0)
            return

        # Cache the address ranges at initialization
        # L1: Use direct allocator query via lowest_occupied_compute_l1_address
        self._l1_addr_range = self._get_l1_addr_range()

        # DRAM: Use temporary buffer allocation approach
        self._dram_addr_range = self._allocate_and_get_addr_range(
            dram_buffer_size, allocate_dram_buffer, "DRAM"
        )

        # Pre-allocate output tensor to determine its memory range
        self._output_addr_range = self._pre_allocate_output_tensor(output_type)

    ###########################################################################
    ############################# Private Methods #############################
    ###########################################################################

    def _round_down_to_alignment(self, value: int, is_l1: bool = True) -> int:
        """Round down to the nearest multiple of ALIGNMENT."""
        if is_l1:
            return (value // self.ALIGNMENT_L1) * self.ALIGNMENT_L1
        else:
            return (value // self.ALIGNMENT_DRAM) * self.ALIGNMENT_DRAM

    def _get_l1_addr_range(self) -> tuple[int, int]:
        """
        Calculate L1 address range using lowest_occupied_compute_l1_address API.
        This directly queries the allocator for the exact available L1 region
        without needing to allocate temporary buffers.

        L1 Memory Layout (per core):
        +----------------------------+ <-- High Address (l1_size_per_core)
        |    Allocated L1 Buffers    |
        +----------------------------+        ( Grow downward )
        |    Allocated L1 Buffers    |
        +----------------------------+ <-- lowest_occupied_compute_l1_address()
        |                            |
        |         FREE SPACE         |
        |                            |
        +----------------------------+
        |            CBs             |
        +----------------------------+ <-- get_l1_base_allocator_addr()
        |  Reserved (firmware, etc.) |
        +----------------------------+ <-- Low Address (0)

        Since at the time of execution of D2M, there's no CBs allocated, the
        free space is essentially [get_l1_base_allocator_addr, lowest_occupied_compute_l1_address).

        Returns:
            tuple[int, int]: (start_addr, end_addr) for available L1 region
        """
        start = get_l1_base_allocator_addr(self.device)
        lowest = get_lowest_occupied_compute_l1_address(self.device)
        # If no allocations exist, lowest is None, use full L1 size
        end = lowest if lowest is not None else get_l1_size_per_core(self.device)
        end = self._round_down_to_alignment(end, is_l1=True)
        return (start, end)

    def _bytes_to_kb(self, bytes_val: int) -> float:
        """Convert bytes to KB."""
        return bytes_val / 1024.0

    def _print_memory_view_stats(
        self, memory_view, prefix: str, addr_range: tuple[int, int]
    ):
        """Print memory statistics from a MemoryView object."""
        if memory_view is None:
            print(f"[{prefix}] No memory view available")
            return

        num_banks = memory_view.num_banks
        total_bytes_per_bank = memory_view.total_bytes_per_bank
        total_allocated_per_bank = memory_view.total_bytes_allocated_per_bank
        total_free_per_bank = memory_view.total_bytes_free_per_bank
        largest_contiguous_free = memory_view.largest_contiguous_bytes_free_per_bank

        total_bytes_kb = self._bytes_to_kb(total_bytes_per_bank)
        total_allocated_kb = self._bytes_to_kb(total_allocated_per_bank)
        total_free_kb = self._bytes_to_kb(total_free_per_bank)
        largest_contiguous_kb = self._bytes_to_kb(largest_contiguous_free)

        # Free memory usable by D2M is the size of the cached address range
        start, end = addr_range
        free_for_d2m_bytes = end - start if end > start else 0
        free_for_d2m_kb = self._bytes_to_kb(free_for_d2m_bytes)

        # Calculate percentage of total
        if total_bytes_per_bank > 0:
            free_percent = (free_for_d2m_bytes / total_bytes_per_bank) * 100
        else:
            free_percent = 0.0

        print(f"[{prefix}] Memory View:")
        print(f"       - Number of banks: {num_banks}")
        print(f"       - Total bytes per bank: {total_bytes_kb:.1f} KB")
        print(f"       - Allocated per bank: {total_allocated_kb:.1f} KB")
        print(f"       - Free per bank: {total_free_kb:.1f} KB")
        print(f"       - Largest contiguous free: {largest_contiguous_kb:.1f} KB")
        print(
            f"       - Free memory usable by D2M: {free_for_d2m_kb:.1f} KB "
            f"({free_percent:.1f}% of total per bank)"
        )
        print(f"       - D2M addr range: [{addr_range[0]}, {addr_range[1]})")

    def _dump_memory_info(self):
        """Dump the memory info of the device to ./generated/reports/"""
        ttnn.device.dump_device_memory_state(self.device)

    def _allocate_and_get_addr_range(
        self, buffer_size: int, allocate_fn, prefix: str
    ) -> tuple[int, int]:
        """
        Common logic to allocate a temporary buffer, get address, and return range.

        Args:
            buffer_size: Size of buffer to allocate
            allocate_fn: Function to call (allocate_l1_buffer or allocate_dram_buffer)
            prefix: Buffer type string for logging ("L1" or "DRAM")

        Returns:
            tuple[int, int]: (start_addr, end_addr) or (0, 0) if failed
        """
        assert self.enabled, "Memory analysis should be enabled"
        assert buffer_size > 0, "Buffer size must be positive"
        # page_size is equal to buffer_size: this relies on the lockstep allocation pattern in metal:
        page_size = buffer_size

        buffer = allocate_fn(self.device, buffer_size, page_size)

        start_addr = buffer.address()
        end_addr = start_addr + buffer_size
        end_addr = self._round_down_to_alignment(end_addr, is_l1=False)

        # Deallocate the buffer before returning
        buffer.deallocate()

        # Validate the range makes sense
        if start_addr >= end_addr:
            print(
                f"[MemoryAnalyzer] Warning: Invalid {prefix} range: "
                f"start={start_addr} >= end={end_addr}"
            )
            self.enabled = False
            return (0, 0)

        return (start_addr, end_addr)

    def _range_to_str(self, addr_range: tuple[int, int]) -> str:
        """
        Convert an address range to a comma-separated string.

        Returns:
            str: "start,end" or empty string if disabled or range is invalid
        """
        if not self.enabled:
            return ""

        start, end = addr_range

        if start == 0 and end == 0:
            return ""
        if start >= end:
            return ""
        if start < 0 or end < 0:
            return ""

        return f"{start},{end}"

    def _create_memory_config_from_encoding(self, shape, mlir_layout):
        """
        Create a ttnn memory config from MLIR layout encoding.

        Args:
            shape: tensor shape
            mlir_layout: TTNNLayoutAttr from MLIR encoding (may be None for ND tensors)

        Returns:
            ttnn.MemoryConfig matching the MLIR encoding, or DRAM_MEMORY_CONFIG as fallback
        """
        # Handle case where layout encoding is not a TTNNLayoutAttr
        # (e.g., N-dimensional sharded tensors with different encoding)
        if mlir_layout is None:
            return ttnn.DRAM_MEMORY_CONFIG

        buffer_type = ttnn_dialect.ir.BufferTypeAttr.maybe_downcast(
            mlir_layout.memref.memory_space
        )
        is_dram = buffer_type.value == ttnn_dialect.BufferType.DRAM.value
        mem_layout_int = mlir_layout.tensor_memory_layout_as_int

        # Interleaved layout (0)
        if mem_layout_int == ttnn_dialect.TensorMemoryLayout.Interleaved.value:
            return ttnn.DRAM_MEMORY_CONFIG if is_dram else ttnn.L1_MEMORY_CONFIG

        # Sharded layouts - create sharded memory config
        grid_shape = mlir_layout.grid_shape
        core_grid = ttnn.CoreGrid(x=grid_shape[1], y=grid_shape[0])

        if mem_layout_int == ttnn_dialect.TensorMemoryLayout.BlockSharded.value:
            strategy = ttnn.ShardStrategy.BLOCK
        elif mem_layout_int == ttnn_dialect.TensorMemoryLayout.HeightSharded.value:
            strategy = ttnn.ShardStrategy.HEIGHT
        elif mem_layout_int == ttnn_dialect.TensorMemoryLayout.WidthSharded.value:
            strategy = ttnn.ShardStrategy.WIDTH
        else:
            # Fallback to interleaved
            return ttnn.DRAM_MEMORY_CONFIG if is_dram else ttnn.L1_MEMORY_CONFIG

        return ttnn.create_sharded_memory_config(
            shape=tuple(shape),
            core_grid=core_grid,
            strategy=strategy,
            use_height_and_width_as_shard_shape=False,
        )

    def _pre_allocate_output_tensor(self, output_type) -> tuple[int, int]:
        """
        Pre-allocate a tensor matching the output type to determine memory range.
        Uses ttnn.empty to create a tensor with matching properties.

        Args:
            output_type: MLIR RankedTensorType with shape, element_type, encoding

        Returns:
            tuple[int, int]: (start_addr, end_addr) for the output tensor
        """
        if not self.enabled:
            return (0, 0)

        # Extract info from output_type
        shape = list(output_type.shape)
        dtype = ttnn_dtype_from_mlir_element_type(output_type.element_type)

        if output_type.encoding is not None:
            mlir_layout = ttnn_dialect.ir.TTNNLayoutAttr.maybe_downcast(
                output_type.encoding
            )
        else:
            mlir_layout = None

        # Create memory config from MLIR encoding
        memory_config = self._create_memory_config_from_encoding(shape, mlir_layout)

        # Create an empty tensor that can hold the hypothetical output
        tensor = ttnn.empty(
            shape,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=memory_config,
        )

        # Get address and calculate size using Metal's APIs
        start = tensor.buffer_address()

        # Calculate size based on tile layout and shard spec
        tile_size = tensor.tile.get_tile_size(tensor.dtype)
        mc = tensor.memory_config()

        if mc.is_sharded() and mc.shard_spec is not None:
            # For sharded tensors: calculate per-core shard size in bytes
            shard_shape = mc.shard_spec.shape
            tiles_in_shard = (shard_shape[0] // TILE_WIDTH) * (
                shard_shape[1] // TILE_HEIGHT
            )
            size_bytes = tiles_in_shard * tile_size
        else:
            # For interleaved tensors: calculate total size from volume
            num_tiles = (tensor.volume() + TILE_WIDTH * TILE_HEIGHT - 1) // (
                TILE_WIDTH * TILE_HEIGHT
            )  # ceil division by tile elements
            size_bytes = num_tiles * tile_size

        end = start + size_bytes

        # Deallocate
        ttnn.deallocate(tensor)

        return (start, end)

    ###########################################################################
    ############################### Public APIs ###############################
    ###########################################################################

    def print_stats(self):
        self._print_memory_view_stats(self._memory_view_l1, "L1", self._l1_addr_range)
        self._print_memory_view_stats(
            self._memory_view_dram, "DRAM", self._dram_addr_range
        )

    def get_l1_range_str(self) -> str:
        """
        Get the L1 address range as a compile-time option string for D2M.

        Returns:
            str: Option string like " available-l1-addr-range=1024,4096"
                 or empty string if memory analysis is disabled or range is invalid.
        """
        range_str = self._range_to_str(self._l1_addr_range)
        return f" available-l1-addr-range={range_str}" if range_str else ""

    def get_dram_range_str(self) -> str:
        """
        Get the DRAM address range as a compile-time option string (not currently used by D2M).

        Returns:
            str: Option string like " not-implemented-dram-addr-range=1024,4096"
                 or empty string if memory analysis is disabled or range is invalid.
        """
        range_str = self._range_to_str(self._dram_addr_range)
        return f" not-implemented-dram-addr-range={range_str}" if range_str else ""

    def get_output_range_str(self) -> str:
        """
        Get the pre-allocated output tensor address range as a compile-time option string.

        Returns:
            str: Option string like " not-implemented-preallocated-output-tensor-addr-range=1024,4096"
                 or empty string if memory analysis is disabled or range is invalid.
        """
        range_str = self._range_to_str(self._output_addr_range)
        return (
            f" not-implemented-preallocated-output-tensor-addr-range={range_str}"
            if range_str
            else ""
        )
