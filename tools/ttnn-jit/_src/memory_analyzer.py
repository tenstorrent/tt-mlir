# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import ttnn.device


class MemoryAnalyzer:
    """
    Extracts memory addresses that can be used by D2M allocator by taking into account the already-in-use memory blocks prior to jit.
    """

    def __init__(self, device):
        self.device = device
        self.memory_info_l1 = ttnn.device.get_statistics(device, ttnn.BufferType.L1)
        self.memory_info_dram = ttnn.device.get_statistics(device, ttnn.BufferType.DRAM)
        self._perfrom_initial_checks()

    def _perfrom_initial_checks(self):
        """
        Note:
        The memory_view().block_table shows addresses relative to the start of the 'allocatable' region
        and should not be used by allocator. However, get_statistics().largest_free_block_addrs returns
        addresses (relative to bank base address) that can hold the largest_free_block_bytes (the actual
        hardware address offset within the L1/DRAM bank). Since D2M only accepts a single [start, end)
        range, we just collect one of the largest free blocks for one bank.
        """
        assert (
            len(self.memory_info_l1.largest_free_block_addrs) > 0
        ), "No largest free block found for L1"
        assert (
            len(self.memory_info_dram.largest_free_block_addrs) > 0
        ), "No largest free block found for DRAM"

    def _print_stats_helper(self, stats, prefix):
        print(
            f"[{prefix}], total allocatable size bytes: {stats.total_allocatable_size_bytes}"
        )
        print(f"[{prefix}], total allocated bytes: {stats.total_allocated_bytes}")
        print(f"[{prefix}], total free bytes: {stats.total_free_bytes}")
        print(f"[{prefix}], largest free block bytes: {stats.largest_free_block_bytes}")
        print(
            f"[{prefix}], largest free block addresses: {stats.largest_free_block_addrs}"
        )

    def print_stats(self):
        self._print_stats_helper(self.memory_info_l1, "L1")
        self._print_stats_helper(self.memory_info_dram, "DRAM")

    def dump_memory_info(self):
        "Dump the memory info of the device to ./generated/reports/"
        ttnn.device.dump_device_memory_state(self.device)

    def get_largest_free_l1_block(self):
        """
        Returns the largest allocatable memory address for bank 0 of L1 as a tuple of [start, end)
        Note:
        The largest_free_block_addrs field stores the starting addresses of all free memory blocks
        that have the largest size (equal to largest_free_block_bytes).
        Since D2M only accepts a single [start, end) range, we just collect one of the largest free
        blocks for one bank.
        """
        start_addr = self.memory_info_l1.largest_free_block_addrs[0]
        end_addr = start_addr + self.memory_info_l1.largest_free_block_bytes
        return (start_addr, end_addr)

    def get_largest_free_dram_block(self):
        """
        Returns the largest allocatable memory address for bank 0 of DRAM as a tuple of [start, end)
        Note:
        The largest_free_block_addrs field stores the starting addresses of all free memory blocks
        that have the largest size (equal to largest_free_block_bytes).
        Since D2M only accepts a single [start, end) range, we just collect one of the largest free
        blocks for one bank.
        """
        start_addr = self.memory_info_dram.largest_free_block_addrs[0]
        end_addr = start_addr + self.memory_info_dram.largest_free_block_bytes
        return (start_addr, end_addr)
