/*
 * SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Linker script for X280 bare-metal firmware.
 *
 * L2CPU0 has 4 GiB of AXI-attached DRAM starting at 0x400030000000 in both
 * the X280's physical address space and the NOC tile view (same address).
 * We place code at the base so the reset vector can jump directly to it.
 *
 * Stack lives in the upper portion of L2CPU0 DRAM (256 MiB region starting
 * at DRAM_BASE + 256 MiB), well above the mailbox / task descriptor at
 * +1 MiB and the data area at +2 MiB. Step 5 declares large stack-resident
 * tensor staging buffers; this layout gives ample headroom (~256 MiB)
 * without introducing a heap.
 */
ENTRY(_start)

_stack_top = 0x400040000000;  /* DRAM_BASE + 256 MiB */

SECTIONS
{
    . = 0x400030000000;

    .text.start : { KEEP(*(.text.start)) }
    .text       : { *(.text*) }
    .rodata     : { *(.rodata*) }
    .data       : { *(.data*) }
    .bss        : { *(.bss*) *(COMMON) }
}

ASSERT(_stack_top < 0x400100000000,
       "stack top must fit within L2CPU0's 4 GiB DRAM aperture")
