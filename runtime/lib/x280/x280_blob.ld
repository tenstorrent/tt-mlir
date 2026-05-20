/*
 * SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Linker script for X280 kernel code blobs.
 *
 * Linked at CODE_LOAD_ADDR (L2CPU0 DRAM + 4 MiB) so that the firmware
 * can call the dispatch entry point directly at this address. The
 * .text.start section (containing x280_cpu_dispatch) is placed first.
 */
SECTIONS
{
    . = 0x400030400000;

    .text.start : { *(.text.start) }
    .text       : { *(.text*) }
    .rodata     : { *(.rodata*) }
    .data       : { *(.data*) }
    .bss        : { *(.bss*) *(COMMON) }

    /DISCARD/ : { *(.eh_frame*) *(.comment) *(.riscv.attributes) }
}
