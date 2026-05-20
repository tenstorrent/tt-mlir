/*
 * SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Linker script for the dynamically-loaded kernel code blob.
 *
 * Linked at CODE_LOAD_ADDR (L2CPU0 DRAM + 4 MiB) so that all
 * addresses are absolute and the firmware can call the entry point
 * directly at CODE_LOAD_ADDR. The .text.start section (containing the
 * dispatch function) is placed first.
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
