/*
 * SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Linker script for X280 bare-metal firmware.
 *
 * L2CPU0 has 4 GiB of AXI-attached DRAM starting at 0x400030000000 in both
 * the X280's physical address space and the NOC tile view (same address).
 * We place code at the base so the reset vector can jump directly to it.
 * Stack sits above code within the first 1 MiB so it never collides with
 * the mailbox / task descriptor at +1 MiB or the data area at +2 MiB.
 */
ENTRY(_start)

SECTIONS
{
    . = 0x400030000000;

    .text.start : { KEEP(*(.text.start)) }
    .text       : { *(.text*) }
    .rodata     : { *(.rodata*) }
    .data       : { *(.data*) }
    .bss        : { *(.bss*) *(COMMON) }

    . = ALIGN(16);
    . = . + 0x4000;           /* 16 KiB stack */
    _stack_top = .;
}

ASSERT(_stack_top < 0x400030100000,
       "firmware + stack must fit below mailbox at DRAM_BASE + 1 MiB")
