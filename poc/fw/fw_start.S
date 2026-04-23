// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * X280 startup: the L2CPU tile has 4 harts. The boot flow in tt-bh-linux
 * programs all four reset vectors to the same address, so every hart
 * enters _start on reset. We run main() on hart 0 and park the rest.
 */
    .section .text.start, "ax"
    .globl _start
_start:
    csrr    a0, mhartid
    bnez    a0, park

    /* Enable the FP unit. mstatus.FS resets to 0 (Off) on the X280, so any
     * FP instruction (flw, fadd.s, fsw, fmv.*) traps with illegal-instruction.
     * mtvec is unset at boot, so a trap dead-loops at address 0. We set FS=11
     * (Dirty) so we never have to think about it again. */
    li      t0, 3 << 13         /* mstatus.FS = 0b11 = Dirty */
    csrs    mstatus, t0

    la      sp, _stack_top
    call    fw_main
    /* fw_main should never return; if it does, fall through. */

park:
    wfi
    j       park
