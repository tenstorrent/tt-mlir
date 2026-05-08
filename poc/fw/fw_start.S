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

    /* Enable the FP and Vector units. mstatus.FS and mstatus.VS both reset
     * to 0 (Off) on the X280, so the first FP instruction (flw, fadd.s,
     * fsw, fmv.*) OR the first vector / vector-CSR instruction (vsetvli,
     * vle*.v, csrr vlenb, ...) traps with illegal-instruction. mtvec is
     * unset at boot, so a trap dead-loops at address 0. We set both fields
     * to 11 (Dirty) so we never have to think about it again. The Vector
     * bit matters even for code that "looks scalar": clang auto-vectorises
     * memcpy / strided copies into whole-register vl*r.v / vs*r.v under
     * -mcpu=sifive-x280, and the very first such instruction reads vlenb. */
    li      t0, (3 << 13) | (3 << 9)  /* mstatus.FS = mstatus.VS = Dirty */
    csrs    mstatus, t0

    la      sp, _stack_top
    call    fw_main
    /* fw_main should never return; if it does, fall through. */

park:
    wfi
    j       park
