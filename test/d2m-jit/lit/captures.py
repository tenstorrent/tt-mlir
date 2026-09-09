# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# REQUIRES: d2m-jit

"""Closure-captured int free variables get collected into
CompiledKernel._captures and passed through to D2MCompiler.captures as
index constants at the top of the emitted kernel func."""

import d2m_jit as d2m


def test_int_freevar_recorded():
    LIMIT = 7

    @d2m.kernel
    def k(in_t, out_t):
        for _ in range(LIMIT):
            pass

    assert "LIMIT" in k._captures, f"expected LIMIT in captures, got {k._captures}"
    assert k._captures["LIMIT"] == LIMIT


def test_multiple_int_freevars():
    A = 3
    B = 5

    @d2m.kernel
    def k(in_t, out_t):
        for _ in range(A):
            for _ in range(B):
                pass

    assert k._captures.get("A") == A
    assert k._captures.get("B") == B


def test_no_freevars_means_empty_captures():
    @d2m.kernel
    def k(in_t, out_t):
        for _ in range(1):
            pass

    # No captured variables; _captures should be empty.
    assert k._captures == {}, f"expected empty captures, got {k._captures}"


test_int_freevar_recorded()
test_multiple_int_freevars()
test_no_freevars_means_empty_captures()
print("PASS captures")
