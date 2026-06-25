# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Prefill pseudo-kernels — SPEC, NOT RUNNABLE.

These modules specify the kernels in the LLM-prefill milestone plan
(../PREFILL_MILESTONES.md). They are written as close to real d2m-jit as
possible, but several reference primitives that do not exist yet; those are
tagged inline with `# ⛔ NEEDS[<ID>]` where `<ID>` maps to the
missing-primitive registry in PREFILL_MILESTONES.md.

This package intentionally does NOT import its modules — importing them will
fail on the missing primitives. They are read as implementation specs, not
executed. Grep for `NEEDS[<ID>]` to find every call site of a given gap.
"""
