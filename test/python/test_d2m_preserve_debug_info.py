# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: d2m-jit
# RUN: %python -m pytest -q %s

"""Verify d2m-jit text rewrite APIs can preserve MLIR debug locations."""

from __future__ import annotations

from d2m_jit._src.rewrite import apply_patterns_text

_INPUT = """\
module {
  func.func @forward(%arg0: f32) -> f32 {
    %0 = arith.addf %arg0, %arg0 : f32 loc(fused<{tt.profile.region = "decoder.sdpa"}>["a.mlir":1:1])
    return %0 : f32
  }
}
"""


def test_apply_patterns_text_drops_debug_info_by_default():
    out = apply_patterns_text(_INPUT, pattern_paths=[])
    assert "tt.profile.region" not in out
    assert "arith.addf" in out


def test_apply_patterns_text_preserve_debug_info_keeps_profile_region():
    out = apply_patterns_text(_INPUT, pattern_paths=[], preserve_debug_info=True)
    assert 'tt.profile.region = "decoder.sdpa"' in out
    # Round-trip: re-parse and confirm the region is still attached.
    out2 = apply_patterns_text(out, pattern_paths=[], preserve_debug_info=True)
    assert 'tt.profile.region = "decoder.sdpa"' in out2
