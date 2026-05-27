# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""E2E test for the QKV projection + RoPE / KV-cache pattern.

The QKV matmul's result encoding is rewritten to bfp_bf8; a ttnn.typecast
back to bf16 is inserted before each rotary_embedding (K, Q) and
update_cache (V) consumer because those ops require bf16 in Tile layout.

Scaffold — requires multi-device hardware AND builder support for
`rotary_embedding` (not currently exposed in tools/builder/ttir/ttir_builder.py).
Also note that the exit-side ttnn.typecast is slow on hardware today, so
this matcher is under-validated until tt-metal exposes a fast
bfp_bf8(Tile) -> bf16(Tile) kernel.
"""
import pytest

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.skip(
    reason=(
        "Scaffold — needs (1) builder API for rotary_embedding, "
        "(2) multi-device hardware. Fill in once both are available."
    )
)
def test_qkv_rope_dtype_lowering():
    """Placeholder for the QKV → RoPE end-to-end test.

    Intended graph (TTIR):
        qkv = matmul(act, w_qkv)
        qkv = reshape(qkv, ...)
        slices = slice_static(qkv, ...)  # K, V, Q
        for slc in slices:
            slc = reshape(slc, ...)
            slc = reduce_scatter(slc, ...)
            slc = reshape(slc, ...)
            slc = all_gather(slc, ...)
            slc = reshape(slc, ...)
        K = rotary_embedding(slices[0], cos, sin)
        Q = rotary_embedding(slices[2], cos, sin)
        V = update_cache(cache, slices[1], update_index)

    Run with `pipeline_options=["enable-activation-dtype-lowering=true"]` and
    compare PCC against a bf16 reference (pcc ≥ 0.99).
    """
    pass
