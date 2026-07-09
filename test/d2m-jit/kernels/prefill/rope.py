# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Rotary position embedding (RoPE) prefill kernel.

    rope(x) = x * cos + rotate_half(x) * sin

`rotate_half` is expressed as a *view-roll* of x plus a precomputed sign tile,
so no `concat` primitive is needed:

    rotate_half(x) = roll_view(x) * sign        # sign = [-1...,+1...]
    rope(x)        = x * cos + roll_view(x) * (sign * sin)

cos / (sign*sin) are precomputed host-side and passed as input tiles. The roll
is a free `view_layout` (metadata, no data movement).

The first implementation only supports tile-aligned half-split RoPE. The final
logical dimension must be a multiple of 64 so the half-roll boundary never cuts
through a 32-wide tile.
"""

import torch

import d2m_jit as d2m
from runner import KernelBench, TensorSpec, d2m_dtype


def _feature_half_roll_view(x_lt):
    """Return RoPE's tile-level half-roll view over the final dimension."""
    layout = x_lt.layout
    head_dim = layout.logical_shape[-1]
    src_shape = list(x_lt.value.type.shape)
    if len(src_shape) != 4:
        raise ValueError(
            "RoPE half-roll expects physical rank 4 "
            f"[grid_y, grid_x, block_y, block_x], got shape {src_shape}"
        )

    grid_x = src_shape[1]
    block_x = src_shape[3]
    total_feature_tiles = grid_x * block_x
    expected_feature_tiles = head_dim // 32
    if total_feature_tiles != expected_feature_tiles:
        raise ValueError(
            "RoPE half-roll expected physical feature tiles "
            f"{expected_feature_tiles}, got {total_feature_tiles} from shape "
            f"{src_shape}"
        )
    if total_feature_tiles % 2 != 0:
        raise ValueError(
            "RoPE half-roll requires an even number of feature tiles, "
            f"got {total_feature_tiles}"
        )

    def roll_dim(d1, d3):
        feature_tile = d1 * block_x + d3 + (total_feature_tiles // 2)
        rolled = feature_tile % total_feature_tiles
        return rolled // block_x, rolled % block_x

    return d2m.view_layout(
        x_lt,
        lambda d0, d1, d2, d3: (d0, roll_dim(d1, d3)[0], d2, roll_dim(d1, d3)[1]),
    )


def _validate_rope_layouts(x_lt, cos_lt, sin_signed_lt, out_layout):
    x_shape = list(x_lt.layout.logical_shape)
    if len(x_shape) != 2:
        raise ValueError(f"RoPE expects a 2D layout, got shape {x_shape}")
    if x_shape[-1] % 64 != 0:
        raise ValueError(
            "RoPE requires the final logical dimension to be a multiple of 64, "
            f"got {x_shape[-1]}"
        )

    for name, lt in (("cos", cos_lt), ("sin_signed", sin_signed_lt)):
        if list(lt.layout.logical_shape) != x_shape:
            raise ValueError(
                f"{name} layout shape {lt.layout.logical_shape} must match x shape "
                f"{x_shape}"
            )
        if lt.layout.tiled != x_lt.layout.tiled:
            raise ValueError(f"{name} layout tiled flag must match x layout")

    if list(out_layout.logical_shape) != x_shape:
        raise ValueError(
            f"output layout shape {out_layout.logical_shape} must match x shape {x_shape}"
        )
    if out_layout.tiled != x_lt.layout.tiled:
        raise ValueError("output layout tiled flag must match x layout")
    if not x_lt.layout.tiled:
        raise ValueError("RoPE requires tiled input layouts")


def build_rope_tables(
    seq_len,
    head_dim,
    *,
    start_pos=0,
    theta=10000.0,
    dtype=torch.float32,
    device=None,
):
    """Build row-major host tables for half-split RoPE prefill.

    Returns `(cos, sin_signed)` with shape `[seq_len, head_dim]`. `sin_signed`
    folds rotate_half's `[-x_hi, x_lo]` sign into the sine table so the kernel
    can use only multiply/add tile ops.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    freqs = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    positions = torch.arange(
        start_pos, start_pos + seq_len, dtype=torch.float32, device=device
    )
    angles = torch.outer(positions, freqs)
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)

    cos = torch.cat([cos_half, cos_half], dim=-1)
    sin_signed = torch.cat([-sin_half, sin_half], dim=-1)
    return cos.to(dtype=dtype), sin_signed.to(dtype=dtype)


@d2m.kernel
def rope(x, x_rolled, cos, sin_signed, out, m_blocks, n_blocks):
    # x_rolled: a view of x produced host-side via d2m.view_layout (the
    #           half-rotation index permutation). Passed in so the kernel
    #           body stays a pure eltwise pass.
    # cos:        precomputed cos table tile(s).
    # sin_signed: precomputed (sign * sin) so rotate_half's negate folds in.
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            xx = remote_load(x, [m_off + m, n_off + n])
            xr = remote_load(x_rolled, [m_off + m, n_off + n])
            c = remote_load(cos, [m_off + m, n_off + n])
            s = remote_load(sin_signed, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], xx * c + xr * s)


def apply_rope(x_lt, cos_lt, sin_signed_lt, L_io, grid, m_blocks, n_blocks):
    """Host driver: build the half-rotation view, dispatch the kernel."""
    _validate_rope_layouts(x_lt, cos_lt, sin_signed_lt, L_io)
    x_rolled = _feature_half_roll_view(x_lt)
    out = d2m.empty(L_io)
    rope(x_lt, x_rolled, cos_lt, sin_signed_lt, out, m_blocks, n_blocks, grid=grid)
    return out


def rope_materializer(kernel, inputs, tensors, grid_shape):
    """Materializer for rope: inputs → layout → kernel dispatch → host."""
    x_torch, cos_torch, sin_signed_torch = inputs

    ts = tensors[0]
    gy, gx = grid_shape
    seq_len, head_dim = ts.shape

    seq_len_tiles = seq_len // 32
    head_dim_tiles = head_dim // 32
    block_y, block_x = ts.block_shape

    L = d2m.Layout(
        shape=(seq_len, head_dim),
        dtype=d2m_dtype(ts.dtype),
        block_shape=[block_y, block_x],
        grid_shape=[gy, gx],
    )

    x_lt = d2m.to_layout(x_torch, L)
    cos_lt = d2m.to_layout(cos_torch, L)
    sin_signed_lt = d2m.to_layout(sin_signed_torch, L)

    m_blocks = (seq_len_tiles // gy) // block_y
    n_blocks = (head_dim_tiles // gx) // block_x

    _validate_rope_layouts(x_lt, cos_lt, sin_signed_lt, L)
    x_rolled = _feature_half_roll_view(x_lt)
    out_lt = d2m.empty(L)
    kernel(
        x_lt,
        x_rolled,
        cos_lt,
        sin_signed_lt,
        out_lt,
        m_blocks,
        n_blocks,
        grid=(gy, gx),
    )

    return out_lt.to_host()


def _golden(x, cos, sin_signed):
    """Torch reference matching the kernel exactly: x*cos + roll_half(x)*sin_signed.

    roll_half shifts the feature dimension by head_dim/2: [x_lo, x_hi] → [x_hi, x_lo].
    This directly mirrors `_feature_half_roll_view` and the kernel body.
    """
    half = x.shape[-1] // 2
    x_hi, x_lo = x[..., half:], x[..., :half]
    x_rolled = torch.cat([x_hi, x_lo], dim=-1)
    return x * cos + x_rolled * sin_signed


KERNEL_BENCHES = {
    "rope": KernelBench(
        kernel=rope,
        golden=_golden,
        run=rope_materializer,
        tensors=[
            TensorSpec(
                shape=(64, 64),
                block_shape=[2, 2],
                dtype=torch.float32,
                dist="uniform(-1,1)",
            ),
            TensorSpec(
                shape=(64, 64),
                block_shape=[2, 2],
                dtype=torch.float32,
                dist=lambda shape, td, gen: build_rope_tables(
                    shape[0], shape[1], dtype=td
                )[0],
            ),
            TensorSpec(
                shape=(64, 64),
                block_shape=[2, 2],
                dtype=torch.float32,
                dist=lambda shape, td, gen: build_rope_tables(
                    shape[0], shape[1], dtype=td
                )[1],
            ),
        ],
        grid_shape=(1, 1),
    )
}
