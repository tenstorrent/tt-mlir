# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fused ttir.moe_compute op + weight prep.

Only the ``compute_only`` path is supported (no A2A selective-reduce-combine,
no multi-device routing): the op runs the selective tilize + W0/W1 + activation
+ W2 expert FFN on a single card and returns matmul_output as the final result.
The full fused (combine) path is intentionally not wired and is rejected by the
op verifier.

The single test below verifies all compute_only outputs against
``ttir_moe_compute_golden`` (tools/golden/mapping.py), which reproduces
tt-metal's device output layouts, on a single Blackhole card:
  * routing metadata (outputs 0-2): byte-exact, PCC=1.0;
  * matmul_output (output 4): the expert FFN scattered into the combine-core
    HEIGHT_SHARDED layout (the exact inverse of tt-metal's
    prepare_output_tensor_from_combine_writer), verified at the relaxed bf4 PCC
    floor (~0.99; bf4 device weights vs raw-bf16 golden).
Slot 3 (tilize_output) is a TILE reinterpret of slot 4's row-major buffer and is
not separately verified (tt-metal ignores it too); slot 5 (combine_output) is
out of scope for compute_only. Executing on device
also exercises the full integration contract: the pipeline compiles, the
OpModel-deduced result types match what the runtime allocates
(checkTensorRefMatchesTTNNTensor), and the op runs without crashing.
"""

import pytest
import torch
from collections import OrderedDict

from builder.base.builder_utils import DeferredDevice
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from golden.mapping import moe_combine_core_rows, moe_combine_scatter_positions
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")

# Verification config: E_VER experts per device, K_VER selected. The golden's
# matmul tile walk only models E_per_device <= 2 (the [.,2,32,.] staging axis),
# and on a single card every global expert is local (mapping all-zeros), so
# E_total == E_VER.
L = 1  # layers
E_VER = 2  # experts per device (== E_total on a single card)
K_VER = 2  # top-k selected experts


def _moe_compute_valid_masks(
    valid_indices,
    valid_mapping,
    E_per_device,
    E_total,
    K_sel,
    total_tokens,
    tc_shape,
    act_elements,
    act_row_stride,
    et_entry_size,
    et_row_elements,
    tile_shape,
    hidden_size,
    height_shard_dim,
):
    """Per-position {0,1} masks for the moe_compute outputs.

    Single-card (1x1 mesh) port of _compute_moe_gpt_valid_masks: builds masks
    over per_expert_total_tokens (output 0), expert_activation (output 1),
    expert_to_token (output 2) and the matmul/tilize combine buffer (outputs
    3/4). The walks mirror ttir_moe_compute_golden so an on-device
    multiply(output, mask) zeroes the don't-care (uninitialized) positions
    before PCC.
    """
    import torch

    mapping_row = valid_mapping.reshape(-1, E_total)[0]
    full_idx = valid_indices.reshape(total_tokens, K_sel)

    tc_mask = torch.zeros(tc_shape, dtype=torch.int32)
    act_mask = torch.zeros(1, act_elements, dtype=torch.int32)
    et_mask = torch.zeros(E_per_device, et_row_elements, dtype=torch.int32)

    local_experts = sorted(
        [e for e in range(E_total) if int(mapping_row[e].item()) == 0]
    )[:E_per_device]

    counts = [0] * E_per_device
    act_row_idx = 0
    for t in range(total_tokens):
        activated = False
        for le, ge in enumerate(local_experts):
            for k in range(K_sel):
                if int(full_idx[t, k].item()) == ge:
                    counts[le] += 1
                    activated = True
                    break
        if activated:
            act_row_idx += 1

    # per_expert_total_tokens is HEIGHT_SHARDED across worker cores but the
    # counts live on a single core's shard; the valid columns are [:E_per_device]
    # of whichever row(s) hold real data (resolved empirically below).
    tc_mask[:, :E_per_device] = 1
    for rec in range(act_row_idx):
        offs = rec * act_row_stride
        act_mask[0, offs : offs + (2 * E_per_device + 1)] = 1
    sentinel_off = act_row_idx * act_row_stride
    if sentinel_off < act_elements:
        act_mask[0, sentinel_off] = 1
    for e in range(E_per_device):
        for idx_c in range(counts[e]):
            et_mask[e, idx_c * et_entry_size] = 1
        sp = counts[e] * et_entry_size
        if sp < et_row_elements:
            et_mask[e, sp] = 1

    # matmul_output combine-core mask (outputs 3/4). Marks exactly the positions
    # the matmul writer scatters per active token, using the same shared walk as
    # ttir_moe_compute_golden (the rest of each L1 shard is uninitialized poison).
    TILE = 32
    width_shard_dim = next(
        (d for d in range(4, 0, -1) if (hidden_size // TILE) % d == 0), 1
    )
    wcols = hidden_size // width_shard_dim
    combine_rows = moe_combine_core_rows(height_shard_dim, width_shard_dim)
    tile_mask = torch.zeros(tile_shape, dtype=torch.bfloat16)
    num_buffers = tile_shape[1]
    for le in range(E_per_device):
        if le >= num_buffers:
            break
        for _t_act, r, dev_t, col0, _lo in moe_combine_scatter_positions(
            counts[le], height_shard_dim, width_shard_dim, wcols, combine_rows
        ):
            tile_mask[r, le, dev_t, col0 : col0 + wcols] = 1

    return tc_mask, act_mask, et_mask, tile_mask


# Self-consistent configs covering every generality axis (NOT a cross-product:
# the op verifier requires matmul_num_cores (= bh_ring_size on BH) % width == 0,
# so width=3 is pinned to ring=12). hidden selects width via
# auto_output_width_shard_dim (largest divisor of hidden/32 that is <= 4):
# hidden=512 -> width=4, hidden=576 -> width=3. hidden and intermediate must
# each be >= bh_ring_size*32 (the W2 output is sharded across the ring;
# w2_shard_tiles in moe_ring_common.h).
_MOE_CONFIGS = [
    # id, hidden, intermediate, activation, has_bias, bh_ring_size
    ("w4_silu_nobias_r16", 512, 512, "silu", False, 16),
    ("w4_silu_nobias_r8", 512, 512, "silu", False, 8),
    ("w4_swiglu_bias_r12", 512, 512, "swiglu", True, 12),
    ("w3_silu_bias_r12", 576, 576, "silu", True, 12),
    ("w3_swiglu_nobias_r12", 576, 576, "swiglu", False, 12),
    # Real-model-derived configs: same structural fingerprint as the shipped
    # single-card MoE models (activation, output width_shard_dim, has_bias,
    # hidden/intermediate ratio), scaled down to hidden <= 2048 with a balanced Ht
    # or a single W2 group.
    ("ds_w4_silu_bias_r8", 2048, 1024, "silu", True, 8),  # DeepSeek-like, H > N
    ("ds_w4_silu_bias_r16", 2048, 2048, "silu", True, 16),  # DeepSeek-like, ring 16
    ("gptoss_w3_swiglu_bias_r12", 1344, 1344, "swiglu", True, 12),  # GPT-OSS-like
]


@pytest.mark.skip(
    "The test will reland completely refactored as part of tt-mlir #8795."
)
@pytest.mark.skip_exec(
    ("n150",),
    reason="Single-card moe_compute doesn't work on WH architecture",
)
@pytest.mark.parametrize(
    "h_size, n_inter, activation, has_bias, bh_ring_size",
    [c[1:] for c in _MOE_CONFIGS],
    ids=[c[0] for c in _MOE_CONFIGS],
)
def test_moe_compute_compute_only_verify(
    h_size, n_inter, activation, has_bias, bh_ring_size, request, system_desc
):
    """Verify moe_compute compute_only outputs 0-4 on a single Blackhole card.

    Feeds valid routing inputs (sparse bf16 buffer, uint16 indices/mapping,
    normalized bf16 scores) and the raw MLP weights (w0/w1/w2) as input goldens;
    ttir_moe_compute_golden reproduces tt-metal's device layouts directly from
    those weights (TTIRToTTNN inserts the device weight prepacking). Per-position
    {0,1} masks (computed the same way as the golden) are passed as func inputs
    and applied with an on-device multiply so the PCC framework compares only the
    meaningful slots (the rest of each device buffer is uninitialized). Outputs
    0-2 match byte-exact (PCC=1.0); output 4 (matmul_output) matches at the
    relaxed bf4 floor (~0.99). Slot 3 (tilize_output) is a TILE reinterpret of
    slot 4 and is not separately verified (tt-metal ignores it too).
    """

    import torch
    from golden.mapping import GoldenMapTensor

    # hidden_size / intermediate_size come from the parametrization. Both must be
    # >= matmul_num_cores tiles (BH ring = 12 -> 12*32 = 384): the W2 output is
    # sharded across the 12-core matmul ring (w2_shard_tiles in moe_ring_common.h),
    # so with fewer tiles than ring cores only the first hidden tile is computed
    # and matmul_output is garbage. See the parametrize list for the constraints.
    H_v = h_size
    N_INTER_v = n_inter
    TOKENS = 32  # tokens_per_device; total_tokens = shape[0]*shape[1] of input
    total_tokens = TOKENS
    E_total = E_VER

    NUM_WORKER_CORES = system_desc.get_num_cores()  # 110 on BH COL grid
    L1_ALIGN = 16

    def _align(v, a):
        return ((v + a - 1) // a) * a

    per_expert_elems = _align(E_VER * 4, L1_ALIGN) // 4
    act_row_stride = _align((2 * E_VER + 1) * 4, L1_ALIGN) // 4
    et_entry = _align(4, L1_ALIGN) // 4
    et_row_elems = (total_tokens + 1) * et_entry

    out_per_expert_total_tokens = (NUM_WORKER_CORES, per_expert_elems)
    out_expert_activation = (1, total_tokens * act_row_stride)
    out_expert_to_token = (E_VER, et_row_elems)
    out_tilize = (NUM_WORKER_CORES, 2, 32, H_v)
    out_matmul = (NUM_WORKER_CORES, 2, 32, H_v)
    out_combine = out_matmul

    w0_shape = (L, E_VER, H_v, N_INTER_v)
    w1_shape = (L, E_VER, H_v, N_INTER_v)
    w2_shape = (L, E_VER, N_INTER_v, H_v)

    # tt-metal moe_compute derives total_tokens = input_shape[0]*input_shape[1]
    # and hidden_size = input_shape[-1], so the sparse buffer is rank-3
    # (1, total_tokens, H). Indices/scores are (1, total_tokens, K); the mapping
    # is (1, E_total). dtypes mirror the tt-metal single-card test: uint16 for
    # indices/mapping, bf16 for sparse/scores.
    tilize_input_shape = (1, TOKENS, H_v)
    tilize_idx_shape = (1, TOKENS, K_VER)
    tilize_scores_shape = (1, TOKENS, K_VER)
    tilize_mapping_shape = (1, E_total)

    # Valid routing inputs (tt-metal gen_sparse_buffer_and_indices pattern).
    torch.manual_seed(42)
    valid_input = torch.rand(1, TOKENS, H_v, dtype=torch.bfloat16)
    valid_indices = torch.zeros(1, TOKENS, K_VER, dtype=torch.uint16)
    for t in range(TOKENS):
        sel = torch.randperm(E_total)[:K_VER].sort().values
        valid_indices[0, t, :] = sel.to(torch.uint16)
    valid_scores = torch.rand(1, TOKENS, K_VER, dtype=torch.float32) + 1e-5
    valid_scores = (valid_scores / valid_scores.sum(dim=-1, keepdim=True)).to(
        torch.bfloat16
    )
    valid_mapping = torch.zeros(tilize_mapping_shape, dtype=torch.uint16)

    raw_w0 = torch.rand(w0_shape, dtype=torch.bfloat16) - 0.5
    raw_w1 = torch.rand(w1_shape, dtype=torch.bfloat16) - 0.5
    raw_w2 = torch.rand(w2_shape, dtype=torch.bfloat16) - 0.5

    # Optional per-expert biases (op verifier: b0/b1 broadcast over the
    # intermediate dim, b2 over hidden). They move only the MLP outputs (3/4);
    # routing outputs 0-2 and all masks are unaffected.
    b0_shape = (L, E_VER, N_INTER_v)
    b1_shape = (L, E_VER, N_INTER_v)
    b2_shape = (L, E_VER, H_v)
    raw_b0 = (torch.rand(b0_shape, dtype=torch.bfloat16) - 0.5) if has_bias else None
    raw_b1 = (torch.rand(b1_shape, dtype=torch.bfloat16) - 0.5) if has_bias else None
    raw_b2 = (torch.rand(b2_shape, dtype=torch.bfloat16) - 0.5) if has_bias else None

    # Per-position valid masks for the routing outputs (0-2) and the
    # matmul/tilize combine buffer (3/4).
    tc_mask, act_mask, et_mask, tile_mask = _moe_compute_valid_masks(
        valid_indices,
        valid_mapping,
        E_VER,
        E_total,
        K_VER,
        total_tokens,
        out_per_expert_total_tokens,
        out_expert_activation[1],
        act_row_stride,
        et_entry,
        et_row_elems,
        out_matmul,
        H_v,
        4,
    )

    # Func inputs in fixed order; the three biases are appended only when this
    # config uses them (golden_inputs carries the matching golden tensor for
    # each operand, in the same order).
    input_shapes = [
        w0_shape,
        w1_shape,
        w2_shape,
        tilize_input_shape,
        tilize_idx_shape,
        tilize_scores_shape,
        tilize_mapping_shape,
        out_per_expert_total_tokens,
        out_expert_activation,
        out_expert_to_token,
        out_matmul,
    ]
    input_types = [
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.uint16,
        torch.bfloat16,
        torch.uint16,
        torch.int32,
        torch.int32,
        torch.int32,
        torch.bfloat16,
    ]
    golden_inputs = [
        raw_w0,
        raw_w1,
        raw_w2,
        valid_input,
        valid_indices,
        valid_scores,
        valid_mapping,
        tc_mask,
        act_mask,
        et_mask,
        tile_mask,
    ]
    if has_bias:
        input_shapes += [b0_shape, b1_shape, b2_shape]
        input_types += [torch.bfloat16, torch.bfloat16, torch.bfloat16]
        golden_inputs += [raw_b0, raw_b1, raw_b2]

    def module(builder: TTIRBuilder):
        @builder.func(input_shapes, input_types)
        def moe_compute_pipeline(*args):
            *operands, builder = args
            ms = builder._mesh_shape
            for operand, golden in zip(operands, golden_inputs):
                builder._set_golden_tensor(
                    operand, GoldenMapTensor({0: golden}, mesh_shape=ms)
                )
            (
                w0,
                w1,
                w2,
                tilize_input,
                tilize_idx,
                tilize_scores,
                tilize_mapping,
                tc_mask_in,
                act_mask_in,
                et_mask_in,
                tile_mask_in,
                *bias_ops,
            ) = operands
            bias_0, bias_1, bias_2 = bias_ops if has_bias else (None, None, None)

            results = builder.moe_compute(
                tilize_input,
                tilize_idx,
                tilize_scores,
                tilize_mapping,
                w0,
                w1,
                w2,
                layer_id=0,
                output_height_shard_dim=4,
                intermediate_size=N_INTER_v,
                bias_0=bias_0,
                bias_1=bias_1,
                bias_2=bias_2,
                activation_function=activation,
                compute_only=True,
                bh_ring_size=bh_ring_size,
                output_shapes=[
                    out_per_expert_total_tokens,
                    out_expert_activation,
                    out_expert_to_token,
                    out_tilize,
                    out_matmul,
                    out_combine,
                ],
                output_types=[
                    torch.int32,
                    torch.int32,
                    torch.int32,
                    torch.bfloat16,
                    torch.bfloat16,
                    torch.bfloat16,
                ],
            )
            # Zero the uninitialized/padding slots so PCC compares only the
            # meaningful positions: routing metadata (0-2) and matmul_output (4).
            # Slot 3 (tilize_output) is a TILE reinterpret of slot 4's row-major
            # backing buffer, so untilizing it scrambles the bytes; tt-metal's
            # validate_matmul ignores it too. Slot 5 (combine_output) is out of
            # scope for compute_only.
            per_expert = builder.multiply(results[0], tc_mask_in)
            act_out = builder.multiply(results[1], act_mask_in)
            e_t = builder.multiply(results[2], et_mask_in)
            matmul_out = builder.multiply(results[4], tile_mask_in)
            return (per_expert, act_out, e_t, matmul_out)

    compile_and_execute_ttir(
        module,
        target="ttnn",
        mesh_name="mesh",
        device=DeferredDevice(request),
        mesh_dict=OrderedDict([("x", 1), ("y", 1)]),
        pipeline_options=["optimization-level=1"],
        pcc=0.98,
        **get_request_kwargs(request),
    )
