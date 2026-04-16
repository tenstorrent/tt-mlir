# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from collections import OrderedDict
from typing import List, Optional, Tuple

import _ttmlir_runtime as tt_runtime
from builder.base.builder_utils import Operand, Shape
from builder.base.builder_enums import MeshShardDirection, MeshShardType
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")


def shape_str(val):
    if isinstance(val, tuple):
        return "x".join(str(v) for v in val)
    return str(val)


# --- MoE CCL pipeline test (multi-device, dispatch → combine) ---


@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
@pytest.mark.parametrize("mesh_shape", [(1, 2)], ids=shape_str)
def test_moe_dispatch_combine(
    target: str, mesh_shape: Tuple[int, int], request, device
):
    """
    MoE CCL round-trip: dispatch → combine on (1,2) mesh.

    FullToShard(Devices) splits inputs across devices, dispatch routes tokens,
    combine gathers expert outputs, ShardToFull(Devices) assembles full result.

    Shard pattern for expert_output and combine result: dim 1 (B*D → B per device).
    """
    D = mesh_shape[0] * mesh_shape[1]
    E = 8
    K = 4
    S = 32
    H = 64
    H_out = 128
    B = 1

    # For (1,2) mesh: 2 shard_dims needed (one per mesh dim).
    # mesh dim 0 (size 1) → tensor dim 2 (no-op, size 1)
    # mesh dim 1 (size D) → tensor dim that gets sharded

    # activations/indices: shard tensor dim 0 across mesh dim 1
    shard_shape_batch = [1, 1, 1, 1]
    shard_shape_batch[2] = mesh_shape[0]  # mesh dim 0 → tensor dim 2
    shard_shape_batch[0] = mesh_shape[1]  # mesh dim 1 → tensor dim 0
    shard_dims_batch = [2, 0]

    # expert_output/combine_result: shard tensor dim 1 across mesh dim 1
    shard_shape_bd = [1, 1, 1, 1]
    shard_shape_bd[2] = mesh_shape[0]  # mesh dim 0 → tensor dim 2
    shard_shape_bd[1] = mesh_shape[1]  # mesh dim 1 → tensor dim 1
    shard_dims_bd = [2, 1]

    def module(builder: TTIRBuilder):
        @builder.func(
            [
                (B * D, 1, S, H),  # activations full [B*D, 1, S, H]
                (B * D, 1, S, K),  # expert_indices full [B*D, 1, S, K]
                (1, 1, E, D),  # expert_mapping [1, 1, E, D]
                (K, B * D, S, H_out),  # expert_output full [K, B*D, S, H']
            ],
            [torch.bfloat16] * 4,
        )
        def moe_pipeline(
            activations: Operand,
            expert_indices: Operand,
            expert_mapping: Operand,
            expert_output: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # Shard activations/indices along dim 0 (batch)
            act = builder.mesh_shard(
                activations,
                shard_type=MeshShardType.Devices,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=shard_shape_batch,
                shard_dims=shard_dims_batch,
            )
            idx = builder.mesh_shard(
                expert_indices,
                shard_type=MeshShardType.Devices,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=shard_shape_batch,
                shard_dims=shard_dims_batch,
            )
            # Replicate expert_mapping
            emap = builder.mesh_shard(
                expert_mapping,
                shard_type=MeshShardType.Replicate,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=[1, 1, 1, 1],
                shard_dims=[],
            )
            # Shard expert_output along dim 1 (B*D -> B)
            exp_out = builder.mesh_shard(
                expert_output,
                shard_type=MeshShardType.Devices,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=shard_shape_bd,
                shard_dims=shard_dims_bd,
            )

            # dispatch: per-device [B, 1, S, H] → [1, B*D, S, H], [1, B*D, S, K]
            dispatched, metadata = builder.all_to_all_dispatch(
                act,
                idx,
                emap,
                num_devices=D,
                cluster_axis=1,
                dispatched_shape=(1, B * D, S, H),
                dispatched_type=torch.bfloat16,
                metadata_shape=(1, B * D, S, K),
                metadata_type=torch.bfloat16,
                unit_attrs=unit_attrs,
            )

            # combine: per-device [K, B*D, S, H'] → [K, B, S, H']
            result = builder.all_to_all_combine(
                exp_out,
                metadata,
                emap,
                num_devices=D,
                cluster_axis=1,
                num_experts_per_tok=K,
                output_shard_dim=1,
                output_shape=(K, B, S, H_out),
                output_type=torch.bfloat16,
                unit_attrs=unit_attrs,
            )

            # ShardToFull along dim 1 (same shard config as expert_output)
            return builder.mesh_shard(
                result,
                shard_type=MeshShardType.Devices,
                shard_direction=MeshShardDirection.ShardToFull,
                shard_shape=shard_shape_bd,
                shard_dims=shard_dims_bd,
            )

    compile_and_execute_ttir(
        module,
        target=target,
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        check_pcc=True,
        **get_request_kwargs(request),
    )


# --- MoE CCL dispatch_metadata test (GPT-OSS 120B-like shapes, 4x8 mesh) ---


def _build_expert_mapping(E_total, D_total, mesh_cols, cluster_axis=0):
    """Build expert-to-device mapping tensor [1, 1, D_total, E_total].

    Mirrors tt-metal's gen_expert_mapping / get_linearized_mesh_coord.
    Each entry mapping[d, e] = linearized device ID that owns expert e.
    All rows are identical (every device sees the same mapping).
    Uses int32 dtype matching tt-metal's integer-typed generation.
    """
    experts_per_cluster = E_total // mesh_cols
    experts_per_device = E_total // D_total

    # Build one row of the mapping
    row = torch.zeros(E_total, dtype=torch.int32)
    for e in range(E_total):
        if cluster_axis == 0:
            cluster_id = e // experts_per_cluster
            device_in_cluster = (e % experts_per_cluster) // experts_per_device
            row[e] = device_in_cluster * mesh_cols + cluster_id
        else:
            row[e] = e // experts_per_device

    # Replicate across all devices, wrap in [1, 1, D, E]
    mapping = row.unsqueeze(0).repeat(D_total, 1).unsqueeze(0).unsqueeze(0)
    return mapping


def _build_expert_indices(batch, S, K, E_total):
    """Build valid expert index tensors [batch, 1, S, K] with values in [0, E_total).

    Uses int32 dtype matching tt-metal's integer-typed generation (torch.int16 → ttnn.uint16).
    """
    indices = torch.zeros(batch, 1, S, K, dtype=torch.int32)
    for b in range(batch):
        for s in range(S):
            sel = torch.randperm(E_total)[:K].sort().values
            indices[b, 0, s, :] = sel.to(torch.int32)
    return indices


def _build_expert_scores(batch, S, K):
    """Build valid normalized expert scores [batch, 1, S, K].

    Matches tt-metal: torch.rand → float32 → normalize → bfloat16.
    """
    scores = torch.rand(batch, 1, S, K, dtype=torch.float32)
    scores = scores + 1e-5  # avoid zeros (tt-metal uses 1e-5)
    scores = scores / scores.sum(dim=-1, keepdim=True)
    return scores.to(torch.bfloat16)


@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("mesh_shape", [(4, 8)], ids=shape_str)
@pytest.mark.parametrize(
    "fabric_config",
    [tt_runtime.runtime.FabricConfig.FABRIC_1D_RING],
    ids=["fabric_1d_ring"],
)
def test_moe_dispatch_metadata(
    target: str, mesh_shape: Tuple[int, int], fabric_config, request, device
):
    """
    MoE CCL dispatch_metadata on (4,8) mesh, mimicking GPT-OSS 120B decode.

    Matches tt-metal test_moe_gpt_e2e.py::test_dispatch layout:
    - Full input: [1, 1, tokens_global, H] sharded along token dim across rows,
      replicated across columns.
    - tokens_global = M * ring_devices (total tokens across the ring)
    - Each ring device gets [1, 1, M, H] after sharding.

    Uses full GPT-OSS 120B parameters:
    - H=2880, E_total=128, M=32, top-k=4
    """
    ring_devices = mesh_shape[0]  # 4 (dispatch ring along dim 0)
    mesh_cols = mesh_shape[1]  # 8
    M = 32  # tokens per ring device (matches GPT-OSS 120B)
    H = 2880  # hidden_size (matches GPT-OSS 120B)
    selected_experts_k = 4  # top-k experts (matches GPT-OSS 120B)
    E_total = 128  # total global experts (matches GPT-OSS 120B)
    D_total = mesh_shape[0] * mesh_shape[1]  # 32 total devices
    tokens_global = M * ring_devices  # 128 total tokens across the ring

    # Pre-build valid input data matching tt-metal's generation pattern.
    torch.manual_seed(42)
    valid_activations = torch.rand(1, 1, tokens_global, H, dtype=torch.bfloat16)
    valid_mapping = _build_expert_mapping(E_total, D_total, mesh_cols, cluster_axis=0)
    valid_indices = _build_expert_indices(1, tokens_global, selected_experts_k, E_total)
    valid_scores = _build_expert_scores(1, tokens_global, selected_experts_k)

    # Build per-device routing mask [rows, cols, tokens_global, 1].
    # mask[r, c, t, 0] = 1.0 if token t is routed to device (r*cols + c).
    # After mesh_shard (dim 0 across rows, dim 1 across cols), each device
    # gets its own [1, 1, tokens_global, 1] mask. Multiplying the dispatched
    # output by this mask zeros out unrouted (garbage) slots so PCC works.
    mapping_row = valid_mapping.reshape(-1, E_total)[0]  # [E_total]
    routed_mask = torch.zeros(
        mesh_shape[0], mesh_shape[1], tokens_global, 1, dtype=torch.bfloat16
    )
    flat_idx = valid_indices.reshape(tokens_global, selected_experts_k)
    for t in range(tokens_global):
        for k in range(selected_experts_k):
            expert_id = int(flat_idx[t, k].item())
            target_dev = int(mapping_row[expert_id].item())
            r, c = target_dev // mesh_cols, target_dev % mesh_cols
            routed_mask[r, c, t, 0] = 1.0

    # Shard patterns
    shard_shape_tokens = [1, 1, 1, 1]
    shard_shape_tokens[2] = mesh_shape[0]
    shard_dims_tokens = [2, -1]  # dim 2 across rows, replicate across cols

    shard_shape_mask = [1, 1, 1, 1]
    shard_shape_mask[0] = mesh_shape[0]  # dim 0 across rows
    shard_shape_mask[1] = mesh_shape[1]  # dim 1 across cols
    shard_dims_mask = [0, 1]

    def module(builder: TTIRBuilder):
        @builder.func(
            [
                (1, 1, tokens_global, H),
                (1, 1, tokens_global, selected_experts_k),
                (1, 1, tokens_global, selected_experts_k),
                (1, 1, D_total, E_total),
                (mesh_shape[0], mesh_shape[1], tokens_global, 1),  # routing mask
            ],
            [torch.bfloat16, torch.int32, torch.bfloat16, torch.int32, torch.bfloat16],
        )
        def moe_dispatch_metadata_gpt_oss(
            activations: Operand,
            expert_indices: Operand,
            expert_scores: Operand,
            expert_mapping: Operand,
            routing_mask: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            from golden.mapping import GoldenMapTensor

            ms = builder._mesh_shape
            builder._set_golden_tensor(
                activations,
                GoldenMapTensor({0: valid_activations}, mesh_shape=ms),
            )
            builder._set_golden_tensor(
                expert_indices,
                GoldenMapTensor({0: valid_indices}, mesh_shape=ms),
            )
            builder._set_golden_tensor(
                expert_scores,
                GoldenMapTensor({0: valid_scores}, mesh_shape=ms),
            )
            builder._set_golden_tensor(
                expert_mapping,
                GoldenMapTensor({0: valid_mapping}, mesh_shape=ms),
            )
            builder._set_golden_tensor(
                routing_mask,
                GoldenMapTensor({0: routed_mask}, mesh_shape=ms),
            )

            act = builder.mesh_shard(
                activations,
                shard_type=MeshShardType.Devices,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=shard_shape_tokens,
                shard_dims=shard_dims_tokens,
            )
            idx = builder.mesh_shard(
                expert_indices,
                shard_type=MeshShardType.Devices,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=shard_shape_tokens,
                shard_dims=shard_dims_tokens,
            )
            scr = builder.mesh_shard(
                expert_scores,
                shard_type=MeshShardType.Devices,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=shard_shape_tokens,
                shard_dims=shard_dims_tokens,
            )
            emap = builder.mesh_shard(
                expert_mapping,
                shard_type=MeshShardType.Replicate,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=[1, 1, 1, 1],
                shard_dims=[],
            )
            # Shard mask: each device gets its own [1, 1, tokens_global, 1]
            mask = builder.mesh_shard(
                routing_mask,
                shard_type=MeshShardType.Devices,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=shard_shape_mask,
                shard_dims=shard_dims_mask,
            )

            dispatched, indices_out, scores_out = builder.all_to_all_dispatch_metadata(
                act,
                idx,
                scr,
                emap,
                num_devices=ring_devices,
                cluster_axis=0,
                dispatched_shape=(1, tokens_global, H),
                dispatched_type=torch.bfloat16,
                indices_shape=(1, tokens_global, selected_experts_k),
                indices_type=torch.int32,
                scores_shape=(1, tokens_global, selected_experts_k),
                scores_type=torch.bfloat16,
                unit_attrs=unit_attrs,
            )

            # Reshape mask [1, 1, tokens_global, 1] → [1, tokens_global, 1]
            # to match dispatched shape [1, tokens_global, H], then multiply
            # to zero out unrouted garbage slots for clean PCC comparison.
            mask_reshaped = builder.reshape(mask, (1, tokens_global, 1))
            masked_dispatched = builder.multiply(dispatched, mask_reshaped)

            return masked_dispatched, indices_out, scores_out

    compile_and_execute_ttir(
        module,
        target=target,
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        check_pcc=True,
        **get_request_kwargs(request),
    )


# --- moe_gpt weight preparation (ported from tt-metal test_moe_gpt_e2e.py) ---

_MOE_GPT_MAX_TILES_PER_CORE = 8
_MOE_GPT_PAD_CORES = {2, 3, 6, 7, 10, 11}
_MOE_GPT_TILE_SIZE = 32


def _moe_gpt_tiles_for_core(ring_pos: int) -> int:
    return 7 if ring_pos in _MOE_GPT_PAD_CORES else 8


def _moe_gpt_static_ring2cores(num_cores: int = 12) -> dict:
    """Build the ring2cores mapping for weight preparation.

    The compiler places HEIGHT_SHARDED DRAM tensors with shard[i] in bank i
    sequentially. The moe_gpt kernel's program factory assigns each matmul
    core a dram_bank_id based on the device's ring ordering. Since the kernel
    reads from bank_id directly, tensor dim-0 index must equal the bank that
    will read it.

    The pad-core pattern {2,3,6,7,10,11} determines which cores have 7 vs 8
    tiles. This pattern is fixed for Wormhole's 12 DRAM banks regardless of
    ring ordering — it depends on how 90 tiles are distributed across 12 cores.
    """
    return {i: (None, i, 1 if i in _MOE_GPT_PAD_CORES else 0) for i in range(num_cores)}


def _moe_gpt_prepare_w0_w1(
    torch_w0: torch.Tensor,
    torch_w1: torch.Tensor,
    L: int,
    E: int,
    K: int,
    N: int,
    ring2cores: dict,
    torch_b0: torch.Tensor = None,
    torch_b1: torch.Tensor = None,
) -> torch.Tensor:
    """Interleave, shard, and pad w0/w1 weights (with optional bias) for moe_gpt.

    Returns tensor of shape (num_cores, L, E, groups_per_core, K_bias, 4*TILE_SIZE)
    where K_bias = K + 32 when biases are provided.
    """
    num_cores = len(ring2cores)
    if torch_b0 is None:
        torch_b0 = torch.zeros(L, E, _MOE_GPT_TILE_SIZE, N, dtype=torch_w0.dtype)
    if torch_b1 is None:
        torch_b1 = torch.zeros(L, E, _MOE_GPT_TILE_SIZE, N, dtype=torch_w1.dtype)

    torch_w0_b0 = torch.cat([torch_w0, torch_b0], dim=2)
    torch_w1_b1 = torch.cat([torch_w1, torch_b1], dim=2)
    K_new = torch_w0_b0.shape[2]
    Nt = N // _MOE_GPT_TILE_SIZE

    w0_chunks = torch_w0_b0.view(L, E, K_new, Nt, _MOE_GPT_TILE_SIZE)
    w1_chunks = torch_w1_b1.view(L, E, K_new, Nt, _MOE_GPT_TILE_SIZE)
    stacked = torch.stack([w0_chunks, w1_chunks], dim=4)
    interleaved = stacked.view(L, E, K_new, Nt, 2 * _MOE_GPT_TILE_SIZE)
    permuted = interleaved.permute(0, 1, 3, 2, 4)

    each_shard = []
    start_tile = 0
    for ring_pos in range(num_cores):
        num_tiles = _moe_gpt_tiles_for_core(ring_pos)
        shard = permuted[:, :, start_tile : start_tile + num_tiles, :, :]
        start_tile += num_tiles
        if num_tiles < _MOE_GPT_MAX_TILES_PER_CORE:
            pad_tiles = _MOE_GPT_MAX_TILES_PER_CORE - num_tiles
            padding = torch.zeros(
                L, E, pad_tiles, K_new, 2 * _MOE_GPT_TILE_SIZE, dtype=torch_w0.dtype
            )
            shard = torch.cat([shard, padding], dim=2)
        each_shard.append(shard)

    reordered = torch.cat(each_shard, dim=2)
    groups_per_core = _MOE_GPT_MAX_TILES_PER_CORE // 2

    all_groups = reordered.view(
        L, E, num_cores, _MOE_GPT_MAX_TILES_PER_CORE, K_new, 2 * _MOE_GPT_TILE_SIZE
    )
    all_groups = all_groups.permute(2, 0, 1, 3, 4, 5)

    pair_view = all_groups.view(
        num_cores, L, E, groups_per_core, 2, K_new, 2 * _MOE_GPT_TILE_SIZE
    )
    pair_view = pair_view.permute(0, 1, 2, 3, 5, 4, 6)
    return pair_view.reshape(
        num_cores, L, E, groups_per_core, K_new, 4 * _MOE_GPT_TILE_SIZE
    )


def _moe_gpt_prepare_w2(
    torch_w2: torch.Tensor,
    L: int,
    E: int,
    N: int,
    K: int,
    ring2cores: dict,
    torch_b2: torch.Tensor = None,
) -> torch.Tensor:
    """Shard, pad, ring-rotate w2 weights (with optional bias) for moe_gpt.

    Returns tensor of shape (num_cores, L, E, 2, N_bias, 4*TILE_SIZE)
    where N_bias = N + 32 when bias is provided.
    """
    num_cores = len(ring2cores)
    if torch_b2 is None:
        torch_b2 = torch.zeros(L, E, _MOE_GPT_TILE_SIZE, K, dtype=torch_w2.dtype)

    torch_w2_b2 = torch.cat([torch_w2, torch_b2], dim=2)
    N_new = N + _MOE_GPT_TILE_SIZE

    each_shard = []
    start_col = 0
    for ring_pos in range(num_cores):
        (_, _, pad_flag) = ring2cores[ring_pos]
        if pad_flag:
            each_shard.append(
                torch_w2_b2[:, :, :, start_col : start_col + 4 * _MOE_GPT_TILE_SIZE]
            )
            start_col += 4 * _MOE_GPT_TILE_SIZE
            each_shard.append(
                torch_w2_b2[:, :, :, start_col : start_col + 3 * _MOE_GPT_TILE_SIZE]
            )
            start_col += 3 * _MOE_GPT_TILE_SIZE
            each_shard.append(
                torch.zeros(L, E, N_new, 1 * _MOE_GPT_TILE_SIZE, dtype=torch_w2.dtype)
            )
        else:
            each_shard.append(
                torch_w2_b2[:, :, :, start_col : start_col + 4 * _MOE_GPT_TILE_SIZE]
            )
            start_col += 4 * _MOE_GPT_TILE_SIZE
            each_shard.append(
                torch_w2_b2[:, :, :, start_col : start_col + 4 * _MOE_GPT_TILE_SIZE]
            )
            start_col += 4 * _MOE_GPT_TILE_SIZE

    torch_w2_reordered = torch.cat(each_shard, dim=-1)
    all_groups = torch_w2_reordered.view(
        L, E, N_new, num_cores, 2, 4 * _MOE_GPT_TILE_SIZE
    )
    all_groups = all_groups.permute(3, 0, 1, 4, 2, 5)

    Nt_all = N_new // _MOE_GPT_TILE_SIZE
    Nt_weight = N // _MOE_GPT_TILE_SIZE
    N_grouped = all_groups.view(
        num_cores, L, E, 2, Nt_all, _MOE_GPT_TILE_SIZE, 4 * _MOE_GPT_TILE_SIZE
    )

    N_weight = N_grouped[:, :, :, :, :Nt_weight, :, :]
    N_bias = N_grouped[:, :, :, :, Nt_weight:, :, :]

    core_chunk_order = torch.tensor(list(reversed(range(num_cores)))).roll(1)
    chunk_sizes = [_moe_gpt_tiles_for_core(i) for i in range(num_cores)]
    chunk_start_positions = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(torch.tensor(chunk_sizes, dtype=torch.int32), dim=0),
        ]
    )

    each_shard = []
    for core_id in range(num_cores):
        each_chunk = []
        for chunk_id in core_chunk_order:
            start_pos = chunk_start_positions[chunk_id]
            end_pos = chunk_start_positions[chunk_id + 1]
            this_chunk = N_weight[core_id, :, :, :, start_pos:end_pos, :, :]
            each_chunk.append(this_chunk)
        each_chunk.append(N_bias[core_id])
        each_shard.append(torch.cat(each_chunk, dim=3))
        core_chunk_order = core_chunk_order.roll(1)

    return torch.stack(each_shard).view(num_cores, L, E, 2, -1, 4 * _MOE_GPT_TILE_SIZE)


def _compute_moe_gpt_valid_masks(
    valid_indices: torch.Tensor,
    valid_mapping: torch.Tensor,
    mesh_shape: Tuple[int, int],
    E_per_device: int,
    E_total: int,
    K_sel: int,
    tokens_global: int,
    tc_elements: int,
    act_elements: int,
    act_row_stride: int,
    et_entry_size: int,
    et_row_elements: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # moe_gpt writes structured int32 records; L1-alignment pads each record
    # and leaves the buffer tail uninitialized. Build per-device {0,1} int32
    # masks marking positions the op is required to write, so on-device
    # `multiply(output, mask)` zeroes out don't-care bytes to match the golden
    # (which initializes with zeros) before PCC.
    mesh_rows, mesh_cols = mesh_shape
    D_total = mesh_rows * mesh_cols
    mapping_row = valid_mapping.reshape(-1, E_total)[0]
    full_idx = valid_indices.reshape(tokens_global, K_sel)

    tc_mask = torch.zeros(mesh_rows, mesh_cols, 1, tc_elements, dtype=torch.int32)
    act_mask = torch.zeros(mesh_rows, mesh_cols, 1, act_elements, dtype=torch.int32)
    et_mask = torch.zeros(
        mesh_rows, mesh_cols, E_per_device, et_row_elements, dtype=torch.int32
    )

    for dev_id in range(D_total):
        r, c = dev_id // mesh_cols, dev_id % mesh_cols
        local_experts = sorted(
            [e for e in range(E_total) if int(mapping_row[e].item()) == dev_id]
        )[:E_per_device]

        counts = [0] * E_per_device
        act_row_idx = 0
        for t in range(tokens_global):
            activated = False
            for le, ge in enumerate(local_experts):
                for k in range(K_sel):
                    if int(full_idx[t, k].item()) == ge:
                        counts[le] += 1
                        activated = True
                        break
            if activated:
                act_row_idx += 1

        tc_mask[r, c, 0, :E_per_device] = 1
        for rec in range(act_row_idx):
            offs = rec * act_row_stride
            act_mask[r, c, 0, offs : offs + (2 * E_per_device + 1)] = 1
        sentinel_off = act_row_idx * act_row_stride
        if sentinel_off < act_elements:
            act_mask[r, c, 0, sentinel_off] = 1
        for e in range(E_per_device):
            for idx_c in range(counts[e]):
                et_mask[r, c, e, idx_c * et_entry_size] = 1
            sp = counts[e] * et_entry_size
            if sp < et_row_elements:
                et_mask[r, c, e, sp] = 1

    return tc_mask, act_mask, et_mask


# --- Chained MoE pipeline: dispatch_metadata → moe_gpt (GPT-OSS fused decode) ---


@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("mesh_shape", [(4, 8)], ids=shape_str)
@pytest.mark.parametrize(
    "fabric_config",
    [tt_runtime.runtime.FabricConfig.FABRIC_1D_RING],
    ids=["fabric_1d_ring"],
)
def test_moe_dispatch_metadata_moe_gpt(
    target: str,
    mesh_shape: Tuple[int, int],
    fabric_config,
    request,
    device,
    system_desc,
):
    """
    Chained MoE pipeline: all_to_all_dispatch_metadata → moe_gpt on (4,8) mesh.

    Mirrors GPT-OSS 120B fused_decode.py:
      1. all_to_all_dispatch_metadata routes tokens to expert-owning devices
      2. Reshape dispatch 3D outputs to 2D
      3. moe_gpt performs fused expert compute
         (tilize → W0/W1 → SwiGLU → A2A ring → W2 → combine)

    Uses GPT-OSS 120B production dimensions (M=32, K=N=2880, E=4, K_sel=4).
    """
    ring_devices = mesh_shape[0]  # 4
    mesh_cols = mesh_shape[1]  # 8
    D_total = ring_devices * mesh_cols  # 32

    # GPT-OSS 120B production dimensions
    M = 32  # tokens per ring device
    H = 2880  # hidden_size
    K_sel = 4  # selected experts per token (top-k)
    E_per_device = 4  # experts per device
    E_total = E_per_device * D_total  # 128 global experts
    E_ring = E_per_device * ring_devices  # 16 experts per ring
    tokens_global = M * ring_devices  # 128 total tokens across the ring

    # moe_gpt weight dimensions (num_cores=12 fixed by DRAM bank count on WH)
    num_cores = 12
    # Worker grid size (arch-dependent: 9x8=72 on Wormhole). The moe_gpt kernel
    # allocates tilize outputs across all worker cores.
    num_worker_cores = system_desc.get_num_cores()
    L_layers = 1
    groups_per_core = 4  # MAX_W0_W1_TILES_PER_CORE // 2 = 8 // 2
    TILE_SIZE = 32
    K_bias = H + TILE_SIZE  # 2912 (hidden_size + 1 bias tile row)
    N_bias = H + TILE_SIZE  # 2912 (intermediate_size + 1 bias tile row)

    # moe_gpt output shapes — must match compute_output_specs() in
    # moe_gpt_device_operation.cpp.  All sizes are L1-alignment-padded.
    L1_ALIGN = 16  # hal::get_l1_alignment() on Wormhole

    def _align(n, a):
        return (n + a - 1) // a * a

    # Output 0: [1, align(E_per_device * sizeof(u32), L1_ALIGN) / sizeof(u32)]
    tc_elements = _align(E_per_device * 4, L1_ALIGN) // 4
    # Output 1: [1, (total_tokens+1) * align((2*E_per_device+1)*sizeof(u32), L1_ALIGN) / sizeof(u32)]
    act_row_stride = _align((2 * E_per_device + 1) * 4, L1_ALIGN) // 4
    act_elements = (tokens_global + 1) * act_row_stride
    # Output 2: [E_per_device, (total_tokens+1) * align(sizeof(u32), L1_ALIGN) / sizeof(u32)]
    et_entry_size = _align(4, L1_ALIGN) // 4
    et_row_elements = (tokens_global + 1) * et_entry_size

    # Pre-build valid dispatch input data
    torch.manual_seed(42)
    valid_activations = torch.rand(1, 1, tokens_global, H, dtype=torch.bfloat16)
    valid_mapping = _build_expert_mapping(E_total, D_total, mesh_cols, cluster_axis=0)
    valid_indices = _build_expert_indices(1, tokens_global, K_sel, E_total)
    valid_scores = _build_expert_scores(1, tokens_global, K_sel)

    # Per-device masks to zero out structured-record padding before PCC.
    tc_mask_full, act_mask_full, et_mask_full = _compute_moe_gpt_valid_masks(
        valid_indices,
        valid_mapping,
        mesh_shape,
        E_per_device,
        E_total,
        K_sel,
        tokens_global,
        tc_elements,
        act_elements,
        act_row_stride,
        et_entry_size,
        et_row_elements,
    )

    # moe_gpt expert mapping: same data as dispatch mapping, shape [D_total, E_total].
    # Matches tt-metal create_fused_moe_gpt_config which passes mapping_data (uint16,
    # ROW_MAJOR, L1) to moe_gpt. The tilize_reader reads one page per device.
    valid_moe_gpt_mapping = valid_mapping.reshape(D_total, E_total)

    # Pre-build properly formatted moe_gpt weight tensors
    ring2cores = _moe_gpt_static_ring2cores(num_cores)
    raw_w0 = torch.rand(L_layers, E_per_device, H, H, dtype=torch.bfloat16) - 0.5
    raw_w1 = torch.rand(L_layers, E_per_device, H, H, dtype=torch.bfloat16) - 0.5
    raw_w2 = torch.rand(L_layers, E_per_device, H, H, dtype=torch.bfloat16) - 0.5
    valid_w0_w1 = _moe_gpt_prepare_w0_w1(
        raw_w0, raw_w1, L_layers, E_per_device, H, H, ring2cores
    )
    valid_w2 = _moe_gpt_prepare_w2(raw_w2, L_layers, E_per_device, H, H, ring2cores)

    # Shard dispatch inputs: dim 2 across rows, replicate across cols
    shard_shape_tokens = [1, 1, 1, 1]
    shard_shape_tokens[2] = mesh_shape[0]
    shard_dims_tokens = [2, -1]

    def module(builder: TTIRBuilder):
        @builder.func(
            [
                # Dispatch inputs (4D)
                (1, 1, tokens_global, H),
                (1, 1, tokens_global, K_sel),
                (1, 1, tokens_global, K_sel),
                (1, 1, D_total, E_total),
                # moe_gpt additional inputs
                (D_total, E_total),
                (
                    num_cores,
                    L_layers,
                    E_per_device,
                    groups_per_core,
                    K_bias,
                    4 * TILE_SIZE,
                ),
                (num_cores, L_layers, E_per_device, 2, N_bias, 4 * TILE_SIZE),
                # Per-device valid-position masks for structured int32 outputs
                (mesh_shape[0], mesh_shape[1], 1, tc_elements),
                (mesh_shape[0], mesh_shape[1], 1, act_elements),
                (mesh_shape[0], mesh_shape[1], E_per_device, et_row_elements),
            ],
            [
                torch.bfloat16,
                torch.int32,
                torch.bfloat16,
                torch.int32,
                torch.int32,
                torch.bfloat16,
                torch.bfloat16,
                torch.int32,
                torch.int32,
                torch.int32,
            ],
        )
        def moe_pipeline(
            activations: Operand,
            expert_indices: Operand,
            expert_scores: Operand,
            dispatch_mapping: Operand,
            moe_gpt_mapping: Operand,
            w0_w1: Operand,
            w2: Operand,
            tc_mask_in: Operand,
            act_mask_in: Operand,
            et_mask_in: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            from golden.mapping import GoldenMapTensor

            ms = builder._mesh_shape
            builder._set_golden_tensor(
                activations,
                GoldenMapTensor({0: valid_activations}, mesh_shape=ms),
            )
            builder._set_golden_tensor(
                expert_indices,
                GoldenMapTensor({0: valid_indices}, mesh_shape=ms),
            )
            builder._set_golden_tensor(
                expert_scores,
                GoldenMapTensor({0: valid_scores}, mesh_shape=ms),
            )
            builder._set_golden_tensor(
                dispatch_mapping,
                GoldenMapTensor({0: valid_mapping}, mesh_shape=ms),
            )
            builder._set_golden_tensor(
                moe_gpt_mapping,
                GoldenMapTensor({0: valid_moe_gpt_mapping}, mesh_shape=ms),
            )
            builder._set_golden_tensor(
                w0_w1,
                GoldenMapTensor({0: valid_w0_w1}, mesh_shape=ms),
            )
            builder._set_golden_tensor(
                w2,
                GoldenMapTensor({0: valid_w2}, mesh_shape=ms),
            )
            builder._set_golden_tensor(
                tc_mask_in,
                GoldenMapTensor({0: tc_mask_full}, mesh_shape=ms),
            )
            builder._set_golden_tensor(
                act_mask_in,
                GoldenMapTensor({0: act_mask_full}, mesh_shape=ms),
            )
            builder._set_golden_tensor(
                et_mask_in,
                GoldenMapTensor({0: et_mask_full}, mesh_shape=ms),
            )

            # Provide raw (unprepared) weights for the moe_gpt golden
            # reference. The golden function needs these because the op
            # inputs are the prepared (interleaved/sharded/padded) tensors
            # which are impractical to unpack.
            builder._moe_gpt_raw_weights = (raw_w0, raw_w1, raw_w2)

            # --- Shard dispatch inputs ---
            act = builder.mesh_shard(
                activations,
                shard_type=MeshShardType.Devices,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=shard_shape_tokens,
                shard_dims=shard_dims_tokens,
            )
            idx = builder.mesh_shard(
                expert_indices,
                shard_type=MeshShardType.Devices,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=shard_shape_tokens,
                shard_dims=shard_dims_tokens,
            )
            scr = builder.mesh_shard(
                expert_scores,
                shard_type=MeshShardType.Devices,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=shard_shape_tokens,
                shard_dims=shard_dims_tokens,
            )
            emap = builder.mesh_shard(
                dispatch_mapping,
                shard_type=MeshShardType.Replicate,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=[1, 1, 1, 1],
                shard_dims=[],
            )

            # --- Replicate moe_gpt inputs across mesh ---
            moe_map = builder.mesh_shard(
                moe_gpt_mapping,
                shard_type=MeshShardType.Replicate,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=[1, 1],
                shard_dims=[],
            )
            w0w1 = builder.mesh_shard(
                w0_w1,
                shard_type=MeshShardType.Replicate,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=[1, 1, 1, 1, 1, 1],
                shard_dims=[],
            )
            w2_rep = builder.mesh_shard(
                w2,
                shard_type=MeshShardType.Replicate,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=[1, 1, 1, 1, 1, 1],
                shard_dims=[],
            )

            # --- Shard per-device valid-position masks (one shard per device) ---
            mask_shard_shape = [mesh_shape[0], mesh_shape[1], 1, 1]
            mask_shard_dims = [0, 1]
            tc_m = builder.mesh_shard(
                tc_mask_in,
                shard_type=MeshShardType.Devices,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=mask_shard_shape,
                shard_dims=mask_shard_dims,
            )
            act_m = builder.mesh_shard(
                act_mask_in,
                shard_type=MeshShardType.Devices,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=mask_shard_shape,
                shard_dims=mask_shard_dims,
            )
            et_m = builder.mesh_shard(
                et_mask_in,
                shard_type=MeshShardType.Devices,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=mask_shard_shape,
                shard_dims=mask_shard_dims,
            )

            # --- Step 1: all_to_all_dispatch_metadata ---
            dispatched, indices_out, scores_out = builder.all_to_all_dispatch_metadata(
                act,
                idx,
                scr,
                emap,
                num_devices=ring_devices,
                cluster_axis=0,
                dispatched_shape=(1, tokens_global, H),
                dispatched_type=torch.bfloat16,
                indices_shape=(1, tokens_global, K_sel),
                indices_type=torch.bfloat16,
                scores_shape=(1, tokens_global, K_sel),
                scores_type=torch.bfloat16,
                unit_attrs=unit_attrs,
            )

            # Reshape dispatched 3D → 2D for moe_gpt input_tensor.
            # Indices/scores stay 3D — metal kernel takes them directly
            # from dispatch (HEIGHT_SHARDED L1), verifier only needs rank >= 2.
            input_2d = builder.reshape(dispatched, (tokens_global, H))

            # --- Step 2: moe_gpt (fused expert compute) ---
            (
                token_counts,
                activation_records,
                token_indices,
                tilize_out,
                tilize_out_rm,
            ) = builder.moe_gpt(
                input_2d,
                indices_out,
                scores_out,
                moe_map,
                w0w1,
                w2_rep,
                hidden_size=H,
                cluster_axis=0,
                token_counts_shape=(1, tc_elements),
                token_counts_type=torch.int32,
                activation_records_shape=(1, act_elements),
                activation_records_type=torch.int32,
                token_indices_shape=(E_per_device, et_row_elements),
                token_indices_type=torch.int32,
                tilize_out_shape=(num_worker_cores, 2, TILE_SIZE, H),
                tilize_out_type=torch.bfloat16,
                tilize_out_rm_shape=(num_worker_cores, 2, TILE_SIZE, H),
                tilize_out_rm_type=torch.bfloat16,
                unit_attrs=unit_attrs,
            )

            # Zero out don't-care padding/tail in the structured int32 outputs
            # so both sides match at those positions for a clean PCC compare.
            tc_mask_2d = builder.reshape(tc_m, (1, tc_elements))
            act_mask_2d = builder.reshape(act_m, (1, act_elements))
            et_mask_2d = builder.reshape(et_m, (E_per_device, et_row_elements))
            token_counts = builder.multiply(token_counts, tc_mask_2d)
            activation_records = builder.multiply(activation_records, act_mask_2d)
            token_indices = builder.multiply(token_indices, et_mask_2d)

            # tilize_out / tilize_out_rm omitted: they share a HEIGHT_SHARDED
            # buffer across all 72 workers but only the combine cores write;
            # golden layout doesn't match the kernel's combine-core assignment,
            # so validate them in moe_gpt's own unit test, not this chained E2E.
            return token_counts, activation_records, token_indices

    compile_and_execute_ttir(
        module,
        target=target,
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        **get_request_kwargs(request),
    )
