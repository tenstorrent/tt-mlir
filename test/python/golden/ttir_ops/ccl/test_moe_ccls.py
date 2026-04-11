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

    Mirrors tt-metal's _gen_expert_mapping_6u / get_linearized_mesh_coord.
    Each entry mapping[d, e] = linearized device ID that owns expert e.
    All rows are identical (every device sees the same mapping).

    For cluster_axis=0 on a (rows, cols) mesh:
      - experts_per_cluster = E_total / cols
      - experts_per_device  = E_total / D_total
      - owner(e) = (e % experts_per_cluster) // experts_per_device * cols + e // experts_per_cluster
    """
    experts_per_cluster = E_total // mesh_cols
    experts_per_device = E_total // D_total

    # Build one row of the mapping
    row = torch.zeros(E_total, dtype=torch.bfloat16)
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
    """Build valid expert index tensors [batch, 1, S, K] with values in [0, E_total)."""
    indices = torch.zeros(batch, 1, S, K, dtype=torch.bfloat16)
    for b in range(batch):
        for s in range(S):
            sel = torch.randperm(E_total)[:K].sort().values
            indices[b, 0, s, :] = sel.to(torch.bfloat16)
    return indices


def _build_expert_scores(batch, S, K):
    """Build valid normalized expert scores [batch, 1, S, K]."""
    scores = torch.rand(batch, 1, S, K, dtype=torch.bfloat16) + 1e-3
    # Normalize each token's scores to sum to 1
    scores = scores / scores.sum(dim=-1, keepdim=True)
    return scores


@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
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

    Scaled down for testing:
    - H=64 (instead of 2880), E_total=32 (instead of 128)
    """
    ring_devices = mesh_shape[0]  # 4 (dispatch ring along dim 0)
    mesh_cols = mesh_shape[1]  # 8
    M = 32  # tokens per ring device (matches GPT-OSS 120B)
    H = 2880  # hidden_size (matches GPT-OSS 120B)
    selected_experts_k = 4  # top-k experts (matches GPT-OSS 120B)
    E_total = 128  # total global experts (matches GPT-OSS 120B)
    D_total = mesh_shape[0] * mesh_shape[1]  # 32 total devices
    tokens_global = M * ring_devices  # 128 total tokens across the ring

    # Pre-build valid routing data
    torch.manual_seed(42)
    valid_mapping = _build_expert_mapping(E_total, D_total, mesh_cols, cluster_axis=0)
    valid_indices = _build_expert_indices(1, tokens_global, selected_experts_k, E_total)
    valid_scores = _build_expert_scores(1, tokens_global, selected_experts_k)

    # Shard pattern: mesh dim 0 (4 rows) shards tensor dim 2 (token dim),
    #                mesh dim 1 (8 cols) replicates.
    # [1, 1, tokens_global, C] → per device [1, 1, M, C]
    shard_shape_tokens = [1, 1, 1, 1]
    shard_shape_tokens[2] = mesh_shape[0]  # mesh dim 0 → tensor dim 2
    shard_dims_tokens = [2, -1]  # shard dim 2 across rows, replicate across cols

    def module(builder: TTIRBuilder):
        @builder.func(
            [
                (1, 1, tokens_global, H),  # activations [1, 1, tokens_global, H]
                (1, 1, tokens_global, selected_experts_k),  # expert_indices
                (1, 1, tokens_global, selected_experts_k),  # expert_scores
                (1, 1, D_total, E_total),  # expert_mapping [1, 1, D, E]
            ],
            [torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.bfloat16],
        )
        def moe_dispatch_metadata_gpt_oss(
            activations: Operand,
            expert_indices: Operand,
            expert_scores: Operand,
            expert_mapping: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # Override random goldens with valid routing data to avoid
            # fabric deadlocks from nonsensical expert routing.
            from golden.mapping import GoldenMapTensor

            ms = builder._mesh_shape
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

            # Shard activations/indices/scores along token dim across rows,
            # replicate across columns.
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
            # Replicate expert_mapping across all devices
            emap = builder.mesh_shard(
                expert_mapping,
                shard_type=MeshShardType.Replicate,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=[1, 1, 1, 1],
                shard_dims=[],
            )

            # dispatch_metadata along cluster_axis=0 (ring of 4 devices)
            # Per-device input: [1, 1, M, H]
            # Per-device output: [1, tokens_global, H] (3D from metal,
            #   reshaped to [1, ring_devices, M, H] by TTIRToTTNN)
            dispatched, indices_out, scores_out = builder.all_to_all_dispatch_metadata(
                act,
                idx,
                scr,
                emap,
                num_devices=ring_devices,
                cluster_axis=0,
                dispatched_shape=(1, ring_devices, M, H),
                dispatched_type=torch.bfloat16,
                indices_shape=(1, ring_devices, M, selected_experts_k),
                indices_type=torch.bfloat16,
                scores_shape=(1, ring_devices, M, selected_experts_k),
                scores_type=torch.bfloat16,
                unit_attrs=unit_attrs,
            )

            return dispatched

    compile_and_execute_ttir(
        module,
        target=target,
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        check_pcc=True,
        **get_request_kwargs(request),
    )
