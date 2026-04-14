# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from collections import OrderedDict
from typing import List, Optional, Tuple

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


# --- Selective reduce combine test (multi-device) ---
# Shapes derived from tt-metal tests:
# - GPT-OSS pipeline: hidden=2880, batch=128, seq=1, K=4, experts=128, mesh=(4,8)


@pytest.mark.skip(reason="Golden not yet implemented")
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
@pytest.mark.parametrize(
    "mesh_shape,hidden_size,batch,seq,select_experts_k,experts",
    [
        ((4, 8), 2880, 128, 1, 4, 128),  # GPT-OSS (32-device galaxy)
    ],
    ids=["gpt_oss_4x8"],
)
def test_selective_reduce_combine(
    target: str,
    mesh_shape: Tuple[int, int],
    hidden_size: int,
    batch: int,
    seq: int,
    select_experts_k: int,
    experts: int,
    request,
    device,
):
    """
    Selective reduce combine on galaxy mesh.

    FullToShard(Devices) splits all 4 input tensors across devices,
    selective_reduce_combine runs per-device, ShardToFull(Devices) assembles the
    full result.
    """
    D = mesh_shape[0] * mesh_shape[1]
    experts_per_device = experts // D

    # Shard tensor dims 0 and 1 across mesh dims 0 and 1
    shard_shape_bd = [1, 1, 1, 1]
    shard_shape_bd[0] = mesh_shape[0]  # mesh dim 0 -> tensor dim 0
    shard_shape_bd[1] = mesh_shape[1]  # mesh dim 1 -> tensor dim 1
    shard_dims_bd = [0, 1]

    # For metadata/counts tensors: same sharding pattern
    shard_shape_meta = [1, 1, 1, 1]
    shard_shape_meta[0] = mesh_shape[0]  # mesh dim 0 -> tensor dim 0
    shard_shape_meta[1] = mesh_shape[1]  # mesh dim 1 -> tensor dim 1
    shard_dims_meta = [0, 1]

    def module(builder: TTIRBuilder):
        @builder.func(
            [
                (experts_per_device, batch * D, seq, hidden_size),
                (experts_per_device, batch * D, seq, hidden_size),
                (1, batch * D, seq, select_experts_k),
                (1, batch * D, seq, 1),
            ],
            [torch.bfloat16, torch.bfloat16, torch.int64, torch.int64],
        )
        def selective_reduce_combine_fn(
            dense_input: Operand,
            dense_activations: Operand,
            dense_token_maps: Operand,
            dense_token_counts: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # Shard all inputs along dim 1
            inp = builder.mesh_shard(
                dense_input,
                shard_type=MeshShardType.Devices,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=shard_shape_bd,
                shard_dims=shard_dims_bd,
            )
            act = builder.mesh_shard(
                dense_activations,
                shard_type=MeshShardType.Devices,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=shard_shape_bd,
                shard_dims=shard_dims_bd,
            )
            tok_maps = builder.mesh_shard(
                dense_token_maps,
                shard_type=MeshShardType.Devices,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=shard_shape_meta,
                shard_dims=shard_dims_meta,
            )
            tok_counts = builder.mesh_shard(
                dense_token_counts,
                shard_type=MeshShardType.Devices,
                shard_direction=MeshShardDirection.FullToShard,
                shard_shape=shard_shape_meta,
                shard_dims=shard_dims_meta,
            )

            result = builder.selective_reduce_combine(
                inp,
                act,
                tok_maps,
                tok_counts,
                hidden_size=hidden_size,
                batch_size=batch,
                seq_size=seq,
                select_experts_k=select_experts_k,
                experts=experts,
                axis=1,
                output_shape=(experts_per_device, batch, seq, hidden_size),
                output_type=torch.bfloat16,
                unit_attrs=unit_attrs,
            )

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
