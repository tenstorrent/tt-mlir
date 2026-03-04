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


# --- sparse_matmul tests (single device, golden enabled) ---


@pytest.mark.parametrize("target", ["ttnn"])
def test_sparse_matmul_b_sparse(target: str, request, device):
    """
    sparse_matmul with is_input_b_sparse=True (column-parallel MoE gate/up).
    a: [A, B, M, K] = [2, 4, 32, 64]
    b: [1, E, K, N] = [1, 4, 64, 128]
    sparsity: [A, B, 1, E] = [2, 4, 1, 4]
    output: [A, B, 1, E, M, N] = [2, 4, 1, 4, 32, 128]
    """

    def module(builder: TTIRBuilder):
        @builder.func(
            [(2, 4, 32, 64), (1, 4, 64, 128), (2, 4, 1, 4)],
            [torch.bfloat16, torch.bfloat16, torch.bfloat16],
        )
        def sparse_matmul_b_sparse(
            a: Operand,
            b: Operand,
            sparsity: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.sparse_matmul(
                a,
                b,
                sparsity,
                is_input_a_sparse=False,
                is_input_b_sparse=True,
                nnz=0,
                output_shape=(2, 4, 1, 4, 32, 128),
                output_type=torch.bfloat16,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        device=device,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("target", ["ttnn"])
def test_sparse_matmul_a_sparse(target: str, request, device):
    """
    sparse_matmul with is_input_a_sparse=True, is_input_b_sparse=False (row-parallel).
    a: [A, E, M, K] = [2, 4, 32, 64]
    b: [1, E, K, N] = [1, 4, 64, 128]
    sparsity: [1, 1, A, E] = [1, 1, 2, 4]
    output: [A, E, M, N] = [2, 4, 32, 128]
    """

    def module(builder: TTIRBuilder):
        @builder.func(
            [(2, 4, 32, 64), (1, 4, 64, 128), (1, 1, 2, 4)],
            [torch.bfloat16, torch.bfloat16, torch.bfloat16],
        )
        def sparse_matmul_a_sparse(
            a: Operand,
            b: Operand,
            sparsity: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.sparse_matmul(
                a,
                b,
                sparsity,
                is_input_a_sparse=True,
                is_input_b_sparse=False,
                nnz=0,
                output_shape=(2, 4, 32, 128),
                output_type=torch.bfloat16,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        device=device,
        **get_request_kwargs(request),
    )


# --- MoE CCL pipeline test (multi-device, dispatch → combine) ---


@pytest.mark.parametrize("target", ["ttnn"])
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
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        check_pcc=True,
        **get_request_kwargs(request),
    )
