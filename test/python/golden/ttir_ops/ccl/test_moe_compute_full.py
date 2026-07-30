# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fused ttir.moe_compute op + weight prep.

moe_compute always runs the full A2A selective-reduce-combine path and exposes a
single result: the combine output. The selective tilize + W0/W1 + activation +
W2 expert FFN and the A2A combine run fused across the mesh; tt-metal's
routing/tilize/matmul intermediates are allocated and freed inside the runtime
handler and are not exposed in the IR.

The test below verifies the combine output against ``ttir_moe_compute_golden``
(tools/golden/mapping.py), which reproduces tt-metal's SelectiveReduceCombine
result, on a galaxy submesh (ring along cluster_axis). Inputs are presharded and
the combine output is a presharded result (per-device local shape), so PCC is
checked per device at the relaxed bf4 floor (bf4 device weights vs raw-bf16
golden). It is parametrized over ``enable_trace`` to cover TTNN trace
capture/replay. Executing on device also exercises the full integration
contract: the pipeline compiles, the result type matches what the runtime
allocates (checkTensorRefMatchesTTNNTensor), and the op runs without crashing.
"""

import os
import pytest
import torch
from collections import OrderedDict
from typing import List, Optional

import _ttmlir_runtime as tt_runtime
from builder.base.builder_utils import DeferredDevice, Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")


# ---------------------------------------------------------------------------
# Full (A2A selective-reduce-combine) path
# ---------------------------------------------------------------------------

# The full-path test runs on a galaxy submesh and has no CI runner
# (filter_valid_mesh_shape deselects multi-chip meshes on n150/n300/p150, and
# llmbox would run with the wrong topology and hang). Each config names the
# mesh-graph descriptor it needs; hard-skip unless that descriptor is active, so
# a config can never accidentally run on the wrong mesh.
def _require_mesh_descriptor(basename):
    desc = os.environ.get("TT_MESH_GRAPH_DESC_PATH", "")
    if not desc.endswith(basename):
        pytest.skip(
            "moe_compute full-path config requires TT_MESH_GRAPH_DESC_PATH ending "
            f"in {basename}; set it (and a matching system_desc) to run on a galaxy."
        )


def _gen_moe_dispatch(
    num_dispatch, tokens_per_device, hidden, experts, k_sel, epd, seed
):
    """Replicate tt-metal gen_sparse_buffer_and_indices (ring orientation-agnostic).

    Returns:
      sparse_buffer: [num_dispatch, total_tokens, hidden] - per-device dispatched
        tokens (token (d,t) is placed at row d*tpd+t on every device that owns one
        of its selected experts; other rows are random garbage).
      indices_flat:  [total_tokens, k_sel] - all-gathered expert ids, ordered
        [d0t0, d0t1, ..., d1t0, ...] (== sparse-buffer row order).
      scores_flat:   [total_tokens, k_sel] - normalized routing scores.
    """
    torch.manual_seed(seed)
    total_tokens = tokens_per_device * num_dispatch
    original = (
        torch.rand(num_dispatch, tokens_per_device, hidden, dtype=torch.bfloat16) - 0.5
    )

    indices = torch.zeros(num_dispatch, tokens_per_device, k_sel, dtype=torch.uint16)
    for d in range(num_dispatch):
        for t in range(tokens_per_device):
            sel = torch.randperm(experts)[:k_sel]
            indices[d, t, :] = sel.to(torch.uint16)
    scores = (
        torch.rand(num_dispatch, tokens_per_device, k_sel, dtype=torch.float32) + 1e-5
    )
    scores = (scores / scores.sum(dim=-1, keepdim=True)).to(torch.bfloat16)

    sparse = torch.rand(num_dispatch, total_tokens, hidden, dtype=torch.bfloat16)
    for d in range(num_dispatch):
        for t in range(tokens_per_device):
            token_idx = d * tokens_per_device + t
            for k in range(k_sel):
                ge = int(indices[d, t, k].item())
                target = ge // epd
                sparse[target, token_idx, :] = original[d, t, :]

    indices_flat = indices.reshape(total_tokens, k_sel)
    scores_flat = scores.reshape(total_tokens, k_sel)
    return sparse, indices_flat, scores_flat


_RING = tt_runtime.runtime.FabricConfig.FABRIC_1D_RING
_LINEAR = tt_runtime.runtime.FabricConfig.FABRIC_1D

_DESC_1x8 = "single_galaxy_1x8_torus_graph_descriptor.textproto"
_DESC_1x8_LINEAR = "single_galaxy_1x8_linear_graph_descriptor.textproto"

_FULL_CONFIGS = [
    # id, descriptor, fabric_config, mesh_shape, cluster_axis, hidden, intermediate,
    # activation, has_bias, tokens_per_device, epd, k_sel
    #
    # The combine topology is resolved at runtime from the MGD + fabric_config
    # (torus MGD + FABRIC_1D_RING -> Ring; linear MGD + FABRIC_1D -> Linear).
    #
    # 1x8 torus ring (cluster_axis=1): tt-metal's correctness-validated _MODELS_1x8
    # shapes, restricted to the activations the op supports (silu/swiglu). gpt_oss
    # exercises a distinct routing (epd=4, k=4) and a larger hidden.
    (
        "h1280_n896_silu-1x8",
        _DESC_1x8,
        _RING,
        (1, 8),
        1,
        1280,
        896,
        "silu",
        False,
        32,
        2,
        6,
    ),
    (
        "h1280_n896_swiglu_bias-1x8",
        _DESC_1x8,
        _RING,
        (1, 8),
        1,
        1280,
        896,
        "swiglu",
        True,
        32,
        2,
        6,
    ),
    (
        "gpt_oss_h2880_swiglu_bias-1x8",
        _DESC_1x8,
        _RING,
        (1, 8),
        1,
        2880,
        2880,
        "swiglu",
        True,
        32,
        4,
        4,
    ),
    # 1x8 linear (cluster_axis=1): same shapes on tt-metal's 1x8-linear mesh
    # (FABRIC_1D), exercising the non-ring combine path.
    (
        "h1280_n896_silu-1x8-linear",
        _DESC_1x8_LINEAR,
        _LINEAR,
        (1, 8),
        1,
        1280,
        896,
        "silu",
        False,
        32,
        2,
        6,
    ),
]


@pytest.mark.parametrize("enable_trace", [False, True], ids=["notrace", "trace"])
@pytest.mark.parametrize(
    "mesh_desc, fabric_config, mesh_shape, cluster_axis, h_size, n_inter, activation, has_bias, tokens_per_device, epd, k_sel",
    [c[1:] for c in _FULL_CONFIGS],
    ids=[c[0] for c in _FULL_CONFIGS],
)
def test_moe_compute_full_path_verify(
    mesh_desc,
    fabric_config,
    mesh_shape,
    cluster_axis,
    h_size,
    n_inter,
    activation,
    has_bias,
    tokens_per_device,
    epd,
    k_sel,
    enable_trace,
    request,
):
    """Verify the moe_compute full (combine) path on a galaxy submesh.

    Runs selective tilize + W0/W1 + activation + W2 + the A2A
    selective-reduce-combine across the device ring (along cluster_axis) and
    checks the combine output against ttir_moe_compute_golden's combine branch.

    Inputs are PRESHARDED (no mesh_shard): production-faithful, and required for
    trace (mesh_shard host inputs land in the trace region, which forbids host
    tensors). The combine output is a presharded result (per-device local shape;
    no ShardToFull gather), so PCC is checked per device. Weights + mapping are
    tagged ``parameter`` so const-eval hoists the weight packers out of the trace.

    ``enable_trace`` runs the same path with TTNN trace capture/replay. bf4 device
    weights vs raw-bf16 golden put the correlation at the tt-metal SILU/SwiGLU
    block-PCC floor (~0.986).
    """
    _require_mesh_descriptor(mesh_desc)

    from golden.mapping import GoldenMapTensor, apply_sharding

    H_v = h_size
    N_v = n_inter
    num_devices = mesh_shape[0] * mesh_shape[1]
    num_dispatch = mesh_shape[cluster_axis]  # ring devices along cluster_axis
    experts = epd * num_devices
    tpd = tokens_per_device
    total_tokens = tpd * num_dispatch

    L = 1

    sparse, indices_flat, scores_flat = _gen_moe_dispatch(
        num_dispatch, tpd, H_v, experts, k_sel, epd, seed=7
    )

    # expert_mapping[d, e] = device owning expert e (= e // epd), per-device rows.
    # MUST be [num_devices, experts] (the tilize_writer consumes num_devices pages
    # of the mapping; a [1, experts] mapping deadlocks the pipeline). Matches
    # tt-metal gen_expert_mapping (rows identical, replicated to every device).
    mapping_full = torch.zeros(num_devices, experts, dtype=torch.uint16)
    for e in range(experts):
        mapping_full[:, e] = e // epd

    raw_w0 = torch.rand(L, epd, H_v, N_v, dtype=torch.bfloat16) - 0.5
    raw_w1 = torch.rand(L, epd, H_v, N_v, dtype=torch.bfloat16) - 0.5
    raw_w2 = torch.rand(L, epd, N_v, H_v, dtype=torch.bfloat16) - 0.5
    raw_b0 = (torch.rand(L, epd, N_v, dtype=torch.bfloat16) - 0.5) if has_bias else None
    raw_b1 = (torch.rand(L, epd, N_v, dtype=torch.bfloat16) - 0.5) if has_bias else None
    raw_b2 = (torch.rand(L, epd, H_v, dtype=torch.bfloat16) - 0.5) if has_bias else None

    sparse_full = sparse.reshape(num_devices, total_tokens, H_v)
    indices_full = indices_flat.reshape(1, total_tokens, k_sel)
    scores_full = scores_flat.reshape(1, total_tokens, k_sel)

    # moe_compute's per-device combine output: {k_sel, total_tokens/num_dispatch, H}.
    out_combine = (k_sel, total_tokens // num_dispatch, H_v)

    # Presharded shard_dims (len == mesh rank): mesh-dim -> tensor-dim, -1 = replicated.
    # sparse: cluster-axis mesh dim shards tensor dim 0; everything else replicated.
    # combine output: cluster-axis mesh dim shards token dim 1.
    sd_sparse = [-1, -1]
    sd_sparse[cluster_axis] = 0
    sd_rep = [-1, -1]
    sd_combine = [-1, -1]
    sd_combine[cluster_axis] = 1

    # GLOBAL shapes (the builder derives per-device local shapes for presharded args).
    in_shapes = [
        (num_devices, total_tokens, H_v),  # sparse
        (1, total_tokens, k_sel),  # indices
        (1, total_tokens, k_sel),  # scores
        (num_devices, experts),  # mapping
        (L, epd, H_v, N_v),  # w0
        (L, epd, H_v, N_v),  # w1
        (L, epd, N_v, H_v),  # w2
    ]
    in_types = [
        torch.bfloat16,
        torch.uint16,
        torch.bfloat16,
        torch.uint16,
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
    ]
    in_goldens = [
        sparse_full,
        indices_full,
        scores_full,
        mapping_full,
        raw_w0,
        raw_w1,
        raw_w2,
    ]
    # sparse is data-parallel sharded; everything else replicated.
    arg_sd = [sd_sparse, sd_rep, sd_rep, sd_rep, sd_rep, sd_rep, sd_rep]
    # input / parameter tags drive const-eval (weight packers hoisted out of trace).
    arg_types = [
        "input",
        "input",
        "input",
        "parameter",
        "parameter",
        "parameter",
        "parameter",
    ]
    if has_bias:
        in_shapes += [(L, epd, N_v), (L, epd, N_v), (L, epd, H_v)]
        in_types += [torch.bfloat16, torch.bfloat16, torch.bfloat16]
        in_goldens += [raw_b0, raw_b1, raw_b2]
        arg_sd += [sd_rep, sd_rep, sd_rep]
        arg_types += ["parameter", "parameter", "parameter"]

    presharded_args = {i: tuple(sd) for i, sd in enumerate(arg_sd)}

    def module(builder: TTIRBuilder):
        @builder.func(
            in_shapes,
            in_types,
            presharded_args=presharded_args,
            presharded_results={0: tuple(sd_combine)},
        )
        def moe_compute_full(*args):
            *operands, builder = args
            ms = builder._mesh_shape
            # Goldens set on GLOBAL tensors; distributed per presharded shard_dims
            # (sparse sharded, the rest replicated across the mesh).
            for operand, golden, sd in zip(operands, in_goldens, arg_sd):
                gmt = apply_sharding(
                    GoldenMapTensor({0: golden}, mesh_shape=ms), ms, tuple(sd)
                )
                builder._set_golden_tensor(operand, gmt)
            (
                sparse_op,
                idx_op,
                scr_op,
                map_op,
                w0_op,
                w1_op,
                w2_op,
                *bias_ops,
            ) = operands
            bias_0, bias_1, bias_2 = bias_ops if has_bias else (None, None, None)

            # Presharded inputs feed moe_compute directly; the presharded result is
            # returned with no ShardToFull gather.
            return builder.moe_compute(
                sparse_op,
                idx_op,
                scr_op,
                map_op,
                w0_op,
                w1_op,
                w2_op,
                layer_id=0,
                output_height_shard_dim=4,
                intermediate_size=N_v,
                cluster_axis=cluster_axis,
                output_shape=out_combine,
                output_type=torch.bfloat16,
                bias_0=bias_0,
                bias_1=bias_1,
                bias_2=bias_2,
                activation_function=activation,
            )

    pipeline_options = ["optimization-level=1"]
    if enable_trace:
        pipeline_options.append("enable-trace=true")

    compile_and_execute_ttir(
        module,
        target="ttnn",
        mesh_name="mesh",
        device=DeferredDevice(request),
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        # Greedy layout optimizer leaves moe_compute's combine output at the
        # ROW_MAJOR/DRAM spec set by its combine_output operand workaround.
        pipeline_options=pipeline_options,
        argument_types_string="moe_compute_full=" + ",".join(arg_types),
        # bf4 device weights vs raw-bf16 golden -> tt-metal SILU/SwiGLU block-PCC
        # floor (~0.986). Relax the program-level PCC accordingly.
        pcc=0.98,
        **get_request_kwargs(request),
    )
