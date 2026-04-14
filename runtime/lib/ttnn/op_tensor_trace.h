// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

namespace tt::target::ttnn {
struct Operation;
} // namespace tt::target::ttnn

namespace tt::runtime::ttnn {

class ProgramContext;

// When TT_RUNTIME_OP_TENSOR_TRACE is set to 1/true/yes, logs each TTNN op's
// inputs (captured just before the op) and outputs (after the op), using
// host copies for numeric all-zero checks.
//
// When TT_RUNTIME_OP_TENSOR_TRACE_FAST=1, input capture and output host reads
// are skipped for most ops; full per-device tracing still runs for mesh and
// collective ops (mesh_shard, mesh_partition, all_gather, all_reduce,
// reduce_scatter, distribute_tensor, all_to_all_*, point_to_point,
// layer_norm_pre_all_gather, aggregate_tensor).
//
// When TT_RUNTIME_OP_TENSOR_TRACE_MESH_DETAIL is also set, mesh_shard
// (FullToShardShape) and mesh_partition ops emit extra MESH lines: formatted
// tensor previews, per-device slice ranges vs the global input, fnv1a64 payload
// checksums, absolute-value sums (|Σ|), and conservation checks.
//
// For traced ops (or all ops when FAST is off), OPSUM lines list per-device
// |Σ| in rows of eight devices, plus all_gather |Σ| checks when applicable.
//
// Row/column tensor previews inside mesh_shard / mesh_partition (logTensorPreview)
// are omitted unless TT_RUNTIME_OP_TENSOR_TRACE_MESH_PREVIEWS=1; they are not
// enabled by TT_RUNTIME_OP_TENSOR_TRACE_VERBOSE.
bool opTensorTraceEnvEnabled();

std::vector<std::optional<bool>> opTensorTraceCaptureInputZeroState(
    const ::tt::target::ttnn::Operation *op, ProgramContext *programContext);

void opTensorTraceLogCompletedOp(
    const ::tt::target::ttnn::Operation *op, ProgramContext *programContext,
    const std::vector<std::optional<bool>> &inputAllZero);

// When TT_RUNTIME_OP_TENSOR_TRACE_TOPK_DUMP_DIR is set to a directory path,
// dumps topk inputs/outputs as .npy files loadable by numpy.load().
// Works independently of TT_RUNTIME_OP_TENSOR_TRACE — no per-op logging needed.
bool opTensorTraceTopKDumpEnabled();

void opTensorTraceTopKDump(const ::tt::target::ttnn::Operation *op,
                           ProgramContext *programContext);

// General op dump: dumps inputs/outputs as .npy for topk and all ops
// downstream of topk (following global_id links through the dataflow graph).
// Writes op-specific parameters to *_meta.json alongside the .npy files.
// Uses the same TT_RUNTIME_OP_TENSOR_TRACE_TOPK_DUMP_DIR directory.
bool opTensorTraceOpDumpEnabled();

void opTensorTraceOpDump(const ::tt::target::ttnn::Operation *op,
                         ProgramContext *programContext);

} // namespace tt::runtime::ttnn
