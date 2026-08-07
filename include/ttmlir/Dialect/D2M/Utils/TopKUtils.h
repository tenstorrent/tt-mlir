// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_TOPKUTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_TOPKUTILS_H

#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include <utility>

namespace mlir::tt::d2m::utils {

// The index element type the topk's user asked for, set by ttir-to-d2m because
// every topk buffer the lowering builds carries i32 indices regardless.
constexpr llvm::StringLiteral kTopkIndexTypeAttr = "d2m.topk_index_type";
// Topk scratch allocations, set by d2m-insert-scratch-buffers and consumed by
// d2m-decompose-topk.
constexpr llvm::StringLiteral kTopkIndexBufferAttr = "d2m.topk_index_buffer";
constexpr llvm::StringLiteral kTopkLaneBufferAttr = "d2m.topk_lane_buffer";
// Set on a leaf generic by d2m-grid-selection and consumed by d2m-lower-topk.
constexpr llvm::StringLiteral kTopKPlanAttr = "d2m.topk_plan";
// Set by d2m-grid-selection on the op ending the laid-out (and masked) input
// chain it emits, so d2m-lower-topk can find it without the placeholder leaf
// having to hold it as an operand.
constexpr llvm::StringLiteral kTopKInputAttr = "d2m.topk_input";
// Must exceed 1: a single row folds arange_block's compute root loop away.
constexpr int64_t kTopkLaneTileRows = 2;

inline void setTopkIndexType(Operation *op, Type indexElementType) {
  op->setAttr(kTopkIndexTypeAttr, TypeAttr::get(indexElementType));
}

// Any field left unplaced is computed from `layoutedInput`'s own type instead.
struct LeafTopKBuffers {
  // Placed only when `dim == 1`; leaving it unplaced skips the transpose.
  PlacedBuffer transpose;
  PlacedBuffer scratch;
  PlacedBuffer values;
  PlacedBuffer indices;
};

// Emits the per-core topk over an already tiled+layouted `layoutedInput`. A
// placed `buffers.transpose` transposes first, since `topk_block` sorts down
// tile columns and `dim == 1` puts the sort dim on tile rows.
std::pair<Value, Value> emitLeafTopk(RewriterBase &rewriter, Location loc,
                                     Value layoutedInput, int32_t k,
                                     int32_t dim, int64_t reductionDimSize,
                                     const LeafTopKBuffers &buffers = {});

} // namespace mlir::tt::d2m::utils

#endif
