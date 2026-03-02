// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_DSTREGISTERANALYSIS_H
#define TTMLIR_DIALECT_D2M_UTILS_DSTREGISTERANALYSIS_H

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::tt::d2m {
class GenericOp;
} // namespace mlir::tt::d2m

namespace mlir::tt::d2m::utils {

struct DSTPackingInfo {
  int64_t num_tiles_per_flip = 0;
  int64_t num_dst_flips = 0;
  int64_t num_outer_loop_iters = 0;
};

using DSTPackingResult = DSTPackingInfo;
using DSTPackingResultsMap = llvm::SmallDenseMap<Value, DSTPackingResult>;

// Analyze linalg.generic ops in a unified d2m.generic region and compute legal
// DST packing values that maximize common outer loop iterations.
DSTPackingResultsMap analyzeGenericForDSTPacking(d2m::GenericOp generic);

} // namespace mlir::tt::d2m::utils

#endif
