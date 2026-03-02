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

struct DSTPackingPerResultInfo {
  int64_t numDstFlips = 0;
  int64_t numTilesPerFlip = 0;
};

struct DSTPackingInfo {
  llvm::SmallDenseMap<Value, DSTPackingPerResultInfo> perResult;
  int64_t numTilesPerResult = 0;
  int64_t numOuterLoopIters = 0;
};

// Analyze linalg.generic ops in a unified d2m.generic region and compute legal
// DST packing values that maximize common outer loop iterations.
DSTPackingInfo analyzeGenericForDSTPacking(d2m::GenericOp generic);

} // namespace mlir::tt::d2m::utils

#endif
