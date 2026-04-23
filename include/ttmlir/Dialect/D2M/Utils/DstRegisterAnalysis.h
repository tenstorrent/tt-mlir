// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_DSTREGISTERANALYSIS_H
#define TTMLIR_DIALECT_D2M_UTILS_DSTREGISTERANALYSIS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::tt::d2m {
class GenericOp;
} // namespace mlir::tt::d2m

namespace mlir::tt::d2m::utils {

struct DSTPackingPerResultInfo {
  int64_t numDstFlips = 0;
  int64_t numTilesPerFlip = 0;
  // Per-result tile count per outer iteration. May differ across results in
  // a region with heterogeneous output shapes (e.g. eltwise + reduction);
  // for homogeneous outputs it equals the region's numTilesPerResult.
  int64_t numTilesPerResult = 0;
};

struct DSTPackingRegionInfo {
  llvm::SmallDenseMap<Value, DSTPackingPerResultInfo> perResult;
  int64_t numTilesPerResult = 0;
  int64_t numOuterLoopIters = 0;
};

class DSTPackingInfo {
public:
  bool empty() const { return perRegion.empty(); }
  size_t size() const { return perRegion.size(); }

  const DSTPackingRegionInfo *lookup(Region *region) const;

private:
  friend class DstRegisterAnalysis;
  llvm::DenseMap<Region *, DSTPackingRegionInfo> perRegion;
};

class DstRegisterAnalysis {
public:
  /// \param op                      Root operation to analyze.
  /// \param maxDstPhysicalSizeTiles If non-zero, upper-bounds numTilesPerFlip
  ///                                for every op, overriding the device
  ///                                descriptor value.
  DstRegisterAnalysis(Operation *op, unsigned maxDstPhysicalSizeTiles = 0);

  const DSTPackingInfo *lookup(d2m::GenericOp generic) const;

private:
  llvm::DenseMap<Operation *, DSTPackingInfo> packingInfoMap;
};

} // namespace mlir::tt::d2m::utils

#endif
