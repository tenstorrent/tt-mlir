// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DstCapacityAnalysis.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"

#include <limits>

using namespace mlir;
using namespace mlir::tt;

/// Constructs a DstCapacityAnalysis by examining all GenericOp operations
/// within the given operation and computing the minimum DST capacity based
/// on the largest element size used in DST accesses.
///
/// \p op The root operation (typically a func::FuncOp) to analyze.
d2m::DstCapacityAnalysis::DstCapacityAnalysis(Operation *op) {
  uint32_t minCapacity = std::numeric_limits<uint32_t>::max();
  bool foundGenericOp = false;

  // Walk all GenericOp operations in the operation tree.
  op->walk([&](d2m::GenericOp genericOp) {
    foundGenericOp = true;
    for (uint32_t regionIndex = 0; regionIndex < genericOp.getNumRegions();
         regionIndex++) {
      // Only consider compute regions (skip data movement and other regions).
      if (genericOp.getRegionThreadType(regionIndex) !=
          d2m::ThreadType::Compute) {
        continue;
      }
      Region *region = &genericOp.getRegion(regionIndex);

      Type largestDstType = d2m::utils::getRegionLargestDstElemType(*region);

      // Query the hardware configuration to get DST capacity for largest
      // element size.
      uint32_t currentCapacity =
          ttcore::getOpChipDescAttr(genericOp).getDstLogicalSizeTiles(
              largestDstType, false, 0);

      minCapacity = std::min(minCapacity, currentCapacity);
    }
  });

  // If no GenericOp operations were found, use a conservative default.
  // This provides a reasonable fallback for functions with no compute regions.
  if (!foundGenericOp) {
    minDstCapacity = kDefaultDstCapacity;
  } else {
    minDstCapacity = minCapacity;
  }
}
