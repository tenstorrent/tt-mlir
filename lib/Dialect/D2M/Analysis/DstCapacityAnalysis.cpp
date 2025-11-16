// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DstCapacityAnalysis.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"

#include <limits>

#define DEBUG_TYPE "dst-capacity-analysis"

using namespace mlir;
using namespace mlir::tt;

/// Constructs a DstCapacityAnalysis by examining all GenericOp operations
/// within the given operation and computing the minimum DST capacity based
/// on the largest element size used in DST accesses.
///
/// \p op The root operation (typically a func::FuncOp) to analyze.
/// \p fullSyncEn Enable full sync mode (true = max capacity, false = half capacity).
/// \p overridePhysicalSize Override physical DST size, or 0 to use chip default.
d2m::DstCapacityAnalysis::DstCapacityAnalysis(Operation *op, bool fullSyncEn,
                                              unsigned overridePhysicalSize) {
  uint32_t minCapacity = std::numeric_limits<uint32_t>::max();
  bool foundGenericOp = false;
  Type largestDstType = nullptr;

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

      largestDstType = d2m::utils::getRegionLargestDstElemType(*region);

      // Query the hardware configuration to get DST capacity for largest
      // element size.
      uint32_t currentCapacity =
          ttcore::getOpChipDescAttr(genericOp).getDstLogicalSizeTiles(
              largestDstType, fullSyncEn, overridePhysicalSize);

      minCapacity = std::min(minCapacity, currentCapacity);
    }
    LLVM_DEBUG(llvm::dbgs()
               << "DST Capacity Analysis: Largest DST type: " << largestDstType
               << " fullSyncEn: " << fullSyncEn
               << " Capacity: " << minCapacity << "\n");
  });

  // If no GenericOp operations were found, use a conservative default.
  // This provides a reasonable fallback for functions with no compute regions.
  if (!foundGenericOp) {
    minDstCapacity = kDefaultDstCapacity;
  } else {
    minDstCapacity = minCapacity;
  }
}
