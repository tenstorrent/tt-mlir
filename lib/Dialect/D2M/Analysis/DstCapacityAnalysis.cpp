// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DstCapacityAnalysis.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include <limits>

#define DEBUG_TYPE "dst-capacity-analysis"

/// Constructs a DstCapacityAnalysis by examining all GenericOp operations
/// within the given operation and computing the minimum DST capacity based
/// on the largest element size used in DST accesses.
///
/// \p op The root operation (typically a func::FuncOp) to analyze.
/// \p fullSyncEn Enable full sync mode (true = max capacity, false = half
///   capacity).
///   true: f16/bf16=16 tiles, f32=8 tiles
///   false: f16/bf16=8 tiles, f32=4 tiles
/// \p overridePhysicalSize Override physical DST size, or 0 to use chip
///   default.
mlir::tt::d2m::DstCapacityAnalysis::DstCapacityAnalysis(
    mlir::Operation *op, bool fullSyncEn, unsigned overridePhysicalSize) {
  uint32_t minCapacity = std::numeric_limits<uint32_t>::max();
  bool foundGenericOp = false;
  mlir::Type largestDstType = nullptr;

  op->walk([&](mlir::tt::d2m::GenericOp genericOp) {
    foundGenericOp = true;
    for (uint32_t regionIndex = 0; regionIndex < genericOp.getNumRegions();
         regionIndex++) {
      // Only consider compute regions (skip data movement regions).
      if (genericOp.getRegionThreadType(regionIndex) !=
          mlir::tt::d2m::ThreadType::Compute) {
        continue;
      }
      mlir::Region *region = &genericOp.getRegion(regionIndex);

      largestDstType =
          mlir::tt::d2m::utils::getRegionLargestDstElemType(*region);

      // Query the hardware configuration to get DST capacity for largest
      // element size.
      uint32_t currentCapacity =
          mlir::tt::ttcore::getOpChipDescAttr(genericOp).getDstLogicalSizeTiles(
              largestDstType, fullSyncEn, overridePhysicalSize);

      minCapacity = std::min(minCapacity, currntCapacity);
    }

    // Note: Do not remove this debug statement, it is used for testing. It is
    // only emitted when the DEBUG_TYPE is set to "dst-capacity-analysis"
    // (command line option --debug-only=dst-capacity-analysis).
    LLVM_DEBUG(llvm::dbgs() << "DST Capacity Analysis: Largest DST type: "
                            << largestDstType << " fullSyncEn: " << fullSyncEn
                            << " Capacity: " << minCapacity << "\n");
  });

  if (!foundGenericOp) {
    minDstCapacity = kDefaultDstCapacity;
  } else {
    minDstCapacity = minCapacity;
  }
}
