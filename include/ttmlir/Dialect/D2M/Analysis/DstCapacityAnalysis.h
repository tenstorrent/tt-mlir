// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_DSTCAPACITYANALYSIS_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_DSTCAPACITYANALYSIS_H

#include "mlir/IR/Operation.h"

#include <cstdint>

namespace mlir::tt::d2m {

/// Default DST capacity if no GenericOp operations are found.
constexpr uint32_t kDefaultDstCapacity = 8;

/// Analyzes the minimum DST (Destination register) capacity available for a
/// function based on all contained GenericOp operations.
///
/// This analysis walks all d2m::GenericOp operations within a given function
/// and computes the minimum DST tile capacity across all of them. The capacity
/// varies depending on the element types used in each operation, since
/// different data types consume different amounts of DST space on Tenstorrent
/// hardware.
///
/// This information is required by the graph coloring register allocation pass
/// (D2MInsertDstRegisterGC) to determine the number of available colors
/// (physical DST slices) that can be allocated. We use the minimum capacity
/// across all ops to find the most constrained operation (typically the one
/// with the largest element type). This ensures the allocation never exceeds
/// what any operation can actually use, preventing allocation failures.
///
/// Usage:
///   DstCapacityAnalysis analysis(funcOp, /*fullSyncEn=*/true,
///   /*overridePhysicalSize=*/0); uint32_t numColors =
///   analysis.getMinDstCapacity();
class DstCapacityAnalysis {
public:
  /// Constructs the analysis by walking the operation to find the minimum
  /// DST capacity available.
  ///
  /// \p op The operation (typically a FuncOp) to analyze. All nested
  ///      d2m::GenericOp operations are examined.
  /// \p fullSyncEn Enable full sync mode (true = max capacity, false = half
  /// capacity).
  ///               true: f16/bf16=16 tiles, f32=8 tiles
  ///               false: f16/bf16=8 tiles, f32=4 tiles
  /// \p overridePhysicalSize Override physical DST size (in tiles), or 0 to use
  /// chip default.
  DstCapacityAnalysis(Operation *op, bool fullSyncEn = true,
                      unsigned overridePhysicalSize = 0);

  /// Returns the minimum DST capacity (in tiles) available for this function.
  ///
  /// This value represents the safe upper bound on the number of DST tiles
  /// that can be simultaneously allocated. It is computed by finding the most
  /// constrained GenericOp (the one with the smallest capacity, typically due
  /// to using larger element types). Using this minimum ensures that the
  /// allocation strategy will work correctly for all operations in the
  /// function.
  ///
  /// \return The minimum DST capacity in tiles.
  uint32_t getMinDstCapacity() const { return minDstCapacity; }

private:
  uint32_t minDstCapacity;
};

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_DSTCAPACITYANALYSIS_H
