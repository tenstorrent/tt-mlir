// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_DSTCAPACITYANALYSIS_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_DSTCAPACITYANALYSIS_H

#include "mlir/IR/Operation.h"
#include "mlir/Pass/AnalysisManager.h"

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
/// (physical DST slices) that can be allocated. By using the minimum across all
/// ops, we ensure that the allocation strategy works correctly regardless of
/// which operation's region is executing.
///
/// Usage:
///   DstCapacityAnalysis analysis(funcOp);
///   uint32_t numColors = analysis.getMinDstCapacity();
class DstCapacityAnalysis {
public:
  /// Constructs the analysis by walking the operation to find the minimum
  /// DST capacity available.
  ///
  /// \p op The operation (typically a FuncOp) to analyze. All nested
  ///      d2m::GenericOp operations are examined.
  DstCapacityAnalysis(Operation *op);

  /// Returns the minimum DST capacity (in tiles) available for this function.
  ///
  /// This value represents the maximum number of DST tiles that can be
  /// simultaneously allocated across all operations in the function. It is
  /// computed as the minimum capacity across all GenericOp compute regions,
  /// ensuring correctness regardless of operation type or data types used.
  ///
  /// \return The minimum DST capacity in tiles.
  uint32_t getMinDstCapacity() const { return minDstCapacity; }

private:
  uint32_t minDstCapacity;
};

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_DSTCAPACITYANALYSIS_H
