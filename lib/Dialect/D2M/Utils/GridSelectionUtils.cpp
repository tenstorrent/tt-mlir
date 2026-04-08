// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/GridSelectionUtils.h"
#include "ttmlir/Asserts.h"

namespace mlir::tt::d2m::utils {

llvm::SmallVector<int64_t>
computeOptimalBlockShardedGrid(ArrayRef<int64_t> physicalShape,
                               ArrayRef<int64_t> targetSquareGridShape) {
  llvm::SmallVector<int64_t> grid(physicalShape.size(), 1);

  TT_assert(targetSquareGridShape.size() == 2u);
  TT_assert(physicalShape.size() >= targetSquareGridShape.size());

  // For tensors with rank > 2, only shard across the last two dimensions
  // (which correspond to the 2D worker grid).
  const size_t dimOffset = physicalShape.size() - targetSquareGridShape.size();

  for (size_t i = 0; i < targetSquareGridShape.size(); ++i) {
    const int64_t dim = physicalShape[dimOffset + i];
    TT_assert(dim > 0);
    // Search downward from the target grid size to find the largest divisor.
    for (int64_t g = targetSquareGridShape[i]; g > 0; g--) {
      if (dim % g == 0) {
        grid[dimOffset + i] = g;
        break;
      }
    }
  }

  TT_assert(grid.size() == physicalShape.size());
  return grid;
}

} // namespace mlir::tt::d2m::utils
