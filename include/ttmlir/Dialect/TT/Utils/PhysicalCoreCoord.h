// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TT_UTILS_PHYSICALCORECOORD_H
#define TTMLIR_DIALECT_TT_UTILS_PHYSICALCORECOORD_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

namespace mlir::tt {
struct PhysicalCoreCoord {
  std::int64_t d = 0;
  std::int64_t y = 0;
  std::int64_t x = 0;

  PhysicalCoreCoord() = default;
  PhysicalCoreCoord(std::int64_t d, std::int64_t y, std::int64_t x)
      : d(d), y(y), x(x) {}
  PhysicalCoreCoord(ArrayRef<std::int64_t> coord)
      : d(coord[0]), y(coord[1]), x(coord[2]) {}

  std::int64_t &operator[](std::size_t i) {
    assert(i < 3);
    return i == 0 ? d : i == 1 ? y : x;
  }

  std::int64_t operator[](std::size_t i) const {
    assert(i < 3);
    return i == 0 ? d : i == 1 ? y : x;
  }

  bool operator==(PhysicalCoreCoord const &other) const {
    return d == other.d && y == other.y && x == other.x;
  }
};

class PhysicalCoreCoordMapping {
public:
  PhysicalCoreCoordMapping(ArrayRef<tt::ChipDescAttr> chipDescs) {
    ArrayRef<int64_t> firstChipGrid = chipDescs.front().getGrid();
    assert(firstChipGrid.size() == 2);
    grid = {firstChipGrid[0], firstChipGrid[1]};

    workers.reserve(chipDescs.size() * grid[0] * grid[1]);
    for (auto chipDesc : chipDescs) {
      auto chipGrid = chipDesc.getGrid();
      assert(chipGrid == firstChipGrid);
      ChipPhysicalCoresAttr chipPhysicalCores = chipDesc.getChipPhysicalCores();
      assert(chipPhysicalCores.getWorker().size() ==
             static_cast<size_t>(grid[0] * grid[1]));
      for (auto worker : chipPhysicalCores.getWorker()) {
        workers.push_back({worker.getY(), worker.getX()});
      }
    }
    assert(workers.size() == chipDescs.size() * grid[0] * grid[1]);
  }

  std::array<int64_t, 2> operator[](PhysicalCoreCoord coord) const {
    return workers[coord.d * grid[0] * grid[1] + coord.y * grid[1] + coord.x];
  }

private:
  std::array<int64_t, 2> grid;
  SmallVector<std::array<int64_t, 2>> workers;
};
} // namespace mlir::tt

// Make PhysicalCoreCoord hashable.
namespace llvm {
template <> struct DenseMapInfo<mlir::tt::PhysicalCoreCoord> {
  static mlir::tt::PhysicalCoreCoord getEmptyKey() {
    return mlir::tt::PhysicalCoreCoord{-1, -1, -1};
  }

  static mlir::tt::PhysicalCoreCoord getTombstoneKey() {
    return mlir::tt::PhysicalCoreCoord{-2, -2, -2};
  }

  static unsigned getHashValue(mlir::tt::PhysicalCoreCoord coord) {
    return llvm::hash_combine(coord.d, coord.y, coord.x);
  }

  static bool isEqual(mlir::tt::PhysicalCoreCoord lhs,
                      mlir::tt::PhysicalCoreCoord rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif
