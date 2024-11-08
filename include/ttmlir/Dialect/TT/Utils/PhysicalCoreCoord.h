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
    switch (i) {
    case 0:
      return d;
    case 1:
      return y;
    case 2:
      return x;
    default:
      llvm_unreachable("invalid index");
    }
  }

  std::int64_t operator[](std::size_t i) const {
    return (*const_cast<PhysicalCoreCoord *>(this))[i];
  }

  bool operator==(PhysicalCoreCoord const &other) const {
    return d == other.d && y == other.y && x == other.x;
  }

  std::string toString() const {
    return std::to_string(d) + " " + std::to_string(y) + " " +
           std::to_string(x);
  }
};

class PhysicalCoreCoordMapping {
public:
  static PhysicalCoreCoordMapping
  getWorkerMapping(ArrayRef<unsigned> chipIds,
                   ArrayRef<tt::ChipDescAttr> chipDescs) {
    SmallVector<std::array<int64_t, 2>> physCores;
    ArrayRef<int64_t> firstChipGrid = chipDescs[chipIds.front()].getGrid();
    assert(firstChipGrid.size() == 2);
    std::array<int64_t, 2> grid = {firstChipGrid[0], firstChipGrid[1]};

    physCores.reserve(chipIds.size() * grid[0] * grid[1]);
    for (auto chipId : chipIds) {
      auto chipDesc = chipDescs[chipId];
      auto chipGrid = chipDesc.getGrid();
      assert(chipGrid == firstChipGrid);
      ChipPhysicalCoresAttr chipPhysicalCores = chipDesc.getChipPhysicalCores();
      assert(chipPhysicalCores.getWorker().size() ==
             static_cast<size_t>(grid[0] * grid[1]));
      for (auto worker : chipPhysicalCores.getWorker()) {
        physCores.push_back({worker.getY(), worker.getX()});
      }
    }
    assert(physCores.size() == chipIds.size() * grid[0] * grid[1]);
    return PhysicalCoreCoordMapping(grid, physCores);
  }

  static PhysicalCoreCoordMapping
  getDramMapping(ArrayRef<unsigned> chipIds,
                 ArrayRef<tt::ChipDescAttr> chipDescs) {
    ArrayRef<CoreCoordAttr> firstChipDramCores =
        chipDescs[chipIds.front()].getChipPhysicalCores().getDram();

    std::array<int64_t, 2> grid = {
        1, static_cast<int64_t>(firstChipDramCores.size())};
    SmallVector<std::array<int64_t, 2>> physCores;
    physCores.reserve(chipIds.size() * grid[0] * grid[1]);
    for (auto chipId : chipIds) {
      auto chipDesc = chipDescs[chipId];
      ChipPhysicalCoresAttr chipPhysicalCores = chipDesc.getChipPhysicalCores();
      assert(chipPhysicalCores.getDram().size() ==
             static_cast<size_t>(grid[0] * grid[1]));
      for (auto dram : chipPhysicalCores.getDram()) {
        physCores.push_back({dram.getY(), dram.getX()});
      }
    }
    assert(physCores.size() == chipIds.size() * grid[0] * grid[1]);
    return PhysicalCoreCoordMapping(grid, physCores);
  }

  static PhysicalCoreCoordMapping
  getMemorySpaceMapping(ArrayRef<unsigned> chipIds,
                        ArrayRef<tt::ChipDescAttr> chipDescs,
                        MemorySpace memorySpace) {
    switch (memorySpace) {
    case MemorySpace::DeviceL1:
      return getWorkerMapping(chipIds, chipDescs);
    case MemorySpace::DeviceDRAM:
      return getDramMapping(chipIds, chipDescs);
    default:
      llvm_unreachable("unsupported memory space");
    }
  }

  std::array<int64_t, 2> operator[](PhysicalCoreCoord coord) const {
    return physCores[coord.d * grid[0] * grid[1] + coord.y * grid[1] + coord.x];
  }

private:
  PhysicalCoreCoordMapping(std::array<int64_t, 2> grid,
                           SmallVector<std::array<int64_t, 2>> physCores)
      : grid(grid), physCores(physCores) {}

private:
  std::array<int64_t, 2> grid;
  SmallVector<std::array<int64_t, 2>> physCores;
};
} // namespace mlir::tt

// Make PhysicalCoreCoord hashable.
namespace llvm {
template <>
struct DenseMapInfo<mlir::tt::PhysicalCoreCoord> {
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
