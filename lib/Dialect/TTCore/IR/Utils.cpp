// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinOps.h"

#include <cmath>

namespace mlir::tt::ttcore {

SystemDescAttr getCurrentScopeSystemDesc(mlir::Operation *op) {
  // Find the top level ModuleOp which carries the system desc.
  ModuleOp moduleOp = dyn_cast<ModuleOp>(op);
  if (!moduleOp) {
    moduleOp = op->getParentOfType<ModuleOp>();
  }
  auto systemDesc =
      moduleOp->getAttrOfType<SystemDescAttr>(SystemDescAttr::name);
  assert(systemDesc && "expected system desc to be present on the module");
  return systemDesc;
}

DeviceOp lookupDeviceOp(Operation *op, SymbolRefAttr deviceName) {
  return SymbolTable::lookupNearestSymbolFrom<DeviceOp>(op, deviceName);
}

DeviceOp lookupDeviceOp(Operation *op, llvm::StringRef deviceName) {
  return lookupDeviceOp(op, SymbolRefAttr::get(op->getContext(), deviceName));
}

DeviceAttr lookupDevice(Operation *op, SymbolRefAttr deviceName) {
  auto deviceOp = lookupDeviceOp(op, deviceName);
  assert(deviceOp && "expected device op to be present");
  return deviceOp.getDeviceAttr();
}

DeviceAttr lookupDevice(Operation *op, llvm::StringRef deviceName) {
  auto deviceOp = lookupDeviceOp(op, deviceName);
  assert(deviceOp && "expected device op to be present");
  return deviceOp.getDeviceAttr();
}

ChipDescAttr getOpChipDescAttr(Operation *op) {
  auto device = ttcore::lookupDevice(op);
  auto chipIds = device.getChipIds();
  auto systemDesc = ttcore::getCurrentScopeSystemDesc(op);
  return systemDesc.getChipDesc(chipIds[0]);
}

mlir::memref::GlobalOp createGlobal(ModuleOp moduleOp, StringRef name,
                                    mlir::MemRefType type, ElementsAttr value,
                                    bool constant, bool privateVisibility,
                                    size_t alignment) {
  SymbolTable symbolTable(moduleOp);

  if (constant && privateVisibility) {
    // Check if a global with the same value already exists.
    for (Operation &op : moduleOp.getRegion().getOps()) {
      auto globalOp = dyn_cast<memref::GlobalOp>(&op);
      if (!globalOp) {
        continue;
      }
      if (!globalOp.getInitialValue().has_value()) {
        continue;
      }
      bool isConstant = globalOp.getConstant();
      if (!isConstant) {
        continue;
      }
      uint64_t opAlignment = globalOp.getAlignment().value_or(0);
      Attribute initialValue = globalOp.getInitialValue().value();
      if (opAlignment == alignment && initialValue == value) {
        return globalOp;
      }
    }
  }

  auto getUniqueSymbolName = [&]() {
    if (!symbolTable.lookup(name)) {
      return name.str();
    }

    int uid = 0;
    while (symbolTable.lookup((Twine(name) + "_" + Twine(uid)).str())) {
      uid++;
    }
    return (Twine(name) + "_" + Twine(uid)).str();
  };

  auto symbolName = getUniqueSymbolName();

  OpBuilder builder(moduleOp.getRegion());
  auto global = builder.create<memref::GlobalOp>(
      moduleOp->getLoc(), symbolName,
      /*sym_visibility*/
      builder.getStringAttr(privateVisibility ? "private" : "public"), type,
      value, constant,
      alignment ? builder.getI64IntegerAttr(alignment) : nullptr);

  symbolTable.insert(global);

  // Move the global to the beginning of the module, just after the device.
  global->moveAfter(lookupDeviceOp(moduleOp));

  return global;
}

mlir::memref::GlobalOp createGlobal(ModuleOp moduleOp, mlir::MemRefType type,
                                    ElementsAttr value, bool constant,
                                    bool privateVisibility, size_t alignment) {
  SmallString<64> symbolName;
  llvm::raw_svector_ostream os(symbolName);
  if (privateVisibility) {
    os << "__";
  }
  if (constant) {
    os << "constant_";
  }
  llvm::interleave(type.getShape(), os, "x");
  os << "x" << type.getElementType();
  return createGlobal(moduleOp, symbolName, type, value, constant,
                      privateVisibility, alignment);
}

bool isTiled(RankedTensorType tensorType) {
  return mlir::isa<TileType>(tensorType.getElementType());
}

ArrayRef<int64_t> getTensorTileShape(RankedTensorType tensorType) {
  auto tileType = mlir::cast<TileType>(tensorType.getElementType());
  return tileType.getShape();
}

ArrayRef<int64_t> getTensorTileShapeOrEmpty(RankedTensorType tensorType) {
  return isTiled(tensorType) ? getTensorTileShape(tensorType)
                             : ArrayRef<int64_t>{};
}

Type getOperandInnerElementType(const mlir::Value operand) {
  auto elemType = operand.getType();
  if (auto memRefType = mlir::dyn_cast<MemRefType>(elemType);
      memRefType != nullptr) {
    elemType = memRefType.getElementType();
  }
  // We could have a memref of tiles, so this needs to be the second query.
  if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elemType);
      tileType != nullptr) {
    elemType = tileType.getElementType();
  }
  assert(elemType.isIntOrFloat());
  return elemType;
}

llvm::SmallVector<int64_t, 2> collapseGridTo2D(ArrayRef<int64_t> gridShape) {
  if (gridShape.size() <= 2) {
    return llvm::to_vector(gridShape);
  }

  // Collapse all leading dimensions into the first dimension.
  // e.g., [3, 2, 4] -> [6, 4].
  int64_t collapsedHeight = 1;
  for (size_t i = 0; i < gridShape.size() - 1; ++i) {
    collapsedHeight *= gridShape[i];
  }
  int64_t width = gridShape.back();

  return {collapsedHeight, width};
}

llvm::SmallVector<int64_t>
findLegalPhysicalGridForVolume(int64_t gridVolume,
                               ArrayRef<int64_t> targetGridShape) {
  assert(gridVolume > 0 && "Grid volume must be positive");
  assert(targetGridShape.size() >= 2u &&
         "Target grid shape must provide at least two dimensions");
  assert((targetGridShape[0] > 0 && targetGridShape[1] > 0) &&
         "Target grid dimensions must be positive");

  auto fitsTarget = [&](int64_t dimY, int64_t dimX) {
    return dimY <= targetGridShape[0] && dimX <= targetGridShape[1];
  };

  int64_t y = 1;
  // Find the largest factor of grid volume that is <= sqrt(gridVolume).
  for (int64_t i = static_cast<int64_t>(std::sqrt(gridVolume)); i > 0; --i) {
    if (gridVolume % i == 0) {
      int64_t candidateY = i;
      int64_t candidateX = gridVolume / i;
      if (fitsTarget(candidateY, candidateX)) {
        return {candidateY, candidateX};
      }
      if (fitsTarget(candidateX, candidateY)) {
        return {candidateX, candidateY};
      }
      if (y == 1) {
        y = candidateY;
      }
    }
  }
  return {};
}

llvm::SmallVector<int64_t, 2>
collapseToPhysicalGrid2D(ArrayRef<int64_t> gridShape,
                         ArrayRef<int64_t> deviceGridShape) {
  // Compute the volume of the virtual grid
  int64_t volume = 1;
  for (int64_t dim : gridShape) {
    volume *= dim;
  }

  // Try to find an optimal factorization (matches main's behavior).
  // This finds factors near sqrt to balance Y and X dimensions.
  auto result = findLegalPhysicalGridForVolume(volume, deviceGridShape);
  if (!result.empty()) {
    return result;
  }

  // Fallback: if no factorization fits (time-multiplexing needed),
  // use leading-dimension collapse which may exceed device bounds.
  auto physGrid = collapseGridTo2D(gridShape);
  assert(physGrid.size() == 2 && "Expected 2D grid after collapse");
  return physGrid;
}

static MemRefType getMemRefType(Type type, bool isView,
                                std::optional<MetalLayoutAttr> hostInfo) {
  auto tensorType = mlir::cast<mlir::RankedTensorType>(type);
  MLIRContext *ctx = tensorType.getContext();
  auto tensorMeshAttr =
      mlir::dyn_cast_if_present<TensorMeshAttr>(tensorType.getEncoding());
  HostLayoutAttr hostLayout = nullptr;

  if (hostInfo.has_value()) {
    // Calculate host layout for I/O with potentially unaligned host memref.
    hostLayout = HostLayoutAttr::get(ctx, tensorType.getShape(),
                                     hostInfo->getHostStride(),
                                     hostInfo->getHostVolume(), tensorMeshAttr);
  } else if (tensorMeshAttr) {
    // Create host layout with tensor mesh info and default
    // shape/strides/volume.
    hostLayout = HostLayoutAttr::get(
        ctx, tensorType.getShape(),
        ttmlir::utils::calculateStrides(tensorType.getShape()),
        ttmlir::utils::volume(tensorType.getShape()), tensorMeshAttr);
  }

  // If there is no encoding or encoding with TensorMesh info, return with the
  // host layout attribute.
  if (!tensorType.getEncoding() || tensorMeshAttr) {
    return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                           hostLayout);
  }

  auto layout = mlir::cast<MetalLayoutAttr>(tensorType.getEncoding());

  SmallVector<int64_t> fullMemrefShape(tensorType.getShape());

  // Create the appropriate layout attribute based on whether this is a view or
  // a materialized buffer. Views use affine maps for flexible indexing, while
  // materialized buffers use shard or interleaved layouts depending on the
  // tensor's memory layout strategy.
  MemRefLayoutAttrInterface layoutAttr;
  if (isView) {
    layoutAttr = ViewLayoutAttr::get(ctx, tensorType.getRank());
  } else {
    SmallVector<int64_t> shardStride = layout.getShardStride(tensorType);
    if (layout.getMemoryLayout() == TensorMemoryLayout::Sharded) {
      SmallVector<int64_t> shardStride = layout.getShardStride(tensorType);
      layoutAttr = ttcore::ShardLayoutAttr::get(ctx, shardStride,
                                                /*buffered=*/1);

    } else if (layout.getMemoryLayout() == TensorMemoryLayout::Interleaved) {
      layoutAttr = InterleavedLayoutAttr::get(ctx, shardStride);
    } else {
      llvm_unreachable("Unsupported memory layout");
    }
  }

  return MemRefType::get(fullMemrefShape, tensorType.getElementType(),
                         layoutAttr,
                         MemorySpaceAttr::get(ctx, layout.getMemorySpace()));
}

bufferization::BufferLikeType
getBufferType(Type type, bool isView, std::optional<MetalLayoutAttr> hostInfo) {
  return mlir::cast<bufferization::BufferLikeType>(
      getMemRefType(type, isView, hostInfo));
}

} // namespace mlir::tt::ttcore
