// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TT_IR_TTOPSTYPES_H
#define TTMLIR_DIALECT_TT_IR_TTOPSTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#include "ttmlir/Dialect/TT/IR/TTOpsEnums.h.inc"

namespace mlir::tt {
struct PhysGridResultIdx {
  enum : int64_t {
    DeviceIdx = 0,
    CoreCoordY = 1,
    CoreCoordX = 2,
    NumIndices = 3,
  };
};

struct MemoryMapResultIdx {
  enum : int64_t {
    DeviceIdx = 0,
    CoreCoordY = 1,
    CoreCoordX = 2,
    ShardOffset = 3,
    NumIndices = 4,
  };
};

inline bool isSystemMemorySpace(MemorySpace memorySpace) {
  return memorySpace == MemorySpace::System ||
         memorySpace == MemorySpace::SystemMMIO;
}

inline bool isDeviceMemorySpace(MemorySpace memorySpace) {
  return memorySpace == MemorySpace::DeviceDRAM ||
         memorySpace == MemorySpace::DeviceL1;
}

inline bool isL1MemorySpace(MemorySpace memorySpace) {
  return memorySpace == MemorySpace::DeviceL1;
}

inline void printDimensionList(::mlir::AsmPrinter &printer,
                               ::llvm::ArrayRef<int64_t> shape) {
  printer.printDimensionList(shape);
}

inline ::mlir::ParseResult
parseDimensionList(::mlir::AsmParser &odsParser,
                   ::llvm::SmallVector<int64_t> &dimensions) {
  return odsParser.parseDimensionList(dimensions, false, false);
}

template <typename... Args>
inline void printVargDimensionList(::mlir::AsmPrinter &printer, Args... dims) {
  printDimensionList(printer, ::llvm::SmallVector<int64_t>({dims...}));
}

template <typename... Args>
inline ::mlir::ParseResult parseVargDimensionList(::mlir::AsmParser &odsParser,
                                                  Args &...dims) {
  ::llvm::SmallVector<int64_t> dimensions;
  ::mlir::ParseResult result = parseDimensionList(odsParser, dimensions);
  if (succeeded(result)) {
    ::llvm::SmallVector<std::tuple_element_t<0, std::tuple<Args...>> *> copy(
        {&dims...});
    assert(dimensions.size() == sizeof...(dims));
    for (size_t i = 0; i < dimensions.size(); ++i) {
      *copy[i] = dimensions[i];
    }
  }
  return result;
}

inline DataType elementTypeToDataType(Type elementType) {
  DataType dtype = DataType::Float32;
  if (isa<FloatType>(elementType)) {
    auto floatType = mlir::cast<FloatType>(elementType);
    if (floatType.isF32()) {
      dtype = DataType::Float32;
    } else if (floatType.isF16()) {
      dtype = DataType::Float16;
    } else if (floatType.isBF16()) {
      dtype = DataType::BFloat16;
    } else {
      assert(false && "unsupported float type");
    }
  } else if (isa<IntegerType>(elementType)) {
    auto intType = mlir::cast<IntegerType>(elementType);
    if (intType.getWidth() == 32) {
      dtype = DataType::UInt32;
    } else if (intType.getWidth() == 16) {
      dtype = DataType::UInt16;
    } else if (intType.getWidth() == 8) {
      dtype = DataType::UInt8;
    } else {
      assert(false && "unsupported integer type");
    }
  }
  return dtype;
}

inline Type dataTypeToElementType(::mlir::MLIRContext *context,
                                  DataType dtype) {
  switch (dtype) {
  case DataType::Float32:
    return Float32Type::get(context);
  case DataType::Float16:
    return Float16Type::get(context);
  case DataType::BFloat16:
    return BFloat16Type::get(context);
  case DataType::BFP_Float8:
    return Float16Type::get(context);
  case DataType::BFP_BFloat8:
    return BFloat16Type::get(context);
  case DataType::BFP_Float4:
    return Float16Type::get(context);
  case DataType::BFP_BFloat4:
    return BFloat16Type::get(context);
  case DataType::BFP_Float2:
    return Float16Type::get(context);
  case DataType::BFP_BFloat2:
    return BFloat16Type::get(context);
  case DataType::UInt32:
    return IntegerType::get(context, 32,
                            IntegerType::SignednessSemantics::Unsigned);
  case DataType::UInt16:
    return IntegerType::get(context, 16,
                            IntegerType::SignednessSemantics::Unsigned);
  case DataType::UInt8:
    return IntegerType::get(context, 8,
                            IntegerType::SignednessSemantics::Unsigned);
  }
}
} // namespace mlir::tt

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TT/IR/TTOpsAttrDefs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h.inc"

namespace mlir::tt {
SystemDescAttr getCurrentScopeSystemDesc(Operation *op);
DeviceAttr getCurrentScopeDevice(Operation *op);
} // namespace mlir::tt

mlir::AffineMap collapsedLinearAffineMap(
    ::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape,
    ::llvm::ArrayRef<int64_t> gridShape,
    ::llvm::ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals);

mlir::SmallVector<std::int64_t>
calculateLogicalShardShape(mlir::ArrayRef<int64_t> tensorShape,
                           mlir::AffineMap linear, mlir::tt::GridAttr grid);

template <typename T, typename TAttr>
mlir::MemRefType buildMemRef(::mlir::MLIRContext *context,
                             ::llvm::ArrayRef<int64_t> shardShape,
                             ::mlir::Type elementType, T memorySpace,
                             mlir::tt::StreamLayoutAttr layout = {}) {
  ::llvm::SmallVector<int64_t> scalarShardShape(shardShape);
  if (mlir::isa<mlir::tt::TileType>(elementType)) {
    scalarShardShape = mlir::cast<mlir::tt::TileType>(elementType)
                           .getTiledShape(scalarShardShape);
  }
  return layout ? mlir::MemRefType::get(scalarShardShape, elementType, layout,
                                        TAttr::get(context, memorySpace))
                : mlir::MemRefType::get(scalarShardShape, elementType,
                                        mlir::AffineMap::getMultiDimIdentityMap(
                                            scalarShardShape.size(), context),
                                        TAttr::get(context, memorySpace));
}

#endif
