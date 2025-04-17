// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TT_IR_TTOPSTYPES_H
#define TTMLIR_DIALECT_TT_IR_TTOPSTYPES_H

#include "mlir/Dialect/Quant/IR/QuantTypes.h"
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

inline void printIdentityAffineMap(mlir::AsmPrinter &printer,
                                   mlir::AffineMap affineMap) {
  if (affineMap.isIdentity()) {
    printer << "map(";
    printer << affineMap.getNumResults() << ")";
    return;
  }

  affineMap.print(printer.getStream());
}

inline ::mlir::ParseResult parseIdentityAffineMap(::mlir::AsmParser &odsParser,
                                                  mlir::AffineMap &affineMap) {
  if (odsParser.parseOptionalKeyword("map").succeeded()) {
    if (odsParser.parseLParen().failed()) {
      return failure();
    }

    unsigned rank;
    if (odsParser.parseInteger(rank).failed()) {
      return failure();
    }

    if (odsParser.parseRParen().failed()) {
      return failure();
    }

    affineMap =
        mlir::AffineMap::getMultiDimIdentityMap(rank, odsParser.getContext());

    return success();
  }
  return odsParser.parseAffineMap(affineMap);
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

inline std::optional<DataType> elementTypeToDataTypeImpl(Type elementType) {
  if (auto quant = dyn_cast<quant::QuantizedType>(elementType)) {
    elementType = quant.getExpressedType();
  }

  if (isa<BFloat16Type>(elementType)) {
    return DataType::BFloat16;
  }

  if (auto floatType = dyn_cast<mlir::FloatType>(elementType)) {
    switch (floatType.getWidth()) {
    // Treat f64 as f32.
    case 32:
    case 64:
      return DataType::Float32;
    case 16:
      return DataType::Float16;
    default:
      return {};
    }
  } else if (auto intType = dyn_cast<mlir::IntegerType>(elementType)) {
    switch (intType.getWidth()) {
    // Booleans treated as bfloat16.
    case 1:
      return DataType::BFloat16;
    case 8:
      return DataType::UInt8;
    case 16:
      return DataType::UInt16;
    case 32:
    case 64:
      return (intType.isSigned() || intType.isSignless()) ? DataType::Int32
                                                          : DataType::UInt32;
    default:
      return {};
    }
  }

  return {};
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
  case DataType::Int32:
    return IntegerType::get(context, 32,
                            IntegerType::SignednessSemantics::Signed);
  }
}

// Convenience function to convert any type to TTMLIR supported type.
inline ::mlir::Type toTTMLIRSupportedDataType(Type elementType) {
  std::optional<DataType> dataType = elementTypeToDataTypeImpl(elementType);

  if (dataType) {
    return dataTypeToElementType(elementType.getContext(), *dataType);
  }

  return {};
}

inline DataType elementTypeToDataType(Type elementType) {
  std::optional<DataType> dataType = elementTypeToDataTypeImpl(elementType);

  if (dataType) {
    return *dataType;
  }

  llvm_unreachable("Unsupported element type.");
}
} // namespace mlir::tt

#include "ttmlir/Dialect/TT/IR/TTAttrInterfaces.h.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TT/IR/TTOpsAttrDefs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h.inc"

namespace mlir::tt {

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
                             ::mlir::Type elementType, T memorySpace) {
  ::llvm::SmallVector<int64_t> scalarShardShape(shardShape);
  if (mlir::isa<mlir::tt::TileType>(elementType)) {
    scalarShardShape = mlir::cast<mlir::tt::TileType>(elementType)
                           .getTiledShape(scalarShardShape);
  }
  return mlir::MemRefType::get(
      scalarShardShape, elementType,
      mlir::AffineMap::getMultiDimIdentityMap(scalarShardShape.size(), context),
      TAttr::get(context, memorySpace));
}

inline uint64_t getElementSizeBytes(mlir::Type elementType) {
  if (mlir::isa<TileType>(elementType)) {
    auto tileType = mlir::cast<TileType>(elementType);
    return tileType.getSizeBytes();
  }
  return elementType.getIntOrFloatBitWidth() / 8;
}

} // namespace mlir::tt

#endif
