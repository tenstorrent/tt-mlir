// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTCORE_IR_TTCOREOPSTYPES_H
#define TTMLIR_DIALECT_TTCORE_IR_TTCOREOPSTYPES_H

#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#include <numeric>

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsEnums.h.inc"

#include "ttmlir/Dialect/TTCore/IR/TTCoreAttrInterfaces.h.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsAttrDefs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h.inc"

namespace mlir::tt::ttcore {
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

inline void printDimensionList(mlir::AsmPrinter &printer,
                               llvm::ArrayRef<int64_t> shape) {
  printer.printDimensionList(shape);
}

inline mlir::ParseResult
parseDimensionList(mlir::AsmParser &odsParser,
                   llvm::SmallVector<int64_t> &dimensions) {
  return odsParser.parseDimensionList(dimensions, false, false);
}

template <typename... Args>
inline void printVargDimensionList(mlir::AsmPrinter &printer, Args &&...dims) {
  printDimensionList(printer,
                     llvm::SmallVector<int64_t>({std::forward<Args>(dims)...}));
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

inline mlir::ParseResult parseIdentityAffineMap(mlir::AsmParser &odsParser,
                                                mlir::AffineMap &affineMap) {
  if (!odsParser.parseOptionalKeyword("map").succeeded()) {
    return odsParser.parseAffineMap(affineMap);
  }

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

template <typename... Args>
inline mlir::ParseResult parseVargDimensionList(mlir::AsmParser &odsParser,
                                                Args &...dims) {
  llvm::SmallVector<int64_t> dimensions;
  mlir::ParseResult result = parseDimensionList(odsParser, dimensions);
  if (succeeded(result)) {
    llvm::SmallVector<std::tuple_element_t<0, std::tuple<Args...>> *> copy(
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
    elementType = quant.getStorageType();
  }

  if (isa<BFloat16Type>(elementType)) {
    return DataType::BFloat16;
  }

  if (auto tileType = dyn_cast<TileType>(elementType)) {
    switch (tileType.getDataType()) {
    case DataType::BFP_BFloat8:
    case DataType::BFP_BFloat4:
    case DataType::BFP_BFloat2:
    case DataType::BFP_Float8:
    case DataType::BFP_Float4:
    case DataType::BFP_Float2:
    case DataType::Float32:
      return tileType.getDataType();
    default:
      assert(false && "Unsupported tile type in elementTypeToDataTypeImpl");
    }
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
  }

  if (auto intType = dyn_cast<mlir::IntegerType>(elementType)) {
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

inline Type dataTypeToElementType(mlir::MLIRContext *context, DataType dtype) {
  switch (dtype) {
  case DataType::Float32:
    return Float32Type::get(context);
  case DataType::Float16:
    return BFloat16Type::get(context);
  case DataType::BFloat16:
    return BFloat16Type::get(context);
  case DataType::BFP_Float8:
    return ttcore::TileType::get(context, ttcore::TileType::getDefaultShape(),
                                 DataType::BFP_Float8);
  case DataType::BFP_BFloat8:
    return ttcore::TileType::get(context, ttcore::TileType::getDefaultShape(),
                                 DataType::BFP_BFloat8);
  case DataType::BFP_Float4:
    return ttcore::TileType::get(context, ttcore::TileType::getDefaultShape(),
                                 DataType::BFP_Float4);
  case DataType::BFP_BFloat4:
    return ttcore::TileType::get(context, ttcore::TileType::getDefaultShape(),
                                 DataType::BFP_BFloat4);
  case DataType::BFP_Float2:
    return ttcore::TileType::get(context, ttcore::TileType::getDefaultShape(),
                                 DataType::BFP_Float2);
  case DataType::BFP_BFloat2:
    return ttcore::TileType::get(context, ttcore::TileType::getDefaultShape(),
                                 DataType::BFP_BFloat2);
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
inline mlir::Type toTTMLIRSupportedDataType(Type elementType) {
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

// The BFP formats are TT home-brew, if not for them we could have used MLIR's
// built-in FloatTypes and getWidth()/getFPMantissaWidth().
inline bool isSignedInteger(const DataType dtype) {
  return dtype == DataType::Int32;
}

inline bool isFloat(const DataType dtype) {
  switch (dtype) {
  case DataType::Float32:
  case DataType::Float16:
  case DataType::BFloat16:
  case DataType::BFP_Float8:
  case DataType::BFP_BFloat8:
  case DataType::BFP_Float4:
  case DataType::BFP_BFloat4:
  case DataType::BFP_Float2:
  case DataType::BFP_BFloat2:
    return true;
  default:
    return false;
  }
}

inline uint8_t getExponentSize(const DataType dtype) {
  assert(isFloat(dtype));
  switch (dtype) {
  case DataType::Float16:
  case DataType::BFP_Float8:
  case DataType::BFP_Float4:
  case DataType::BFP_Float2:
    return 5;
  case DataType::Float32:
  case DataType::BFloat16:
  case DataType::BFP_BFloat8:
  case DataType::BFP_BFloat4:
  case DataType::BFP_BFloat2:
    return 8;
  default:
    return 0;
  }
}

inline uint8_t getMantissaSize(const DataType dtype) {
  assert(isFloat(dtype));
  switch (dtype) {
  case DataType::Float32:
    return 23;
  case DataType::Float16:
    return 10;
  case DataType::BFloat16:
    return 7;
  case DataType::BFP_Float8:
  case DataType::BFP_BFloat8:
    return 7;
  case DataType::BFP_Float4:
  case DataType::BFP_BFloat4:
    return 3;
  case DataType::BFP_Float2:
  case DataType::BFP_BFloat2:
    return 1;
  default:
    return 0;
  }
}

inline uint8_t getNumberOfBits(const DataType dtype) {
  switch (dtype) {
  case DataType::Float32:
  case DataType::UInt32:
  case DataType::Int32:
    return 32;
  case DataType::Float16:
  case DataType::BFloat16:
  case DataType::UInt16:
    return 16;
  case DataType::BFP_Float8:
  case DataType::BFP_BFloat8:
  case DataType::UInt8:
    return 8;
  case DataType::BFP_Float4:
  case DataType::BFP_BFloat4:
    return 4;
  case DataType::BFP_Float2:
  case DataType::BFP_BFloat2:
    return 2;
  }
}

mlir::AffineMap collapsedLinearAffineMap(
    mlir::MLIRContext *context, llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<int64_t> gridShape,
    llvm::ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals);

mlir::SmallVector<std::int64_t>
calculateLogicalShardShape(mlir::ArrayRef<int64_t> tensorShape,
                           mlir::AffineMap linear,
                           mlir::tt::ttcore::GridAttr grid);

template <typename T, typename TAttr>
mlir::MemRefType buildMemRef(mlir::MLIRContext *context,
                             llvm::ArrayRef<int64_t> shardShape,
                             mlir::Type elementType, T memorySpace) {
  llvm::SmallVector<int64_t> scalarShardShape(shardShape);
  if (mlir::isa<mlir::tt::ttcore::TileType>(elementType)) {
    scalarShardShape = mlir::cast<mlir::tt::ttcore::TileType>(elementType)
                           .getTiledShape(scalarShardShape);
  }
  return mlir::MemRefType::get(
      scalarShardShape, elementType,
      mlir::AffineMap::getMultiDimIdentityMap(scalarShardShape.size(), context),
      TAttr::get(context, memorySpace));
}

inline uint64_t getElementSizeBytes(mlir::Type elementType) {
  TileType tileType = mlir::dyn_cast<TileType>(elementType);
  return tileType ? tileType.getSizeBytes()
                  : elementType.getIntOrFloatBitWidth() / 8;
}

inline MemorySpace getMemorySpace(MemorySpaceAttr memorySpaceAttr) {
  return memorySpaceAttr.getValue();
}

inline MemorySpace getMemorySpace(MemRefType memref) {
  return getMemorySpace(mlir::cast<MemorySpaceAttr>(memref.getMemorySpace()));
}

inline MemorySpace getMemorySpace(Type memrefType) {
  return getMemorySpace(mlir::cast<MemRefType>(memrefType));
}

inline MemorySpace getMemorySpace(Value memrefTypedValue) {
  return getMemorySpace(memrefTypedValue.getType());
}

} // namespace mlir::tt::ttcore

#endif
