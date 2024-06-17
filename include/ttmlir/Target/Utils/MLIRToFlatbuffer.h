#ifndef TTMLIR_TARGET_UTILS_MLIRTOFLATBUFFER_H
#define TTMLIR_TARGET_UTILS_MLIRTOFLATBUFFER_H

#include "flatbuffers/flatbuffers.h"
#include "ttmlir/Target/Utils/FlatbufferObjectCache.h"

namespace mlir::tt {
::tt::OOBVal toFlatbuffer(OOBVal oobVal) {
  switch (oobVal) {
  case OOBVal::Undef:
    return ::tt::OOBVal::Undef;
  case OOBVal::Zero:
    return ::tt::OOBVal::Zero;
  case OOBVal::One:
    return ::tt::OOBVal::One;
  case OOBVal::Inf:
    return ::tt::OOBVal::Inf;
  case OOBVal::NegInf:
    return ::tt::OOBVal::NegInf;
  }
}

::tt::DataType toFlatbuffer(DataType dtype) {
  switch (dtype) {
  case DataType::Float32:
    return ::tt::DataType::Float32;
  case DataType::Float16:
    return ::tt::DataType::Float16;
  case DataType::BFloat16:
    return ::tt::DataType::BFloat16;
  case DataType::BC_Float8:
    return ::tt::DataType::BC_Float8;
  case DataType::BC_BFloat8:
    return ::tt::DataType::BC_BFloat8;
  case DataType::BC_Float4:
    return ::tt::DataType::BC_Float4;
  case DataType::BC_BFloat4:
    return ::tt::DataType::BC_BFloat4;
  case DataType::BC_Float2:
    return ::tt::DataType::BC_Float2;
  case DataType::BC_BFloat2:
    return ::tt::DataType::BC_BFloat2;
  case DataType::UInt32:
    return ::tt::DataType::UInt32;
  case DataType::UInt16:
    return ::tt::DataType::UInt16;
  case DataType::UInt8:
    return ::tt::DataType::UInt8;
  }
}

::tt::MemorySpace toFlatbuffer(MemorySpace memspace) {
  switch (memspace) {
  case MemorySpace::System:
    return ::tt::MemorySpace::System;
  case MemorySpace::SystemMMIO:
    return ::tt::MemorySpace::SystemMMIO;
  case MemorySpace::DeviceDRAM:
    return ::tt::MemorySpace::DeviceDRAM;
  case MemorySpace::DeviceL1:
    return ::tt::MemorySpace::DeviceL1;
  }
}

inline DataType elementTypeToDataType(Type elementType) {
  DataType dtype = DataType::Float32;
  if (isa<FloatType>(elementType)) {
    auto floatType = elementType.cast<FloatType>();
    if (floatType.getWidth() == 32) {
      dtype = DataType::Float32;
    } else if (floatType.getWidth() == 16) {
      dtype = DataType::Float16;
    } else {
      assert(false && "unsupported float type");
    }
  } else if (isa<IntegerType>(elementType)) {
    auto intType = elementType.cast<IntegerType>();
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

inline flatbuffers::Offset<::tt::MemoryDesc>
memrefAttrToFlatbuffer(FlatbufferObjectCache &cache, MemRefType memref) {
  auto shapeInt64 = memref.getShape();
  std::vector<int32_t> shape(shapeInt64.begin(), shapeInt64.end());
  DataType dtype = DataType::Float32;
  ::tt::Dim2d tileShape(0, 0);
  Type elementType = memref.getElementType();
  if (isa<TileType>(elementType)) {
    auto tileType = elementType.cast<TileType>();
    dtype = tileType.getDataType();
    tileShape = ::tt::Dim2d(tileType.getHeight(), tileType.getWidth());
  } else {
    dtype = elementTypeToDataType(elementType);
  }

  return ::tt::CreateMemoryDescDirect(
      *cache.fbb, &shape, &tileShape, toFlatbuffer(dtype),
      toFlatbuffer(memref.getMemorySpace().cast<MemorySpaceAttr>().getValue()));
}

inline flatbuffers::Offset<::tt::LayoutDesc>
layoutAttrToFlatbuffer(FlatbufferObjectCache &cache, Attribute attr) {
  assert(attr.isa<LayoutAttr>() && "expected a tensor type");
  auto layoutAttr = attr.cast<LayoutAttr>();
  auto stridesInt64 = layoutAttr.getStrides();
  std::vector<int32_t> strides(stridesInt64.begin(), stridesInt64.end());
  auto gridAttr = layoutAttr.getGrid();
  auto gridShape = gridAttr.getShape();
  assert(gridShape.size() == 2 && "expected a 2D grid");
  ::tt::Dim2dRange grid(::tt::Dim2d(0, 0),
                        ::tt::Dim2d(gridShape[0], gridShape[1]));
  return ::tt::CreateLayoutDescDirect(
      *cache.fbb, &strides, toFlatbuffer(layoutAttr.getOobVal()), &grid,
      cache.getOrCreate(layoutAttr.getMemref(), memrefAttrToFlatbuffer));
}

inline flatbuffers::Offset<::tt::TensorDesc>
tensorTypeToFlatbuffer(FlatbufferObjectCache &cache, Type type) {
  auto tensorType = type.cast<RankedTensorType>();
  auto shapeInt64 = tensorType.getShape();
  std::vector<int32_t> shape(shapeInt64.begin(), shapeInt64.end());
  return ::tt::CreateTensorDescDirect(
      *cache.fbb, &shape,
      cache.getOrCreate(tensorType.getEncoding(), layoutAttrToFlatbuffer));
}

inline flatbuffers::Offset<::tt::TensorRef>
tensorValueToFlatbuffer(FlatbufferObjectCache &cache, Value value,
                        uint64_t address, uint64_t size) {
  auto tensorType = value.getType().cast<RankedTensorType>();
  auto tensorDesc = cache.getOrCreate(tensorType, tensorTypeToFlatbuffer);
  return ::tt::CreateTensorRef(*cache.fbb, cache.global_id++, address, size,
                               tensorDesc);
}
} // namespace mlir::tt

#endif
