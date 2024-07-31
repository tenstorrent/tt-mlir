// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_UTILS_MLIRTOFLATBUFFER_H
#define TTMLIR_TARGET_UTILS_MLIRTOFLATBUFFER_H

#include <type_traits>

#include "flatbuffers/flatbuffers.h"
#include "ttmlir/Target/Common/debug_info_generated.h"
#include "ttmlir/Target/Common/types_generated.h"
#include "ttmlir/Target/Utils/FlatbufferObjectCache.h"

namespace mlir::tt {
inline ::tt::target::OOBVal toFlatbuffer(FlatbufferObjectCache &,
                                         OOBVal oobVal) {
  switch (oobVal) {
  case OOBVal::Undef:
    return ::tt::target::OOBVal::Undef;
  case OOBVal::Zero:
    return ::tt::target::OOBVal::Zero;
  case OOBVal::One:
    return ::tt::target::OOBVal::One;
  case OOBVal::Inf:
    return ::tt::target::OOBVal::Inf;
  case OOBVal::NegInf:
    return ::tt::target::OOBVal::NegInf;
  }
}

inline ::tt::target::DataType toFlatbuffer(FlatbufferObjectCache &,
                                           DataType dtype) {
  switch (dtype) {
  case DataType::Float32:
    return ::tt::target::DataType::Float32;
  case DataType::Float16:
    return ::tt::target::DataType::Float16;
  case DataType::BFloat16:
    return ::tt::target::DataType::BFloat16;
  case DataType::BFP_Float8:
    return ::tt::target::DataType::BFP_Float8;
  case DataType::BFP_BFloat8:
    return ::tt::target::DataType::BFP_BFloat8;
  case DataType::BFP_Float4:
    return ::tt::target::DataType::BFP_Float4;
  case DataType::BFP_BFloat4:
    return ::tt::target::DataType::BFP_BFloat4;
  case DataType::BFP_Float2:
    return ::tt::target::DataType::BFP_Float2;
  case DataType::BFP_BFloat2:
    return ::tt::target::DataType::BFP_BFloat2;
  case DataType::UInt32:
    return ::tt::target::DataType::UInt32;
  case DataType::UInt16:
    return ::tt::target::DataType::UInt16;
  case DataType::UInt8:
    return ::tt::target::DataType::UInt8;
  }
}

inline ::tt::target::MemorySpace toFlatbuffer(FlatbufferObjectCache &,
                                              MemorySpace memspace) {
  switch (memspace) {
  case MemorySpace::System:
    return ::tt::target::MemorySpace::System;
  case MemorySpace::SystemMMIO:
    return ::tt::target::MemorySpace::SystemMMIO;
  case MemorySpace::DeviceDRAM:
    return ::tt::target::MemorySpace::DeviceDRAM;
  case MemorySpace::DeviceL1:
    return ::tt::target::MemorySpace::DeviceL1;
  }
}

inline ::tt::target::Arch toFlatbuffer(FlatbufferObjectCache &, ArchAttr arch) {
  switch (arch.getValue()) {
  case Arch::Grayskull:
    return ::tt::target::Arch::Grayskull;
  case Arch::WormholeB0:
    return ::tt::target::Arch::Wormhole_b0;
  case Arch::Blackhole:
    return ::tt::target::Arch::Blackhole;
  }
}

inline ::tt::target::ChipCapability
toFlatbuffer(FlatbufferObjectCache &, ChipCapabilityAttr capabilityAttr) {
  auto capabilities = capabilityAttr.getValue();
  static_assert(
      static_cast<std::underlying_type_t<ChipCapability>>(
          ChipCapability::PCIE) ==
      static_cast<std::underlying_type_t<::tt::target::ChipCapability>>(
          ::tt::target::ChipCapability::PCIE));
  static_assert(
      static_cast<std::underlying_type_t<ChipCapability>>(
          ChipCapability::HostMMIO) ==
      static_cast<std::underlying_type_t<::tt::target::ChipCapability>>(
          ::tt::target::ChipCapability::HostMMIO));
  assert((static_cast<std::underlying_type_t<ChipCapability>>(capabilities) &
          ~0b11) == 0 &&
         "unsupported chip capabilities");
  return static_cast<::tt::target::ChipCapability>(capabilities);
}

inline ::tt::target::ChipCoord toFlatbuffer(FlatbufferObjectCache &cache,
                                            ChipCoordAttr chipCoord) {
  return ::tt::target::ChipCoord(chipCoord.getRack(), chipCoord.getShelf(),
                                 chipCoord.getY(), chipCoord.getX());
}

inline ::tt::target::ChipChannel toFlatbuffer(FlatbufferObjectCache &cache,
                                              ChipChannelAttr chipChannel) {
  return ::tt::target::ChipChannel(chipChannel.getEndpoint0(),
                                   chipChannel.getEndpoint1());
}

inline ::tt::target::Dim2d toFlatbuffer(FlatbufferObjectCache &cache,
                                        GridAttr arch) {
  assert(arch.getShape().size() == 2 && "expected a 2D grid");
  return ::tt::target::Dim2d(arch.getShape()[0], arch.getShape()[1]);
}

template <typename T, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
T toFlatbuffer(FlatbufferObjectCache &, T arith) {
  return arith;
}

template <typename T>
using ToFlatbufferReturnType = decltype(toFlatbuffer(
    std::declval<FlatbufferObjectCache &>(), std::declval<T>()));

template <typename, typename = void>
struct IsNativeFlatbufferType : std::false_type {};

template <typename T>
struct IsNativeFlatbufferType<
    T, std::void_t<typename ToFlatbufferReturnType<T>::Traits::type>> {
  constexpr static bool value = true;
};

template <typename T,
          std::enable_if_t<IsNativeFlatbufferType<T>::value, int> = 0>
flatbuffers::Offset<flatbuffers::Vector<ToFlatbufferReturnType<T> const *>>
toFlatbuffer(FlatbufferObjectCache &cache, ::llvm::ArrayRef<T> arr) {
  static_assert(std::is_trivially_copyable_v<ToFlatbufferReturnType<T>>);
  ToFlatbufferReturnType<T> *buf;
  auto vec = cache.fbb->CreateUninitializedVectorOfStructs(arr.size(), &buf);
  for (auto elem : arr) {
    auto v = toFlatbuffer(cache, elem);
    std::memcpy(buf, &v, sizeof(v));
    ++buf;
  }
  return vec;
}

template <typename T,
          std::enable_if_t<!IsNativeFlatbufferType<T>::value, int> = 0>
flatbuffers::Offset<flatbuffers::Vector<ToFlatbufferReturnType<T>>>
toFlatbuffer(FlatbufferObjectCache &cache, ::llvm::ArrayRef<T> arr) {
  return cache.fbb->CreateVector<ToFlatbufferReturnType<T>>(
      arr.size(),
      [&cache, arr](size_t i) { return toFlatbuffer(cache, arr[i]); });
}

inline flatbuffers::Offset<::tt::target::ChipDesc>
toFlatbuffer(FlatbufferObjectCache &cache, ChipDescAttr chipDesc) {
  assert(chipDesc.getGrid().size() == 2 && "expected a 2D grid");
  auto grid = ::tt::target::Dim2d(chipDesc.getGrid()[0], chipDesc.getGrid()[1]);
  return ::tt::target::CreateChipDesc(
      *cache.fbb, toFlatbuffer(cache, chipDesc.getArch()), &grid,
      chipDesc.getL1Size(), chipDesc.getNumDramChannels(),
      chipDesc.getDramChannelSize(), chipDesc.getNocL1AddressAlignBytes(),
      chipDesc.getPcieAddressAlignBytes(),
      chipDesc.getNocDRAMAddressAlignBytes());
}

inline flatbuffers::Offset<::tt::target::SystemDesc>
toFlatbuffer(FlatbufferObjectCache &cache, SystemDescAttr systemDesc) {
  auto chipDescs = toFlatbuffer(cache, systemDesc.getChipDescs());
  auto chipDescIndices = toFlatbuffer(cache, systemDesc.getChipDescIndices());
  auto chipCapabilities = toFlatbuffer(cache, systemDesc.getChipCapabilities());
  auto chipCoords = toFlatbuffer(cache, systemDesc.getChipCoords());
  auto chipChannels = toFlatbuffer(cache, systemDesc.getChipChannels());
  return ::tt::target::CreateSystemDesc(*cache.fbb, chipDescs, chipDescIndices,
                                        chipCapabilities, chipCoords,
                                        chipChannels);
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

template <typename AttrType, typename ValueType>
struct ArrayAttrToFlatbufferSerializer {
  static flatbuffers::Offset<flatbuffers::Vector<ValueType>>
  impl(FlatbufferObjectCache &cache, const ArrayAttr &arrayAttr) {
    assert(false && "unsupported array attr to value type serializer");
  }
};

template <typename ValueType>
struct ArrayAttrToFlatbufferSerializer<IntegerAttr, ValueType> {
  static flatbuffers::Offset<flatbuffers::Vector<ValueType>>
  impl(FlatbufferObjectCache &cache, const ::mlir::ArrayAttr &arrayAttr) {
    return cache.fbb->CreateVector<ValueType>(
        arrayAttr.size(), [&arrayAttr](size_t i) {
          return static_cast<ValueType>(
              mlir::cast<IntegerAttr>(arrayAttr[i]).getInt());
        });
  }
};

template <typename AttrType, typename ValueType>
inline flatbuffers::Offset<flatbuffers::Vector<ValueType>>
arrayAttrToFlatbuffer(FlatbufferObjectCache &cache,
                      const ::mlir::ArrayAttr &arrayAttr) {
  return ArrayAttrToFlatbufferSerializer<AttrType, ValueType>::impl(cache,
                                                                    arrayAttr);
}

template <typename AttrType, typename ValueType>
inline flatbuffers::Offset<flatbuffers::Vector<ValueType>>
arrayAttrToFlatbuffer(FlatbufferObjectCache &cache,
                      const std::optional<::mlir::ArrayAttr> &arrayAttrOpt) {
  return arrayAttrOpt.has_value() ? arrayAttrToFlatbuffer<AttrType, ValueType>(
                                        cache, arrayAttrOpt.value())
                                  : 0;
}

inline flatbuffers::Offset<::tt::target::MemoryDesc>
memrefAttrToFlatbuffer(FlatbufferObjectCache &cache, MemRefType memref) {
  auto shapeInt64 = memref.getShape();
  std::vector<int32_t> shape(shapeInt64.begin(), shapeInt64.end());
  DataType dtype = DataType::Float32;
  ::tt::target::Dim2d tileShape(0, 0);
  Type elementType = memref.getElementType();
  if (isa<TileType>(elementType)) {
    auto tileType = mlir::cast<TileType>(elementType);
    dtype = tileType.getDataType();
    tileShape = ::tt::target::Dim2d(tileType.getHeight(), tileType.getWidth());
  } else {
    dtype = elementTypeToDataType(elementType);
  }

  return ::tt::target::CreateMemoryDescDirect(
      *cache.fbb, &shape, &tileShape, toFlatbuffer(cache, dtype),
      toFlatbuffer(
          cache,
          mlir::cast<MemorySpaceAttr>(memref.getMemorySpace()).getValue()));
}

inline flatbuffers::Offset<::tt::target::LayoutDesc>
layoutAttrToFlatbuffer(FlatbufferObjectCache &cache, Attribute attr,
                       ArrayRef<int64_t> logicalShape) {
  assert(isa<LayoutAttr>(attr) && "expected a tensor type");
  auto layoutAttr = mlir::cast<LayoutAttr>(attr);
  auto strideInt64 = layoutAttr.getStride(logicalShape);
  std::vector<int32_t> stride(strideInt64.begin(), strideInt64.end());
  auto gridAttr = layoutAttr.getGrid();
  auto gridShape = gridAttr.getShape();
  assert(gridShape.size() == 2 && "expected a 2D grid");
  ::tt::target::Dim2dRange grid(
      ::tt::target::Dim2d(0, 0),
      ::tt::target::Dim2d(gridShape[0], gridShape[1]));
  return ::tt::target::CreateLayoutDescDirect(
      *cache.fbb, &stride, toFlatbuffer(cache, layoutAttr.getOobVal()), &grid,
      cache.getOrCreate(layoutAttr.getMemref(), memrefAttrToFlatbuffer));
}

inline flatbuffers::Offset<::tt::target::TensorDesc>
tensorTypeToFlatbuffer(FlatbufferObjectCache &cache, Type type) {
  auto tensorType = mlir::cast<RankedTensorType>(type);
  auto shapeInt64 = tensorType.getShape();
  std::vector<int32_t> shape(shapeInt64.begin(), shapeInt64.end());
  return ::tt::target::CreateTensorDescDirect(
      *cache.fbb, &shape,
      cache.getOrCreate(tensorType.getEncoding(), layoutAttrToFlatbuffer,
                        shapeInt64));
}

inline flatbuffers::Offset<::tt::target::TensorRef>
tensorValueToFlatbuffer(FlatbufferObjectCache &cache, Value value,
                        uint64_t address, uint64_t size) {
  auto tensorType = mlir::cast<RankedTensorType>(value.getType());
  auto tensorDesc = cache.getOrCreate(tensorType, tensorTypeToFlatbuffer);
  return ::tt::target::CreateTensorRef(*cache.fbb, cache.global_id++, address,
                                       size, tensorDesc);
}

inline flatbuffers::Offset<::tt::target::MLIR>
toDebugInfo(::flatbuffers::FlatBufferBuilder &fbb, std::string const &name,
            ModuleOp module) {
  std::string source;
  llvm::raw_string_ostream os(source);
  module->print(os);
  return ::tt::target::CreateMLIRDirect(fbb, name.c_str(), source.c_str());
}
} // namespace mlir::tt

#endif
