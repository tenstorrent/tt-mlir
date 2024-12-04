// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_UTILS_MLIRTOFLATBUFFER_H
#define TTMLIR_TARGET_UTILS_MLIRTOFLATBUFFER_H

#include <numeric>
#include <type_traits>

#include "flatbuffers/flatbuffers.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Target/Common/Target.h"
#include "ttmlir/Target/Utils/FlatbufferObjectCache.h"
#include "ttmlir/Utils.h"

namespace mlir::tt {

flatbuffers::Offset<::tt::target::LayoutDesc>
metalLayoutAttrToFlatbuffer(FlatbufferObjectCache &cache, MetalLayoutAttr attr,
                            ArrayRef<int64_t> logicalShape,
                            DeviceAttr deviceAttr);

flatbuffers::Offset<::tt::target::LayoutDesc> ttnnLayoutAttrToFlatbuffer(
    FlatbufferObjectCache &cache, ttnn::TTNNLayoutAttr attr,
    ArrayRef<int64_t> logicalShape, DeviceAttr deviceAttr);

struct GoldenTensor {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  ::tt::target::DataType dtype;
  std::uint8_t *data;

  GoldenTensor(std::string name, std::vector<int64_t> shape,
               std::vector<int64_t> strides, ::tt::target::DataType dtype,
               std::uint8_t *data)
      : name(name), shape(shape), strides(strides), dtype(dtype), data(data) {}

  std::vector<std::uint8_t> convertDataToVector() {
    int totalDataSize = std::accumulate(this->shape.begin(), this->shape.end(),
                                        1, std::multiplies<int64_t>()) *
                        sizeof(float);

    std::vector<std::uint8_t> dataVec(totalDataSize);
    std::memcpy(dataVec.data(), this->data, totalDataSize);

    return dataVec;
  }
};

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

inline ::tt::target::TensorMemoryLayout
toFlatbuffer(FlatbufferObjectCache &, TensorMemoryLayout memLayout) {
  switch (memLayout) {
  case TensorMemoryLayout::None:
    return ::tt::target::TensorMemoryLayout::None;
  case TensorMemoryLayout::Interleaved:
    return ::tt::target::TensorMemoryLayout::Interleaved;
  case TensorMemoryLayout::SingleBank:
    return ::tt::target::TensorMemoryLayout::SingleBank;
  case TensorMemoryLayout::HeightSharded:
    return ::tt::target::TensorMemoryLayout::HeightSharded;
  case TensorMemoryLayout::WidthSharded:
    return ::tt::target::TensorMemoryLayout::WidthSharded;
  case TensorMemoryLayout::BlockSharded:
    return ::tt::target::TensorMemoryLayout::BlockSharded;
  }
}

inline std::uint64_t getElementSizeBytes(DataType dtype) {
  switch (dtype) {
  case DataType::Float32:
    return 4;
  case DataType::Float16:
    return 2;
  case DataType::BFloat16:
    return 2;
  case DataType::UInt32:
    return 4;
  case DataType::UInt16:
    return 2;
  case DataType::UInt8:
    return 1;
  default:
    assert(false && "unsupported data type");
    break;
  }
  assert(false && "unsupported data type");
  return 0;
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

// Overloaded function for DataTypeAttr
inline ::tt::target::DataType toFlatbuffer(FlatbufferObjectCache &cache,
                                           const DataTypeAttr &dtypeAttr) {
  return toFlatbuffer(cache, dtypeAttr.getValue());
}

inline ::tt::target::Dim2d toFlatbuffer(FlatbufferObjectCache &cache,
                                        TileSizeAttr tileSize) {
  return ::tt::target::Dim2d(tileSize.getY(), tileSize.getX());
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
  return ::tt::target::ChipChannel(
      chipChannel.getDeviceId0(),
      ::tt::target::Dim2d(chipChannel.getEthernetCoreCoord0()[0],
                          chipChannel.getEthernetCoreCoord0()[1]),
      chipChannel.getDeviceId1(),
      ::tt::target::Dim2d(chipChannel.getEthernetCoreCoord1()[0],
                          chipChannel.getEthernetCoreCoord1()[1]));
}

inline ::tt::target::Dim2d toFlatbuffer(FlatbufferObjectCache &cache,
                                        GridAttr arch) {
  assert(arch.getShape().size() == 2 && "expected a 2D grid");
  return ::tt::target::Dim2d(arch.getShape()[0], arch.getShape()[1]);
}

inline flatbuffers::Offset<::tt::target::ChipPhysicalCores>
toFlatbuffer(FlatbufferObjectCache &cache,
             ChipPhysicalCoresAttr chipPhysicalCores) {

  // Create a Flatbuffer Dim2d struct for each type of core.
  std::vector<::tt::target::Dim2d> workerCores, dramCores, ethCores,
      ethInactiveCores;

  for (auto const &coreCoord : chipPhysicalCores.getWorker()) {
    workerCores.emplace_back(coreCoord.getY(), coreCoord.getX());
  }
  for (auto const &coreCoord : chipPhysicalCores.getDram()) {
    dramCores.emplace_back(coreCoord.getY(), coreCoord.getX());
  }
  for (auto const &coreCoord : chipPhysicalCores.getEth()) {
    ethCores.emplace_back(coreCoord.getY(), coreCoord.getX());
  }
  for (auto const &coreCoord : chipPhysicalCores.getEthInactive()) {
    ethInactiveCores.emplace_back(coreCoord.getY(), coreCoord.getX());
  }

  // Create and return the ChipPhysicalCores flatbuffer object
  return ::tt::target::CreateChipPhysicalCores(
      *cache.fbb,
      cache.fbb->CreateVectorOfStructs<::tt::target::Dim2d>(workerCores),
      cache.fbb->CreateVectorOfStructs<::tt::target::Dim2d>(dramCores),
      cache.fbb->CreateVectorOfStructs<::tt::target::Dim2d>(ethCores),
      cache.fbb->CreateVectorOfStructs<::tt::target::Dim2d>(ethInactiveCores));
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
      chipDesc.getNocDRAMAddressAlignBytes(), chipDesc.getL1UnreservedBase(),
      chipDesc.getEriscL1UnreservedBase(), chipDesc.getDramUnreservedBase(),
      chipDesc.getDramUnreservedEnd(),
      toFlatbuffer(cache, chipDesc.getChipPhysicalCores()),
      toFlatbuffer(cache, chipDesc.getSupportedDataTypes()),
      toFlatbuffer(cache, chipDesc.getSupportedTileSizes()),
      chipDesc.getNumCBs());
}

inline ::tt::target::CPURole toFlatbuffer(FlatbufferObjectCache &,
                                          CPURole memLayout) {
  switch (memLayout) {
  case CPURole::Host:
    return ::tt::target::CPURole::Host;
  case CPURole::Device:
    return ::tt::target::CPURole::Device;
  }
}

inline flatbuffers::Offset<::tt::target::CPUDesc>
toFlatbuffer(FlatbufferObjectCache &cache, CPUDescAttr cpuDesc) {
  return ::tt::target::CreateCPUDesc(
      *cache.fbb, toFlatbuffer(cache, cpuDesc.getRole()),
      cache.fbb->CreateString(cpuDesc.getTargetTriple().getValue().str()));
}

inline flatbuffers::Offset<::tt::target::SystemDesc>
toFlatbuffer(FlatbufferObjectCache &cache, SystemDescAttr systemDesc) {
  auto cpuDescs = toFlatbuffer(cache, systemDesc.getCpuDescs());
  auto chipDescs = toFlatbuffer(cache, systemDesc.getChipDescs());
  auto chipDescIndices = toFlatbuffer(cache, systemDesc.getChipDescIndices());
  auto chipCapabilities = toFlatbuffer(cache, systemDesc.getChipCapabilities());
  auto chipCoords = toFlatbuffer(cache, systemDesc.getChipCoords());
  auto chipChannels = toFlatbuffer(cache, systemDesc.getChipChannels());
  return ::tt::target::CreateSystemDesc(*cache.fbb, cpuDescs, chipDescs,
                                        chipDescIndices, chipCapabilities,
                                        chipCoords, chipChannels);
}

inline std::vector<::tt::target::Dim2dRange>
toFlatbuffer(FlatbufferObjectCache &cache, GridAttr tensorGrid,
             GridAttr deviceGrid) {
  std::vector<::tt::target::Dim2dRange> coreRangeSet;
  SmallVector<std::int64_t> tensorGridShape(tensorGrid.getShape());
  AffineMap mapping = deviceGrid.getMapping();
  ::ttmlir::utils::sample(
      tensorGridShape, [&](ArrayRef<std::int64_t> virtualCoreCoord) {
        SmallVector<std::int64_t> coreCoord = mapping.compose(virtualCoreCoord);
        assert(coreCoord.size() == PhysGridResultIdx::NumIndices &&
               "expected a 2D core");
        assert(coreCoord[PhysGridResultIdx::DeviceIdx] == 0 &&
               "expected single device");
        if (!coreRangeSet.empty() &&
            ((coreRangeSet.back().loc().y() ==
              coreCoord[PhysGridResultIdx::CoreCoordY]) &&
             (coreRangeSet.back().loc().x() + coreRangeSet.back().size().x()) ==
                 coreCoord[PhysGridResultIdx::CoreCoordX])) {
          coreRangeSet.back() = ::tt::target::Dim2dRange(
              coreRangeSet.back().loc(),
              ::tt::target::Dim2d(coreRangeSet.back().size().y(),
                                  coreRangeSet.back().size().x() + 1));
        } else {
          coreRangeSet.push_back(::tt::target::Dim2dRange(
              ::tt::target::Dim2d(coreCoord[PhysGridResultIdx::CoreCoordY],
                                  coreCoord[PhysGridResultIdx::CoreCoordX]),
              ::tt::target::Dim2d(1, 1)));
        }
        if (coreRangeSet.size() > 1 &&
            (coreRangeSet[coreRangeSet.size() - 2].loc().x() ==
             coreRangeSet.back().loc().x()) &&
            (coreRangeSet[coreRangeSet.size() - 2].size().x() ==
             coreRangeSet.back().size().x()) &&
            ((coreRangeSet[coreRangeSet.size() - 2].loc().y() +
              coreRangeSet[coreRangeSet.size() - 2].size().y()) ==
             coreRangeSet.back().loc().y())) {
          assert(coreRangeSet.back().size().y() == 1);
          coreRangeSet[coreRangeSet.size() - 2] = ::tt::target::Dim2dRange(
              coreRangeSet[coreRangeSet.size() - 2].loc(),
              ::tt::target::Dim2d(
                  coreRangeSet[coreRangeSet.size() - 2].size().y() + 1,
                  coreRangeSet[coreRangeSet.size() - 2].size().x()));
          coreRangeSet.pop_back();
        }
      });
  return coreRangeSet;
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

inline flatbuffers::Offset<flatbuffers::Vector<uint32_t>>
toFlatbuffer(FlatbufferObjectCache &cache, ElementsAttr elementsAttr) {
  assert(elementsAttr.getElementType().isIntOrIndexOrFloat() &&
         "unsupported elements attr type");
  assert(elementsAttr.isSplat() && "expected a splat elements attr");
  assert(elementsAttr.getElementType().getIntOrFloatBitWidth() == 32 &&
         "unsupported elements attr bit width");
  uint32_t value = 0;
  if (elementsAttr.getElementType().isInteger()) {
    value = elementsAttr.getSplatValue<int>();
  } else {
    *(reinterpret_cast<float *>(&value)) = elementsAttr.getSplatValue<float>();
  }
  SmallVector<uint32_t> data({value});
  return toFlatbuffer(cache, ArrayRef<uint32_t>(data));
}

inline flatbuffers::Offset<::tt::target::LayoutDesc>
encodingToFlatbuffer(FlatbufferObjectCache &cache, Attribute attr,
                     ArrayRef<int64_t> logicalShape, DeviceAttr deviceAttr) {
  if (isa<MetalLayoutAttr>(attr)) {
    return metalLayoutAttrToFlatbuffer(cache, cast<MetalLayoutAttr>(attr),
                                       logicalShape, deviceAttr);
  }

  assert(isa<ttnn::TTNNLayoutAttr>(attr) && "unsupported layout attr");
  return ttnnLayoutAttrToFlatbuffer(cache, cast<ttnn::TTNNLayoutAttr>(attr),
                                    logicalShape, deviceAttr);
}

inline flatbuffers::Offset<::tt::target::TensorDesc>
tensorTypeToFlatbuffer(FlatbufferObjectCache &cache, Type type,
                       DeviceAttr deviceAttr) {
  auto tensorType = mlir::cast<RankedTensorType>(type);
  auto shapeInt64 = tensorType.getShape();
  std::vector<int32_t> shape(shapeInt64.begin(), shapeInt64.end());
  return ::tt::target::CreateTensorDescDirect(
      *cache.fbb, &shape,
      cache.getOrCreate(tensorType.getEncoding(), encodingToFlatbuffer,
                        shapeInt64, deviceAttr));
}

inline flatbuffers::Offset<::tt::target::TensorRef>
tensorValueToFlatbuffer(FlatbufferObjectCache &cache, Value value,
                        uint64_t address, uint64_t size) {
  auto deviceAttr =
      getCurrentScopeDevice(value.getParentBlock()->getParentOp());
  assert(deviceAttr);
  auto tensorType = mlir::cast<RankedTensorType>(value.getType());
  auto tensorDesc =
      cache.getOrCreate(tensorType, tensorTypeToFlatbuffer, deviceAttr);
  return ::tt::target::CreateTensorRef(*cache.fbb, cache.global_id++, address,
                                       size, tensorDesc);
}

inline flatbuffers::Offset<::tt::target::MLIR>
toDebugInfo(::flatbuffers::FlatBufferBuilder &fbb, std::string const &name,
            ModuleOp module) {
  std::string source;
  llvm::raw_string_ostream os(source);

  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(); // Enable the loc dumping
  module->print(os, flags);

  return ::tt::target::CreateMLIRDirect(fbb, name.c_str(), source.c_str());
}
} // namespace mlir::tt

#endif
