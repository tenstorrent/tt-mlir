// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_UTILS_MLIRTOFLATBUFFER_H
#define TTMLIR_TARGET_UTILS_MLIRTOFLATBUFFER_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TT/Utils/CoreRangeSet.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Target/Common/Target.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Target/TTNN/utils.h"
#include "ttmlir/Target/Utils/FlatbufferObjectCache.h"
#include "ttmlir/Utils.h"

#include "flatbuffers/buffer.h"
#include "llvm/ADT/STLForwardCompat.h"

#include <optional>
#include <type_traits>

namespace mlir::tt {

struct GoldenTensor {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  ::tt::target::DataType dtype;
  std::vector<std::uint8_t> data;

  GoldenTensor(std::string name, std::vector<int64_t> shape,
               std::vector<int64_t> strides, ::tt::target::DataType dtype,
               std::vector<std::uint8_t> &&_data)
      : name(name), shape(shape), strides(strides), dtype(dtype),
        data(std::move(_data)) {}

  // Create an explicit empty constructor
  GoldenTensor() = default;
};

inline flatbuffers::Offset<::tt::target::MLIR>
toDebugInfo(::flatbuffers::FlatBufferBuilder &fbb, const std::string &name,
            ModuleOp module) {
  std::string source;
  llvm::raw_string_ostream os(source);

  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(); // Enable the loc dumping
  module->print(os, flags);

  return ::tt::target::CreateMLIRDirect(fbb, name.c_str(), source.c_str());
}

inline flatbuffers::Offset<::tt::target::DebugInfo> debugInfoToFlatbuffer(
    flatbuffers::FlatBufferBuilder &fbb, const std::string &name,
    ModuleOp module,
    const std::unordered_map<std::string, GoldenTensor> &goldenMap,
    const std::vector<std::pair<std::string, std::string>> &moduleCache,
    const char *cpp = nullptr) {
  std::vector<flatbuffers::Offset<::tt::target::GoldenKV>> goldenKVList;
  goldenKVList.reserve(goldenMap.size());

  for (const auto &[key, value] : goldenMap) {
    auto goldenTensor = ::tt::target::CreateGoldenTensorDirect(
        fbb, value.name.c_str(), &value.shape, &value.strides, value.dtype,
        &value.data);
    auto goldenKV =
        ::tt::target::CreateGoldenKVDirect(fbb, key.c_str(), goldenTensor);
    goldenKVList.push_back(goldenKV);
  }

  auto goldenInfo = ::tt::target::CreateGoldenInfoDirect(fbb, &goldenKVList);

  // Load the ModuleCache if present and populate DebugInfo
  std::vector<flatbuffers::Offset<::tt::target::MLIR>> moduleCacheList;
  moduleCacheList.reserve(moduleCache.size());

  for (const auto &item : moduleCache) {
    // Here the Name is the Pass Name and Source is the IR itself
    auto moduleCacheItem = ::tt::target::CreateMLIRDirect(
        fbb, item.first.c_str(), item.second.c_str());
    moduleCacheList.push_back(moduleCacheItem);
  }

  return ::tt::target::CreateDebugInfoDirect(
      fbb, toDebugInfo(fbb, name, module), cpp, &moduleCacheList, goldenInfo);
}

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

inline std::uint64_t getElementSizeBytes(DataType dtype) {
  switch (dtype) {
  case DataType::Float32:
    return 4;
  case DataType::Float16:
    return 2;
  case DataType::BFloat16:
    return 2;
  case DataType::UInt32:
  case DataType::Int32:
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
  case DataType::Int32:
    return ::tt::target::DataType::Int32;
  }
}

inline ::tt::target::MathFidelity
toFlatbuffer(FlatbufferObjectCache &, ttnn::MathFidelity mathFidelity) {
  switch (mathFidelity) {
  case ttnn::MathFidelity::LoFi:
    return ::tt::target::MathFidelity::LoFi;
  case ttnn::MathFidelity::HiFi2:
    return ::tt::target::MathFidelity::HiFi2;
  case ttnn::MathFidelity::HiFi3:
    return ::tt::target::MathFidelity::HiFi3;
  case ttnn::MathFidelity::HiFi4:
    return ::tt::target::MathFidelity::HiFi4;
  }
}

inline ::tt::target::ttnn::TensorMemoryLayout
toFlatbuffer(FlatbufferObjectCache &, ttnn::TensorMemoryLayout memLayout) {
  switch (memLayout) {
  case ttnn::TensorMemoryLayout::SingleBank:
    return ::tt::target::ttnn::TensorMemoryLayout::SingleBank;
  case ttnn::TensorMemoryLayout::Interleaved:
    return ::tt::target::ttnn::TensorMemoryLayout::Interleaved;
  case ttnn::TensorMemoryLayout::HeightSharded:
    return ::tt::target::ttnn::TensorMemoryLayout::HeightSharded;
  case ttnn::TensorMemoryLayout::WidthSharded:
    return ::tt::target::ttnn::TensorMemoryLayout::WidthSharded;
  case ttnn::TensorMemoryLayout::BlockSharded:
    return ::tt::target::ttnn::TensorMemoryLayout::BlockSharded;
  }
}

inline ::tt::target::ttnn::TensorMemoryLayout
toFlatbuffer(FlatbufferObjectCache &cache,
             ttnn::TensorMemoryLayoutAttr memLayoutAttr) {
  return toFlatbuffer(cache, memLayoutAttr.getValue());
}

inline ::tt::target::MemorySpace toFlatbuffer(FlatbufferObjectCache &,
                                              ttnn::BufferType bufferType) {
  switch (bufferType) {
  case ttnn::BufferType::SystemMemory:
    return ::tt::target::MemorySpace::System;
  case ttnn::BufferType::DRAM:
    return ::tt::target::MemorySpace::DeviceDRAM;
  case ttnn::BufferType::L1:
    return ::tt::target::MemorySpace::DeviceL1;
  default:
    llvm_unreachable("unhandled buffer type");
  }
}

inline ::tt::target::TensorLayout toFlatbuffer(FlatbufferObjectCache &cache,
                                               ttnn::Layout layout) {
  switch (layout) {
  case ttnn::Layout::RowMajor:
    return ::tt::target::TensorLayout::RowMajor;
  case ttnn::Layout::Tile:
    return ::tt::target::TensorLayout::Tile;
  case ttnn::Layout::Invalid:
    return ::tt::target::TensorLayout::Invalid;
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
  case MemorySpace::RegisterDst:
    llvm_unreachable("MemorySpace::RegisterDst not supported");
  }
}

inline ::tt::target::ttnn::ShardOrientation
toFlatbuffer(FlatbufferObjectCache &, ttnn::ShardOrientation orientation) {
  switch (orientation) {
  case ttnn::ShardOrientation::RowMajor:
    return ::tt::target::ttnn::ShardOrientation::RowMajor;
  case ttnn::ShardOrientation::ColMajor:
    return ::tt::target::ttnn::ShardOrientation::ColMajor;
  }
}

inline ::tt::target::ttnn::ShardMode toFlatbuffer(FlatbufferObjectCache &,
                                                  ttnn::ShardMode mode) {
  switch (mode) {
  case ttnn::ShardMode::Physical:
    return ::tt::target::ttnn::ShardMode::Physical;
  case ttnn::ShardMode::Logical:
    return ::tt::target::ttnn::ShardMode::Logical;
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
  static_assert(llvm::to_underlying(ChipCapability::PCIE) ==
                llvm::to_underlying(::tt::target::ChipCapability::PCIE));
  static_assert(llvm::to_underlying(ChipCapability::HostMMIO) ==
                llvm::to_underlying(::tt::target::ChipCapability::HostMMIO));
  assert((llvm::to_underlying(capabilities) & ~0b11) == 0 &&
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

inline flatbuffers::Offset<::tt::target::ChipPhysicalHelperCores>
toFlatbuffer(FlatbufferObjectCache &cache,
             ChipPhysicalHelperCoresAttr chipPhysicalHelperCores) {

  // Create a Flatbuffer Dim2d struct for each type of core.
  std::vector<::tt::target::Dim2d> dramCores, ethCores, ethInactiveCores;

  for (const auto &coreCoord : chipPhysicalHelperCores.getDram()) {
    dramCores.emplace_back(coreCoord.getY(), coreCoord.getX());
  }
  for (const auto &coreCoord : chipPhysicalHelperCores.getEth()) {
    ethCores.emplace_back(coreCoord.getY(), coreCoord.getX());
  }
  for (const auto &coreCoord : chipPhysicalHelperCores.getEthInactive()) {
    ethInactiveCores.emplace_back(coreCoord.getY(), coreCoord.getX());
  }

  // Create and return the ChipPhysicalHelperCores flatbuffer object
  return ::tt::target::CreateChipPhysicalHelperCores(
      *cache.fbb,
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
    T, std::void_t<typename ToFlatbufferReturnType<T>::Traits::type>>
    : std::true_type {};

template <typename T>
constexpr bool IsNativeFlatbufferTypeV = IsNativeFlatbufferType<T>::value;

template <typename T>
flatbuffers::Optional<ToFlatbufferReturnType<T>>
toFlatbuffer(FlatbufferObjectCache &cache, const std::optional<T> &optValue) {
  return llvm::transformOptional(
      optValue, [&](const T &val) { return toFlatbuffer(cache, val); });
}

template <typename T, std::enable_if_t<IsNativeFlatbufferTypeV<T>, int> = 0>
flatbuffers::Offset<flatbuffers::Vector<const ToFlatbufferReturnType<T> *>>
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

template <typename T, std::enable_if_t<!IsNativeFlatbufferTypeV<T>, int> = 0>
flatbuffers::Offset<flatbuffers::Vector<ToFlatbufferReturnType<T>>>
toFlatbuffer(FlatbufferObjectCache &cache, ::llvm::ArrayRef<T> arr) {
  return cache.fbb->CreateVector<ToFlatbufferReturnType<T>>(
      arr.size(),
      [&cache, arr](size_t i) { return toFlatbuffer(cache, arr[i]); });
}

inline flatbuffers::Offset<flatbuffers::String>
toFlatbuffer(FlatbufferObjectCache &cache, llvm::StringRef str) {
  return cache.fbb->CreateString(str.data(), str.size());
}

inline flatbuffers::Offset<::tt::target::ChipDesc>
toFlatbuffer(FlatbufferObjectCache &cache, ChipDescAttr chipDesc) {
  assert(chipDesc.getGrid().size() == 2 && "expected a 2D grid");
  auto grid = ::tt::target::Dim2d(chipDesc.getGrid()[0], chipDesc.getGrid()[1]);
  auto coordTranslationOffsets =
      ::tt::target::Dim2d(chipDesc.getCoordTranslationOffsets()[0],
                          chipDesc.getCoordTranslationOffsets()[1]);
  return ::tt::target::CreateChipDesc(
      *cache.fbb, toFlatbuffer(cache, chipDesc.getArch()), &grid,
      &coordTranslationOffsets, chipDesc.getL1Size(),
      chipDesc.getNumDramChannels(), chipDesc.getDramChannelSize(),
      chipDesc.getNocL1AddressAlignBytes(), chipDesc.getPcieAddressAlignBytes(),
      chipDesc.getNocDRAMAddressAlignBytes(), chipDesc.getL1UnreservedBase(),
      chipDesc.getEriscL1UnreservedBase(), chipDesc.getDramUnreservedBase(),
      chipDesc.getDramUnreservedEnd(),
      toFlatbuffer(cache, chipDesc.getChipPhysicalHelperCores()),
      toFlatbuffer(cache, chipDesc.getSupportedDataTypes()),
      toFlatbuffer(cache, chipDesc.getSupportedTileSizes()),
      chipDesc.getDstRegisterSizeTiles(), chipDesc.getNumCBs(),
      chipDesc.getNumComputeThreads(), chipDesc.getNumDatamovementThreads());
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
toFlatbuffer(FlatbufferObjectCache &cache, llvm::ArrayRef<int64_t> tensorGrid,
             mlir::AffineMap mapping) {
  std::vector<::tt::target::Dim2dRange> coreRangeSet;

  for (const auto &locsize2d : utils::toCoreRangeSet(tensorGrid, mapping)) {
    const auto &[loc, size] = locsize2d;
    coreRangeSet.push_back(
        ::tt::target::Dim2dRange(::tt::target::Dim2d(loc[1], loc[0]),
                                 ::tt::target::Dim2d(size[1], size[0])));
  }

  return coreRangeSet;
}

// Compatibility function for TTNN flatbuffer serialization.
inline std::vector<::tt::target::Dim2dRange>
toFlatbuffer(FlatbufferObjectCache &cache, GridAttr tensorGrid,
             GridAttr deviceGrid) {
  auto mapping = tensorGrid.getMapping().isEmpty() ? deviceGrid.getMapping()
                                                   : tensorGrid.getMapping();
  return toFlatbuffer(cache, tensorGrid.getShape(), mapping);
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

inline double toFlatbuffer(FlatbufferObjectCache &, mlir::FloatAttr attr) {
  return attr.getValueAsDouble();
}

inline ::tt::target::ttnn::CoreCoord
toFlatbuffer(FlatbufferObjectCache &cache, ttnn::CoreCoordAttr coreCoordAttr) {
  return ::tt::target::ttnn::CoreCoord(coreCoordAttr.getX(),
                                       coreCoordAttr.getY());
}

inline ::tt::target::ttnn::CoreRange
toFlatbuffer(FlatbufferObjectCache &cache, ttnn::CoreRangeAttr coreRangeAttr) {
  return ::tt::target::ttnn::CoreRange(
      toFlatbuffer(cache, coreRangeAttr.getStartCoord()),
      toFlatbuffer(cache, coreRangeAttr.getEndCoord()));
}

inline ::flatbuffers::Offset<::tt::target::ttnn::CoreRangeSet>
toFlatbuffer(FlatbufferObjectCache &cache,
             ttnn::CoreRangeSetAttr coreRangeSetAttr) {
  if (!coreRangeSetAttr) {
    return 0;
  }

  return ::tt::target::ttnn::CreateCoreRangeSet(
      *cache.fbb, toFlatbuffer(cache, coreRangeSetAttr.getCoreRanges()));
}

inline ::tt::target::ttnn::UnaryOpType
toFlatbuffer(FlatbufferObjectCache &, ttnn::UnaryOpType unaryOpType) {
  using MlirUnaryOpType = ::mlir::tt::ttnn::UnaryOpType;
  using FbUnaryOpType = ::tt::target::ttnn::UnaryOpType;

  static const std::unordered_map<MlirUnaryOpType, FbUnaryOpType> opTypeMap = {
      {MlirUnaryOpType::Exp, FbUnaryOpType::Exp},
      {MlirUnaryOpType::Recip, FbUnaryOpType::Recip},
      {MlirUnaryOpType::Gelu, FbUnaryOpType::Gelu},
      {MlirUnaryOpType::Relu, FbUnaryOpType::Relu},
      {MlirUnaryOpType::Sqrt, FbUnaryOpType::Sqrt},
      {MlirUnaryOpType::Sigmoid, FbUnaryOpType::Sigmoid},
      {MlirUnaryOpType::Log, FbUnaryOpType::Log},
      {MlirUnaryOpType::Tanh, FbUnaryOpType::Tanh},
      {MlirUnaryOpType::Log2, FbUnaryOpType::Log2},
      {MlirUnaryOpType::Log10, FbUnaryOpType::Log10},
      {MlirUnaryOpType::Sin, FbUnaryOpType::Sin},
      {MlirUnaryOpType::Cos, FbUnaryOpType::Cos},
      {MlirUnaryOpType::Abs, FbUnaryOpType::Abs},
      {MlirUnaryOpType::AbsInt32, FbUnaryOpType::AbsInt32},
      {MlirUnaryOpType::Sign, FbUnaryOpType::Sign},
      {MlirUnaryOpType::Square, FbUnaryOpType::Square},
      {MlirUnaryOpType::Eqz, FbUnaryOpType::Eqz},
      {MlirUnaryOpType::Nez, FbUnaryOpType::Nez},
      {MlirUnaryOpType::Gtz, FbUnaryOpType::Gtz},
      {MlirUnaryOpType::Ltz, FbUnaryOpType::Ltz},
      {MlirUnaryOpType::Gez, FbUnaryOpType::Gez},
      {MlirUnaryOpType::Lez, FbUnaryOpType::Lez},
      {MlirUnaryOpType::ReluMax, FbUnaryOpType::ReluMax},
      {MlirUnaryOpType::ReluMin, FbUnaryOpType::ReluMin},
      {MlirUnaryOpType::Power, FbUnaryOpType::Power},
      {MlirUnaryOpType::LeakyRelu, FbUnaryOpType::LeakyRelu},
      {MlirUnaryOpType::Elu, FbUnaryOpType::Elu},
      {MlirUnaryOpType::Exp2, FbUnaryOpType::Exp2},
      {MlirUnaryOpType::Heaviside, FbUnaryOpType::Heaviside},
      {MlirUnaryOpType::Expm1, FbUnaryOpType::Expm1},
      {MlirUnaryOpType::Signbit, FbUnaryOpType::Signbit},
      {MlirUnaryOpType::Asin, FbUnaryOpType::Asin},
      {MlirUnaryOpType::Acos, FbUnaryOpType::Acos},
      {MlirUnaryOpType::Rsqrt, FbUnaryOpType::Rsqrt},
      {MlirUnaryOpType::Relu6, FbUnaryOpType::Relu6},
      {MlirUnaryOpType::Atan, FbUnaryOpType::Atan},
      {MlirUnaryOpType::Erf, FbUnaryOpType::Erf},
      {MlirUnaryOpType::Erfc, FbUnaryOpType::Erfc},
      {MlirUnaryOpType::IsInf, FbUnaryOpType::Isinf},
      {MlirUnaryOpType::IsPosInf, FbUnaryOpType::Isposinf},
      {MlirUnaryOpType::IsNegInf, FbUnaryOpType::Isneginf},
      {MlirUnaryOpType::IsNan, FbUnaryOpType::Isnan},
      {MlirUnaryOpType::LogicalNotUnary, FbUnaryOpType::LogicalNotUnary},
      {MlirUnaryOpType::IsFinite, FbUnaryOpType::Isfinite},
      {MlirUnaryOpType::Erfinv, FbUnaryOpType::Erfinv},
      {MlirUnaryOpType::I0, FbUnaryOpType::I0},
      {MlirUnaryOpType::I1, FbUnaryOpType::I1},
      {MlirUnaryOpType::Tan, FbUnaryOpType::Tan},
      {MlirUnaryOpType::Rsub, FbUnaryOpType::Rsub},
      {MlirUnaryOpType::Rdiv, FbUnaryOpType::Rdiv},
      {MlirUnaryOpType::Silu, FbUnaryOpType::Silu},
      {MlirUnaryOpType::SoftPlus, FbUnaryOpType::Softplus},
      {MlirUnaryOpType::Identity, FbUnaryOpType::Identity},
      {MlirUnaryOpType::Neg, FbUnaryOpType::Neg},
      {MlirUnaryOpType::AddUnarySfpu, FbUnaryOpType::AddUnarySfpu},
      {MlirUnaryOpType::SubUnarySfpu, FbUnaryOpType::SubUnarySfpu},
      {MlirUnaryOpType::MulUnarySfpu, FbUnaryOpType::MulUnarySfpu},
      {MlirUnaryOpType::DivUnarySfpu, FbUnaryOpType::DivUnarySfpu},
      {MlirUnaryOpType::IdentityUint32, FbUnaryOpType::IdentityUint32},
      {MlirUnaryOpType::UnaryNe, FbUnaryOpType::UnaryNe},
      {MlirUnaryOpType::UnaryGt, FbUnaryOpType::UnaryGt},
      {MlirUnaryOpType::UnaryLt, FbUnaryOpType::UnaryLt},
      {MlirUnaryOpType::TiledProd, FbUnaryOpType::TiledProd},
      {MlirUnaryOpType::Typecast, FbUnaryOpType::Typecast},
      {MlirUnaryOpType::BitwiseXor, FbUnaryOpType::BitwiseXor},
      {MlirUnaryOpType::BitwiseNot, FbUnaryOpType::BitwiseNot},
      {MlirUnaryOpType::BitwiseAnd, FbUnaryOpType::BitwiseAnd},
      {MlirUnaryOpType::BitwiseOr, FbUnaryOpType::BitwiseOr},
      {MlirUnaryOpType::RightShift, FbUnaryOpType::RightShift},
      {MlirUnaryOpType::Floor, FbUnaryOpType::Floor},
      {MlirUnaryOpType::Ceil, FbUnaryOpType::Ceil},
      {MlirUnaryOpType::Round, FbUnaryOpType::Round},
      {MlirUnaryOpType::LeftShift, FbUnaryOpType::LeftShift},
      {MlirUnaryOpType::Remainder, FbUnaryOpType::Remainder},
      {MlirUnaryOpType::Fmod, FbUnaryOpType::Fmod},
      {MlirUnaryOpType::Dropout, FbUnaryOpType::Dropout},
      {MlirUnaryOpType::Fill, FbUnaryOpType::Fill},
      {MlirUnaryOpType::PreluSfpu, FbUnaryOpType::PreluSfpu},
      {MlirUnaryOpType::ZeroPoint, FbUnaryOpType::ZeroPoint}};

  auto it = opTypeMap.find(unaryOpType);
  if (it != opTypeMap.end()) {
    return it->second;
  }

  llvm_unreachable("Unsupported unary op type");
}

inline ::flatbuffers::Offset<
    ::tt::target::ttnn::MatmulMultiCoreReuseProgramConfig>
toFlatbuffer(FlatbufferObjectCache &cache,
             ttnn::MatmulMultiCoreReuseProgramConfigAttr matmulConfigAttr) {
  ::tt::target::ttnn::CoreCoord computeWithStorageGridSize =
      toFlatbuffer(cache, matmulConfigAttr.getComputeWithStorageGridSize());
  return ::tt::target::ttnn::CreateMatmulMultiCoreReuseProgramConfig(
      *cache.fbb, &computeWithStorageGridSize, matmulConfigAttr.getIn0BlockW(),
      matmulConfigAttr.getOutSubblockH(), matmulConfigAttr.getOutSubblockW(),
      matmulConfigAttr.getPerCoreM(), matmulConfigAttr.getPerCoreN());
}

inline ::flatbuffers::Offset<::tt::target::ttnn::UnaryWithParam>
toFlatbuffer(FlatbufferObjectCache &cache,
             ttnn::UnaryWithParamAttr unaryWithParam) {
  return ::tt::target::ttnn::CreateUnaryWithParam(
      *cache.fbb, toFlatbuffer(cache, unaryWithParam.getOpType()),
      toFlatbuffer(cache, unaryWithParam.getParams()));
}

inline ::flatbuffers::Offset<
    ::tt::target::ttnn::MatmulMultiCoreReuseMultiCastProgramConfig>
toFlatbuffer(
    FlatbufferObjectCache &cache,
    ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr matmulConfigAttr) {
  ::tt::target::ttnn::CoreCoord computeWithStorageGridSize =
      toFlatbuffer(cache, matmulConfigAttr.getComputeWithStorageGridSize());
  ::flatbuffers::Offset<::tt::target::ttnn::UnaryWithParam> fusedActivation;
  if (matmulConfigAttr.getFusedActivation()) {
    fusedActivation =
        toFlatbuffer(cache, matmulConfigAttr.getFusedActivation());
  }
  return ::tt::target::ttnn::CreateMatmulMultiCoreReuseMultiCastProgramConfig(
      *cache.fbb, &computeWithStorageGridSize, matmulConfigAttr.getIn0BlockW(),
      matmulConfigAttr.getOutSubblockH(), matmulConfigAttr.getOutSubblockW(),
      matmulConfigAttr.getOutBlockH(), matmulConfigAttr.getOutBlockW(),
      matmulConfigAttr.getPerCoreM(), matmulConfigAttr.getPerCoreN(),
      matmulConfigAttr.getTransposeMcast(), fusedActivation,
      matmulConfigAttr.getFuseBatch());
}

inline ::flatbuffers::Offset<
    ::tt::target::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfig>
toFlatbuffer(
    FlatbufferObjectCache &cache,
    ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr matmulConfigAttr) {
  ::tt::target::ttnn::CoreCoord computeWithStorageGridSize =
      toFlatbuffer(cache, matmulConfigAttr.getComputeWithStorageGridSize());
  ::flatbuffers::Offset<::tt::target::ttnn::UnaryWithParam> fusedActivation;
  if (matmulConfigAttr.getFusedActivation()) {
    fusedActivation =
        toFlatbuffer(cache, matmulConfigAttr.getFusedActivation());
  }
  return ::tt::target::ttnn::CreateMatmulMultiCoreReuseMultiCast1DProgramConfig(
      *cache.fbb, &computeWithStorageGridSize, matmulConfigAttr.getIn0BlockW(),
      matmulConfigAttr.getOutSubblockH(), matmulConfigAttr.getOutSubblockW(),
      matmulConfigAttr.getOutBlockH(), matmulConfigAttr.getOutBlockW(),
      matmulConfigAttr.getPerCoreM(), matmulConfigAttr.getPerCoreN(),
      matmulConfigAttr.getFuseBatch(), fusedActivation,
      matmulConfigAttr.getMcastIn0(), matmulConfigAttr.getGatherIn0(),
      toFlatbuffer(cache, matmulConfigAttr.getHopCores()),
      matmulConfigAttr.getNumGlobalCbReceivers(),
      matmulConfigAttr.getUntilizeOut());
}

inline ::flatbuffers::Offset<
    ::tt::target::ttnn::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>
toFlatbuffer(FlatbufferObjectCache &cache,
             ttnn::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr
                 matmulConfigAttr) {
  ::flatbuffers::Offset<::tt::target::ttnn::UnaryWithParam> fusedActivation;
  if (matmulConfigAttr.getFusedActivation()) {
    fusedActivation =
        toFlatbuffer(cache, matmulConfigAttr.getFusedActivation());
  }
  return ::tt::target::ttnn::
      CreateMatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
          *cache.fbb, matmulConfigAttr.getIn0BlockW(),
          matmulConfigAttr.getPerCoreM(), matmulConfigAttr.getPerCoreN(),
          fusedActivation);
}

inline ::flatbuffers::Offset<::flatbuffers::String>
toFlatbuffer(FlatbufferObjectCache &cache, StringAttr strAttr) {
  if (strAttr) {
    return toFlatbuffer(cache, strAttr.getValue());
  }

  return 0;
}

inline ::flatbuffers::Optional<bool> toFlatbuffer(FlatbufferObjectCache &cache,
                                                  BoolAttr attr) {
  if (attr) {
    return attr.getValue();
  }

  return ::flatbuffers::nullopt;
}

inline ::flatbuffers::Offset<::tt::target::ttnn::Conv2dConfig>
toFlatbuffer(FlatbufferObjectCache &cache, ttnn::Conv2dConfigAttr config) {
  return ::tt::target::ttnn::CreateConv2dConfig(
      *cache.fbb, toFlatbuffer(cache, config.getDtype()),
      toFlatbuffer(cache, config.getWeightsDtype()),
      toFlatbuffer(cache, config.getActivation()),
      toFlatbuffer(cache, config.getDeallocateActivation()),
      toFlatbuffer(cache, config.getReallocateHaloOutput()),
      toFlatbuffer(cache, config.getActBlockHOverride()),
      toFlatbuffer(cache, config.getActBlockWDiv()),
      toFlatbuffer(cache, config.getReshardIfNotOptimal()),
      toFlatbuffer(cache, config.getOverrideShardingConfig()),
      toFlatbuffer(cache, config.getShardLayout()),
      toFlatbuffer(cache, config.getCoreGrid()),
      toFlatbuffer(cache, config.getTransposeShards()),
      toFlatbuffer(cache, config.getOutputLayout()),
      toFlatbuffer(cache, config.getPreprocessWeightsOnDevice()),
      toFlatbuffer(cache, config.getAlwaysPreprocessWeights()),
      toFlatbuffer(cache, config.getEnableActDoubleBuffer()),
      toFlatbuffer(cache, config.getEnableWeightsDoubleBuffer()),
      toFlatbuffer(cache, config.getEnableSplitReader()),
      toFlatbuffer(cache, config.getEnableSubblockPadding()));
}

inline ::flatbuffers::Offset<::tt::target::ttnn::DeviceComputeKernelConfig>
toFlatbuffer(FlatbufferObjectCache &cache,
             ttnn::DeviceComputeKernelConfigAttr computeConfigAttr) {
  return ::tt::target::ttnn::CreateDeviceComputeKernelConfig(
      *cache.fbb, toFlatbuffer(cache, computeConfigAttr.getMathFidelity()),
      toFlatbuffer(cache, computeConfigAttr.getMathApproxMode()),
      toFlatbuffer(cache, computeConfigAttr.getFp32DestAccEn()),
      toFlatbuffer(cache, computeConfigAttr.getPackerL1Acc()),
      toFlatbuffer(cache, computeConfigAttr.getDstFullSyncEn()));
}

template <typename T>
static flatbuffers::Offset<::flatbuffers::Vector<uint8_t>>
toFlatbufferByteVector(FlatbufferObjectCache &cache,
                       mlir::DenseElementsAttr &attr) {
  size_t sizeBytes = attr.getNumElements() * sizeof(T);
  cache.fbb->StartVector<flatbuffers::Offset<uint8_t>>(sizeBytes);

  // Iterate over the values to make sure splat is unrolled correctly
  for (auto i = attr.value_begin<T>(); i != attr.value_end<T>(); i++) {
    T value = *i;
    uint8_t *buf = reinterpret_cast<uint8_t *>(&value);
    cache.fbb->PushBytes(buf, sizeof(T));
  }
  return cache.fbb->EndVector(sizeBytes);
}

inline ::flatbuffers::Offset<::tt::target::ttnn::ShardSpec>
toFlatbuffer(FlatbufferObjectCache &cache,
             ::mlir::tt::ttnn::ShardSpecAttr shardSpec) {
  auto coreRangeSet = toFlatbuffer(cache, shardSpec.getCoreRangeSet());
  llvm::ArrayRef<int64_t> shardShapeArr = shardSpec.getShape().getShape();
  assert(shardShapeArr.size() == 2);
  std::vector<int32_t> shardShape;
  shardShape.reserve(shardShapeArr.size());
  std::transform(shardShapeArr.begin(), shardShapeArr.end(),
                 std::back_inserter(shardShape), [](int64_t val) -> int32_t {
                   return static_cast<int32_t>(val);
                 });
  auto shardOrientation =
      toFlatbuffer(cache, shardSpec.getShardOrientation().getValue());
  auto shardMode = toFlatbuffer(cache, shardSpec.getShardMode().getValue());

  return ::tt::target::ttnn::CreateShardSpecDirect(
      *cache.fbb, coreRangeSet, &shardShape, shardOrientation, shardMode);
}

inline ::flatbuffers::Offset<::tt::target::ttnn::MemoryConfig>
toFlatbuffer(FlatbufferObjectCache &cache,
             ::mlir::tt::ttnn::MemoryConfigAttr memoryConfigAttr) {
  ttnn::TensorMemoryLayoutAttr tensorMemoryLayoutAttr =
      memoryConfigAttr.getTensorMemoryLayout();
  ::tt::target::ttnn::TensorMemoryLayout tensorMemoryLayout =
      toFlatbuffer(cache, tensorMemoryLayoutAttr);
  ::tt::target::BufferType bufferType =
      ::mlir::tt::ttnn::utils::toTargetBufferType(
          memoryConfigAttr.getBufferType().getValue());

  ::flatbuffers::Offset<::tt::target::ttnn::ShardSpec> shardSpec = 0;
  if (memoryConfigAttr.getShardSpec()) {
    assert(tensorMemoryLayoutAttr && mlir::tt::ttnn::isShardedMemoryLayout(
                                         tensorMemoryLayoutAttr.getValue()));
    shardSpec = toFlatbuffer(cache, *memoryConfigAttr.getShardSpec());
  }
  ::flatbuffers::Offset<::tt::target::ttnn::MemoryConfig> memoryConfig =
      ::tt::target::ttnn::CreateMemoryConfig(*cache.fbb, tensorMemoryLayout,
                                             bufferType, shardSpec);
  return memoryConfig;
}

inline flatbuffers::Offset<::tt::target::ttnn::MemoryDesc>
toFlatbuffer(FlatbufferObjectCache &cache, mlir::MemRefType memref,
             tt::TensorMeshShardingAttr tensorMeshSharding,
             ttnn::BufferType bufferType,
             ttnn::TensorMemoryLayoutAttr memLayoutAttr, tt::GridAttr shardGrid,
             tt::GridAttr deviceGrid) {
  auto shapeInt64 = memref.getShape();
  std::vector<int32_t> shape(shapeInt64.begin(), shapeInt64.end());
  DataType dtype = DataType::Float32;
  ::tt::target::Dim2d tileShape(1, 1);
  mlir::Type elementType = memref.getElementType();
  std::uint64_t elementSize = 0;
  if (mlir::isa<TileType>(elementType)) {
    auto tileType = mlir::cast<TileType>(elementType);
    dtype = tileType.getDataType();
    tileShape = ::tt::target::Dim2d(tileType.getHeight(), tileType.getWidth());
    elementSize = tileType.getSizeBytes();
  } else {
    dtype = elementTypeToDataType(elementType);
    elementSize = getElementSizeBytes(dtype);
  }

  std::uint64_t size = elementSize;
  for (auto dim : shapeInt64) {
    size *= dim;
  }

  ::tt::target::ttnn::StorageType storageType;
  if (tensorMeshSharding) {
    storageType = bufferType == ttnn::BufferType::SystemMemory
                      ? ::tt::target::ttnn::StorageType::MultiDeviceHost
                      : ::tt::target::ttnn::StorageType::Device;
  } else {
    storageType = bufferType == ttnn::BufferType::SystemMemory
                      ? ::tt::target::ttnn::StorageType::Host
                      : ::tt::target::ttnn::StorageType::Device;
  }

  ::flatbuffers::Offset<::tt::target::ttnn::MemoryConfig> memoryConfig = 0;

  // Only device tensors should have a memory config
  if (bufferType != ttnn::BufferType::SystemMemory) {
    ::mlir::MLIRContext *ctx = memref.getContext();
    auto bufferTypeAttr = ttnn::BufferTypeAttr::get(ctx, bufferType);
    std::optional<mlir::tt::ttnn::ShardSpecAttr> shardSpecAttr = std::nullopt;
    if (isShardedMemoryLayout(memLayoutAttr.getValue())) {
      llvm::SmallVector<int64_t> shape(memref.getShape().begin(),
                                       memref.getShape().end());
      assert(shape.size() == 2);
      shape[0] *= tileShape.y();
      shape[1] *= tileShape.x();
      shardSpecAttr = ttnn::ShardSpecAttr::get(
          ctx, ttnn::ShapeAttr::get(ctx, shape), shardGrid, deviceGrid);
    }
    auto memoryConfigAttr = ::mlir::tt::ttnn::MemoryConfigAttr::get(
        ctx, memLayoutAttr, bufferTypeAttr, shardSpecAttr);

    memoryConfig = toFlatbuffer(cache, memoryConfigAttr);
  }

  return ::tt::target::ttnn::CreateMemoryDesc(
      *cache.fbb, storageType, &tileShape, toFlatbuffer(cache, dtype),
      memoryConfig, size);
}

inline flatbuffers::Offset<::tt::target::ttnn::LayoutDesc>
ttnnLayoutAttrToFlatbuffer(FlatbufferObjectCache &cache,
                           ttnn::TTNNLayoutAttr layoutAttr,
                           DeviceAttr deviceAttr) {
  // TODO (jnie): Memory reference alone is insufficient to determine LayoutDesc
  // uniquely. Using `cache.getOrCreate()` is unsafe because identical memory
  // references can produce different LayoutDesc objects.
  // Current state: Removed cache.getOrCreate() to prevent inconsistencies
  // Ideally, we establish one-to-one mapping between MLIR and FlatBuffer
  // that guarantees identical memrefs will always produce identical
  // flatbuffer LayoutDescs.
  return ::tt::target::ttnn::CreateLayoutDesc(
      *cache.fbb, toFlatbuffer(cache, OOBVal::Undef),
      toFlatbuffer(cache, layoutAttr.getMemref(),
                   layoutAttr.getTensorMeshSharding(),
                   layoutAttr.getBufferType(), layoutAttr.getMemLayout(),
                   layoutAttr.getGrid(), deviceAttr.getWorkerGrid()));
}

} // namespace mlir::tt

#endif
