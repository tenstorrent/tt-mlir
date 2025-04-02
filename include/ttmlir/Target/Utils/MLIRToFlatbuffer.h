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
#include "ttmlir/Target/Utils/FlatbufferObjectCache.h"
#include "ttmlir/Utils.h"

#include "flatbuffers/flatbuffers.h"

#include <numeric>
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

inline ::flatbuffers::Optional<::tt::target::DataType>
toFlatbufferOptional(FlatbufferObjectCache &cache,
                     ::std::optional<::mlir::tt::DataType> dataType) {
  return dataType.has_value() ? ::flatbuffers::Optional<::tt::target::DataType>(
                                    toFlatbuffer(cache, dataType.value()))
                              : ::flatbuffers::nullopt;
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

inline ::flatbuffers::Optional<::tt::target::TensorLayout>
toFlatbufferOptional(FlatbufferObjectCache &cache,
                     ::std::optional<mlir::tt::ttnn::Layout> layout) {
  return layout.has_value()
             ? ::flatbuffers::Optional<::tt::target::TensorLayout>(
                   toFlatbuffer(cache, layout.value()))
             : ::flatbuffers::nullopt;
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

inline flatbuffers::Offset<flatbuffers::String>
toFlatbuffer(FlatbufferObjectCache &cache, llvm::StringRef str) {
  return cache.fbb->CreateString(str.data(), str.size());
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
      chipDesc.getNumCBs(), chipDesc.getNumComputeThreads(),
      chipDesc.getNumDatamovementThreads());
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

  auto mapping = (tensorGrid.getMapping().isEmpty() == true)
                     ? deviceGrid.getMapping()
                     : tensorGrid.getMapping();
  for (const auto &locsize2d :
       utils::toCoreRangeSet(tensorGrid.getShape(), mapping)) {
    const auto &[loc, size] = locsize2d;
    coreRangeSet.push_back(
        ::tt::target::Dim2dRange(::tt::target::Dim2d(loc[1], loc[0]),
                                 ::tt::target::Dim2d(size[1], size[0])));
  }

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
      {MlirUnaryOpType::FloorFloat32, FbUnaryOpType::FloorFloat32},
      {MlirUnaryOpType::Ceil, FbUnaryOpType::Ceil},
      {MlirUnaryOpType::CeilFloat32, FbUnaryOpType::CeilFloat32},
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
      matmulConfigAttr.getNumGlobalCbReceivers());
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

} // namespace mlir::tt

#endif
