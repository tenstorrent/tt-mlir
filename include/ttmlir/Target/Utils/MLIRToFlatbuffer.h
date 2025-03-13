// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_UTILS_MLIRTOFLATBUFFER_H
#define TTMLIR_TARGET_UTILS_MLIRTOFLATBUFFER_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TT/Utils/CoreRangeSet.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Target/Common/Target.h"
#include "ttmlir/Target/TTNN/types_generated.h"
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

inline int64_t toFlatbuffer(FlatbufferObjectCache &, mlir::IntegerAttr attr) {
  return attr.getInt();
}

inline double toFlatbuffer(FlatbufferObjectCache &, mlir::FloatAttr attr) {
  return attr.getValueAsDouble();
}

inline bool toFlatbuffer(FlatbufferObjectCache &, mlir::BoolAttr attr) {
  return attr.getValue();
}

inline ::tt::target::ttnn::CoreCoord
toFlatbuffer(FlatbufferObjectCache &cache, ttnn::CoreCoordAttr coreCoordAttr) {
  return ::tt::target::ttnn::CoreCoord(
      toFlatbuffer(cache, coreCoordAttr.getX()),
      toFlatbuffer(cache, coreCoordAttr.getY()));
}

inline ::tt::target::ttnn::UnaryOpType
toFlatbuffer(FlatbufferObjectCache &, ttnn::UnaryOpType unaryOpType) {
  switch (unaryOpType) {
  case ttnn::UnaryOpType::Exp:
    return ::tt::target::ttnn::UnaryOpType::Exp;
  case ttnn::UnaryOpType::Recip:
    return ::tt::target::ttnn::UnaryOpType::Recip;
  case ttnn::UnaryOpType::Gelu:
    return ::tt::target::ttnn::UnaryOpType::Gelu;
  case ttnn::UnaryOpType::Relu:
    return ::tt::target::ttnn::UnaryOpType::Relu;
  case ttnn::UnaryOpType::Sqrt:
    return ::tt::target::ttnn::UnaryOpType::Sqrt;
  case ttnn::UnaryOpType::Sigmoid:
    return ::tt::target::ttnn::UnaryOpType::Sigmoid;
  case ttnn::UnaryOpType::Log:
    return ::tt::target::ttnn::UnaryOpType::Log;
  case ttnn::UnaryOpType::Tanh:
    return ::tt::target::ttnn::UnaryOpType::Tanh;
  case ttnn::UnaryOpType::Log2:
    return ::tt::target::ttnn::UnaryOpType::Log2;
  case ttnn::UnaryOpType::Log10:
    return ::tt::target::ttnn::UnaryOpType::Log10;
  case ttnn::UnaryOpType::Sin:
    return ::tt::target::ttnn::UnaryOpType::Sin;
  case ttnn::UnaryOpType::Cos:
    return ::tt::target::ttnn::UnaryOpType::Cos;
  case ttnn::UnaryOpType::Abs:
    return ::tt::target::ttnn::UnaryOpType::Abs;
  case ttnn::UnaryOpType::AbsInt32:
    return ::tt::target::ttnn::UnaryOpType::AbsInt32;
  case ttnn::UnaryOpType::Sign:
    return ::tt::target::ttnn::UnaryOpType::Sign;
  case ttnn::UnaryOpType::Square:
    return ::tt::target::ttnn::UnaryOpType::Square;
  case ttnn::UnaryOpType::Eqz:
    return ::tt::target::ttnn::UnaryOpType::Eqz;
  case ttnn::UnaryOpType::Nez:
    return ::tt::target::ttnn::UnaryOpType::Nez;
  case ttnn::UnaryOpType::Gtz:
    return ::tt::target::ttnn::UnaryOpType::Gtz;
  case ttnn::UnaryOpType::Ltz:
    return ::tt::target::ttnn::UnaryOpType::Ltz;
  case ttnn::UnaryOpType::Gez:
    return ::tt::target::ttnn::UnaryOpType::Gez;
  case ttnn::UnaryOpType::Lez:
    return ::tt::target::ttnn::UnaryOpType::Lez;
  case ttnn::UnaryOpType::ReluMax:
    return ::tt::target::ttnn::UnaryOpType::ReluMax;
  case ttnn::UnaryOpType::ReluMin:
    return ::tt::target::ttnn::UnaryOpType::ReluMin;
  case ttnn::UnaryOpType::Power:
    return ::tt::target::ttnn::UnaryOpType::Power;
  case ttnn::UnaryOpType::LeakyRelu:
    return ::tt::target::ttnn::UnaryOpType::LeakyRelu;
  case ttnn::UnaryOpType::Elu:
    return ::tt::target::ttnn::UnaryOpType::Elu;
  case ttnn::UnaryOpType::Exp2:
    return ::tt::target::ttnn::UnaryOpType::Exp2;
  case ttnn::UnaryOpType::Heaviside:
    return ::tt::target::ttnn::UnaryOpType::Heaviside;
  case ttnn::UnaryOpType::Expm1:
    return ::tt::target::ttnn::UnaryOpType::Expm1;
  case ttnn::UnaryOpType::Signbit:
    return ::tt::target::ttnn::UnaryOpType::Signbit;
  case ttnn::UnaryOpType::Asin:
    return ::tt::target::ttnn::UnaryOpType::Asin;
  case ttnn::UnaryOpType::Acos:
    return ::tt::target::ttnn::UnaryOpType::Acos;
  case ttnn::UnaryOpType::Rsqrt:
    return ::tt::target::ttnn::UnaryOpType::Rsqrt;
  case ttnn::UnaryOpType::Relu6:
    return ::tt::target::ttnn::UnaryOpType::Relu6;
  case ttnn::UnaryOpType::Atan:
    return ::tt::target::ttnn::UnaryOpType::Atan;
  case ttnn::UnaryOpType::Erf:
    return ::tt::target::ttnn::UnaryOpType::Erf;
  case ttnn::UnaryOpType::Erfc:
    return ::tt::target::ttnn::UnaryOpType::Erfc;
  case ttnn::UnaryOpType::IsInf:
    return ::tt::target::ttnn::UnaryOpType::Isinf;
  case ttnn::UnaryOpType::IsPosInf:
    return ::tt::target::ttnn::UnaryOpType::Isposinf;
  case ttnn::UnaryOpType::IsNegInf:
    return ::tt::target::ttnn::UnaryOpType::Isneginf;
  case ttnn::UnaryOpType::IsNan:
    return ::tt::target::ttnn::UnaryOpType::Isnan;
  case ttnn::UnaryOpType::LogicalNotUnary:
    return ::tt::target::ttnn::UnaryOpType::LogicalNotUnary;
  case ttnn::UnaryOpType::IsFinite:
    return ::tt::target::ttnn::UnaryOpType::Isfinite;
  case ttnn::UnaryOpType::Erfinv:
    return ::tt::target::ttnn::UnaryOpType::Erfinv;
  case ttnn::UnaryOpType::I0:
    return ::tt::target::ttnn::UnaryOpType::I0;
  case ttnn::UnaryOpType::I1:
    return ::tt::target::ttnn::UnaryOpType::I1;
  case ttnn::UnaryOpType::Tan:
    return ::tt::target::ttnn::UnaryOpType::Tan;
  case ttnn::UnaryOpType::Rsub:
    return ::tt::target::ttnn::UnaryOpType::Rsub;
  case ttnn::UnaryOpType::Rdiv:
    return ::tt::target::ttnn::UnaryOpType::Rdiv;
  case ttnn::UnaryOpType::Silu:
    return ::tt::target::ttnn::UnaryOpType::Silu;
  case ttnn::UnaryOpType::SoftPlus:
    return ::tt::target::ttnn::UnaryOpType::Softplus;
  case ttnn::UnaryOpType::Identity:
    return ::tt::target::ttnn::UnaryOpType::Identity;
  case ttnn::UnaryOpType::Neg:
    return ::tt::target::ttnn::UnaryOpType::Neg;
  case ttnn::UnaryOpType::AddUnarySfpu:
    return ::tt::target::ttnn::UnaryOpType::AddUnarySfpu;
  case ttnn::UnaryOpType::SubUnarySfpu:
    return ::tt::target::ttnn::UnaryOpType::SubUnarySfpu;
  case ttnn::UnaryOpType::MulUnarySfpu:
    return ::tt::target::ttnn::UnaryOpType::MulUnarySfpu;
  case ttnn::UnaryOpType::DivUnarySfpu:
    return ::tt::target::ttnn::UnaryOpType::DivUnarySfpu;
  case ttnn::UnaryOpType::IdentityUint32:
    return ::tt::target::ttnn::UnaryOpType::IdentityUint32;
  case ttnn::UnaryOpType::UnaryNe:
    return ::tt::target::ttnn::UnaryOpType::UnaryNe;
  case ttnn::UnaryOpType::UnaryGt:
    return ::tt::target::ttnn::UnaryOpType::UnaryGt;
  case ttnn::UnaryOpType::UnaryLt:
    return ::tt::target::ttnn::UnaryOpType::UnaryLt;
  case ttnn::UnaryOpType::TiledProd:
    return ::tt::target::ttnn::UnaryOpType::TiledProd;
  case ttnn::UnaryOpType::Typecast:
    return ::tt::target::ttnn::UnaryOpType::Typecast;
  case ttnn::UnaryOpType::BitwiseXor:
    return ::tt::target::ttnn::UnaryOpType::BitwiseXor;
  case ttnn::UnaryOpType::BitwiseNot:
    return ::tt::target::ttnn::UnaryOpType::BitwiseNot;
  case ttnn::UnaryOpType::BitwiseAnd:
    return ::tt::target::ttnn::UnaryOpType::BitwiseAnd;
  case ttnn::UnaryOpType::BitwiseOr:
    return ::tt::target::ttnn::UnaryOpType::BitwiseOr;
  case ttnn::UnaryOpType::RightShift:
    return ::tt::target::ttnn::UnaryOpType::RightShift;
  case ttnn::UnaryOpType::Floor:
    return ::tt::target::ttnn::UnaryOpType::Floor;
  case ttnn::UnaryOpType::FloorFloat32:
    return ::tt::target::ttnn::UnaryOpType::FloorFloat32;
  case ttnn::UnaryOpType::Ceil:
    return ::tt::target::ttnn::UnaryOpType::Ceil;
  case ttnn::UnaryOpType::CeilFloat32:
    return ::tt::target::ttnn::UnaryOpType::CeilFloat32;
  case ttnn::UnaryOpType::LeftShift:
    return ::tt::target::ttnn::UnaryOpType::LeftShift;
  case ttnn::UnaryOpType::Remainder:
    return ::tt::target::ttnn::UnaryOpType::Remainder;
  case ttnn::UnaryOpType::Fmod:
    return ::tt::target::ttnn::UnaryOpType::Fmod;
  case ttnn::UnaryOpType::Dropout:
    return ::tt::target::ttnn::UnaryOpType::Dropout;
  case ttnn::UnaryOpType::Fill:
    return ::tt::target::ttnn::UnaryOpType::Fill;
  case ttnn::UnaryOpType::PreluFspu:
    return ::tt::target::ttnn::UnaryOpType::PreluSfpu;
  case ttnn::UnaryOpType::ZeroPoint:
    return ::tt::target::ttnn::UnaryOpType::ZeroPoint;
  }
}

inline ::flatbuffers::Offset<
    ::tt::target::ttnn::MatmulMultiCoreReuseProgramConfig>
toFlatbuffer(FlatbufferObjectCache &cache,
             ttnn::MatmulMultiCoreReuseProgramConfigAttr matmulConfigAttr) {
  ::tt::target::ttnn::CoreCoord computeWithStorageGridSize =
      toFlatbuffer(cache, matmulConfigAttr.getComputeWithStorageGridSize());
  return ::tt::target::ttnn::CreateMatmulMultiCoreReuseProgramConfig(
      *cache.fbb, &computeWithStorageGridSize,
      toFlatbuffer(cache, matmulConfigAttr.getIn0BlockW()),
      toFlatbuffer(cache, matmulConfigAttr.getOutSubblockH()),
      toFlatbuffer(cache, matmulConfigAttr.getOutSubblockW()),
      toFlatbuffer(cache, matmulConfigAttr.getPerCoreM()),
      toFlatbuffer(cache, matmulConfigAttr.getPerCoreN()));
}

inline ::flatbuffers::Offset<::tt::target::ttnn::UnaryWithParam>
toFlatbuffer(FlatbufferObjectCache &cache,
             ttnn::UnaryWithParamAttr unaryWithParam) {
  ::tt::target::ttnn::UnaryOpType opType =
      toFlatbuffer(cache, unaryWithParam.getOpType());
  ::flatbuffers::Offset<::flatbuffers::Vector<double>> params =
      toFlatbuffer(cache, unaryWithParam.getParams());
  return ::tt::target::ttnn::CreateUnaryWithParam(*cache.fbb, opType, params);
}

inline ::flatbuffers::Offset<
    ::tt::target::ttnn::MatmulMultiCoreReuseMultiCastProgramConfig>
toFlatbuffer(
    FlatbufferObjectCache &cache,
    ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr matmulConfig) {
  ::tt::target::ttnn::CoreCoord computeWithStorageGridSize =
      toFlatbuffer(cache, matmulConfig.getComputeWithStorageGridSize());
  return ::tt::target::ttnn::CreateMatmulMultiCoreReuseMultiCastProgramConfig(
      *cache.fbb, &computeWithStorageGridSize,
      toFlatbuffer(cache, matmulConfig.getIn0BlockW()),
      toFlatbuffer(cache, matmulConfig.getOutSubblockH()),
      toFlatbuffer(cache, matmulConfig.getOutSubblockW()),
      toFlatbuffer(cache, matmulConfig.getPerCoreM()),
      toFlatbuffer(cache, matmulConfig.getPerCoreN()));
}

} // namespace mlir::tt

#endif
