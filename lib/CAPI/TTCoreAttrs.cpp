// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "ttmlir-c/TTAttrs.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

using namespace mlir::tt::ttcore;

//===----------------------------------------------------------------------===//
// TileType
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool ttmlirIsTileType(MlirType type) {
  return mlir::isa<TileType>(unwrap(type));
}

MLIR_CAPI_EXPORTED MlirType ttmlirTileTypeGet(MlirContext ctx, int64_t height,
                                              int64_t width,
                                              MlirAttribute dataType) {
  return wrap(
      TileType::get(unwrap(ctx), llvm::SmallVector<std::int64_t>{height, width},
                    mlir::cast<DataTypeAttr>(unwrap(dataType)).getValue()));
}

MLIR_CAPI_EXPORTED int64_t ttmlirTileTypeGetHeight(MlirType type) {
  return mlir::cast<TileType>(unwrap(type)).getHeight();
}

MLIR_CAPI_EXPORTED int64_t ttmlirTileTypeGetWidth(MlirType type) {
  return mlir::cast<TileType>(unwrap(type)).getWidth();
}

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTileTypeGetDataType(MlirType type) {
  TileType tileType = mlir::cast<TileType>(unwrap(type));
  mlir::MLIRContext *ctx = tileType.getContext();
  return wrap(DataTypeAttr::get(ctx, tileType.getDataType()));
}

//===----------------------------------------------------------------------===//
// GridAttr
//===----------------------------------------------------------------------===//

bool ttmlirIsGridAttr(MlirAttribute attr) {
  return mlir::isa<GridAttr>(unwrap(attr));
}

MlirAttribute ttmlirGridAttrGet(MlirContext ctx, intptr_t nShape,
                                const int64_t *shape) {
  return wrap(
      GridAttr::get(unwrap(ctx), llvm::ArrayRef<int64_t>(shape, nShape)));
}

intptr_t ttmlirGridAttrGetShapeSize(MlirAttribute attr) {
  return mlir::cast<GridAttr>(unwrap(attr)).getShape().size();
}

int64_t ttmlirGridAttrGetShapeElem(MlirAttribute attr, intptr_t pos) {
  return mlir::cast<GridAttr>(unwrap(attr)).getShape()[pos];
}

//===----------------------------------------------------------------------===//
// ReduceTypeAttr
//===----------------------------------------------------------------------===//

std::optional<ReduceType>
enumTypeToReduceType(MlirReduceTypeEnum reduceTypeEnum) {
  switch (reduceTypeEnum) {
  case MlirReduceTypeSum:
    return ReduceType::Sum;
  case MlirReduceTypeMean:
    return ReduceType::Mean;
  case MlirReduceTypeMax:
    return ReduceType::Max;
  case MlirReduceTypeMin:
    return ReduceType::Min;
  case MlirReduceTypeStd:
    return ReduceType::Std;
  case MlirReduceTypeVar:
    return ReduceType::Var;
  case MlirReduceTypeProd:
    return ReduceType::Prod;
  case MlirReduceTypeInvalid:
    return ReduceType::Invalid;
  }

  return std::nullopt;
}

std::optional<MlirReduceTypeEnum> reduceTypeToEnumType(ReduceType reduceType) {
  switch (reduceType) {
  case ReduceType::Sum:
    return MlirReduceTypeSum;
  case ReduceType::Mean:
    return MlirReduceTypeMean;
  case ReduceType::Max:
    return MlirReduceTypeMax;
  case ReduceType::Min:
    return MlirReduceTypeMin;
  case ReduceType::Std:
    return MlirReduceTypeStd;
  case ReduceType::Var:
    return MlirReduceTypeVar;
  case ReduceType::Prod:
    return MlirReduceTypeProd;
  case ReduceType::Invalid:
    return MlirReduceTypeInvalid;
  }

  return std::nullopt;
}

MLIR_CAPI_EXPORTED bool ttmlirIsReduceTypeAttr(MlirAttribute attr) {
  return mlir::isa<ReduceTypeAttr>(unwrap(attr));
}

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirReduceTypeAttrGet(MlirContext ctx, MlirReduceTypeEnum reduceType) {
  std::optional<ReduceType> rt = enumTypeToReduceType(reduceType);

  if (!rt) {
    llvm::report_fatal_error("Invalid value.");
  }

  return wrap(ReduceTypeAttr::get(unwrap(ctx), rt.value()));
}

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirReduceTypeAttrFromStringGet(MlirContext ctx, MlirStringRef value) {
  std::optional<ReduceType> reduceType = symbolizeReduceType(unwrap(value));

  if (!reduceType) {
    llvm::report_fatal_error("Invalid value.");
  }

  return wrap(ReduceTypeAttr::get(unwrap(ctx), reduceType.value()));
}

MLIR_CAPI_EXPORTED MlirReduceTypeEnum
ttmlirReduceTypeAttrGetValue(MlirAttribute attr) {
  ReduceType reduceType = mlir::cast<ReduceTypeAttr>(unwrap(attr)).getValue();
  std::optional<MlirReduceTypeEnum> reduceTypeEnum =
      reduceTypeToEnumType(reduceType);

  if (!reduceTypeEnum) {
    llvm::report_fatal_error("Invalid value.");
  }

  return reduceTypeEnum.value();
}

//===----------------------------------------------------------------------===//
// DataTypeAttr
//===----------------------------------------------------------------------===//

std::optional<DataType> enumTypeToDataType(MlirDataTypeEnum dataTypeEnum) {
  switch (dataTypeEnum) {
  case MlirDataTypeFloat32:
    return DataType::Float32;
  case MlirDataTypeFloat16:
    return DataType::Float16;
  case MlirDataTypeBFloat16:
    return DataType::BFloat16;
  case MlirDataTypeBFP_Float8:
    return DataType::BFP_Float8;
  case MlirDataTypeBFP_BFloat8:
    return DataType::BFP_BFloat8;
  case MlirDataTypeBFP_Float4:
    return DataType::BFP_Float4;
  case MlirDataTypeBFP_BFloat4:
    return DataType::BFP_BFloat4;
  case MlirDataTypeBFP_Float2:
    return DataType::BFP_Float2;
  case MlirDataTypeBFP_BFloat2:
    return DataType::BFP_BFloat2;
  case MlirDataTypeUInt32:
    return DataType::UInt32;
  case MlirDataTypeUInt16:
    return DataType::UInt16;
  case MlirDataTypeUInt8:
    return DataType::UInt8;
  case MlirDataTypeInt32:
    return DataType::Int32;
  case MlirDataTypeBool:
    return DataType::Bool;
  }

  return std::nullopt;
}

std::optional<MlirDataTypeEnum> dataTypeToEnumType(DataType dataType) {
  switch (dataType) {
  case DataType::Float32:
    return MlirDataTypeFloat32;
  case DataType::Float16:
    return MlirDataTypeFloat16;
  case DataType::BFloat16:
    return MlirDataTypeBFloat16;
  case DataType::BFP_Float8:
    return MlirDataTypeBFP_Float8;
  case DataType::BFP_BFloat8:
    return MlirDataTypeBFP_BFloat8;
  case DataType::BFP_Float4:
    return MlirDataTypeBFP_Float4;
  case DataType::BFP_BFloat4:
    return MlirDataTypeBFP_BFloat4;
  case DataType::BFP_Float2:
    return MlirDataTypeBFP_Float2;
  case DataType::BFP_BFloat2:
    return MlirDataTypeBFP_BFloat2;
  case DataType::UInt32:
    return MlirDataTypeUInt32;
  case DataType::UInt16:
    return MlirDataTypeUInt16;
  case DataType::UInt8:
    return MlirDataTypeUInt8;
  case DataType::Int32:
    return MlirDataTypeInt32;
  case DataType::Bool:
    return MlirDataTypeBool;
  }

  return std::nullopt;
}

MLIR_CAPI_EXPORTED bool ttmlirIsDataTypeAttr(MlirAttribute attr) {
  return mlir::isa<DataTypeAttr>(unwrap(attr));
}

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirDataTypeAttrGet(MlirContext ctx, MlirDataTypeEnum dataTypeEnum) {
  std::optional<DataType> dataType = enumTypeToDataType(dataTypeEnum);

  if (!dataType) {
    llvm::report_fatal_error("Invalid value.");
  }

  return wrap(DataTypeAttr::get(unwrap(ctx), dataType.value()));
}

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirDataTypeAttrFromStringGet(MlirContext ctx, MlirStringRef value) {
  std::optional<DataType> dataType = DataTypeStringToEnum(unwrap(value));

  if (!dataType) {
    llvm::report_fatal_error("Invalid value.");
  }

  return wrap(DataTypeAttr::get(unwrap(ctx), dataType.value()));
}

MLIR_CAPI_EXPORTED MlirAttribute ttmlirDataTypeAttrFromIntGet(MlirContext ctx,
                                                              uint32_t value) {
  std::optional<DataType> dataType = symbolizeDataType(value);

  if (!dataType) {
    llvm::report_fatal_error("Invalid value.");
  }

  return wrap(DataTypeAttr::get(unwrap(ctx), dataType.value()));
}

MLIR_CAPI_EXPORTED MlirDataTypeEnum
ttmlirDataTypeAttrGetValue(MlirAttribute attr) {
  DataType dataType = mlir::cast<DataTypeAttr>(unwrap(attr)).getValue();
  std::optional<MlirDataTypeEnum> dataTypeEnum = dataTypeToEnumType(dataType);

  if (!dataTypeEnum) {
    llvm::report_fatal_error("Invalid value.");
  }

  return dataTypeEnum.value();
}

//===----------------------------------------------------------------------===//
// ChipCapabilityAttr
//===----------------------------------------------------------------------===//

MlirAttribute ttmlirTTChipCapabilityAttrGet(MlirContext ctx,
                                            uint32_t chipCapability) {
  return wrap(ChipCapabilityAttr::get(
      unwrap(ctx), static_cast<ChipCapability>(chipCapability)));
}

//===----------------------------------------------------------------------===//
// ArchAttr
//===----------------------------------------------------------------------===//

MlirAttribute ttmlirTTArchAttrGet(MlirContext ctx, uint32_t arch) {
  return wrap(ArchAttr::get(unwrap(ctx), static_cast<Arch>(arch)));
}

//===----------------------------------------------------------------------===//
// ChipDescAttr
//===----------------------------------------------------------------------===//

MlirAttribute ttmlirTTChipDescAttrGet(
    MlirContext ctx, MlirAttribute arch, int64_t *grid, size_t gridSize,
    int64_t *coordTranslationOffsets, size_t coordTranslationOffsetsSize,
    unsigned l1Size, unsigned numDramChannels, unsigned dramChannelSize,
    unsigned nocL1AddressAlignBytes, unsigned pcieAddressAlignBytes,
    unsigned nocDRAMAddressAlignBytes, unsigned l1UnreservedBase,
    unsigned eriscL1UnreservedBase, unsigned dramUnreservedBase,
    unsigned dramUnreservedEnd, MlirAttribute *supportedDataTypes,
    MlirAttribute *supportedTileSizes, unsigned dstPhysicalSizeTiles,
    unsigned numCBs, unsigned numComputeThreads,
    unsigned numDatamovementThreads) {
  std::vector<int64_t> gridVec(grid, grid + gridSize);
  std::vector<int64_t> coordTranslationOffsetsVec(
      coordTranslationOffsets,
      coordTranslationOffsets + coordTranslationOffsetsSize);
  return wrap(ChipDescAttr::get(
      unwrap(ctx), mlir::dyn_cast<ArchAttr>(unwrap(arch)), gridVec,
      coordTranslationOffsetsVec, l1Size, numDramChannels, dramChannelSize,
      nocL1AddressAlignBytes, pcieAddressAlignBytes, nocDRAMAddressAlignBytes,
      l1UnreservedBase, eriscL1UnreservedBase, dramUnreservedBase,
      dramUnreservedEnd,
      mlir::dyn_cast<DataTypeAttr>(unwrap(*supportedDataTypes)),
      mlir::dyn_cast<TileSizeAttr>(unwrap(*supportedTileSizes)),
      dstPhysicalSizeTiles, numCBs, numComputeThreads, numDatamovementThreads));
}

//===----------------------------------------------------------------------===//
// ChipCoordAttr
//===----------------------------------------------------------------------===//

MlirAttribute ttmlirTTChipCoordAttrGet(MlirContext ctx, unsigned rack,
                                       unsigned shelf, unsigned y, unsigned x) {
  return wrap(ChipCoordAttr::get(unwrap(ctx), rack, shelf, y, x));
}

//===----------------------------------------------------------------------===//
// ChipChannelAttr
//===----------------------------------------------------------------------===//

MlirAttribute ttmlirTTChipChannelAttrGet(MlirContext ctx, unsigned deviceId0,
                                         int64_t *ethernetCoreCoord0,
                                         size_t ethernetCoreCoord0Size,
                                         unsigned deviceId1,
                                         int64_t *ethernetCoreCoord1,
                                         size_t ethernetCoreCoord1Size) {
  std::vector<int64_t> ethCoord0Vec(
      ethernetCoreCoord0, ethernetCoreCoord0 + ethernetCoreCoord0Size);
  std::vector<int64_t> ethCoord1Vec(
      ethernetCoreCoord1, ethernetCoreCoord1 + ethernetCoreCoord1Size);
  return wrap(ChipChannelAttr::get(unwrap(ctx), deviceId0, ethCoord0Vec,
                                   deviceId1, ethCoord1Vec));
}

//===----------------------------------------------------------------------===//
// SystemDescAttr
//===----------------------------------------------------------------------===//

MlirAttribute ttmlirTTSystemDescAttrGet(
    MlirContext ctx, MlirAttribute *cpuDescs, size_t cpuDescsSize,
    MlirAttribute *chipDescs, size_t chipDescsSize, unsigned *chipDescIndices,
    size_t chipDescIndicesSize, MlirAttribute *chipCapabilities,
    size_t chipCapabilitiesSize, MlirAttribute *chipCoords,
    size_t chipCoordsSize, MlirAttribute *chipChannels,
    size_t chipChannelsSize) {
  llvm::ArrayRef<MlirAttribute> cpuDescsRef(cpuDescs, cpuDescsSize),
      chipDescsRef(chipDescs, chipDescsSize),
      chipCapabilitiesRef(chipCapabilities, chipCapabilitiesSize),
      chipCoordsRef(chipCoords, chipCoordsSize),
      chipChannelsRef(chipChannels, chipChannelsSize);
  llvm::ArrayRef<unsigned> chipDescIndicesRef(chipDescIndices,
                                              chipDescIndicesSize);

  // Unwrap all of the MlirAttributes
  std::vector<ChipDescAttr> chipDescsUnwrapped;
  for (auto chipDesc : chipDescsRef) {
    chipDescsUnwrapped.push_back(mlir::cast<ChipDescAttr>(unwrap(chipDesc)));
  }

  std::vector<ChipCapabilityAttr> chipCapabilitiesUnwrapped;
  for (auto chipCapability : chipCapabilitiesRef) {
    chipCapabilitiesUnwrapped.push_back(
        mlir::cast<ChipCapabilityAttr>(unwrap(chipCapability)));
  }

  std::vector<ChipCoordAttr> chipCoordsUnwrapped;
  for (auto chipCoord : chipCoordsRef) {
    chipCoordsUnwrapped.push_back(mlir::cast<ChipCoordAttr>(unwrap(chipCoord)));
  }

  std::vector<ChipChannelAttr> chipChannelsUnwrapped;
  for (auto chipChannel : chipChannelsRef) {
    chipChannelsUnwrapped.push_back(
        mlir::cast<ChipChannelAttr>(unwrap(chipChannel)));
  }

  std::vector<CPUDescAttr> cpuDescsUnwrapped;
  for (auto cpuDesc : cpuDescsRef) {
    cpuDescsUnwrapped.push_back(mlir::cast<CPUDescAttr>(unwrap(cpuDesc)));
  }

  return wrap(SystemDescAttr::get(
      unwrap(ctx), cpuDescsUnwrapped, chipDescsUnwrapped, chipDescIndicesRef,
      chipCapabilitiesUnwrapped, chipCoordsUnwrapped, chipChannelsUnwrapped));
}

//===----------------------------------------------------------------------===//
// MetalLayoutAttr
//===----------------------------------------------------------------------===//

bool ttmlirIsMetalLayoutAttr(MlirAttribute attr) {
  return mlir::isa<MetalLayoutAttr>(unwrap(attr));
}

MlirAttribute ttmlirMetalLayoutAttrGet(MlirContext ctx, intptr_t nLogicalShape,
                                       const int64_t *logicalShape,
                                       uint32_t oobValValue,
                                       uint32_t memorySpaceValue) {
  return wrap(mlir::tt::ttcore::MetalLayoutAttr::get(
      unwrap(ctx), mlir::ArrayRef(logicalShape, nLogicalShape),
      static_cast<mlir::tt::ttcore::OOBVal>(oobValValue),
      static_cast<mlir::tt::ttcore::MemorySpace>(memorySpaceValue),
      mlir::tt::ttcore::TensorMemoryLayout::Sharded));
}

//===----------------------------------------------------------------------===//
// MemorySpaceAttr
//===----------------------------------------------------------------------===//

MlirAttribute ttmlirTTMemorySpaceAttrGet(MlirContext ctx,
                                         uint32_t memorySpace) {
  return wrap(
      MemorySpaceAttr::get(unwrap(ctx), static_cast<MemorySpace>(memorySpace)));
}

//===----------------------------------------------------------------------===//
// OOBValAttr
//===----------------------------------------------------------------------===//

MlirAttribute ttmlirTTOOBValAttrGet(MlirContext ctx, uint32_t oobVal) {
  return wrap(OOBValAttr::get(unwrap(ctx), static_cast<OOBVal>(oobVal)));
}

//===----------------------------------------------------------------------===//
// IteratorTypeAttr
//===----------------------------------------------------------------------===//

MlirAttribute ttmlirTTIteratorTypeAttrGet(MlirContext ctx,
                                          uint32_t iteratorType) {
  return wrap(IteratorTypeAttr::get(unwrap(ctx),
                                    static_cast<IteratorType>(iteratorType)));
}

//===----------------------------------------------------------------------===//
// IteratorTypeArrayAttr
//===----------------------------------------------------------------------===//

MlirAttribute ttmlirTTIteratorTypeArrayAttrGet(MlirContext ctx,
                                               uint32_t *iteratorTypes,
                                               size_t iteratorTypesSize) {
  std::vector<uint32_t> iteratorTypesEnumArray(
      iteratorTypes, iteratorTypes + iteratorTypesSize);
  std::vector<mlir::Attribute> iteratorTypesArray;

  for (auto iteratorEnum : iteratorTypesEnumArray) {
    iteratorTypesArray.push_back(IteratorTypeAttr::get(
        unwrap(ctx), static_cast<IteratorType>(iteratorEnum)));
  }

  return wrap(mlir::ArrayAttr::get(unwrap(ctx), iteratorTypesArray));
}

//===----------------------------------------------------------------------===//
// TileSizeAttr
//===----------------------------------------------------------------------===//

MlirAttribute ttmlirTTTileSizeAttrGet(MlirContext ctx, int64_t y, int64_t x) {
  return wrap(TileSizeAttr::get(unwrap(ctx), y, x));
}

//===----------------------------------------------------------------------===//
// CoreCoordAttr
//===----------------------------------------------------------------------===//

MlirAttribute ttmlirTTCoreCoordAttrGet(MlirContext ctx, int64_t y, int64_t x) {
  return wrap(CoreCoordAttr::get(unwrap(ctx), y, x));
}
