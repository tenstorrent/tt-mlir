// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_C_TTATTRS_H
#define TTMLIR_C_TTATTRS_H

#include "mlir-c/AffineMap.h"
#include "ttmlir-c/Dialects.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// TileType
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool ttmlirIsTileType(MlirType type);

MLIR_CAPI_EXPORTED MlirType ttmlirTileTypeGet(MlirContext ctx, int64_t height,
                                              int64_t width,
                                              MlirAttribute dataType);

MLIR_CAPI_EXPORTED int64_t ttmlirTileTypeGetHeight(MlirType type);

MLIR_CAPI_EXPORTED int64_t ttmlirTileTypeGetWidth(MlirType type);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTileTypeGetDataType(MlirType type);

//===----------------------------------------------------------------------===//
// GridAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool ttmlirIsGridAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirGridAttrGet(MlirContext ctx,
                                                   intptr_t nShape,
                                                   const int64_t *shape);

MLIR_CAPI_EXPORTED intptr_t ttmlirGridAttrGetShapeSize(MlirAttribute attr);

MLIR_CAPI_EXPORTED int64_t ttmlirGridAttrGetShapeElem(MlirAttribute attr,
                                                      intptr_t pos);

//===----------------------------------------------------------------------===//
// ReduceTypeAttr
//===----------------------------------------------------------------------===//

enum MlirReduceTypeEnum {
  MlirReduceTypeSum,
  MlirReduceTypeMean,
  MlirReduceTypeMax,
  MlirReduceTypeMin,
  MlirReduceTypeStd,
  MlirReduceTypeVar,
  MlirReduceTypeProd,
  MlirReduceTypeInvalid,
};
typedef enum MlirReduceTypeEnum MlirReduceTypeEnum;

MLIR_CAPI_EXPORTED bool ttmlirIsReduceTypeAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirReduceTypeAttrGet(MlirContext ctx, MlirReduceTypeEnum reduceType);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirReduceTypeAttrFromStringGet(MlirContext ctx, MlirStringRef value);

MLIR_CAPI_EXPORTED MlirReduceTypeEnum
ttmlirReduceTypeAttrGetValue(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// DataTypeAttr
//===----------------------------------------------------------------------===//

enum MlirDataTypeEnum {
  MlirDataTypeFloat32,
  MlirDataTypeFloat16,
  MlirDataTypeBFloat16,
  MlirDataTypeBFP_Float8,
  MlirDataTypeBFP_BFloat8,
  MlirDataTypeBFP_Float4,
  MlirDataTypeBFP_BFloat4,
  MlirDataTypeBFP_Float2,
  MlirDataTypeBFP_BFloat2,
  MlirDataTypeUInt32,
  MlirDataTypeUInt16,
  MlirDataTypeUInt8,
  MlirDataTypeInt32,
  MlirDataTypeBool
};
typedef enum MlirDataTypeEnum MlirDataTypeEnum;

MLIR_CAPI_EXPORTED bool ttmlirIsDataTypeAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirDataTypeAttrGet(MlirContext ctx, MlirDataTypeEnum dataType);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirDataTypeAttrFromStringGet(MlirContext ctx, MlirStringRef value);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirDataTypeAttrFromIntGet(MlirContext ctx,
                                                              uint32_t value);

MLIR_CAPI_EXPORTED MlirDataTypeEnum
ttmlirDataTypeAttrGetValue(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// ChipCapabilityAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTChipCapabilityAttrGet(MlirContext ctx, uint32_t chipCapability);

//===----------------------------------------------------------------------===//
// ArchAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTArchAttrGet(MlirContext ctx,
                                                     uint32_t arch);

//===----------------------------------------------------------------------===//
// ChipDescAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTChipDescAttrGet(
    MlirContext ctx, MlirAttribute arch, int64_t *grid, size_t gridSize,
    int64_t *coordTranslationOffsets, size_t coordTranslationOffsetsSize,
    unsigned l1Size, unsigned numDramChannels, unsigned dramChannelSize,
    unsigned nocL1AddressAlignBytes, unsigned pcieAddressAlignBytes,
    unsigned nocDRAMAddressAlignBytes, unsigned l1UnreservedBase,
    unsigned eriscL1UnreservedBase, unsigned dramUnreservedBase,
    unsigned dramUnreservedEnd, MlirAttribute *supportedDataTypes,
    MlirAttribute *supportedTileSizes, unsigned dstPhysicalSizeTiles,
    unsigned numCBs, unsigned numComputeThreads,
    unsigned numDatamovementThreads);

//===----------------------------------------------------------------------===//
// ChipCoordAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTChipCoordAttrGet(
    MlirContext ctx, unsigned rack, unsigned shelf, unsigned y, unsigned x);

//===----------------------------------------------------------------------===//
// ChipChannelAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTChipChannelAttrGet(
    MlirContext ctx, unsigned deviceId0, int64_t *ethernetCoreCoord0,
    size_t ethernetCoreCoord0Size, unsigned deviceId1,
    int64_t *ethernetCoreCoord1, size_t ethernetCoreCoord1Size);

//===----------------------------------------------------------------------===//
// SystemDescAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTSystemDescAttrGet(
    MlirContext ctx, MlirAttribute *cpuDescs, size_t cpuDescsSize,
    MlirAttribute *chipDescs, size_t chipDescsSize, unsigned *chipDescIndices,
    size_t chipDescIndicesSize, MlirAttribute *chipCapabilities,
    size_t chipCapabilitiesSize, MlirAttribute *chipCoords,
    size_t chipCoordsSize, MlirAttribute *chipChannels,
    size_t chipChannelsSize);

//===----------------------------------------------------------------------===//
// MetalLayoutAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute ttmlirMetalLayoutAttrGet(
    MlirContext ctx, intptr_t nLogicalShape, const int64_t *logicalShape,
    uint32_t oobValValue, uint32_t memorySpaceValue);

//===----------------------------------------------------------------------===//
// MemorySpaceAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTMemorySpaceAttrGet(MlirContext ctx, uint32_t memorySpace);

//===----------------------------------------------------------------------===//
// OOBValAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTOOBValAttrGet(MlirContext ctx,
                                                       uint32_t oobVal);

//===----------------------------------------------------------------------===//
// IteratorTypeAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTIteratorTypeAttrGet(MlirContext ctx, uint32_t iteratorType);

//===----------------------------------------------------------------------===//
// IteratorTypeArrayAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTIteratorTypeArrayAttrGet(
    MlirContext ctx, uint32_t *iteratorTypes, size_t iteratorTypesSize);

//===----------------------------------------------------------------------===//
// TileSizeAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTTileSizeAttrGet(MlirContext ctx,
                                                         int64_t y, int64_t x);

//===----------------------------------------------------------------------===//
// CoreCoordAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTCoreCoordAttrGet(MlirContext ctx,
                                                          int64_t y, int64_t x);

#ifdef __cplusplus
}
#endif

#endif // TTMLIR_C_TTATTRS_H
