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

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTGridAttrGet(MlirContext ctx,
                                                     int64_t *shape,
                                                     size_t shapeSize);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTChipCapabilityAttrGet(MlirContext ctx, uint32_t chipCapability);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTArchAttrGet(MlirContext ctx,
                                                     uint32_t arch);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTDataTypeAttrGet(MlirContext ctx, uint16_t *supportedDataTypes);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTChipDescAttrGet(
    MlirContext ctx, MlirAttribute arch, int64_t *grid, size_t gridSize,
    unsigned l1Size, unsigned numDramChannels, unsigned dramChannelSize,
    unsigned nocL1AddressAlignBytes, unsigned pcieAddressAlignBytes,
    unsigned nocDRAMAddressAlignBytes, unsigned l1UnreservedBase,
    unsigned eriscL1UnreservedBase, unsigned dramUnreservedBase,
    MlirAttribute chipPhysicalCores, MlirAttribute *supportedDataTypes,
    MlirAttribute *supportedTileSizes, unsigned numCBs);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTChipCoordAttrGet(
    MlirContext ctx, unsigned rack, unsigned shelf, unsigned y, unsigned x);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTChipChannelAttrGet(
    MlirContext ctx, unsigned deviceId0, int64_t *ethernetCoreCoord0,
    size_t ethernetCoreCoord0Size, unsigned deviceId1,
    int64_t *ethernetCoreCoord1, size_t ethernetCoreCoord1Size);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTSystemDescAttrGet(
    MlirContext ctx, MlirAttribute *chipDescs, size_t chipDescsSize,
    unsigned *chipDescIndices, size_t chipDescIndicesSize,
    MlirAttribute *chipCapabilities, size_t chipCapabilitiesSize,
    MlirAttribute *chipCoords, size_t chipCoordsSize,
    MlirAttribute *chipChannels, size_t chipChannelsSize);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTLayoutAttrGet(MlirContext ctx, MlirAffineMap linear, unsigned oobVal,
                      MlirAttribute grid, MlirType memref, unsigned memLayout);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTMemorySpaceAttrGet(MlirContext ctx, uint32_t memorySpace);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTOOBValAttrGet(MlirContext ctx,
                                                       uint32_t oobVal);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTTensorMemoryLayoutAttrGet(MlirContext ctx, uint32_t memLayout);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTIteratorTypeAttrGet(MlirContext ctx, uint32_t iteratorType);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTIteratorTypeArrayAttrGet(
    MlirContext ctx, uint32_t *iteratorTypes, size_t iteratorTypesSize);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTOperandConstraintAttrGet(MlirContext ctx, uint32_t OperandConstraint);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTOperandConstraintArrayAttrGet(
    MlirContext ctx, uint32_t *OperandConstraints,
    size_t OperandConstraintsSize);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTTileSizeAttrGet(MlirContext ctx,
                                                         int64_t y, int64_t x);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTChipPhysicalCoresAttrGet(
    MlirContext ctx, MlirAttribute *worker, size_t workerSize,
    MlirAttribute *dram, size_t dramSize, MlirAttribute *eth, size_t ethSize,
    MlirAttribute *eth_inactive, size_t eth_inactiveSize);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTCoreCoordAttrGet(MlirContext ctx,
                                                          int64_t y, int64_t x);

#ifdef __cplusplus
}
#endif

#endif // TTMLIR_C_TTATTRS_H
