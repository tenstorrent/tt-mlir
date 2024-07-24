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

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTChipDescAttrGet(
    MlirContext ctx, MlirAttribute arch, MlirAttribute grid, unsigned l1Size,
    unsigned numDramChannels, unsigned dramChannelSize,
    unsigned nocL1AddressAlignBytes, unsigned pcieAddressAlignBytes,
    unsigned nocDRAMAddressAlignBytes);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTChipCoordAttrGet(
    MlirContext ctx, unsigned rack, unsigned shelf, unsigned y, unsigned x);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTChipChannelAttrGet(MlirContext ctx,
                                                            unsigned endpoint0,
                                                            unsigned endpoint1);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTSystemDescAttrGet(
    MlirContext ctx, MlirAttribute *chipDescs, size_t chipDescsSize,
    unsigned *chipDescIndices, size_t chipDescIndicesSize,
    MlirAttribute *chipCapabilities, size_t chipCapabilitiesSize,
    MlirAttribute *chipCoords, size_t chipCoordsSize,
    MlirAttribute *chipChannels, size_t chipChannelsSize);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTLayoutAttrGet(MlirContext ctx,
                                                       MlirAffineMap linear,
                                                       unsigned oobVal,
                                                       MlirAttribute grid,
                                                       MlirType memref);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTMemorySpaceAttrGet(MlirContext ctx, uint32_t memorySpace);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTOOBValAttrGet(MlirContext ctx,
                                                       uint32_t oobVal);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTIteratorTypeAttrGet(MlirContext ctx, uint32_t iteratorType);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTIteratorTypeArrayAttrGet(
    MlirContext ctx, uint32_t *iteratorTypes, size_t iteratorTypesSize);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTOperandConstraintAttrGet(MlirContext ctx, uint32_t OperandConstraint);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTOperandConstraintArrayAttrGet(
    MlirContext ctx, uint32_t *OperandConstraints,
    size_t OperandConstraintsSize);

#ifdef __cplusplus
}
#endif

#endif // TTMLIR_C_TTATTRS_H
