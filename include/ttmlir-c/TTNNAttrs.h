// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_C_TTNNATTRS_H
#define TTMLIR_C_TTNNATTRS_H

#include "mlir-c/AffineMap.h"
#include "ttmlir-c/Dialects.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTNNCoreCoordAttrGet(MlirContext ctx,
                                                            uint64_t y,
                                                            uint64_t x);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTNNCoreRangeSetAttrGet(MlirContext ctx, MlirAttribute *coreRangesAttrs,
                              size_t coreRangesAttrsSize);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTNNUnaryWithParamAttr(MlirContext ctx, uint32_t opTypeEnum,
                             MlirAttribute *params, size_t paramsSize);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTNNMatmulMultiCoreReuseProgramConfigAttr(
    MlirContext ctx, MlirAttribute computeWithStorageGridSize,
    uint64_t in0BlockW, uint64_t outSubblockH, uint64_t outSubblockW,
    uint64_t perCoreM, uint64_t perCoreN);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTNNMatmulMultiCoreReuseMultiCastProgramConfigAttr(
    MlirContext ctx, MlirAttribute computeWithStorageGridSize,
    uint64_t in0BlockW, uint64_t outSubblockH, uint64_t outSubblockW,
    uint64_t outBlockH, uint64_t outBlockW, uint64_t perCoreM,
    uint64_t perCoreN, bool transposeMcast, MlirAttribute fusedActivation,
    bool fuseBatch);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTNNMatmulMultiCoreReuseMultiCast1DProgramConfigAttrGet(
    MlirContext ctx, MlirAttribute computeWithStorageGridSize,
    uint64_t in0BlockW, uint64_t outSubblockH, uint64_t outSubblockW,
    uint64_t outBlockH, uint64_t outBlockW, uint64_t perCoreM,
    uint64_t perCoreN, bool fuseBatch, MlirAttribute fusedActivation,
    bool mcastIn0, bool gatherIn0, MlirAttribute hopCores,
    uint64_t numGlobalCbReceivers, bool untilizeOut);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTNNMatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttrGet(
    MlirContext ctx, uint64_t in0BlockW, uint64_t perCoreM, uint64_t perCoreN,
    MlirAttribute fusedActivation);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTNNCoreRangeAttrGet(
    MlirContext ctx, MlirAttribute startCoord, MlirAttribute endCoord);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTNNCoreRangeArrayAttrGet(
    MlirContext ctx, MlirAttribute *coreRangeAttrs, size_t coreRangeAttrsSize);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTNNLayoutAttrGet(MlirContext ctx,
                                                         uint32_t layout);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTNNTensorMemoryLayoutAttrGet(
    MlirContext ctx, uint32_t tensorMemoryLayout);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTNNBufferTypeAttrGet(MlirContext ctx, uint32_t bufferType);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTNNShardSpecAttrGet(
    MlirContext ctx, MlirAttribute coreRangeSetAttr, MlirAttribute shapeAttr,
    MlirAttribute shardOrientationAttr);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTNNMemoryConfigAttrGet(
    MlirContext ctx, MlirAttribute tensorMemoryLayoutAttr,
    MlirAttribute bufferTypeAttr, MlirAttribute shardSpecAttr);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTNNShapeAttrGet(MlirContext ctx,
                                                        int64_t *shape,
                                                        size_t shapeSize);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTNNMeshShapeAttrGet(MlirContext ctx,
                                                            int64_t y,
                                                            int64_t x);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTNNTTNNLayoutAttrGet(
    MlirContext ctx, MlirAffineMap linear, MlirAttribute grid, MlirType memref,
    unsigned memLayout);

#ifdef __cplusplus
}
#endif

#endif // TTMLIR_C_TTNNATTRS_H
