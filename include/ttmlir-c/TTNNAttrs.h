// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_C_TTNNATTRS_H
#define TTMLIR_C_TTNNATTRS_H

#include "ttmlir-c/Dialects.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTNNCoreRangeAttrGet(MlirContext ctx,
                                                            int64_t *offset,
                                                            size_t offsetSize,
                                                            int64_t *size,
                                                            size_t sizeSize);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTNNCoreRangeArrayAttrGet(
    MlirContext ctx, MlirAttribute *coreRangeAttrs, size_t coreRangeAttrsSize);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTNNLayoutAttrGet(MlirContext ctx,
                                                         uint32_t layout);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTNNTensorMemoryLayoutAttrGet(
    MlirContext ctx, uint32_t tensorMemoryLayout);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTNNBufferTypeAttrGet(MlirContext ctx, uint32_t bufferType);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTNNMemoryConfigAttrGet(
    MlirContext ctx, MlirAttribute tensorMemoryLayoutAttr,
    MlirAttribute bufferTypeAttr);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTNNShapeAttrGet(MlirContext ctx,
                                                        int64_t *shape,
                                                        size_t shapeSize);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTNNMeshShapeAttrGet(MlirContext ctx,
                                                            int64_t y,
                                                            int64_t x);

#ifdef __cplusplus
}
#endif

#endif // TTMLIR_C_TTNNATTRS_H
