// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_C_TTKERNELTYPES_H
#define TTMLIR_C_TTKERNELTYPES_H

#include "ttmlir-c/Dialects.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirType ttmlirTTKernelCBTypeGet(MlirContext ctx,
                                                    MlirType memrefType);

MLIR_CAPI_EXPORTED MlirType ttmlirTTKernelSemaphoreTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirType ttmlirTTKernelNocAddrTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTKernelThreadTypeAttrGet(MlirContext ctx, uint32_t enumValue);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTKernelReduceTypeAttrGet(MlirContext ctx, uint32_t enumValue);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTKernelReduceDimAttrGet(MlirContext ctx, uint32_t enumValue);

MLIR_CAPI_EXPORTED MlirType ttmlirTTKernelL1AddrTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirType ttmlirTTKernelL1AddrPtrTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirType
ttmlirTTKernelInterleavedAddrGenFastTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirType ttmlirTTKernelDataFormatTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTKernelArgAttrGet(MlirContext ctx,
                                                          MlirType argType,
                                                          size_t operandIndex,
                                                          bool isUniform);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTKernelArgSpecAttrGet(
    MlirContext ctx, MlirAttribute *rtArgs, size_t rtArgsSize,
    MlirAttribute *ctArgs, size_t ctArgsSize);

#ifdef __cplusplus
}
#endif

#endif // TTMLIR_C_TTKERNELTYPES_H
