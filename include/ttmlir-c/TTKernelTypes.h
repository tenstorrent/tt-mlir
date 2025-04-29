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
                                                    uint64_t address,
                                                    uint64_t port,
                                                    MlirType memrefType);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTKernelThreadTypeAttrGet(MlirContext ctx, uint32_t enumValue);

#ifdef __cplusplus
}
#endif

#endif // TTMLIR_C_TTKERNELTYPES_H
