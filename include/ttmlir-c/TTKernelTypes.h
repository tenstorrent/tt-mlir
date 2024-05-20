// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
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

#ifdef __cplusplus
}
#endif

#endif // TTMLIR_C_TTKERNELTYPES_H
