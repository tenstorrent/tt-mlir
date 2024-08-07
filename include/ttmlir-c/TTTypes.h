// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_C_TTKERNELTYPES_H
#define TTMLIR_C_TTKERNELTYPES_H

#include "ttmlir-c/Dialects.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirType ttmlirTTTileTypeGet(MlirContext ctx,
                                                unsigned height, unsigned width,
                                                uint32_t dataType);

MLIR_CAPI_EXPORTED MlirType ttmlirTTDeviceTypeGet(MlirContext ctx,
                                                  MlirAttribute deviceAttr);

#ifdef __cplusplus
}
#endif

#endif // TTMLIR_C_TTKERNELTYPES_H
