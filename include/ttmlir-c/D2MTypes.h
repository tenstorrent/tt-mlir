// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_C_D2MTYPES_H
#define TTMLIR_C_D2MTYPES_H

#include "ttmlir-c/Dialects.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirAttribute ttmlirD2MThreadTypeAttrGet(MlirContext ctx,
                                                            uint32_t enumValue);

#ifdef __cplusplus
}
#endif

#endif // TTMLIR_C_D2MTYPES_H
