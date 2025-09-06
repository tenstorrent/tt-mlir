// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_C_TTIRTYPES_H
#define TTMLIR_C_TTIRTYPES_H

#include "ttmlir-c/Dialects.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTIRThreadTypeAttrGet(MlirContext ctx, uint32_t enumValue);

#ifdef __cplusplus
}
#endif

#endif // TTMLIR_C_TTIRTYPES_H
