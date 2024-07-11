// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_C_TTTYPES_H
#define TTMLIR_C_TTTYPES_H

#include "ttmlir-c/Dialects.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTOperandConstraintAttrGet(MlirContext ctx, uint32_t OperandConstraint);

#ifdef __cplusplus
}
#endif

#endif // TTMLIR_C_TTKERNELTYPES_H
