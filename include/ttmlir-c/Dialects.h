// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#ifndef TTMLIR_C_DIALECTS_H
#define TTMLIR_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TT, tt);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TTIR, ttir);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TTKernel, ttkernel);

#ifdef __cplusplus
}
#endif

#endif // TTMLIR_C_DIALECTS_H
