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
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TTNN, ttnn);

/// Registers all ttmlir and standard dialects and their extensions into the
/// provided dialect registry. This is safe to call from a Python extension
/// that links against the common CAPI library.
MLIR_CAPI_EXPORTED void
ttmlirRegisterAllDialects(MlirDialectRegistry dialectRegistry);

#ifdef __cplusplus
}
#endif

#endif // TTMLIR_C_DIALECTS_H
