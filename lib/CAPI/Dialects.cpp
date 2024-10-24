// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir-c/Dialects.h"

#include "mlir/CAPI/Registration.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TT, tt, mlir::tt::TTDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TTIR, ttir, mlir::tt::ttir::TTIRDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TTKernel, ttkernel,
                                      mlir::tt::ttkernel::TTKernelDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TTNN, ttnn, mlir::tt::ttnn::TTNNDialect)
