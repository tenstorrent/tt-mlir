// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_PASSES_H
#define TTMLIR_DIALECT_TTNN_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DECL
#include "ttmlir/Dialect/TTNN/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/TTNN/Passes.h.inc"

void createTTIRToTTNNBackendPipeline(OpPassManager &pm);
} // namespace mlir::tt::ttnn

#endif
