// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTMETAL_PASSES_H
#define TTMLIR_DIALECT_TTMETAL_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"
#include <memory>

namespace mlir::tt::ttmetal {
#define GEN_PASS_DECL
#include "ttmlir/Dialect/TTMetal/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/TTMetal/Passes.h.inc"

void createTTIRToTTMetalBackendPipeline(OpPassManager &pm);
} // namespace mlir::tt::ttmetal

#endif
