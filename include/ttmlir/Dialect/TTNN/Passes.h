// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_PASSES_H
#define TTMLIR_DIALECT_TTNN_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include <memory>

namespace mlir::tt::ttnn {
#define GEN_PASS_DECL
#include "ttmlir/Dialect/TTNN/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/TTNN/Passes.h.inc"
} // namespace mlir::tt::ttnn

#endif
