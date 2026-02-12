// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_PASSES_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/GreedyL1SpillManagement.h"
#include "ttmlir/Dialect/TTNN/Transforms/GreedyLayoutPropagation.h"
#include "ttmlir/Dialect/TTNN/Transforms/Optimizer.h"
#include "ttmlir/Dialect/TTNN/Utils/MathFidelityParser.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerOverrides.h"

namespace mlir::tt::ttnn {

// TTNN Passes
#define GEN_PASS_DECL
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

} // namespace mlir::tt::ttnn

#endif
