// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_TRANSFORMS_PASSES_H
#define TTMLIR_DIALECT_TTIR_TRANSFORMS_PASSES_H

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"

namespace mlir::bufferization {
struct OneShotBufferizationOptions;
} // namespace mlir::bufferization

namespace mlir::tt::ttir {
#define GEN_PASS_DECL
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

} // namespace mlir::tt::ttir

#endif
