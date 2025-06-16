// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_JSONGRAPHINGESTPASS_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_JSONGRAPHINGESTPASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_DECL_JSONNETWORKINGEST
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createJSONNetworkIngestPass(StringRef jsonFilePath);

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_JSONGRAPHINGESTPASS_H
