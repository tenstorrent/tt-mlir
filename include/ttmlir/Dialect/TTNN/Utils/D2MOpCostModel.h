// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_D2MOPCOSTMODEL_H
#define TTMLIR_DIALECT_TTNN_UTILS_D2MOPCOSTMODEL_H

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/TTNNOpConstraints.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <vector>

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::tt::ttnn {

/// Back-of-envelope cost model for D2M internal ops. Estimates L1 memory
/// constraints and runtime for unary, binary, reduction, and matmul ops
/// without calling the TTNN backend. Used by D2MSubgraphOp::getOpConstraints
/// and getOpRuntime so that one failing internal op does not invalidate the
/// whole subgraph.

/// Returns estimated OpConstraints for the given op when the op is supported
/// (unary, binary, reduction, matmul). Returns an error when the op is not
/// supported so the caller can fall back to the backend.
llvm::Expected<op_model::OpConstraints>
estimateOpConstraints(Operation *op, const std::vector<TTNNLayoutAttr> &inputs,
                      const OpConfig &opConfig);

/// Returns estimated runtime for the given op when supported. Returns an
/// error when the op is not supported.
llvm::Expected<size_t>
estimateOpRuntime(Operation *op, const std::vector<TTNNLayoutAttr> &inputs,
                  const OpConfig &opConfig);

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_UTILS_D2MOPCOSTMODEL_H
