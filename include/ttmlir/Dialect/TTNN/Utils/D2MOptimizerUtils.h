// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_D2MOPTIMIZERUTILS_H
#define TTMLIR_DIALECT_TTNN_UTILS_D2MOPTIMIZERUTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::d2m_optimizer_utils {

/// Applies the chosen layout to a D2MSubgraphOp: result type(s), output
/// buffer(s) (Empty op), and the referenced D2M subgraph function.
///
/// This updates:
/// - The dispatch op's result type(s)
/// - The EmptyOp output buffer type and memory config
/// - The D2M function body argument types (to match dispatch inputs)
/// - Inserts a trailing to_layout in the D2M function body if the return
///   type differs from the chosen layout
/// - The D2M function signature (input and result types)
void applyChosenLayoutToD2MSubgraphOp(D2MSubgraphOp dispatchOp,
                                      RankedTensorType newTensorType,
                                      TTNNLayoutAttr layoutAttr,
                                      ttcore::GridAttr deviceGrid);

/// Convenience wrapper: computes newTensorType from the dispatch op's current
/// result type and the chosen layout, then delegates to the overload above.
/// Handles quantized element type preservation.
void applyChosenLayoutToD2MSubgraphOp(D2MSubgraphOp dispatchOp,
                                      TTNNLayoutAttr chosenLayout,
                                      ttcore::GridAttr deviceGrid);

/// Sync the D2M subgraph function's argument and function types to the
/// dispatch op's current input types (e.g. after spill/reshard, operand
/// types may have changed to DRAM or different sharding).
void syncD2MFuncTypesToDispatchInputs(D2MSubgraphOp dispatchOp);

/// Walk all D2MSubgraphOps within a function and sync their function types
/// to match dispatch input types.
void syncAllD2MFuncTypes(func::FuncOp func);

} // namespace mlir::tt::ttnn::d2m_optimizer_utils

#endif // TTMLIR_DIALECT_TTNN_UTILS_D2MOPTIMIZERUTILS_H
