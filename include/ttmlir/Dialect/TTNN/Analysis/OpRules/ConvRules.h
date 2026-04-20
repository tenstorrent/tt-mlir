// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_CONVRULES_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_CONVRULES_H

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// Conv2d / ConvTranspose2d rules:
//
// Output hints:
//   Use full legal configs (carry Conv2dConfig tied to output hint).
//   Attempt L1 sharding.
//
// Op-specific attributes:
//   Set Conv2dConfig + DeviceComputeKernelConfig from candidate.
//
// Candidate tiebreaker:
//   Prefer lower act_block_h_override (0=auto is optimal).
//
// Pre-optimization fixup:
//   Set L1Full slice config on Conv2d ops before validation.
//
// Post-propagation fixup:
//   Disable deallocate_activation for Conv2d/ConvTranspose2d with multi-use
//   inputs (prevents premature deallocation crashes).
//===----------------------------------------------------------------------===//

struct Conv2dRuleBook : OpRuleBook {
  /// Output hints: full legal configs with Conv2dConfig.
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;

  /// Apply Conv2dConfig + DeviceComputeKernelConfig from candidate.
  void applyOpSpecificAttrs(Operation *op,
                            const BeamCandidate &candidate) const override;

  /// Tiebreaker: prefer lower act_block_h_override.
  bool preferCandidate(Operation *op, const BeamCandidate &a,
                       const BeamCandidate &b) const override;

  /// Reject output hints that would mix sharding types with the input.
  bool isValidOutputHintForInputs(
      const OpConfig &hint,
      llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const override;
};

/// Set L1Full slice config on all Conv2d ops before validation.
void applyConvSliceConfig(ModuleOp moduleOp);

/// Disable deallocate_activation for conv ops with multi-use inputs.
void fixupConvDeallocate(func::FuncOp func);

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_CONVRULES_H
