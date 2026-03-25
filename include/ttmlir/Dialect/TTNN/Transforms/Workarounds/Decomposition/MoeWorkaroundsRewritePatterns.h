// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MOEWORKAROUNDSREWRITEPATTERNS_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MOEWORKAROUNDSREWRITEPATTERNS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Populate all MoE-specific decomposition workarounds.
//
// This is the single entry point for MoE hardware workarounds, including:
// - all_to_all_dispatch input-rank canonicalization to [B, 1, S, H]
// - all_to_all_combine input-layout canonicalization to [E, BD, S, H]
// - moe_expert_token_remap topk-rank canonicalization
// - sparse_matmul tile-dimension reshaping/decomposition
// - all_to_all_combine decode shard-dim runtime workaround
void populateMoeWorkaroundPatterns(RewritePatternSet &patterns);

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MOEWORKAROUNDSREWRITEPATTERNS_H
