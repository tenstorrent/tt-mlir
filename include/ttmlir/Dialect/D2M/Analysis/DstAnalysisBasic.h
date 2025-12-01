// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_DSTANALYSISBASIC_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_DSTANALYSISBASIC_H

#include "ttmlir/Dialect/D2M/Analysis/DstAnalysis.h"

#include <memory>

namespace mlir::tt::d2m {

/// Create a basic DST analysis strategy.
///
/// This strategy assigns each DST access its own slice with no reuse.
/// Provides an upper bound on requirements but is not optimal.
std::unique_ptr<DstAnalysis> createBasicDstAnalysis();

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_DSTANALYSISBASIC_H
