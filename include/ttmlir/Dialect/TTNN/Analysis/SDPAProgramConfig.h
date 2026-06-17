// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_SDPAPROGRAMCONFIG_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_SDPAPROGRAMCONFIG_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/Operation.h"

#include <optional>

namespace mlir::tt::ttnn {

// Generate a fixed SDPAProgramConfig for a paged SDPA decode op from its
// attributes (head_dim threshold, page table block count, ...) and tensor
// shapes, plus the device worker grid.
//
// This mirrors generateMatmulProgramConfig: it deterministically derives a
// single program config from static information so that the caller can validate
// it against L1 via OpModel and fall back if it does not fit. Returns nullopt
// when a config cannot/should not be generated (e.g. the op is not a paged SDPA
// decode op, or it already carries an explicit program config).
//
// Companion of the opt-level=0 workaround
// PagedScaledDotProductAttentionDecodeProgramConfigRewritePattern; at
// opt-level>=1 OperationValidationAndFallback owns this decision.
//
// Metal issue: https://github.com/tenstorrent/tt-metal/issues/44311
std::optional<SDPAProgramConfigAttr>
generateSDPADecodeProgramConfig(Operation *op);

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_SDPAPROGRAMCONFIG_H
