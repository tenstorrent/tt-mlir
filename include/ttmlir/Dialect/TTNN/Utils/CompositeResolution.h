// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_COMPOSITERESOLUTION_H
#define TTMLIR_DIALECT_TTNN_UTILS_COMPOSITERESOLUTION_H

namespace mlir::tt::ttnn {

// Controls how ttcore.composite ops are resolved in TTNNResolveComposites:
//   Auto         — pipeline decides: Validate when optimizer+OpModel available,
//                  else Inline. This is the default when no option is given.
//   Inline       — always inline the decomposition body; never upgraded.
//   Validate     — promote to typed op if OpModel validates, else inline.
//   ForcePromote — unconditionally promote (testing only).
enum class CompositeResolution { Auto, Inline, Validate, ForcePromote };

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_UTILS_COMPOSITERESOLUTION_H
