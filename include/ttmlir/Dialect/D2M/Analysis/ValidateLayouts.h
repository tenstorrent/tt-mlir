// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLLIR_DIALECT_D2M_TRANSFORMS_VALIDATELAYOUTS_H
#define TTMLLIR_DIALECT_D2M_TRANSFORMS_VALIDATELAYOUTS_H

#include "mlir/Pass/Pass.h"

namespace ttmlir::d2m {

/// Creates a pass that validates D2M operations have proper MetalLayoutAttr
/// encoding on their tensor operands and results.
std::unique_ptr<mlir::Pass> createD2MValidateLayoutsPass();

} // namespace ttmlir::d2m

#endif // TTMLLIR_DIALECT_D2M_TRANSFORMS_VALIDATELAYOUTS_H
