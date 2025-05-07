// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_STABLEHLOTOTTIR_MULTIDEVICETENSORCLEANUP_H
#define TTMLIR_CONVERSION_STABLEHLOTOTTIR_MULTIDEVICETENSORCLEANUP_H

#include "mlir/Pass/Pass.h"

namespace mlir::tt {

#ifdef TTMLIR_ENABLE_STABLEHLO

std::unique_ptr<mlir::Pass> createTTIRMultideviceTensorAnnotationCleanupPass();

#endif

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_STABLEHLOTOTTIR_MULTIDEVICETENSORCLEANUP_H
