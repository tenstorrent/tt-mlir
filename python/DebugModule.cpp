// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

#include "ttmlir/Dialect/Debug/IR/Debug.h"
#include "ttmlir/Dialect/Debug/IR/DebugOps.h"

#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/AffineMap.h"

#include <cstdint>
#include <vector>

namespace mlir::ttmlir::python {
void populateDebugModule(nb::module_ &m) {
  // Populate Debug dialect bindings here when there are any.
}
} // namespace mlir::ttmlir::python
