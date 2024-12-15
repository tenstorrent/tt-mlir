// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_TTKERNELTOCPP_H
#define TTMLIR_TARGET_TTKERNEL_TTKERNELTOCPP_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttkernel {

// Translates a TTKernel operation to C++ and writes it to the given
// stream.
LogicalResult translateTTKernelToCpp(
    Operation *op, llvm::raw_ostream &os);
} // namespace mlir::tt::ttkernel

#endif
