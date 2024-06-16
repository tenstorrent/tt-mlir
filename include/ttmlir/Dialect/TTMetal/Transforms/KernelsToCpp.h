// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTMETAL_TRANSFORMS_KERNELSTOCPP_H
#define TTMLIR_DIALECT_TTMETAL_TRANSFORMS_KERNELSTOCPP_H

#include "mlir/Support/LogicalResult.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

namespace mlir::tt::ttmetal {
LogicalResult emitDispatchOpRegionAsCpp(DispatchOp dispatchOp,
                                        unsigned regionNumber,
                                        llvm::raw_ostream &os);
} // namespace mlir::tt::ttmetal
#endif
