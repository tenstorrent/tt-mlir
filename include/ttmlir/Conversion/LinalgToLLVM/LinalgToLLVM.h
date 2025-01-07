// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Pass/Pass.h"

#include "ttmlir/Dialect/TT/IR/TTOps.h"

namespace mlir::tt {
    std::unique_ptr<OperationPass<tt::CPUModuleOp>> createConvertLinalgToLLVMPass();
}