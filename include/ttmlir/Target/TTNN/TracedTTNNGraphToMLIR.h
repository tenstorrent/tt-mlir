// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTNN_TTNNTOFLATBUFFER_H
#define TTMLIR_TARGET_TTNN_TTNNTOFLATBUFFER_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir::tt::ttnn {

// Translate a traced TTNN graph to MLIR.
OwningOpRef<ModuleOp> translateTracedTTNNGraphToMLIR(llvm::SourceMgr &sourceMgr,
                                                     MLIRContext *context);

} // namespace mlir::tt::ttnn

#endif
