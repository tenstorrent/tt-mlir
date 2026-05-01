// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRPRUNETOOUTPUT_TTIRPRUNETOOUTPUT_H
#define TTMLIR_CONVERSION_TTIRPRUNETOOUTPUT_TTIRPRUNETOOUTPUT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt {

struct TTIRPruneToOutputOptions {
  /// Keep only the result with this hw.port_name.
  std::string keepOutput;
};

std::unique_ptr<OperationPass<ModuleOp>>
createTTIRPruneToOutputPass(const TTIRPruneToOutputOptions &options = {});

std::unique_ptr<OperationPass<ModuleOp>> createFuncDropUnusedArgsPass();

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTIRPRUNETOOUTPUT_TTIRPRUNETOOUTPUT_H
