// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRPRUNETOOUTPUT_TTIRPRUNETOOUTPUT_H
#define TTMLIR_CONVERSION_TTIRPRUNETOOUTPUT_TTIRPRUNETOOUTPUT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt {

struct TTIRPruneToOutputOptions {
  /// Keep only the result whose `nameAttr` result-attribute matches this name.
  std::string keepOutput;
  /// The result-attribute key used to identify port names (default:
  /// "ttir.name"). Pass "hw.port_name" when consuming CIRCT-emitted IR.
  std::string nameAttr = "ttir.name";
};

std::unique_ptr<OperationPass<ModuleOp>>
createTTIRPruneToOutputPass(const TTIRPruneToOutputOptions &options = {});

std::unique_ptr<OperationPass<ModuleOp>> createFuncDropUnusedArgsPass();

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTIRPRUNETOOUTPUT_TTIRPRUNETOOUTPUT_H
