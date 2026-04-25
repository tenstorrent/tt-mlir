// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_CONSTRUCTTTIRLEC_H
#define TTMLIR_CONVERSION_CONSTRUCTTTIRLEC_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt {

struct ConstructTTIRLECOptions {
  std::string firstFunc;
  std::string secondFunc;
};

std::unique_ptr<OperationPass<ModuleOp>>
createConstructTTIRLECPass(const ConstructTTIRLECOptions &options = {});

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_CONSTRUCTTTIRLEC_H
