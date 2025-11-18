// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"

int main() {
  // Test that we can compile and link against a basic TTMLIR library.
  mlir::DialectRegistry registry;
  registry.insert<mlir::tt::ttcore::TTCoreDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto gridAttr = mlir::tt::ttcore::GridAttr::get(&context);
  (void)gridAttr;

  return 0;
}
