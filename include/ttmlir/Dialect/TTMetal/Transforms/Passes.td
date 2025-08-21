// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALPASSES_TD

include "mlir/Pass/PassBase.td"

def TTMetalGenerateStridedHostIO: Pass<"ttmetal-generate-strided-host-io", "::mlir::ModuleOp"> {
  let summary = "Foo.";
  let description = [{
    Bar
  }];
  let dependentDialects = ["mlir::tt::ttmetal::TTMetalDialect", "mlir::memref::MemRefDialect"];
}

#endif
