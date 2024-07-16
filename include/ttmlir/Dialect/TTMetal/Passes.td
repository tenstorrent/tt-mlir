// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALPASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertTTIRToTTMetal : Pass<"convert-ttir-to-ttmetal", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    todo
  }];
}

def TTMetalSerializeToBinary: Pass<"ttmetal-serialize-to-binary", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    todo
  }];

  list<Option> options = [
    Option<"output", "output", "std::string", "", "Output binary path">,
  ];
}

#endif
