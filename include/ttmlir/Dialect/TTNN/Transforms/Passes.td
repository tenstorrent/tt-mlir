// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TRANSFORMS_TTNNPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TRANSFORMS_TTNNPASSES_TD

include "mlir/Pass/PassBase.td"

def TTNNOpenDevice: Pass<"ttnn-open-device", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    todo
  }];
}

def ConvertTTIRToTTNN: Pass<"convert-ttir-to-ttnn", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    todo
  }];
}

def TTNNSerializeToBinary: Pass<"ttnn-serialize-to-binary", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    todo
  }];

  list<Option> options = [
    Option<"output", "output", "std::string", "", "Output binary path">,
  ];
}

#endif
