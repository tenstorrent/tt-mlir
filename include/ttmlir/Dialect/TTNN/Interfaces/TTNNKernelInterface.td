// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNKERNELINTERFACE_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNKERNELINTERFACE_TD

include "mlir/IR/Interfaces.td"

def TTNN_KernelInterface: AttrInterface<"KernelInterface"> {
  let summary = "Kernel interface";
  let cppNamespace = "::mlir::tt::ttnn";
  let description = [{
    Kernel interface for TTNN generic operation kernels
  }];
  let methods = [
    InterfaceMethod<
      "Get the symbol name for this kernel",
      "mlir::SymbolRefAttr", "getSymbolRef", (ins)
    >,
    InterfaceMethod<
      "Get the core range for this kernel",
      "mlir::Attribute", "getCoreRanges", (ins)
    >,
    InterfaceMethod<
      "Get common runtime arguments",
      "llvm::ArrayRef<mlir::Attribute>", "getCommonRtArgs", (ins)
    >,
    InterfaceMethod<
      "Get compile time arguments",
      "llvm::ArrayRef<mlir::Attribute>", "getCtArgs", (ins)
    >,
  ];
}

#endif  // TTMLIR_TTMLIR_DIALECT_TTNN_TTNNKERNELINTERFACE_TD
