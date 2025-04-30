// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTMETAL_IR_TTMETALATTRINTERFACES_TD
#define TTMLIR_TTMLIR_DIALECT_TTMETAL_IR_TTMETALATTRINTERFACES_TD

include "mlir/IR/Interfaces.td"
include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsEnums.td"

//===----------------------------------------------------------------------===//
// TTMetal interface definitions
//===----------------------------------------------------------------------===//

def TTMetal_KernelConfigInterface: AttrInterface<"KernelConfigInterface"> {
  let summary = "TT KernelConfigInterface";
  let cppNamespace = "::mlir::tt::ttmetal";
  let description = [{
  }];
  let methods = [
    InterfaceMethod<
      "Get the symbol name for this kernel",
      "mlir::SymbolRefAttr", "getKernelSymbol", (ins)
    >,
    InterfaceMethod<
      "Get the thread type for this KernelConfig",
      "mlir::tt::ttkernel::ThreadType", "getThreadType", (ins)
    >,
    InterfaceMethod<
      "Get the core ranges KernelConfig",
      "mlir::Attribute", "getCoreRange", (ins)
    >,
    InterfaceMethod<
      "Get the kernel arguments for this KernelConfig",
      "mlir::Attribute", "getKernelArgs", (ins)
    >,
  ];
}

#endif
