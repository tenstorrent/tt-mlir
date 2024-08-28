// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License -Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELATTRINTERFACES_TD
#define TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELATTRINTERFACES_TD

include "mlir/IR/Interfaces.td"

//===----------------------------------------------------------------------===//
// TTKernel type definitions
//===----------------------------------------------------------------------===//

def TTKernel_KernelConfigInterface: AttrInterface<"KernelConfigInterface"> {
  let summary = "TT KernelConfigInterface";
  let cppNamespace = "::mlir::tt::ttkernel";
  let description = [{
  }];
  let methods = [
    InterfaceMethod<
      "Get the thread type for this KernelConfig",
      "ThreadType", "getThreadType", (ins)
    >,
    InterfaceMethod<
      "Query the thread type for this KernelConfig",
      "bool", "isThreadType", (ins "ThreadType":$threadType),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return $_attr.getThreadType() == threadType;"
    >,
  ];
}

#endif
