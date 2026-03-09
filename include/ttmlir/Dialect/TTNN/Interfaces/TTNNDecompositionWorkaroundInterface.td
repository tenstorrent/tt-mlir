// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_INTERFACES_TTNNDECOMPOSITIONWORKAROUNDINTERFACE_TD
#define TTMLIR_DIALECT_TTNN_INTERFACES_TTNNDECOMPOSITIONWORKAROUNDINTERFACE_TD

include "mlir/IR/OpBase.td"

def TTNN_DecompositionWorkaroundInterface : OpInterface<"DecompositionWorkaroundInterface"> {
  let cppNamespace = "::mlir::tt::ttnn::wa";

  let description = [{
    Interface for operations that provide decomposition workarounds.
    Operations return a list of workaround objects that can be applied
    to transform the operation to meet hardware constraints.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Get decomposition workarounds for this operation.
        Returns a list of workaround objects that should be tried in order.
        Empty list means no workarounds needed.
      }],
      /*retTy=*/"::mlir::tt::ttnn::wa::DecompositionWorkarounds",
      /*methodName=*/"getDecompositionWorkarounds",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return {};"
    >,
  ];
}

#endif // TTMLIR_DIALECT_TTNN_INTERFACES_TTNNDECOMPOSITIONWORKAROUNDINTERFACE_TD