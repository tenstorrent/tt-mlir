// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_IR_TTNNOPSINTERFACES_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_IR_TTNNOPSINTERFACES_TD

include "mlir/IR/OpBase.td"

def TTNNLayoutUpdatableInterface : OpInterface<"TTNNLayoutUpdatableInterface"> {
  let cppNamespace = "::mlir::tt::ttnn";
  let description = [{
    Interface for operations that have a layout attribute that can be updated.
    This interface also constraints the name used for layout attribute to 'layout'.
  }];
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Utility method which can be used to update layout attribute on operation.
      }],
      /*retTy=*/"void",
      /*methodName==*/"updateLayoutAttribute",
      /*args=*/(ins "ttnn::LayoutAttr":$newLayoutAttr),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        ConcreteOp concreteOp = cast<ConcreteOp>(this->getOperation());
        concreteOp.setLayoutAttr(newLayoutAttr);
      }]
    >
  ];
}

#endif
