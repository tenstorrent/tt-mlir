// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_INTERFACES_TTNNTENSORSPECINTERFACE_TD
#define TTMLIR_DIALECT_TTNN_INTERFACES_TTNNTENSORSPECINTERFACE_TD

include "mlir/IR/OpBase.td"

// This interface is used to specify TTNN ops that have defined DtypeAttr
def TTNN_DtypeOpInterface : OpInterface<"TTNNDtypeOpInterface"> {
  let cppNamespace = "::mlir::tt::ttnn";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        }],
      /*retTy=*/"mlir::tt::ttcore::DataTypeAttr",
      /*methodName=*/"getDtypeAttr",
      /*args=*/(ins),
      /*methodBody=*/[{
        return $_op.getDtypeAttr();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        }],
      /*retTy=*/"void",
      /*methodName=*/"setDtypeAttr",
      /*args=*/(ins "mlir::tt::ttcore::DataTypeAttr": $dataTypeAttr),
      /*methodBody=*/[{
        $_op.setDtypeAttr(dataTypeAttr);
      }]
    >,
  ];

  let verify = [{
    return verifyTTNNDtypeOpInterface<ConcreteOp>($_op);
  }];
}

def TTNN_LayoutOpInterface : OpInterface<"TTNNLayoutOpInterface"> {
  let cppNamespace = "::mlir::tt::ttnn";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        }],
      /*retTy=*/"LayoutAttr",
      /*methodName=*/"getLayoutAttr",
      /*args=*/(ins),
      /*methodBody=*/[{
        return $_op.getLayoutAttr();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        }],
      /*retTy=*/"void",
      /*methodName=*/"setLayoutAttr",
      /*args=*/(ins "LayoutAttr": $layoutAttr),
      /*methodBody=*/[{
        $_op.setLayoutAttr(layoutAttr);
      }]
    >,
  ];

  let verify = [{
    return verifyTTNNLayoutInterface<ConcreteOp>($_op);
  }];
}

def TTNN_MemoryConfigOpInterface : OpInterface<"TTNNMemoryConfigOpInterface"> {
  let cppNamespace = "::mlir::tt::ttnn";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        }],
      /*retTy=*/"MemoryConfigAttr",
      /*methodName=*/"getMemoryConfigAttr",
      /*args=*/(ins),
      /*methodBody=*/[{
        return $_op.getMemoryConfigAttr();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        }],
      /*retTy=*/"void",
      /*methodName=*/"setMemoryConfigAttr",
      /*args=*/(ins "MemoryConfigAttr": $memoryConfigAttr),
      /*methodBody=*/[{
        $_op.setMemoryConfigAttr(memoryConfigAttr);
      }]
    >,
  ];

  let verify = [{
    return verifyTTNNMemoryConfigInterface<ConcreteOp>($_op);
  }];
}

def TTNN_ComputeKernelConfigOpInterface : OpInterface<"TTNNComputeKernelConfigOpInterface"> {
  let cppNamespace = "::mlir::tt::ttnn";
  let description = [{
    Interface for operations that support ComputeKernelConfig attribute.
  }];
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Get the compute kernel config attribute.
        Returns nullptr if not set.
      }],
      /*retTy=*/"DeviceComputeKernelConfigAttr",
      /*methodName=*/"getComputeConfigAttr",
      /*args=*/(ins),
      /*methodBody=*/[{
        return $_op.getComputeConfigAttr();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Set the compute kernel config attribute.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setComputeConfigAttr",
      /*args=*/(ins "DeviceComputeKernelConfigAttr": $computeConfigAttr),
      /*methodBody=*/[{
        $_op.setComputeConfigAttr(computeConfigAttr);
      }]
    >,
  ];

  let verify = [{
    return verifyTTNNComputeKernelConfigInterface<ConcreteOp>($_op);
  }];
}

def TTNN_TensorSpecInterface : OpInterface<"TTNNTensorSpecInterface", [TTNN_DtypeOpInterface, TTNN_LayoutOpInterface, TTNN_MemoryConfigOpInterface]>
{
  let cppNamespace = "::mlir::tt::ttnn";
}

#endif // TTMLIR_DIALECT_TTNN_INTERFACES_TTNNTENSORSPECINTERFACE_TD
