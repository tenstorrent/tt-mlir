// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_INTERFACES_TTNNTENSORSPECINTERFACE_TD
#define TTMLIR_DIALECT_TTNN_INTERFACES_TTNNTENSORSPECINTERFACE_TD

include "mlir/IR/OpBase.td"

// This interface is used to specify TTNN ops whose output data type can be
// queried. The dtype is derived from the first result's TTNNLayoutAttr encoding.
def TTNN_DtypeOpInterface : OpInterface<"TTNNDtypeOpInterface"> {
  let cppNamespace = "::mlir::tt::ttnn";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the output data type as a DataTypeAttr, derived from the
        targeted result's TTNNLayoutAttr encoding.
        }],
      /*retTy=*/"mlir::tt::ttcore::DataTypeAttr",
      /*methodName=*/"getDtypeFromResult",
      /*args=*/(ins "unsigned": $resultIndex),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::tt::ttnn::getDtypeFromResult($_op.getOperation(), resultIndex);
      }]
    >,
        InterfaceMethod<
      /*desc=*/[{
        Returns the output data type as a DataTypeAttr, derived from the
        first result's TTNNLayoutAttr encoding.
        }],
      /*retTy=*/"mlir::tt::ttcore::DataTypeAttr",
      /*methodName=*/"getDtypeAttr",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getDtypeFromResult(0u);
      }]
    >,
  ];
}

def TTNN_LayoutOpInterface : OpInterface<"TTNNLayoutOpInterface"> {
  let cppNamespace = "::mlir::tt::ttnn";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the target page layout as a LayoutAttr, derived from the first
        result's TTNN(ND)LayoutAttr encoding.
        }],
      /*retTy=*/"LayoutAttr",
      /*methodName=*/"getLayoutAttr",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::tt::ttnn::getLayoutAttrFromResult($_op.getOperation());
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
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getMemoryConfigFromResult($_op.getOperation());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Compatibility getter that forwards to getMemoryConfigAttr().
      }],
      /*retTy=*/"std::optional<MemoryConfigAttr>",
      /*methodName=*/"getMemoryConfig",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return getMemoryConfigFromResult($_op.getOperation());
      }]
    >,
  ];
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
