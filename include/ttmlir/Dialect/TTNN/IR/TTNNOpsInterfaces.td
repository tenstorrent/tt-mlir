// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_IR_TTNNOPSINTERFACES_TD
#define TTMLIR_DIALECT_TTNN_IR_TTNNOPSINTERFACES_TD

include "mlir/IR/OpBase.td"

def TTNN_ElementwiseUnary : OpInterface<"ElementwiseUnary"> {
  let cppNamespace = "::mlir::tt::ttnn";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Get the parameters for the elementwise unary operation.
        Returns an empty array by default, but ops can override this
        to return their specific parameters.
      }],
      /*retTy=*/"::llvm::ArrayRef<::mlir::FloatAttr>",
      /*methodName=*/"getParams",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return ::llvm::ArrayRef<::mlir::FloatAttr>();"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the unary operation type for this elementwise unary operation.
        This method must be implemented by each concrete unary op.
      }],
      /*retTy=*/"::mlir::tt::ttnn::UnaryOpType",
      /*methodName=*/"getUnaryOpType",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/""
    >,
  ];
}

def TTNN_ElementwiseBinary : OpInterface<"ElementwiseBinary"> {
  let cppNamespace = "::mlir::tt::ttnn";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Get the LHS operand of the binary operation.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getLhs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return $_op->getOperand(0);"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the RHS side operand of the binary operation.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getRhs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return $_op->getOperand(1);"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Add a lhs activation to the operation. Activations are added at the beginning of the array.
      }],
      /*retTy=*/"void",
      /*methodName=*/"addLhsActivation",
      /*args=*/(ins "::mlir::tt::ttnn::UnaryWithParamAttr":$activation),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        ::mlir::ArrayAttr currentActivations = ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>($_op.getProperties().lhs_activations);
        ::llvm::SmallVector<::mlir::Attribute> activations{activation};

        if (currentActivations) {
          activations.append(currentActivations.begin(), currentActivations.end());
        }

        $_op.getProperties().lhs_activations = ::mlir::ArrayAttr::get($_op.getContext(), activations);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Add a rhs activation to the operation. Activations are added at the beginning of the array.
      }],
      /*retTy=*/"void",
      /*methodName=*/"addRhsActivation",
      /*args=*/(ins "::mlir::tt::ttnn::UnaryWithParamAttr":$activation),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        ::mlir::ArrayAttr currentActivations = ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>($_op.getProperties().rhs_activations);
        ::llvm::SmallVector<::mlir::Attribute> activations{activation};

        if (currentActivations) {
          activations.append(currentActivations.begin(), currentActivations.end());
        }

        $_op.getProperties().rhs_activations = ::mlir::ArrayAttr::get($_op.getContext(), activations);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Add a post activation to the operation. Activations are added at the end of the array.
      }],
      /*retTy=*/"void",
      /*methodName=*/"addPostActivation",
      /*args=*/(ins "::mlir::tt::ttnn::UnaryWithParamAttr":$activation),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        ::mlir::ArrayAttr currentActivations = ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>($_op.getProperties().rhs_activations);
        ::llvm::SmallVector<::mlir::Attribute> activations;

        if (currentActivations) {
          activations.append(currentActivations.begin(), currentActivations.end());
        }
        activations.push_back(activation);

        $_op.getProperties().post_activations = ::mlir::ArrayAttr::get($_op.getContext(), activations);
      }]
    >,
  ];
}

#endif // TTMLIR_DIALECT_TTNN_IR_TTNNOPSINTERFACES_TD
