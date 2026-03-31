// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSINTERFACES_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSINTERFACES_TD

include "mlir/IR/OpBase.td"

def TTNN_ElementwiseUnary : OpInterface<"ElementwiseUnary"> {
  let cppNamespace = "::mlir::tt::ttnn";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Get the input operand of the unary operation.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getInput",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return $_op->getOperand(0);"
         >,
    InterfaceMethod<
      /*desc=*/[{
        Get the parameters for the elementwise unary operation.
        Returns an empty array by default, but ops can override this
        to return their specific parameters.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::FloatAttr>",
      /*methodName=*/"getParams",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/""
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
        Add an input_tensor_a activation to the operation. Activations are added at the beginning of the array.
      }],
      /*retTy=*/"void",
      /*methodName=*/"addInputTensorAActivation",
      /*args=*/(ins "::mlir::tt::ttnn::UnaryOpType":$opType, "::llvm::ArrayRef<::mlir::FloatAttr>":$params),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto activation = ::mlir::tt::ttnn::UnaryWithParamAttr::get($_op.getContext(), opType, params);
        ::mlir::ArrayAttr currentActivations = $_op.getProperties().input_tensor_a_activations;
        ::llvm::SmallVector<::mlir::Attribute> activations;

        activations.push_back(activation);
        if (currentActivations) {
          activations.append(currentActivations.begin(), currentActivations.end());
        }

        $_op.getProperties().input_tensor_a_activations = ::mlir::ArrayAttr::get($_op.getContext(), activations);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Add an input_tensor_b activation to the operation. Activations are added at the beginning of the array.
      }],
      /*retTy=*/"void",
      /*methodName=*/"addInputTensorBActivation",
      /*args=*/(ins "::mlir::tt::ttnn::UnaryOpType":$opType, "::llvm::ArrayRef<::mlir::FloatAttr>":$params),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto activation = ::mlir::tt::ttnn::UnaryWithParamAttr::get($_op.getContext(), opType, params);
        ::mlir::ArrayAttr currentActivations = $_op.getProperties().input_tensor_b_activations;
        ::llvm::SmallVector<::mlir::Attribute> activations;

        activations.push_back(activation);
        if (currentActivations) {
          activations.append(currentActivations.begin(), currentActivations.end());
        }

        $_op.getProperties().input_tensor_b_activations = ::mlir::ArrayAttr::get($_op.getContext(), activations);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Add an output activation to the operation. Activations are added at the end of the array.
      }],
      /*retTy=*/"void",
      /*methodName=*/"addActivation",
      /*args=*/(ins "::mlir::tt::ttnn::UnaryOpType":$opType, "::llvm::ArrayRef<::mlir::FloatAttr>":$params),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto activation = ::mlir::tt::ttnn::UnaryWithParamAttr::get($_op.getContext(), opType, params);
        ::mlir::ArrayAttr currentActivations = $_op.getProperties().activations;
        ::llvm::SmallVector<::mlir::Attribute> activations;

        if (currentActivations) {
          activations.append(currentActivations.begin(), currentActivations.end());
        }
        activations.push_back(activation);

        $_op.getProperties().activations = ::mlir::ArrayAttr::get($_op.getContext(), activations);
      }]
    >,
  ];
}

#endif
