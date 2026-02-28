// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPS_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPS_TD

include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td"
include "ttmlir/Dialect/TTNN/Interfaces/TTNNDeviceOperandInterface.td"
include "ttmlir/Dialect/TTNN/Interfaces/TTNNTensorSpecInterface.td"
include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.td"
include "ttmlir/Dialect/TTNN/IR/TTNNBase.td"
include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.td"
include "ttmlir/Dialect/TTNN/IR/TTNNOpsEnums.td"
include "ttmlir/Dialect/TTNN/IR/TTNNTraits.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"

def TTNN_GetDeviceOp : TTNN_Op<"get_device", [TTCore_DuplicateConstEvalTrait]> {
    let summary = "Get Device op.";
    let description = [{
      This op returns a submesh carved out from the parent runtime device.
      Mesh shape and mesh offset define the size and offset of the submesh.
    }];

    let arguments = (ins OptionalAttr<TTNN_MeshShapeAttr>:$mesh_shape,
                         OptionalAttr<TTNN_MeshOffsetAttr>:$mesh_offset);

    let results = (outs TTNN_Device:$device);
}

def TTNN_ToMemoryConfigOp : TTNN_Op<"to_memory_config", [TTNN_MemoryConfigOpInterface]> {
    let summary = "ToMemoryConfig op.";
    let description = [{
      This op converts the memory config of the input tensor based on the given memory config.
      It handles:
        - Dram to L1
        - L1 to Dram
        - Interleaved to sharded
        - Sharded to interleaved
        - Sharded to sharded (reshard)
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         TTNN_MemoryConfigAttr:$memory_config);
    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_ToLayoutOp : TTNN_Op<"to_layout"> {
    let summary = "ToLayout op.";
    let description = [{
      This op wraps all layout information gathered from ttir.toLayout. It is used/updated by the optimizer
      to perform optimizations, and later broken down into specific memory/layout operations (toDevice, toMemoryConfig etc.).
      Currently in the TTNN backend, we use this op solely for tilize/untilize, therefore marking all other attrs as optional.
      Once ttnn::to_layout supports other attrs, we can remove the optional tag.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         TTNN_LayoutAttr:$layout,
                         OptionalAttr<TTCore_DataTypeAttr>:$dtype,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);
    let results = (outs AnyRankedTensor:$result);

    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

def TTNN_TypecastOp : TTNN_Op<"typecast"> {
    let summary = "Typecast op.";
    let description = [{
      This op converts the data type of the input tensor based on the given data type.
      It handles:
        - conversions of data types.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         TTCore_DataTypeAttr:$dtype,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);
    let results = (outs AnyRankedTensor:$result);

    let builders =
    [
      OpBuilder<(ins "Type": $resultType, "Value": $input, "ttcore::DataTypeAttr": $dtype),
      [{
        build($_builder, $_state, resultType, input, dtype, /*memory_config=*/nullptr);
      }]>
    ];

    let hasFolder = 1;
    let hasCanonicalizeMethod = 1;
    let hasVerifier = 1;
}


def TTNN_ToDeviceOp : TTNN_Op<"to_device"> {
    let summary = "ToDevice op.";
    let description = [{
      This op sends the input tensor to the given device with the given memory config.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         TTNN_Device:$device,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);
    let results = (outs AnyRankedTensor:$result);
}

def TTNN_FromDeviceOp : TTNN_Op<"from_device"> {
    let summary = "FromDevice op.";
    let description = [{
      This op retrieves the input tensor from the given device.
    }];

    let arguments = (ins AnyRankedTensor:$input);
    let results = (outs AnyRankedTensor:$result);
}

class TTNN_ElementwiseUnaryOp<string mnemonic, list<Trait> traits = []> :
    TTNN_Op<mnemonic, [TTNN_MemoryConfigOpInterface] # traits> {
    let summary = "Eltwise unary op.";
    let description = [{
      Eltwise unary op.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);
    let results = (outs AnyRankedTensor:$result);

    let builders =
    [
      OpBuilder<(ins "Value": $input),
      [{
        build($_builder, $_state, {input.getType()}, input, /*memory_config=*/nullptr);
      }]>,
      OpBuilder<(ins "Type": $resultType, "Value": $input),
      [{
        build($_builder, $_state, resultType, input, /*memory_config=*/nullptr);
      }]>
    ];
}

class TTNN_ElementwiseBinaryOp<string mnemonic, list<Trait> traits = []> :
    TTNN_Op<mnemonic, [TTNN_MemoryConfigOpInterface, TTNN_DtypeOpInterface] # traits> {
    let summary = "Eltwise binary op.";
    let description = [{
      Eltwise binary op.
    }];

    let arguments = (ins AnyRankedTensor:$lhs,
                         AnyRankedTensor:$rhs,
                         OptionalAttr<TTCore_DataTypeAttr>:$dtype,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);
    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return
          wa::TTNNOperandsWorkaroundsFactory::createBinaryOpOperandsWorkarounds(*this);
      }
    }];

    let builders =
    [
      OpBuilder<(ins "Type": $resultType, "Value": $lhs, "Value": $rhs),
      [{
        ttcore::DataTypeAttr outputDTypeAttr = ttcore::DataTypeAttr();
        RankedTensorType rankedTensorType = mlir::cast<RankedTensorType>(resultType);
        TTNNLayoutAttr layoutAttr = mlir::dyn_cast_if_present<TTNNLayoutAttr>(rankedTensorType.getEncoding());
        if (layoutAttr)
        {
          outputDTypeAttr = $_builder.getAttr<ttcore::DataTypeAttr>(layoutAttr.getDataType());
        }
        build($_builder, $_state, resultType, lhs, rhs, outputDTypeAttr, /*memory_config=*/nullptr);
      }]>
    ];
}

class TTNN_ElementwiseBinaryCompositeOp<string mnemonic, list<Trait> traits = []> :
    TTNN_Op<mnemonic, [TTNN_MemoryConfigOpInterface] # traits> {
    let summary = "Eltwise binary composite op.";
    let description = [{
      Eltwise binary composite op.
    }];

    let arguments = (ins AnyRankedTensor:$lhs,
                         AnyRankedTensor:$rhs,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);
    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return
          wa::TTNNOperandsWorkaroundsFactory::createBinaryOpOperandsWorkarounds(*this);
      }
    }];

    let builders =
    [
      OpBuilder<(ins "Type": $resultType, "Value": $lhs, "Value": $rhs),
      [{
        build($_builder, $_state, resultType, lhs, rhs, /*memory_config=*/nullptr);
      }]>
    ];
}

class TTNN_ElementwiseBinaryCompositeScalarOp<string mnemonic, list<Trait> traits = []> :
  TTNN_Op<mnemonic, [TTNN_MemoryConfigOpInterface] # traits> {
  let summary = "Eltwise binary composite op.";
  let description = [{
    Eltwise binary composite op where the first input is a tensor and the second is a scalar.
    It applies the op independently to each element of the tensor using the scalar.
  }];

  let arguments = (ins AnyRankedTensor:$lhs,
                        AnyAttrOf<[F32Attr, I32Attr]>:$rhs,
                        OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);
  let results = (outs AnyRankedTensor:$result);

  let builders =
  [
    OpBuilder<(ins "Type": $resultType, "Value": $lhs, "::mlir::Attribute": $rhs),
    [{
      build($_builder, $_state, resultType, lhs, rhs, /*memory_config=*/nullptr);
    }]>
  ];
}

class TTNN_ElementwiseTernaryOp<string mnemonic, list<Trait> traits = []> :
    TTNN_Op<mnemonic, [TTNN_MemoryConfigOpInterface] # traits> {
    let summary = "Eltwise ternary op.";
    let description = [{
      Eltwise ternary op.
    }];

    let arguments = (ins AnyRankedTensor:$first,
                         AnyRankedTensor:$second,
                         AnyRankedTensor:$third,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);
    let results = (outs AnyRankedTensor:$result);

    let builders =
    [
      OpBuilder<(ins "Type": $resultType, "Value": $first, "Value": $second, "Value": $third),
      [{
        build($_builder, $_state, resultType, first, second, third, /*memory_config=*/nullptr);
      }]>
    ];
}

def TTNN_WhereOp : TTNN_ElementwiseTernaryOp<"where", [ExplicateOperandBroadcastsTrait]> {
    let summary = "Eltwise where.";
    let description = [{
      Eltwise where operation.
    }];

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        ::mlir::Operation::operand_range inputs = getOperands();
        return
          wa::TTNNOperandsWorkaroundsFactory::createWhereOpOperandsWorkarounds(
                                                                        inputs);
      }
    }];
}

def TTNN_AbsOp : TTNN_ElementwiseUnaryOp<"abs"> {
    let summary = "Eltwise absolute.";
    let description = [{
      Eltwise absolute operation.
    }];
}

def TTNN_CbrtOp : TTNN_ElementwiseUnaryOp<"cbrt"> {
    let summary = "Eltwise cubic root.";
    let description = [{
      Eltwise cubic root operation.
    }];
}

def TTNN_CeilOp : TTNN_ElementwiseUnaryOp<"ceil"> {
    let summary = "Eltwise ceil.";
    let description = [{
      Eltwise ceil operation.
    }];
}

def TTNN_SignOp: TTNN_ElementwiseUnaryOp<"sign"> {
    let summary = "Eltwise sign operation.";
    let description = [{
      Returns the sign of the `operand` element-wise and produces a `result`
      tensor.

      Example:
        %a: [[3, -2, 0], [1, -4, 4]]
        "ttnn.sign"(%a, %out) -> %out: [[1, -1, 0], [1, -1, 1]]
    }];
}

def TTNN_CosOp : TTNN_ElementwiseUnaryOp<"cos"> {
    let summary = "Eltwise cosine.";
    let description = [{
      Eltwise cosine operation.
    }];
}

def TTNN_ExpOp : TTNN_ElementwiseUnaryOp<"exp"> {
    let summary = "Eltwise exponential.";
    let description = [{
      Eltwise exponential operation.
    }];
}

def TTNN_ErfOp : TTNN_ElementwiseUnaryOp<"erf"> {
    let summary = "Eltwise erf op.";
    let description = [{
        Eltwise erf operation. Calculates erf(x) for each element of the input tensor.
      }];
}

def TTNN_ErfcOp : TTNN_ElementwiseUnaryOp<"erfc"> {
    let summary = "Eltwise erfc op.";
    let description = [{
        Eltwise erfc operation. Calculates erfc(x) for each element of the input tensor.
      }];
}

def TTNN_FloorOp: TTNN_ElementwiseUnaryOp<"floor"> {
    let summary = "Eltwise floor op.";
    let description = [{
      Eltwise floor operation.
    }];
}

def TTNN_GeluOp: TTNN_ElementwiseUnaryOp<"gelu"> {
  let summary = "Eltwise GELU.";
  let description = [{
    Eltwise GELU operation.
  }];
}

def TTNN_IsFiniteOp: TTNN_ElementwiseUnaryOp<"isfinite"> {
    let summary = "Eltwise isfinite op.";
    let description = [{
      Eltwise isfinite operation.
    }];
}

def TTNN_LogicalNotOp: TTNN_ElementwiseUnaryOp<"logical_not"> {
    let summary = "Eltwise logical not op.";
    let description = [{
      Eltwise logical not operation.
    }];
}

def TTNN_BitwiseNotOp : TTNN_ElementwiseUnaryOp<"bitwise_not"> {
    let summary = "Eltwise bitwise NOT.";
    let description = [{
        Performs element-wise NOT of tensor `operand` and produces a `result` tensor.

        Example:
            // Bitwise operation with with integer tensors
            // %operand: [[1, 2], [3, 4]]
            %result = "ttnn.bitwise_not"(%operand) : (tensor<2x2xi32>) -> tensor<2x2xi32>
            // %result: [[-2, -3], [-4, -5]]
    }];
}

def TTNN_NegOp : TTNN_ElementwiseUnaryOp<"neg"> {
    let summary = "Eltwise negate.";
    let description = [{
      Eltwise negate operation.
    }];
}

def TTNN_TanOp: TTNN_ElementwiseUnaryOp<"tan"> {
    let summary = "Eltwise tan op.";
    let description = [{
      Eltwise tan operation.
    }];
}

def TTNN_AtanOp: TTNN_ElementwiseUnaryOp<"atan"> {
    let summary = "Eltwise arctangent op.";
    let description = [{
      Performs an elementwise arctangent (`atan`) operation on the input tensor.
      This operation computes the inverse tangent of each element, returning
      values in the range [-π/2, π/2]. Supports floating-point tensor types.

      Example:

      ```mlir
      %input = tensor<4xf32> {1.0, 0.5, 0.0, -1.0}
      %result = "ttir.atan"(%input) : (tensor<4xf32>) -> tensor<4xf32>
      ```

      Given the input `[1.0, 0.5, 0.0, -1.0]`, the result would be approximately:
      `[0.785, 0.464, 0.0, -0.785]` (values in radians).
    }];
}

def TTNN_TanhOp: TTNN_ElementwiseUnaryOp<"tanh"> {
    let summary = "Eltwise tanh op.";

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return
          wa::TTNNOperandsWorkaroundsFactory::createTanhOpOperandsWorkarounds();
      }
    }];

    let description = [{
      Eltwise tanh operation.
    }];
}

def TTNN_ReciprocalOp : TTNN_ElementwiseUnaryOp<"reciprocal"> {
    let summary = "Eltwise reciprocal.";
    let description = [{
      Eltwise reciprocal operation.
    }];
}

def TTNN_ReluOp : TTNN_ElementwiseUnaryOp<"relu"> {
    let summary = "Eltwise ReLU.";
    let description = [{
      Eltwise ReLU operation.
    }];
}

def TTNN_Relu6Op : TTNN_ElementwiseUnaryOp<"relu6"> {
    let summary = "Eltwise ReLU6.";
    let description = [{
      Eltwise ReLU6 operation.
    }];
}

def TTNN_SinOp : TTNN_ElementwiseUnaryOp<"sin"> {
    let summary = "Eltwise sine.";
    let description = [{
      Eltwise sine operation.
    }];
}

def TTNN_SqrtOp : TTNN_ElementwiseUnaryOp<"sqrt"> {
    let summary = "Eltwise sqrt.";
    let description = [{
      Eltwise sqrt operation.
    }];
}

def TTNN_RsqrtOp : TTNN_ElementwiseUnaryOp<"rsqrt"> {
    let summary = "Eltwise rsqrt.";
    let description = [{
      Eltwise rsqrt operation.
    }];
}

def TTNN_SigmoidOp : TTNN_ElementwiseUnaryOp<"sigmoid"> {
    let summary = "Eltwise sigmoid.";
    let description = [{
      Eltwise sigmoid operation.
    }];
}
def TTNN_HardsigmoidOp : TTNN_ElementwiseUnaryOp<"hardsigmoid"> {
  let summary = "Eltwise hardsigmoid.";
  let description = [{
    Eltwise hardsigmoid operation. Computes hardsigmoid(x) = max(0, min(1, (x + 3) / 6)).
  }];
}

def TTNN_SiluOp : TTNN_ElementwiseUnaryOp<"silu"> {
  let summary = "Eltwise SiLU.";
  let description = [{
    Eltwise SiLU (Swish) operation.
  }];
}

def TTNN_MishOp : TTNN_ElementwiseUnaryOp<"mish"> {
  let summary = "Eltwise Mish.";
  let description = [{
    Eltwise Mish operation.
  }];
}

def TTNN_LogOp : TTNN_ElementwiseUnaryOp<"log"> {
    let summary = "Eltwise logarithm.";
    let description = [{
      Eltwise logarithm operation.
    }];
}

def TTNN_Log1pOp: TTNN_ElementwiseUnaryOp<"log1p"> {
    let summary = "Eltwise log1p operation.";
    let description = [{
        Performs element-wise logarithm plus one operation on `operand` tensor and
        puts the result in the output tensor.

        Example:
          %a: [0.0, -0.999, 7.0, 6.38905621, 15.0]
          "ttnn.logp1"(%a, %out) -> %out: [0.0, -6.90776825, 2.07944155, 2.0, 2.77258873]
      }];
}

def TTNN_Expm1Op: TTNN_ElementwiseUnaryOp<"expm1"> {
  let description = [{
    Performs element-wise exponential minus one operation on `operand` tensor
    and stores the result in the output tensor.

    Example:
        %a: [[0, 1], [0, 0]]
        "ttnn.exmp1"(%a, %out) -> %out: [[0, 1.71828], [0, 0]]
  }];
}

class TTNN_ElementwiseUnaryWithFloatParameterOp<string mnemonic, list<Trait> traits = []> :
    TTNN_ElementwiseUnaryOp<mnemonic, [TTNN_MemoryConfigOpInterface] # traits> {
    let summary = "Eltwise unary op with the float parameter.";
    let description = [{
      Eltwise unary op with the float parameter.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         F32Attr:$parameter,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let builders =
    [
      OpBuilder<(ins "Value": $input, "::llvm::APFloat": $parameter),
      [{
        build($_builder, $_state, {input.getType()}, input, parameter, /* memory_config */ nullptr);
      }]>,
      OpBuilder<(ins "Type": $resultType, "Value": $input, "::llvm::APFloat": $parameter),
      [{
        build($_builder, $_state, resultType, input, parameter, /* memory_config */ nullptr);
      }]>
    ];
}

def TTNN_LeakyReluOp : TTNN_ElementwiseUnaryWithFloatParameterOp<"leaky_relu"> {
    let summary = "Eltwise leaky relu operation.";
    let description = [{
      The Leaky ReLU (Rectified Linear Unit) operation computes an element-wise
      activation function over its input tensor. It is defined as:

      y = x if x > 0
      y = parameter * x if x <= 0

      where `parameter` is a small, user-defined constant that determines the slope for
      negative inputs.

      Attributes:
      - `parameter` (float): The slope for negative values.

      Inputs:
      - `input` (Tensor): The input tensor to be activated.

      Outputs:
      - `output` (Tensor): The tensor after applying the Leaky ReLU activation.
    }];
}

def TTNN_AddOp : TTNN_ElementwiseBinaryOp<"add"> {

    let summary = "Eltwise add.";
    let description = [{
      Eltwise add operation.
    }];
}

def TTNN_DivideOp : TTNN_ElementwiseBinaryOp<"divide"> {
    let summary = "Eltwise divide.";
    let description = [{
      Eltwise divide operation.
    }];
}

def TTNN_MultiplyOp : TTNN_ElementwiseBinaryOp<"multiply"> {

    let summary = "Eltwise multiply.";
    let description = [{
      Eltwise multiply operation.
    }];
}

def TTNN_LogicalRightShiftOp: TTNN_ElementwiseBinaryOp<"logical_right_shift">{
  let summary = "Eltwise Logical Right Shift operation";
  let description = [{
    The `logical_right_shift` operation performs an elementwise logical right shift
    on the elements of the first tensor by the corresponding shift amounts in the
    second tensor.
  }];
  let hasVerifier = 1;
}

def TTNN_SubtractOp : TTNN_ElementwiseBinaryOp<"subtract"> {
    let summary = "Eltwise subtract.";
    let description = [{
      Eltwise subtract operation.
    }];
}

def TTNN_EqualOp : TTNN_ElementwiseBinaryOp<"eq"> {
    let summary = "Eltwise equal to.";
    let description = [{
      Eltwise equal to operation.
    }];
}

def TTNN_NotEqualOp : TTNN_ElementwiseBinaryOp<"ne"> {
    let summary = "Eltwise not equal to.";
    let description = [{
      Eltwise not equal to operation.
    }];
}

def TTNN_GreaterEqualOp : TTNN_ElementwiseBinaryOp<"ge"> {
    let summary = "Eltwise greater than or equal to.";
    let description = [{
      Eltwise greater than or equal to operation.
    }];
}

def TTNN_GreaterThanOp : TTNN_ElementwiseBinaryOp<"gt"> {
    let summary = "Eltwise greater than.";
    let description = [{
      Eltwise greater than operation.
    }];
}

def TTNN_LessEqualOp : TTNN_ElementwiseBinaryOp<"le"> {
    let summary = "Eltwise less than or equal to.";
    let description = [{
      Eltwise less than or equal to operation.
    }];
}

def TTNN_LessThanOp : TTNN_ElementwiseBinaryOp<"lt"> {
    let summary = "Eltwise less than.";
    let description = [{
      Eltwise less than operation.
    }];
}

def TTNN_LogicalAndOp : TTNN_ElementwiseBinaryOp<"logical_and"> {
    let summary = "Eltwise logical and.";
    let description = [{
      Eltwise logical and operation.
    }];
}

def TTNN_LogicalOrOp : TTNN_ElementwiseBinaryOp<"logical_or"> {
    let summary = "Eltwise logical or.";
    let description = [{
      Eltwise logical or operation.
    }];
}

def TTNN_LogicalXorOp : TTNN_ElementwiseBinaryOp<"logical_xor"> {
    let summary = "Eltwise logical xor.";
    let description = [{
      Eltwise logical xor operation.
    }];
}

def TTNN_BitwiseAndOp : TTNN_ElementwiseBinaryCompositeOp<"bitwise_and"> {
    let summary = "Eltwise bitwise AND.";
    let description = [{
        Performs element-wise bitwise AND of two tensors `lhs` and `rhs`
        and produces a `result` tensor.

        Example:
            // %lhs: [[1, 2], [3, 4]]
            // %rhs: [[5, 6], [7, 8]]
            %result = "ttnn.bitwise_and"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
            // %result: [[1, 2], [3, 0]]
    }];
}

def TTNN_LogicalLeftShiftOp: TTNN_ElementwiseBinaryCompositeOp<"logical_left_shift"> {
  let summary = "Eltwise Logical Left Shift operation";
  let description = [{
    The `logical_left_shift` operation performs an elementwise logical left shift
    on the elements of the first tensor by the corresponding shift amounts in the
    second tensor.
  }];
  let hasVerifier = 1;
}

def TTNN_BitwiseOrOp : TTNN_ElementwiseBinaryCompositeOp<"bitwise_or"> {
    let summary = "Eltwise bitwise OR.";
    let description = [{
        Performs element-wise bitwise OR of two tensors `lhs` and `rhs`
        and produces a `result` tensor.

        Example:
            // %lhs: [[1, 2], [3, 4]]
            // %rhs: [[5, 6], [7, 8]]
            %result = "ttnn.bitwise_or"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
            // %result: [[5, 6], [7, 12]]
    }];
}

def TTNN_BitwiseXorOp : TTNN_ElementwiseBinaryCompositeOp<"bitwise_xor"> {
    let summary = "Eltwise bitwise XOR.";
    let description = [{
        Performs element-wise bitwise XOR of two tensors `lhs` and `rhs`
        and produces a `result` tensor.

        Example:
          // %lhs: [[1, 2], [3, 4]]
          // %rhs: [[5, 6], [7, 8]]
          %result = "ttnn.bitwise_xor"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
          // %result: [[4, 4], [4, 12]]
    }];
}

def TTNN_MaximumOp :  TTNN_ElementwiseBinaryCompositeOp<"maximum"> {
    let summary = "Eltwise maximum OP.";
    let description = [{
      Calculates maximum of input tensors' values element-wise and stores result in output tensor.

      Example:
        %lhs: [[3, 2, 7], [1, 4, 4]]
        %rhs: [[1, 4, 2], [1, 2, 3]]
        "ttnn.maximum"(%lhs, %rhs, %out) -> %out: [[3, 4, 7], [1, 4, 4]]
    }];
}

def TTNN_MinimumOp :  TTNN_ElementwiseBinaryCompositeOp<"minimum"> {
    let summary = "Eltwise minimum OP.";
    let description = [{
      Calculates minimum of input tensors' values element-wise and stores result
      in output tensor.

      Example:
        %lhs: [[3, 2, 7], [1, 4, 4]]
        %rhs: [[1, 4, 2], [1, 2, 3]]
        "ttnn.minimum"(%lhs, %rhs, %out) -> %out: [[1, 2, 2], [1, 2, 3]]
    }];
}


def TTNN_RemainderOp : TTNN_ElementwiseBinaryCompositeOp<"remainder"> {
    let summary = "Eltwise remainder.";
    let description = [{
      Performs element-wise remainder of dividend lhs and divisor rhs tensors and produces a
      result tensor.

      Example:

      // %lhs: [17, -17, 17, -17]
      // %rhs: [3, 3, -3, -3]
      %result = "ttnn.remainder"(%lhs, %rhs) : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
      // %result: [2, -2, 2, -2]
    }];
}

def TTNN_PowTensorOp : TTNN_ElementwiseBinaryCompositeOp<"pow_tensor"> {
    let summary = "Eltwise power OP.";
    let description = [{
      Performs element-wise exponentiation of lhs tensor by rhs tensor and produces a
      result tensor. Tensors must be of same shape.

      Example:
      ```
        %result = "ttnn.pow_tensor"(%lhs, %rhs) : (tensor<6xf64>, tensor<6xf64>) -> tensor<6xf64>

        %lhs: [-2.0, -0.0, -36.0, 5.0, 3.0, 10000.0]
        %rhs: [2.0, 2.0, 1.1, 2.0, -1.0, 10.0]
        %result: [4.0, 0.0, -nan, 25.0, 0.333333343, inf]
      ```
    }];
}

def TTNN_PowScalarOp : TTNN_ElementwiseBinaryCompositeScalarOp<"pow_scalar"> {
  let summary = "Eltwise power OP.";
  let description = [{
    The `pow_scalar` operation performs an exponentiation of each element of an
    input tensor with a scalar exponent and returns the result.

    Example:
    ```mlir
    %result = ttnn.pow_scalar(%input) <{exponent = 2.0 : f32}> : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    // Input tensors:
    // %input: [2.0, 3.0, 4.0, 5.0]  // Bases
    // %exponent: 2.0  // Power
    // Output tensor: [4.0, 9.0, 16.0, 25.0]
    ```

    Restriction: TTNN API supports expoenent ≥ 0 only.
  }];

  let hasVerifier = 1;
}

def TTNN_Atan2Op :  TTNN_ElementwiseBinaryCompositeOp<"atan2"> {
    let summary = "Eltwise atan2 OP.";
    let description = [{
      Performs element-wise atan2 operation on lhs and rhs tensor and produces a result
      tensor.

      Example:
      ```
        // %lhs: [0.0, 1.0, -1.0]
        // %rhs: [1.0, 0.0, 0.0]
        %result = "ttnn.atan2"(%lhs, %rhs) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
        // %result: [0.0, 1.57079637, -1.57079637] // [0.0, pi/2, -pi/2]
      ```
    }];
}

def TTNN_GeluBackwardOp : TTNN_ElementwiseBinaryOp<"gelu_bw"> {
  let summary = "Backward pass operation for the GELU activation function.";
  let description = [{
    Computes the gradient of the GELU (Gaussian Error Linear Unit) activation
    function with respect to its input during backpropagation.

    This operation corresponds to ttnn.experimental.gelu_bw.
  }];

  let arguments = (ins AnyRankedTensor:$lhs,
                       AnyRankedTensor:$rhs,
                       OptionalAttr<TTCore_DataTypeAttr>:$dtype,
                       OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                       DefaultValuedAttr<StrAttr, "\"none\"">:$approximate);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

class TTNN_ReductionOp<string mnemonic, list<Trait> traits = []> : TTNN_Op<mnemonic, traits> {
  let summary = "Reduction op.";
  let description = [{
    Reduction op.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       BoolAttr:$keep_dim,
                       OptionalAttr<I32ArrayAttr>:$dim_arg,
                       OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config);
  let results = (outs AnyRankedTensor:$result);

  let builders = [
    OpBuilder<(ins "Type":$result, "Value":$input,
                   "bool":$keep_dim, "ArrayAttr":$dim_arg),
    [{
      build($_builder, $_state, result, input, keep_dim, dim_arg,
            /*compute_config=*/nullptr);
    }]>
  ];

  let extraClassDeclaration = [{
    wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
      return
        wa::TTNNOperandsWorkaroundsFactory::createReductionOpOperandsWorkarounds(*this);
    }
  }];

  let hasVerifier = 1;
}

def TTNN_SumOp : TTNN_ReductionOp<"sum", [TTNN_ComputeKernelConfigOpInterface]> {
    let summary = "Sum reduction op.";
    let description = [{
      Sum reduction op.
    }];
}

def TTNN_MeanOp : TTNN_ReductionOp<"mean", [TTNN_ComputeKernelConfigOpInterface]> {
  let summary = "Mean reduction op.";
  let description = [{
    Mean reduction op.
  }];
}

def TTNN_MaxOp : TTNN_ReductionOp<"max", [TTNN_ComputeKernelConfigOpInterface]> {
  let summary = "Max reduction op.";
  let description = [{
    Max reduction op.
  }];
}

def TTNN_MinOp : TTNN_ReductionOp<"min", [TTNN_ComputeKernelConfigOpInterface]> {
  let summary = "Min reduction op.";
  let description = [{
    This op computes the minimum of all elements of the tensor or along
    specified dimension.

    Example:
      input: [[1, 5, 3],
              [4, 2, 6]]

      // Computing along dim 0
      output: [1, 2, 3]

      // Computing along dim 1
      output: [1, 2]

      // Computing for entire tensor
      output: 1
  }];
}

def TTNN_ArgMaxOp : TTNN_Op<"argmax"> {
  let summary = "Argmax reduction op.";
  let description = [{
    Determine the indices of the maximum values along a specified dimension of a tensor or over all elements in a tensor.

    Parameters:
      - `input`: The input tensor.
      - `dim`: Specifies the dimension along which the argmax is applied.
      - `keep_dim`: If set to true, the output tensor will have the same number of dimensions as the input tensor.
      - `use_multicore`: Whether to use multiple cores or not.

    IR usage:
    // Input tensor of shape (128, 28, 28, 64)
    %input = ... : tensor<128x28x28x64xbf16>

    %empty = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<u32>, ....}> : -> tensor<128x28x28xi32>
    %4 = "ttnn.argmax"(%input, %empty) <{dim = 3 : i32, use_multicore = false}> : (tensor<128x28x28xbf16>, tensor<128x28x28xi32) -> tensor<128x28x28xi32>

    Example:
      input: [[1, 5, 3],
              [2, 4, 6]]

      // Computing along dim 0
      output: [1, 0, 1]

      // Computing along dim 1
      output: [1, 2]

      // Computing for entire tensor
      output: 5
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       OptionalAttr<I32Attr>:$dim,
                       DefaultValuedAttr<BoolAttr, "false">:$keep_dim,
                       BoolAttr:$use_multicore,
                       OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

  let extraClassDeclaration = [{
    wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
      return wa::TTNNOperandsWorkaroundsFactory::
          createArgMaxOpOperandsWorkarounds();
    }
  }];

  let results = (outs AnyRankedTensor:$result);
}

def TTNN_ProdOp : TTNN_Op<"prod"> {
  let summary = "Product reduction op.";
  let description = [{
    This op computes the product of all elements of the tensor (full product)
    or along a specific dimension.

    Example:
      input: [[1, 2, 3],
              [4, 5, 6]]

      // Computing along dim 0
      output: [4, 10, 18]

      // Computing along dim 1
      output: [6, 120]

      // Computing full product
      output: 720
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       OptionalAttr<I64Attr>:$dim_arg,
                       BoolAttr:$keep_dim,
                       OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

  let results = (outs AnyRankedTensor:$result);

  let extraClassDeclaration = [{
    wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
      return
        wa::TTNNOperandsWorkaroundsFactory::createReduceProdOpOperandsWorkarounds();
    }
  }];

  let hasVerifier = 1;
}

def TTNN_EmbeddingOp : TTNN_Op<"embedding"> {
    let summary = "Embedding op.";
    let description = [{
      Embedding operation.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$weight);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return wa::TTNNOperandsWorkaroundsFactory::createEmbeddingOpOperandsWorkarounds();
      }
    }];

    let hasVerifier = 1;
}

def TTNN_UpdateCacheOp : TTNN_InplaceOp<"update_cache"> {
  let summary = "Update static cache tensor.";
  let description = [{
      Updates the `cache` tensor in-place with values from `input` at `update_index` and `batch_offset`.
  }];

  let arguments = (ins Arg<AnyRankedTensor, "cache tensor", [MemWrite]>:$cache,
                       AnyRankedTensor:$input,
                       AnyRankedTensor:$update_index,
                       I32Attr:$batch_offset);

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        RankedTensorType updateIndexType = getUpdateIndex().getType();
        return wa::TTNNOperandsWorkaroundsFactory::createUpdateCacheOpOperandsWorkarounds(updateIndexType);
      }
    }];

  let hasVerifier = 1;
}

def TTNN_PagedUpdateCacheOp : TTNN_InplaceOp<"paged_update_cache"> {
  let summary = "Paged update cache op.";
  let description = [{
      Inputs:
        - `cache`: The cache tensor to be updated. This tensor is modified in place [max_num_blocks, num_heads, block_size, head_dim]
        - `input`: The input tensor containing new values. [1, num_users, num_heads (padded to 32), head_dim]
        - `update_index`: Indices specifying where to update the cache. [num_users]
        - `share_cache`: Whether the cache tensors share memory regions. Defaults to False.
        - `page_table`: The page table for managing memory regions during updates. [num_users, max_num_blocks_per_seq]
  }];

  let arguments = (ins Arg<AnyRankedTensor, "cache tensor", [MemWrite]>:$cache,
                       AnyRankedTensor:$input,
                       AnyRankedTensor:$update_index,
                       DefaultValuedAttr<BoolAttr, "false">:$share_cache,
                       Optional<AnyRankedTensor>:$page_table);

  let extraClassDeclaration = [{
    wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
      return wa::TTNNOperandsWorkaroundsFactory::createPagedUpdateCacheOpOperandsWorkarounds(this->getOperation());
    }
  }];

  let hasVerifier = 1;
}

def TTNN_FillCacheOp : TTNN_InplaceOp<"fill_cache"> {
  let summary = "Fill static cache tensor.";
  let description = [{
      Fills the `cache` tensor in-place with values from `input` at `batch_offset`.
  }];

  let arguments = (ins Arg<AnyRankedTensor, "cache tensor", [MemWrite]>:$cache,
                       AnyRankedTensor:$input,
                       I32Attr:$batch_offset);

  let hasVerifier = 1;
}

def TTNN_PagedFillCacheOp : TTNN_InplaceOp<"paged_fill_cache"> {
  let summary = "Paged fill cache op.";
  let description = [{
    Fills the `cache` tensor in-place with values from `input` at `batch_offset`.
  }];

  let arguments = (ins Arg<AnyRankedTensor, "cache tensor", [MemWrite]>:$cache,
                       AnyRankedTensor:$input,
                       AnyRankedTensor:$page_table,
                       Optional<AnyRankedTensor>:$batch_idx_tensor);

  let extraClassDeclaration = [{
    wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
      return wa::TTNNOperandsWorkaroundsFactory::createPagedFillCacheOpOperandsWorkarounds(getOperation());
    }
  }];

  let hasVerifier = 1;
}

def TTNN_EmbeddingBackwardOp : TTNN_Op<"embedding_bw", [TTNN_DtypeOpInterface, TTNN_MemoryConfigOpInterface]> {
    let summary = "Embedding backward op.";
    let description = [{
      Embedding backward operation. Generates the gradient of the embedding operation with respect to the input.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$weight,
                         AnyRankedTensor:$in_gradient,
                         OptionalAttr<TTCore_DataTypeAttr>:$dtype,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return wa::TTNNOperandsWorkaroundsFactory::createEmbeddingBackwardOpOperandsWorkarounds();
      }
    }];

    let hasVerifier = 1;
}

def TTNN_MorehCumSumOp : TTNN_Op<"moreh_cumsum"> {
  let summary = "Moreh cumulative sum op.";
  let description = [{
    Computes the cumulative sum of elements of a tensor along specified dimension.

    Example:
      input: [[1, 2, 3],
              [4, 5, 6]]

      // Cumulative sum along dim=0:
      output: [[1, 2, 3],
               [5, 7, 9]]

      // Cumulative sum along dim=1:
      output: [[1, 3, 6],
               [4, 9, 15]]
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       I64Attr:$dim,
                       OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

  let results = (outs AnyRankedTensor:$result);
}

def TTNN_ConcatenateHeadsOp: TTNN_Op<"concatenate_heads", [TTNN_MemoryConfigOpInterface]> {
    let summary = "Concatenate heads op used in attention layer.";
    let description = [{
      Takes in a tensor of shape [batch_size, num_heads, sequence_size, head_size],
      concatenates heads back along the width dimension and returns the tensor
      of shape [batch_size, sequence_size, num_heads * head_size].
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_NLPConcatHeadsDecodeOp: TTNN_Op<"nlp_concat_heads_decode", [TTNN_MemoryConfigOpInterface]> {
    let summary = "Concatenate heads op used in attention layer.";
    let description = [{
      Shuffles [S=1, B=32, 32(num_heads), head_dim] tensor into tensor with shape [S=1, 1, B=32, num_heads * head_dim].
      This operation assumes that input num_heads is padded to at most 32. When invoking this op,
      we specify the actual num_heads via the attribute `num_heads` and it should be less than input padded num_heads.
      Operation will unpad the input num_heads to the actual num_heads.
      The output is default width sharded by num heads.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         UI32Attr:$num_heads,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_SplitQueryKeyValueAndSplitHeadsOp: TTNN_Op<"split_query_key_value_and_split_heads", [TTNN_MemoryConfigOpInterface]> {
    let summary = "Split query, key, values and split heads op used in attention layer.";
    let description = [{
      Splits input_tensor of shape [batch_size, sequence_size, 3 * hidden_size] into 3 tensors (Query, Key, Value) of shape [batch_size, sequence_size, hidden_size]. Then, reshapes and permutes the output tensors, to make them ready for computing attention scores.
      If kv_input_tensor is passed in, then input_tensor of shape [batch_size, sequence_size, hidden_size] is only used for Query, and kv_input_tensor of shape [batch_size, sequence_size, 2 * hidden_size] is used for Key and Value.
      For the sharded implementation, the input query, key and value are expected to be concatenated such that the heads are interleaved (q1 k1 v1…qn kn vn).
    }];

    let arguments = (ins AnyRankedTensor:$input_tensor,
                         Optional<AnyRankedTensor>:$kv_input_tensor,
                         UI32Attr:$num_heads,
                         OptionalAttr<UI32Attr>:$num_kv_heads,
                         BoolAttr:$transpose_key,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let results = (outs AnyRankedTensor:$query,
                        AnyRankedTensor:$key,
                        AnyRankedTensor:$value);

    let hasVerifier = 1;
}

def TTNN_SoftmaxOp : TTNN_Op<"softmax", [TTNN_ComputeKernelConfigOpInterface]> {

    let summary = "Softmax op.";
    let description = [{
      Softmax operation.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         SI32Attr: $dimension,
                         DefaultValuedAttr<BoolAttr, "false">:$numericStable,
                         OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config);

    let results = (outs AnyRankedTensor:$result);

    let builders = [
      OpBuilder<(ins "Type":$result, "Value":$input,
                     "int32_t":$dimension, "bool":$numericStable),
      [{
        build($_builder, $_state, result, input, dimension, numericStable,
              /*compute_config=*/nullptr);
      }]>,
      OpBuilder<(ins "Type":$result, "Value":$input,
                     "int32_t":$dimension),
      [{
        build($_builder, $_state, result, input, dimension, /*numericStable=*/false,
              /*compute_config=*/nullptr);
      }]>
    ];

    let hasVerifier = 1;
}

def TTNN_SortOp : TTNN_Op<"sort"> {
  let summary = "Sort op.";
  let description = [{
    Sorts elements of a tensor along a given dimension.

    Input:
      - input: AnyRankedTensor

    Attributes:
      - dim (int8): The dimension to sort along (default: -1, the last dim).
      - descending (bool): If True, sort in descending order (default: False).
      - stable (bool): If True, ensures stable sort (equal elements keep order).

    Returns a tuple:
      - values: the sorted tensor.
      - indices: the original indices of the sorted values.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       DefaultValuedAttr<SI8Attr, "-1">:$dim,
                       DefaultValuedAttr<BoolAttr, "false">:$descending,
                       DefaultValuedAttr<BoolAttr, "false">:$stable,
                       OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

  let results = (outs AnyRankedTensor:$values,
                      AnyRankedTensor:$indices);

  let extraClassDeclaration = [{
    wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
      return
        wa::TTNNOperandsWorkaroundsFactory::createSortOpOperandsWorkarounds(*this);
    }
  }];

  let hasVerifier = 1;
}

def TTNN_TransposeOp : TTNN_Op<"transpose"> {
    let summary = "Transpose op.";
    let description = [{
      Transpose tensor along two given dimensions.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         SI32Attr:$dim0,
                         SI32Attr:$dim1);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_RepeatInterleaveOp : TTNN_Op<"repeat_interleave"> {
    let summary = "Repeat interleave op.";
    let description = [{
      Repeats elements of a tensor along a specified dimension.
      It allows for flexible repetition patterns, where each element can be repeated a different number of times.
      This is particularly useful for tasks that require duplicating elements in a non-uniform manner.

      Parameters:
      - `input`: The input tensor.
      - `repeats`: Specifies the number of repetitions for each element, each element is repeated that number of times.
      - `dim`: The dimension along which to repeat values.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         UI32Attr:$repeats,
                         SI32Attr:$dim,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_ConcatOp : TTNN_Op<"concat", [TTNN_MemoryConfigOpInterface]> {
    let summary = "Concat op.";
    let description = [{
      Concat tensors along a given dimension.
    }];

    let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                         SI32Attr:$dim,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_ReshapeOp : TTNN_Op<"reshape", [TTNN_MemoryConfigOpInterface]> {
    let summary = "Reshape op.";
    let description = [{
      Reshape tensor.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         I32ArrayAttr:$shape,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let hasFolder = 1;

    // Include workaround for ui8 reshape operation.
    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        RankedTensorType inputType = getInput().getType();
        return wa::TTNNOperandsWorkaroundsFactory::createReshapeOpOperandsWorkarounds(inputType);
      }
    }];
}

def TTNN_RepeatOp : TTNN_Op<"repeat", [TTNN_MemoryConfigOpInterface]> {
    let summary = "Repeat op.";
    let description = [{
      Returns a new tensor filled with repetition of input tensor according to number of times specified in repeat_dims.

      Parameters:
        - `input_tensor` (ttnn.Tensor): the input tensor.
        - `repeat_dims` (number): The number of repetitions for each element.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         TTNN_ShapeAttr:$repeat_dims,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let results = (outs AnyRankedTensor:$result);

    let builders =
    [
      OpBuilder<(ins "Type": $resultType, "Value": $input, "ttnn::ShapeAttr": $repeat_dims),
      [{
        build($_builder, $_state, resultType, input, repeat_dims, /*memory_config=*/nullptr);
      }]>
    ];

    let hasVerifier = 1;
}

def TTNN_PadOp: TTNN_Op<"pad"> {
  let summary = "Pad op.";
  let description = [{
    Pad input tensor by padding the input_shape to output_shape using the provided value.

    The `padding` attribute must be a sequence of integers that is twice the size as the rank of the input.
    Each pair of integers in the padding attribute represents the amount of padding to add to the low and high of that dimension.
    I.e: an input tensor of shape <1x30x30x64xf32> with padding attribute <0, 0, 1, 1, 1, 1, 0, 0> will return a tensor of shape <1x32x32x64xf32>,
    and so will a padding attribute of <0, 0, 0, 2, 0, 2, 0, 0>.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                        DenseI32ArrayAttr:$padding,
                        F32Attr:$value,
                        BoolAttr:$use_multicore,
                        OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
      mlir::TypedValue<mlir::RankedTensorType> input = getInput();
      ttnn::TTNNLayoutAttr layoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
            getResult().getType().getEncoding());
      llvm::ArrayRef<int32_t> padding = getPadding();
      return
        wa::TTNNOperandsWorkaroundsFactory::createPadOpOperandsWorkarounds(
                                                    input, layoutAttr, padding);
    }
  }];
}

def TTNN_SliceStaticOp: TTNN_Op<"slice_static"> {
    let summary = "Slice op.";
    let description = [{
      Extract a portion of a tensor based on the specified start (`begins`), stop (`ends`), and step
      indices for each dimension. The `begins` and `ends` parameters are attributes with fixed values.
      Maps to ttnn::slice.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         I32ArrayAttr:$begins,
                         I32ArrayAttr:$ends,
                         I32ArrayAttr:$step);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return wa::TTNNOperandsWorkaroundsFactory::
            createSliceStaticOpOperandsWorkarounds(*this);
      }
    }];

    let hasVerifier = 1;
}

def TTNN_SliceDynamicOp: TTNN_Op<"slice_dynamic"> {
    let summary = "Dynamic slice op.";
    let description = [{
      Extract a portion of a tensor based on the specified start (`begins`), stop (`ends`), and step
      indices for each dimension. Maps to ttnn::slice.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$begins,
                         AnyRankedTensor:$ends,
                         OptionalAttr<I32ArrayAttr>:$step);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return wa::TTNNOperandsWorkaroundsFactory::
            createSliceDynamicOpOperandsWorkarounds(*this);
      }
    }];

    let hasVerifier = 1;
}

def TTNN_LinearOp : TTNN_Op<"linear", [TTNN_ComputeKernelConfigOpInterface]> {
    let summary = "Linear transformation of inputs.";

    let description = [{
      Produces the matmul of tensors `a` and `b` with optional addition with `bias`.

      Example:
        // %a = [[1., 2.], [2., 1.]]
        // %b = [[0., 1.], [1., 0.]]
        // %bias = [[1.]]
        "ttnn.linear"(%a, %b, %bias, %result) : (tensor<2x2xf16>, tensor<2x2xf16>, tensor<1xf16>, tensor<2x2xf16>) -> tensor<2x2xf16>
        // %result = [[3., 2.], [2., 3.]]
    }];

    let arguments = (ins AnyRankedTensor:$a,
                         AnyRankedTensor:$b,
                         Optional<AnyRankedTensor>:$bias,
                         DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
                         DefaultValuedAttr<BoolAttr, "false">:$transpose_b,
                         OptionalAttr<AnyAttrOf<[
                            TTNN_MatmulMultiCoreReuseProgramConfigAttr,
                            TTNN_MatmulMultiCoreReuseMultiCastProgramConfigAttr,
                            TTNN_MatmulMultiCoreReuseMultiCast1DProgramConfigAttr,
                            TTNN_MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr
                         ]>>:$matmul_program_config,
                         OptionalAttr<StrAttr>:$activation,
                         OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config);

    let results = (outs AnyRankedTensor:$result);

    let builders = [
      OpBuilder<(ins "Type":$result, "Value":$a, "Value":$b,
                     "Value":$bias, "bool":$transpose_a, "bool":$transpose_b,
                     "StringAttr":$activation),
      [{
        build($_builder, $_state, result, a, b, bias, transpose_a, transpose_b,
              /*matmul_program_config=*/nullptr, activation, /*compute_config=*/nullptr);
      }]>
    ];

    let hasVerifier = 1;
}


// ANCHOR: adding_an_op_matmul_ttnn
def TTNN_MatmulOp : TTNN_Op<"matmul", [TTNN_ComputeKernelConfigOpInterface]> {
    let arguments = (ins AnyRankedTensor:$a,
                         AnyRankedTensor:$b,
                         DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
                         DefaultValuedAttr<BoolAttr, "false">:$transpose_b,
                         OptionalAttr<AnyAttrOf<[
                            TTNN_MatmulMultiCoreReuseProgramConfigAttr,
                            TTNN_MatmulMultiCoreReuseMultiCastProgramConfigAttr,
                            TTNN_MatmulMultiCoreReuseMultiCast1DProgramConfigAttr,
                            TTNN_MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr
                         ]>>:$matmul_program_config,
                         OptionalAttr<StrAttr>:$activation,
                         OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config);

    let results = (outs AnyRankedTensor:$result);

    let builders = [
      OpBuilder<(ins "Type":$result, "Value":$a, "Value":$b,
                     "bool":$transpose_a, "bool":$transpose_b,
                     "Attribute":$matmul_program_config,
                     "StringAttr":$activation),
      [{
        build($_builder, $_state, result, a, b, transpose_a, transpose_b,
              matmul_program_config, activation, /*compute_config=*/nullptr);
      }]>
    ];

    let hasVerifier = 1;
}
// ANCHOR_END: adding_an_op_matmul_ttnn

def TTNN_PrepareConv2dWeightsOp : TTNN_Op<"prepare_conv2d_weights", [TTNN_ComputeKernelConfigOpInterface]> {
    let summary = "Prepares conv2d weights so that they can be consumed by the conv2d op.";

    let arguments = (ins AnyRankedTensor:$weight_tensor,
                         TTNN_MemoryConfigAttr:$input_memory_config,
                         TTNN_LayoutAttr:$input_tensor_layout,
                         StrAttr:$weights_format,
                         I32Attr:$in_channels,
                         I32Attr:$out_channels,
                         I32Attr:$batch_size,
                         I32Attr:$input_height,
                         I32Attr:$input_width,
                         DenseI32ArrayAttr:$kernel_size,
                         DenseI32ArrayAttr:$stride,
                         DenseI32ArrayAttr:$padding,
                         DenseI32ArrayAttr:$dilation,
                         BoolAttr:$has_bias,
                         I32Attr:$groups,
                         TTNN_Device:$device,
                         TTCore_DataTypeAttr:$input_dtype,
                         OptionalAttr<TTCore_DataTypeAttr>:$output_dtype,
                         OptionalAttr<TTNN_Conv2dConfigAttr>:$conv2d_config,
                         OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config,
                         OptionalAttr<TTNN_Conv2dSliceConfigAttr>:$conv2d_slice_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_PrepareConv2dBiasOp : TTNN_Op<"prepare_conv2d_bias", [TTNN_ComputeKernelConfigOpInterface]> {
    let summary = "Prepares conv2d bias so that it can be consumed by the conv2d op.";

    let arguments = (ins AnyRankedTensor:$bias_tensor,
                         TTNN_MemoryConfigAttr:$input_memory_config,
                         TTNN_LayoutAttr:$input_tensor_layout,
                         I32Attr:$in_channels,
                         I32Attr:$out_channels,
                         I32Attr:$batch_size,
                         I32Attr:$input_height,
                         I32Attr:$input_width,
                         DenseI32ArrayAttr:$kernel_size,
                         DenseI32ArrayAttr:$stride,
                         DenseI32ArrayAttr:$padding,
                         DenseI32ArrayAttr:$dilation,
                         I32Attr:$groups,
                         TTNN_Device:$device,
                         TTCore_DataTypeAttr:$input_dtype,
                         OptionalAttr<TTCore_DataTypeAttr>:$output_dtype,
                         OptionalAttr<TTNN_Conv2dConfigAttr>:$conv2d_config,
                         OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config,
                         OptionalAttr<TTNN_Conv2dSliceConfigAttr>:$conv2d_slice_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_PrepareConvTranspose2dWeightsOp : TTNN_Op<"prepare_conv_transpose2d_weights", [TTNN_ComputeKernelConfigOpInterface]> {
    let summary = "Prepares conv_transpose2d weights so that they can be consumed by the conv_transpose2d op.";

    let arguments = (ins AnyRankedTensor:$weight_tensor,
                         TTNN_MemoryConfigAttr:$input_memory_config,
                         TTNN_LayoutAttr:$input_tensor_layout,
                         StrAttr:$weights_format,
                         I32Attr:$in_channels,
                         I32Attr:$out_channels,
                         I32Attr:$batch_size,
                         I32Attr:$input_height,
                         I32Attr:$input_width,
                         DenseI32ArrayAttr:$kernel_size,
                         DenseI32ArrayAttr:$stride,
                         DenseI32ArrayAttr:$padding,
                         DenseI32ArrayAttr:$dilation,
                         BoolAttr:$has_bias,
                         I32Attr:$groups,
                         TTNN_Device:$device,
                         TTCore_DataTypeAttr:$input_dtype,
                         OptionalAttr<TTCore_DataTypeAttr>:$output_dtype,
                         OptionalAttr<TTNN_Conv2dConfigAttr>:$conv2d_config,
                         OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config,
                         OptionalAttr<TTNN_Conv2dSliceConfigAttr>:$conv2d_slice_config,
                         DefaultValuedAttr<BoolAttr, "true">:$mirror_kernel);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_PrepareConvTranspose2dBiasOp : TTNN_Op<"prepare_conv_transpose2d_bias", [TTNN_ComputeKernelConfigOpInterface]> {
    let summary = "Prepares conv_transpose2d bias so that it can be consumed by the conv_transpose2d op.";

    let arguments = (ins AnyRankedTensor:$bias_tensor,
                         TTNN_MemoryConfigAttr:$input_memory_config,
                         TTNN_LayoutAttr:$input_tensor_layout,
                         I32Attr:$in_channels,
                         I32Attr:$out_channels,
                         I32Attr:$batch_size,
                         I32Attr:$input_height,
                         I32Attr:$input_width,
                         DenseI32ArrayAttr:$kernel_size,
                         DenseI32ArrayAttr:$stride,
                         DenseI32ArrayAttr:$padding,
                         DenseI32ArrayAttr:$dilation,
                         I32Attr:$groups,
                         TTNN_Device:$device,
                         TTCore_DataTypeAttr:$input_dtype,
                         OptionalAttr<TTCore_DataTypeAttr>:$output_dtype,
                         OptionalAttr<TTNN_Conv2dConfigAttr>:$conv2d_config,
                         OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config,
                         OptionalAttr<TTNN_Conv2dSliceConfigAttr>:$conv2d_slice_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_Conv2dOp : TTNN_Op<"conv2d", [TTNN_DtypeOpInterface, TTNN_ComputeKernelConfigOpInterface]> {
    let summary = "Conv2d operation.";
    let description = [{
      Applies a 2D convolution over an input image composed of several input planes.

      Inputs:
      - `input` (AnyRankedTensor): expected in the following flattened format (1, 1, N * H_in * W_in, C) where:
        - N is the batch size
        - H_in is the height of the input planes
        - W_in is the width of the input planes
        - C is the number of channels
      - `weight` (AnyRankedTensor): expected in the following format (O, C/G, K_H, K_W).
      - `bias` (Optional<AnyRankedTensor>): expected in the following format (1, 1, 1, O) where:
        - C is the number of input channels
        - O is the number of output channels
        - G is the number of groups
        - K_H is the height of the kernel
        - K_W is the width of the kernel

      Attributes:
      - `in_channels` (i32): The number of input channels.
      - `out_channels` (i32): The number of output channels.
      - `batch_size` (i32): The batch size.
      - `input_height` (i32): The input height.
      - `input_width` (i32): The input width.
      - `kernel_size` (array<2xi32>): [K_H, K_W] where K_H is the kernel height and K_W is the kernel width.
      - `stride` (array<2xi32>): [sH, sW] where sH is stride for height and sW is stride for width.
      - `padding` (array<2xi32> | array<4xi32>):
        - array<2xi32>: [pH, pW] where pH is padding for height (top/bottom) and pW is padding for width (left/right).
        - array<4xi32>: [pT, pB, pL, pR] for top, bottom, left, and right padding respectively.
      - `dilation` (array<2xi32>): [dH, dW] where dH is dilation for height and dW is dilation for width.
      - `groups` (i32): Number of blocked connections from input channels to output channels. Input and output channels must both be divisible by groups.

      Outputs:
      - `result` (AnyRankedTensor): returned in the following flattened format (1, 1, N * H_out * W_out, O) where:
        - `H_out = (H_in + pT + pB - dH * (K_H - 1) - 1) / sH + 1`
        - `W_out = (W_in + pL + pR - dW * (K_W - 1) - 1) / sW + 1`

      Example:
        %input = ttir.empty() : () -> tensor<1x1x1024x64xbf16>
        %weight = ttir.empty() : () -> tensor<64x64x3x3xbf16>
        %bias = ttir.empty() : () -> tensor<1x1x1x64xbf16>
        %device = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        %0 = "ttnn.conv2d"(%input, %weight, %bias, %device)
          <{
            in_channels = 64: i32,
            out_channels = 64: i32,
            batch_size = 1: i32,
            input_height = 32: i32,
            input_width = 32: i32,
            kernel_size = array<i32: 3, 3>,
            stride = array<i32: 1, 1>,
            padding = array<i32: 0, 0>,
            dilation = array<i32: 1, 1>,
            groups = 1: i32
          }> : (tensor<1x1x1024x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$weight,
                         Optional<AnyRankedTensor>:$bias,
                         TTNN_Device:$device,
                         I32Attr:$in_channels,
                         I32Attr:$out_channels,
                         I32Attr:$batch_size,
                         I32Attr:$input_height,
                         I32Attr:$input_width,
                         DenseI32ArrayAttr:$kernel_size,
                         DenseI32ArrayAttr:$stride,
                         DenseI32ArrayAttr:$padding,
                         DenseI32ArrayAttr:$dilation,
                         I32Attr:$groups,
                         OptionalAttr<TTCore_DataTypeAttr>:$dtype,
                         OptionalAttr<TTNN_Conv2dConfigAttr>:$conv2d_config,
                         OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config,
                         OptionalAttr<TTNN_Conv2dSliceConfigAttr>:$conv2d_slice_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      // Get number of output channels.
      int64_t getOutputChannelSize();
      bool isBiasCompatible(llvm::ArrayRef<int64_t> biasDims);
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return wa::TTNNOperandsWorkaroundsFactory::createConvOpOperandsWorkarounds(*this);
      }
    }];
}

def TTNN_Conv3dOp : TTNN_Op<"conv3d", [TTNN_DtypeOpInterface, TTNN_ComputeKernelConfigOpInterface]> {
    let summary = "Conv3d operation.";
    let description = [{
      Applies a 3D convolution over an input volume composed of several input planes.

      Inputs:
      - `input` (AnyRankedTensor): expected in the following format (N, D, H, W, C) where:
        - N is the batch size
        - D is the depth of the input volume
        - H is the height of the input planes
        - W is the width of the input planes
        - C is the number of input channels
      - `weight` (AnyRankedTensor): expected in the following format (K_D * K_H * K_W * C / G, O) where:
        - K_D is the depth of the kernel
        - K_H is the height of the kernel
        - K_W is the width of the kernel
        - C is the number of input channels
        - O is the number of output channels
        - G is the number of groups
        The spatial kernel dimensions and input channels are flattened together into a 2D tensor.
      - `bias` (Optional<AnyRankedTensor>): expected in the following format (1, O) where:
        - O is the number of output channels

      Attributes:
      - `in_channels` (i32): The number of input channels.
      - `out_channels` (i32): The number of output channels.
      - `batch_size` (i32): The batch size.
      - `input_depth` (i32): The input depth.
      - `input_height` (i32): The input height.
      - `input_width` (i32): The input width.
      - `kernel_size` (array<3xi32>): [K_D, K_H, K_W] where K_D is the kernel depth, K_H is the kernel height and K_W is the kernel width.
      - `stride` (array<3xi32>): [sD, sH, sW] where sD is stride for depth, sH for height and sW is stride for width.
      - `padding` (array<3xi32>): [pD, pH, pW] where pD is padding for depth, pH is padding for height and pW is padding for width.
        Padding is symmetric (same on both sides of each dimension).
      - `padding_mode` (StrAttr): "zeros" or "replicate" - padding fill strategy.
      - `groups` (i32): Number of blocked connections from input channels to output channels. Input and output channels must both be divisible by groups.

      Outputs:
      - `result` (AnyRankedTensor): returned in the following format (N, D_out, H_out, W_out, O) where:
        - `D_out = (D_in + 2*pD - K_D) / sD + 1`
        - `H_out = (H_in + 2*pH - K_H) / sH + 1`
        - `W_out = (W_in + 2*pW - K_W) / sW + 1`

      Example:
        %input = ttir.empty() : () -> tensor<1x28x28x28x32xbf16>
        %weight = ttir.empty() : () -> tensor<864x64xbf16>  // 864 = 3*3*3*32 (2D tensor)
        %bias = ttir.empty() : () -> tensor<1x64xbf16>      // 2D tensor
        %device = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        %0 = "ttnn.conv3d"(%input, %weight, %bias, %device)
          <{
            in_channels = 32: i32,
            out_channels = 64: i32,
            batch_size = 1: i32,
            input_depth = 28: i32,
            input_height = 28: i32,
            input_width = 28: i32,
            kernel_size = array<i32: 3, 3, 3>,
            stride = array<i32: 1, 1, 1>,
            padding = array<i32: 0, 0, 0>,
            padding_mode = "zeros",
            groups = 1: i32
          }> : (tensor<1x28x28x28x32xbf16>, tensor<864x64xbf16>, tensor<1x64xbf16>, !ttnn.device) -> tensor<1x26x26x26x64xbf16>
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$weight,
                         Optional<AnyRankedTensor>:$bias,
                         TTNN_Device:$device,
                         I32Attr:$in_channels,
                         I32Attr:$out_channels,
                         I32Attr:$batch_size,
                         I32Attr:$input_depth,
                         I32Attr:$input_height,
                         I32Attr:$input_width,
                         DenseI32ArrayAttr:$kernel_size,
                         DenseI32ArrayAttr:$stride,
                         DenseI32ArrayAttr:$padding,
                         DefaultValuedAttr<StrAttr, "\"zeros\"">:$padding_mode,
                         I32Attr:$groups,
                         OptionalAttr<TTCore_DataTypeAttr>:$dtype,
                         OptionalAttr<TTNN_Conv3dConfigAttr>:$conv3d_config,
                         OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return wa::TTNNOperandsWorkaroundsFactory::createConv3dOpOperandsWorkarounds(*this);
      }
    }];
}

def TTNN_ConvTranspose2dOp : TTNN_Op<"conv_transpose2d", [TTNN_DtypeOpInterface, TTNN_ComputeKernelConfigOpInterface]> {
    let summary = "ConvTranspose2d operation.";
    let description = [{
      Applies a 2D transposed convolution operator over an input image composed of several input planes.

      Inputs:
        - `input` AnyRankedTensor: expected in the following format (N, H_in, W_in, C) where:
          - N is the batch size
          - H_in is the height of the input planes
          - W_in is the width of the input planes
          - C is the number of channels

        - `weight` AnyRankedTensor: expected in the following format (C, O/G, K_H, K_W).
        - `bias` Optional<AnyRankedTensor>: expected in the following format (1, 1, 1, O) where:
          - C is the number of input channels
          - O is the number of output channels
          - G is the number of groups
          - K_H is the height of the kernel
          - K_W is the width of the kernel

        - `output` AnyRankedTensor: expected in the following format (N, H_out, W_out, O) where:
          - H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (K_H - 1) + output_padding[0] + 1
          - W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (K_W - 1) + output_padding[1] + 1

      Attributes:
        - `in_channels` i32: The number of input channels.
        - `out_channels` i32: The number of output channels.
        - `batch_size` i32: The batch size.
        - `input_height` i32: The input height.
        - `input_width` i32: The input width.
        - `kernel_size` array<2xi32>: The kernel size.
        - `stride` array<2xi32>: Controls the stride for the cross-correlation.
        - `padding` array<2xi32>: Controls the amount of implicit zero padding on both sides for dilation * (kernel_size - 1) - padding number of points.
        - `output_padding` array<2xi32>: Controls the additional size added to one side of the output shape.
        - `dilation` array<2xi32>: Controls the spacing between the kernel points
        - `groups` i32: Controls the connections between inputs and outputs. Must be divisible by input and output channels.

      Example:
        // %input: tensor<3x8x8x256xbf16>
        // %weight: tensor<256x256x3x3xbf16>
        // %bias: tensor<1x1x1x256xbf16>
        // %output: tensor<3x10x10x256xbf16>
        %0 = "ttnn.conv_transpose2d"(%input, %weight, %bias, %output, %device)
          <{
            batch_size = 3: i32,
            dilation = array<i32: 1, 1>,
            groups = 1: i32,
            in_channels = 256: i32,
            input_height = 8: i32,
            input_width = 8: i32,
            kernel_size = array<i32: 3, 3>,
            out_channels = 256: i32,
            output_padding = array<i32: 0, 0>,
            padding = array<i32: 0, 0>,
            stride = array<i32: 1, 1>
          }> : (tensor<3x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<3x10x10x256xbf16>) -> tensor<3x10x10x256xbf16>
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$weight,
                         Optional<AnyRankedTensor>:$bias,
                         TTNN_Device:$device,
                         I32Attr:$in_channels,
                         I32Attr:$out_channels,
                         I32Attr:$batch_size,
                         I32Attr:$input_height,
                         I32Attr:$input_width,
                         DenseI32ArrayAttr:$kernel_size,
                         DenseI32ArrayAttr:$stride,
                         DenseI32ArrayAttr:$padding,
                         DenseI32ArrayAttr:$output_padding,
                         DenseI32ArrayAttr:$dilation,
                         I32Attr:$groups,
                         OptionalAttr<TTCore_DataTypeAttr>:$dtype,
                         OptionalAttr<TTNN_Conv2dConfigAttr>:$conv2d_config,
                         OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                         OptionalAttr<TTNN_Conv2dSliceConfigAttr>:$conv2d_slice_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return wa::TTNNOperandsWorkaroundsFactory::createConvOpOperandsWorkarounds(*this);
      }
    }];
}

def TTNN_AvgPool2dOp : TTNN_Op<"avg_pool2d", [TTNN_MemoryConfigOpInterface]> {
  let summary = "Applies a 2D average pooling over an input signal composed of several input planes.";
  let description = [{
    It is a downsampling operation to reduce the spatial dimensions (height and width) of a input tensor by computing averages with in a window.

    Example:
      // 3x3 input tensor
      input: [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]
      kernel_height: 2
      kernel_width: 2
      stride_height: 1
      stride_width: 1
      dilation_height: 1
      dilation_width: 1
      output: [[3, 4],
               [6, 7]]
  }];

  let arguments = (ins AnyRankedTensor:$input,
                         SI32Attr:$batch_size,
                         SI32Attr:$input_height,
                         SI32Attr:$input_width,
                         SI32Attr:$channels,
                         DenseI32ArrayAttr:$kernel_size,
                         DenseI32ArrayAttr:$stride,
                         DenseI32ArrayAttr:$padding,
                         DenseI32ArrayAttr:$dilation,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                         OptionalAttr<TTNN_TensorMemoryLayoutAttr>:$applied_shard_scheme,
                         BoolAttr:$ceil_mode,
                         DefaultValuedAttr<BoolAttr, "true">:$reallocate_halo_output,
                         DefaultValuedAttr<BoolAttr, "true">:$count_include_pad,
                         OptionalAttr<BoolAttr>:$config_tensors_in_dram);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return wa::TTNNOperandsWorkaroundsFactory::createPool2DOpOperandsWorkarounds();
      }
    }];

    let hasVerifier = 1;
}

def TTNN_MaxPool2dOp : TTNN_Op<"max_pool2d", [TTNN_MemoryConfigOpInterface]> {
    let summary = "Applies a 2D max pooling over an input signal composed of several input planes.";
    let description = [{
      Applies a 2D max pooling over an input signal composed of several input planes.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         SI32Attr:$batch_size,
                         SI32Attr:$input_height,
                         SI32Attr:$input_width,
                         SI32Attr:$channels,
                         DenseI32ArrayAttr:$kernel_size,
                         DenseI32ArrayAttr:$stride,
                         DenseI32ArrayAttr:$padding,
                         DenseI32ArrayAttr:$dilation,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                         OptionalAttr<TTNN_TensorMemoryLayoutAttr>:$applied_shard_scheme,
                         BoolAttr:$ceil_mode,
                         DefaultValuedAttr<BoolAttr, "true">:$reallocate_halo_output,
                         OptionalAttr<BoolAttr>:$config_tensors_in_dram);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return wa::TTNNOperandsWorkaroundsFactory::createPool2DOpOperandsWorkarounds();
      }
    }];

    let hasVerifier = 1;
}

def TTNN_MaxPool2dWithIndicesOp : TTNN_Op<"max_pool2d_with_indices", [TTNN_MemoryConfigOpInterface]> {
    let summary = "Applies a 2D max pooling over an input signal composed of several input planes, returning both values and indices.";
    let description = [{
      Applies a 2D max pooling over an input signal composed of several input planes.
      Returns both the maximum values and the indices of where those values were found in the input tensor.
      The indices can be used for unpooling operations or gradient computation.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         SI32Attr:$batch_size,
                         SI32Attr:$input_height,
                         SI32Attr:$input_width,
                         SI32Attr:$channels,
                         DenseI32ArrayAttr:$kernel_size,
                         DenseI32ArrayAttr:$stride,
                         DenseI32ArrayAttr:$padding,
                         DenseI32ArrayAttr:$dilation,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                         OptionalAttr<TTNN_TensorMemoryLayoutAttr>:$applied_shard_scheme,
                         BoolAttr:$ceil_mode,
                         DefaultValuedAttr<BoolAttr, "true">:$reallocate_halo_output,
                         OptionalAttr<BoolAttr>:$config_tensors_in_dram);

    let results = (outs AnyRankedTensor:$result,
                        AnyRankedTensor:$result_indices);

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return wa::TTNNOperandsWorkaroundsFactory::createPool2DWithIndicesOpOperandsWorkarounds();
      }
    }];

    let hasVerifier = 1;
}

def TTNN_BatchNormInferenceOp : TTNN_Op<"batch_norm_inference", [AttrSizedOperandSegments, TTNN_ComputeKernelConfigOpInterface]> {
    let summary = "Batch normalization inference op.";
    let description = [{
      Batch normalization operation for inference over each channel on input tensor.
      Uses pre-computed mean and variance.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         Optional<AnyRankedTensor>:$running_mean,
                         Optional<AnyRankedTensor>:$running_var,
                         DefaultValuedAttr<F32Attr, "1e-05">:$epsilon,
                         Optional<AnyRankedTensor>:$weight,
                         Optional<AnyRankedTensor>:$bias,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                         OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config);

    let results = (outs AnyRankedTensor:$result);

    let builders = [
      OpBuilder<(ins "Type":$result, "Value":$input,
                     "Value":$running_mean, "Value":$running_var,
                     "llvm::APFloat":$epsilon,
                     "Value":$weight, "Value":$bias,
                     "MemoryConfigAttr":$memory_config),
      [{
        build($_builder, $_state, result, input, running_mean, running_var,
              epsilon, weight, bias, memory_config, /*compute_config=*/nullptr);
      }]>
    ];

    let hasVerifier = 1;
}

def TTNN_BatchNormTrainingOp : TTNN_Op<"batch_norm_training", [AttrSizedOperandSegments, TTNN_ComputeKernelConfigOpInterface]> {
    let summary = "Batch normalization training op.";
    let description = [{
      Batch normalization operation for training over each channel on input tensor.
      Computes batch statistics and updates running mean and variance.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         Arg<Optional<AnyRankedTensor>, "running mean", [MemWrite]>:$running_mean,
                         Arg<Optional<AnyRankedTensor>, "running variance", [MemWrite]>:$running_var,
                         DefaultValuedAttr<F32Attr, "1e-05">:$epsilon,
                         DefaultValuedAttr<F32Attr, "0.1">:$momentum,
                         Optional<AnyRankedTensor>:$weight,
                         Optional<AnyRankedTensor>:$bias,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                         OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config);

    let results = (outs AnyRankedTensor:$result);

    let builders = [
      OpBuilder<(ins "Type":$result, "Value":$input,
                     "Value":$running_mean, "Value":$running_var,
                     "llvm::APFloat":$epsilon, "llvm::APFloat":$momentum,
                     "Value":$weight, "Value":$bias,
                     "MemoryConfigAttr":$memory_config),
      [{
        build($_builder, $_state, result, input, running_mean, running_var,
              epsilon, momentum, weight, bias, memory_config, /*compute_config=*/nullptr);
      }]>
    ];

    let hasVerifier = 1;
}

def TTNN_RMSNormOp : TTNN_Op<"rms_norm", [AttrSizedOperandSegments, TTNN_MemoryConfigOpInterface, TTNN_ComputeKernelConfigOpInterface]> {
    let summary = "RMS normalization op.";
    let description = [{
      RMS (Root Mean Square) normalization operation over the input tensor.
      Normalizes the input by computing the root mean square of elements and
      dividing by that value, optionally scaling and shifting the result.

      This operation performs normalization over the last dimension of the input tensor,
      matching the TTNN runtime implementation.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         Optional<AnyRankedTensor>:$weight,
                         Optional<AnyRankedTensor>:$bias,
                         DefaultValuedAttr<F32Attr, "1e-12">:$epsilon,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                         OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config);

    let results = (outs AnyRankedTensor:$result);

    let builders = [
      OpBuilder<(ins "Type":$result, "Value":$input,
                     "Value":$weight, "Value":$bias,
                     "llvm::APFloat":$epsilon,
                     "MemoryConfigAttr":$memory_config),
      [{
        build($_builder, $_state, result, input, weight, bias,
              epsilon, memory_config, /*compute_config=*/nullptr);
      }]>
    ];

    let hasVerifier = 1;
}

def TTNN_DistributedRMSNormOp : TTNN_Op<"distributed_rms_norm",
    [AttrSizedOperandSegments, TTNN_MemoryConfigOpInterface, TTNN_ComputeKernelConfigOpInterface]> {
    let summary = "Distributed RMS normalization with all-gather op.";
    let description = [{
      Fused distributed RMS normalization operation across mesh devices.
      Computes local RMS statistics (the mean of squared values, E(x²)),
      all-gathers the statistics along the specified cluster_axis to obtain
      globally-correct values, then normalizes each element by
      x / sqrt(E(x²) + epsilon) and applies optional weight scaling locally
      on each device.

      Only statistics are communicated across devices — the input data itself
      is not all-gathered. Each device's output shape equals its input shape.

      Maps to ttnn::fused_rms_minimal at runtime.

      This operation requires the input tensor to be width-sharded across devices.

      Inputs:
        - input: Input tensor. Must be width-sharded in L1 with shape
            (1,1,32,M) where M is a multiple of 32. Tiled layout required.
        - weight: Optional gamma (scale) tensor applied after normalization.
            Must be in ROW_MAJOR layout with width equal to tile_width (32),
            i.e. reshaped from 1D (N,) to 2D (N/32, 32).
        - residual: Optional residual tensor to add to input before
            normalization (x + residual). Must have the same shard spec as
            input.
        - stats: Scratch tensor for intermediate RMS statistics exchanged
            across devices via all-gather. Shape (1,1,32,32), width-sharded
            on core (0,0) in L1. Dtype is Float32 when fp32_dest_acc_en is
            set, otherwise BFloat16.

      Attributes:
        - cluster_axis: Mesh dimension (0 or 1) along which to all-gather
            the RMS statistics across devices.
        - epsilon: Small constant added to the denominator for numerical
            stability. Defaults to 1e-12.
        - sub_device_id: Optional sub-device targeting for kernel placement.
        - memory_config: Output memory configuration. Typically matches the
            input's width-sharded shard spec.
        - num_links: Optional number of links for the all-gather
            communication.
        - topology: CCL topology for all-gather (Linear or Ring).
        - compute_config: Device compute kernel configuration. Controls math
            fidelity, fp32_dest_acc_en, and packer L1 accumulation.
        - program_config: LayerNormShardedMultiCoreProgramConfig derived from
            the input's shard spec (core grid, block_h, block_w).
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         Optional<AnyRankedTensor>:$weight,
                         Optional<AnyRankedTensor>:$residual,
                         Optional<AnyRankedTensor>:$stats,
                         TTNN_Device:$device,
                         UI32Attr:$cluster_axis,
                         DefaultValuedAttr<F32Attr, "1e-12">:$epsilon,
                         OptionalAttr<UI32Attr>:$sub_device_id,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                         OptionalAttr<UI32Attr>:$num_links,
                         OptionalAttr<TTCore_TopologyAttr>:$topology,
                         OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config,
                         OptionalAttr<TTNN_LayerNormShardedMultiCoreProgramConfigAttr>:$program_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_LayerNormOp : TTNN_Op<"layer_norm", [AttrSizedOperandSegments, TTNN_MemoryConfigOpInterface]> {
    let summary = "Layer normalization op.";
    let description = [{
      Performs layer normalization on the input tensor. This operation normalizes
      the input tensor by computing the mean and variance of elements across
      the specified dimensions, then normalizes by subtracting the mean and
      dividing by the standard deviation, optionally scaling and shifting the result.

      This operation performs normalization over the last dimension of the input tensor,
      matching the TTNN runtime implementation.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         Optional<AnyRankedTensor>:$weight,
                         Optional<AnyRankedTensor>:$bias,
                         DefaultValuedAttr<F32Attr, "1e-12">:$epsilon,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_ClampScalarOp : TTNN_Op<"clamp_scalar", [TTNN_MemoryConfigOpInterface]> {
    let summary = "Clamp op.";
    let description = [{
      Clamp tensor values to a specified range.

      Example:
        min: 2.000000+00
        input: [[0, 1, 2, 3, 4, 5, 6, 7]]
        max: 5.000000+00

        "ttnn.clamp_scalar"(%arg0) <{max = 2.000000e+00 : f32, min = 5.000000e+00 : f32}>
        -> %out = [[2, 2, 2, 3, 4, 5, 5, 5]]
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyAttrOf<[F32Attr, I32Attr]>:$min,
                         AnyAttrOf<[F32Attr, I32Attr]>:$max,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let builders =
    [
      OpBuilder<(ins "Type": $resultType, "Value": $input, "::llvm::APFloat": $min, "::llvm::APFloat": $max),
      [{
        auto minAttr = $_builder.getF32FloatAttr(min.convertToFloat());
        auto maxAttr = $_builder.getF32FloatAttr(max.convertToFloat());
        build($_builder, $_state, resultType, input, minAttr, maxAttr, /* memory_config */ nullptr);
      }]>,
      OpBuilder<(ins "Type": $resultType, "Value": $input, "::mlir::IntegerAttr": $min, "::mlir::IntegerAttr": $max),
      [{
        auto minAttr = $_builder.getI32IntegerAttr(min.getInt());
        auto maxAttr = $_builder.getI32IntegerAttr(max.getInt());
        build($_builder, $_state, resultType, input, minAttr, maxAttr, /* memory_config */ nullptr);
      }]>,
      OpBuilder<(ins "Type": $resultType, "Value": $input, "::mlir::Attribute": $min, "::mlir::Attribute": $max),
      [{
        build($_builder, $_state, resultType, input, min, max, /* memory_config */ nullptr);
      }]>
    ];
}

def TTNN_ClampTensorOp : TTNN_Op<"clamp_tensor", [TTNN_MemoryConfigOpInterface]> {
  let summary = "Clamp op.";
  let description = [{
    Clamp tensor values to a specified range using min/max as tensor.

    Example:
      min:   [[2, 2, 2, 3, 3, 3, 0, 0]]
      input: [[0, 1, 2, 3, 4, 5, 6, 7]]
      max:   [[5, 5, 5, 9, 9, 9, 6, 6]]

      "ttnn.clamp_tensor"(%input, %min, %max)
      %out:  [[2, 2, 2, 3, 4, 5, 6, 6]]
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$min,
                       AnyRankedTensor:$max,
                       OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

  let results = (outs AnyRankedTensor:$result);

  let builders =
    [
      OpBuilder<(ins "Type": $resultType, "Value": $input, "Value": $min, "Value": $max),
      [{
        build($_builder, $_state, resultType, input, min, max, /* memory_config */ nullptr);
      }]>
    ];
}

class TTNN_CreationOp<string mnemonic, list<Trait> traits = []> :
    TTNN_Op<mnemonic, [TTCore_CreationOpTrait, TTNN_TensorSpecInterface, TTNN_DeviceOperandInterface] # traits>;

def TTNN_EmptyOp : TTNN_CreationOp<"empty", [TTCore_NonCacheableTrait]> {
    let summary = "Empty op.";
    let description = [{
      Tensor empty operation
    }];

    let arguments = (ins TTNN_Device:$device,
                         TTNN_ShapeAttr:$shape,
                         TTCore_DataTypeAttr:$dtype,
                         TTNN_LayoutAttr:$layout,
                         TTNN_MemoryConfigAttr:$memory_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_ArangeOp : TTNN_CreationOp<"arange", [CanExecuteOnHostTrait]> {
  let summary = "Arange operation.";
  let description = [{
    Tensor arange operation.

    Produces a (1, 1, 1, N)-shaped tensor with values from `start` to `end` (exclusive) with a step size of `step`.

    Examples:
      %0 = "ttnn.arange"() {start = 0 : i64, end = 5 : i64 step = 1 : i64} : () -> tensor<1x1x1x5xi64>
      // %0: [[[[0, 1, 2, 3, 4]]]]

      %1 = "ttnn.arange"() {start = 0 : i64, end = 10 : i64, step = 2 : i64} : () -> tensor<1x1x1x5xf32>
      // %1: [[[[0.0, 2.0, 4.0, 6.0, 8.0]]]]
  }];

  let arguments = (ins Optional<TTNN_Device>:$device,
                       I64Attr:$start,
                       I64Attr:$end,
                       I64Attr:$step,
                       OptionalAttr<TTCore_DataTypeAttr>:$dtype,
                       OptionalAttr<TTNN_LayoutAttr>:$layout,
                       OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def TTNN_RandOp : TTNN_CreationOp<"rand", [TTCore_NonCacheableTrait]> {
  let summary = "Random number generation operation.";
  let description = [{
    Returns a tensor filled with random numbers drawn from a uniform distribution over given interval [low, high) [Default: [0, 1)].

    Example:
      %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
      %1 = "ttnn.rand"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, high = 1.000000e+00 : f32, layout = #ttnn.layout<tile>, low = 0.000000e+00 : f32, memory_config = #ttnn.memory_config<#dram, <interleaved>>, seed = 0 : ui32, size = [32 : i32, 32 : i32]}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout1>

    Attributes:
      - `size` (TTNN_ShapeAttr): The shape of the tensor to create.
      - `device` (TTNN_Device): The device where the trace was captured.
      - `dtype` (mlir:Type): Data type of the returned tensor.
      - `layout` (TTNN_LayoutAttr): The layout for the output tensor.
      - `memory_config` (TTNN_MemoryConfigAttr): The memory configuration for the output tensor.
      - `low` (Float): The lower bound of the range (inclusive) [Default: 0.0].
      - `high` (Float): The upper bound of the range (exclusive) [Default: 1.0].
      - `seed` (Integer): Value to initialize the random number generator for reproducible results [Default: 0].

    Outputs:
      - `result` (Tensor): The generated tensor containing the random values.
  }];

  let arguments = (ins TTNN_Device:$device,
                       TTNN_ShapeAttr:$size,
                       DefaultValuedAttr<F32Attr, "0.0">:$low,
                       DefaultValuedAttr<F32Attr, "1.0">:$high,
                       DefaultValuedAttr<UI32Attr, "0">:$seed,
                       DefaultValuedAttr<TTCore_DataTypeAttr, "mlir::tt::ttcore::DataType::BFloat16">:$dtype,
                       DefaultValuedAttr<TTNN_LayoutAttr, "mlir::tt::ttnn::Layout::Tile">:$layout,
                       TTNN_MemoryConfigAttr:$memory_config);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

def TTNN_DropoutOp : TTNN_Op<"dropout", [TTNN_MemoryConfigOpInterface, TTCore_NonCacheableTrait]> {
  let summary = "Dropout operation.";
  let description = [{
    Applies dropout to the input tensor element-wise.

    Example:
      %result = "ttnn.dropout"(%input) <{prob = 0.2 : f32, scale = 1.25 : f32, seed = 42 : ui32}> : (tensor<64x128xbf16>) -> tensor<64x128xbf16>

    Attributes:
      - `prob` (Float): Dropout probability. Elements are zeroed with this probability [Default: 0.0].
      - `scale` (Float): Scale factor applied to non-zeroed elements. Typically 1/(1-prob) [Default: 1.0].
      - `seed` (Integer): Seed for the random number generator [Default: 0].
      - `use_per_device_seed` (Bool): Whether to use a different seed per device [Default: true].
      - `memory_config` (MemoryConfig): Memory configuration for the output tensor.

    Inputs:
      - `input` (Tensor): The input tensor.

    Output:
      - `result` (Tensor): The output tensor with dropout applied.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       DefaultValuedAttr<F32Attr, "0.0">:$prob,
                       DefaultValuedAttr<F32Attr, "1.0">:$scale,
                       DefaultValuedAttr<UI32Attr, "0">:$seed,
                       DefaultValuedAttr<BoolAttr, "true">:$use_per_device_seed,
                       OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

  let results = (outs AnyRankedTensor:$result);

  let extraClassDeclaration = [{
    wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
      return wa::TTNNOperandsWorkaroundsFactory::createDropoutOpOperandsWorkarounds();
    }
  }];

  let hasVerifier = 1;
}

class TTNN_NamedFullOp<string mnemonic, list<Trait> traits = []> :
  TTNN_CreationOp<mnemonic, [CanExecuteOnHostTrait] # traits> {
  let arguments = (ins Optional<TTNN_Device>:$device,
                       TTNN_ShapeAttr:$shape,
                       OptionalAttr<TTCore_DataTypeAttr>:$dtype,
                       OptionalAttr<TTNN_LayoutAttr>:$layout,
                       OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

def TTNN_ZerosOp : TTNN_NamedFullOp<"zeros"> {
  let summary = "Creates a tensor filled with zeros.";
  let description = [{
    Tensor operation to create a tensor filled with zeros.

    Given a ShapeAttr `shape`, produces a tensor with the same shape, filled with zeros.

    Example:
      %0 = "ttnn.zeros"() <{shape = array<i32:64, 28, 28>}> : () -> tensor<64x28x28xbf16>
      // %0: [[[0, 0, 0, ..., 0], [0, 0, 0, ..., 0], ..., [0, 0, 0, ..., 0]]]
  }];
}

def TTNN_OnesOp : TTNN_NamedFullOp<"ones"> {
  let summary = "Creates a tensor filled with ones.";
  let description = [{
    Tensor operation to create a tensor filled with ones.

    Given a ShapeAttr `shape`, produces a tensor with the same shape, filled with ones.

    Example:
      %0 = "ttnn.ones"() <{shape = array<i32:64, 28, 28>}> : () -> tensor<64x28x28xbf16>
      // %0: [[[1, 1, 1, ..., 1], [1, 1, 1, ..., 1], ..., [1, 1, 1, ..., 1]]]
  }];
}

def TTNN_FullOp : TTNN_CreationOp<"full", [CanExecuteOnHostTrait]> {
  let summary = "Creates a tensor filled with the specified value";
  let description = [{
    Tensor operation to create a tensor filled with a specified value.

    Given a `shape` and a `fill_value`, produces a tensor with the shape, filled with the specified value.

    Example:
      %0 = "ttnn.full"() <{
        dtype = #ttcore.supportedDataTypes<u32>,
        fill_value = 7 : i32,
        layout = #ttnn.layout<tile>,
        shape = #ttnn.shape<64x128>
      }> : () -> tensor<64x128xui32>
      // %0: [[[7, 7, 7, ..., 7], [7, 7, 7, ..., 7], ..., [7, 7, 7, ..., 7]]]
  }];

  let arguments = (ins Optional<TTNN_Device>:$device,
                       TTNN_ShapeAttr:$shape,
                       AnyAttrOf<[F32Attr, I32Attr]>:$fill_value,
                       OptionalAttr<TTCore_DataTypeAttr>:$dtype,
                       OptionalAttr<TTNN_LayoutAttr>:$layout,
                       OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

  let results = (outs AnyRankedTensor:$result);

  let builders =
  [
    OpBuilder<(ins "Type": $resultType, "Attribute": $fillValue, "Value": $device)>
  ];

}

def TTNN_ConstantOp : TTNN_CreationOp<"constant", [AllShapesMatch<["value", "result"]>, CanExecuteOnHostTrait]> {
    let summary = "Constant op.";
    let description = [{
      Produces tensor filled with given constant value.

      Examples:
        %0 = "ttnn.constant"() {value = dense<[[3, 4, 2], [1, 7, 8]]> : tensor<2x3xui16>} : () -> tensor<2x3xui16>
        // %0: [[3, 4, 2], [1, 7, 8]]
        %1 = "ttnn.constant"() {value = dense<[0.2, 1.3]> : tensor<2xf32>} : () -> tensor<2xf32>
        // %1: [0.2, 1.3]
    }];

    let arguments = (ins Optional<TTNN_Device>:$device,
                         ElementsAttr:$value,
                         OptionalAttr<TTCore_DataTypeAttr>:$dtype,
                         OptionalAttr<TTNN_LayoutAttr>:$layout,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return wa::TTNNOperandsWorkaroundsFactory::createConstantOpOperandsWorkarounds();
      }
    }];

    let hasVerifier = 1;
}

def TTNN_AllocOp : TTNN_Op<"alloc", [TTCore_CreationOpTrait]> {
    let summary = "Alloc op.";
    let description = [{
      Tensor Alloc operation
    }];

    let arguments = (ins I64Attr:$address, I64Attr:$size, TTNN_BufferTypeAttr:$buffer_type);
    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_DeallocateOp : TTNN_MemFreeOp<"deallocate"> {
    let summary = "Deallocate op.";
    let description = [{
      Tensor Deallocate operation
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         DefaultValuedAttr<BoolAttr, "false">:$force);
}

def TTNN_ScatterOp : TTNN_Op<"scatter">{
    let summary = "Scatter op.";
    let description = [{
      Embeds the values of the source tensor into the input tensor at locations specified by the index tensor along the given dimension.

      Parameters:
        - `input` (ttnn.Tensor): The tensor being updated.
        - `index` (ttnn.Tensor): Indices where values will be written to.
        - `source` (ttnn.Tensor): The values to scatter into the input tensor.
        - `dim` (int32_t): The dimension along which to scatter.
        - `scatter_reduce_type` (Enum): The scatter reduce type to use (SUM, PROD, MIN, MAX, INVALID).
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$index,
                         AnyRankedTensor:$source,
                         I32Attr:$dim,
                         TTCore_ReduceTypeAttr:$scatter_reduce_type,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return wa::TTNNOperandsWorkaroundsFactory::createScatterOpOperandsWorkarounds(getOperation());
      }
    }];
}

def TTNN_AllGatherOp: TTNN_Op<"all_gather", [TTNN_MemoryConfigOpInterface]> {
    let summary = "All gather op.";
    let description = [{
        Tensor All Gather operation
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         SI32Attr:$all_gather_dim,
                         UI32Attr:$cluster_axis,
                         OptionalAttr<UI32Attr>:$sub_device_id,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                         OptionalAttr<UI32Attr>:$num_links,
                         OptionalAttr<TTCore_TopologyAttr>:$topology);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let hasFolder = 1;
}

def TTNN_ReduceScatterOp: TTNN_Op<"reduce_scatter", [TTNN_MemoryConfigOpInterface]> {
    let summary = "Reduce scatter op.";
    let description = [{
        Tensor Reduce Scatter operation
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         TTCore_ReduceTypeAttr:$reduce_type,
                         SI32Attr:$scatter_dim,
                         UI32Attr:$cluster_axis,
                         OptionalAttr<UI32Attr>:$sub_device_id,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                         OptionalAttr<UI32Attr>:$num_links,
                         OptionalAttr<TTCore_TopologyAttr>:$topology);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let hasFolder = 1;
}

def TTNN_AllReduceOp: TTNN_Op<"all_reduce", [TTNN_MemoryConfigOpInterface]> {
    let summary = "All reduce op.";
    let description = [{
        Tensor All Reduce operation
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         TTCore_ReduceTypeAttr:$reduce_type,
                         UI32Attr:$cluster_axis,
                         OptionalAttr<UI32Attr>:$sub_device_id,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                         OptionalAttr<UI32Attr>:$num_links,
                         OptionalAttr<TTCore_TopologyAttr>:$topology);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let hasFolder = 1;
}

def TTNN_MeshShardOp: TTNN_Op<"mesh_shard"> {
    let summary = "Mesh shard op.";
    let description = [{
        Tensor Mesh Shard operation
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         TTNN_Device:$device,
                         TTCore_MeshShardDirectionAttr:$shard_direction,
                         TTCore_MeshShardTypeAttr:$shard_type,
                         DenseI64ArrayAttr:$shard_shape,
                         DenseI64ArrayAttr:$shard_dims);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return wa::TTNNOperandsWorkaroundsFactory::createMeshShardOpOperandsWorkarounds(getShardType());
      }
    }];

    let hasVerifier = 1;
}

def TTNN_MeshPartitionOp: TTNN_Op<"mesh_partition", [TTNN_MemoryConfigOpInterface]> {
  let summary = "Mesh partition operation.";
  let description = [{
    Mesh partition op.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                       SI32Attr:$dim,
                       OptionalAttr<UI32Attr>:$cluster_axis,
                       OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
      return wa::TTNNOperandsWorkaroundsFactory::createMeshPartitionOpOperandsWorkarounds();
    }
  }];
}

def TTNN_PermuteOp : TTNN_Op<"permute"> {
    let summary = "Permute operation.";
    let description = [{
      Permute input tensor dimensions.

      Attributes:
        - `permutation` array<i64>: The permutation of the input tensor dimensions.

      Example:
      %a = ttir.empty() : () -> tensor<2x3x4xi32>
      %0 = "ttir.permute"(%a) {permutation = array<i64: 1, 2, 0>} : (tensor<2x3x4xi32>) -> tensor<3x4x2xi32>
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         DenseI64ArrayAttr:$permutation,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                         DefaultValuedOptionalAttr<F32Attr, "0.0f">:$pad_value);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return
          wa::TTNNOperandsWorkaroundsFactory::createPermuteOpOperandWorkaround(
            getInput().getType()
          );
      }
    }];

    let hasFolder = 1;
}

def TTNN_UpsampleOp : TTNN_Op<"upsample"> {
    let summary = "Upsample 2D op.";

    let description = [{
      Upsample 2D operation. Input tensor is assumed to be in NHWC format.

      Attributes:
      - `scale_factor` (si32 | array<i32>): The scale factor for upsampling in H and W dimensions respectively.
      - `mode` (str): The upsampling algorithm. Currently only "nearest" and "bilinear" are supported. Default is "nearest".

      Example:
        // %a: tensor<10x64x32xbf16>
        %0 = "ttnn.upsample"(%a) <{scale_factor = array<i32: 2, 4>}> : (tensor<10x64x32x3xbf16>) -> tensor<10x128x128x3xbf16>
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyAttrOf<[SI32Attr, DenseI32ArrayAttr]>:$scale_factor,
                         DefaultValuedAttr<StrAttr, "\"nearest\"">:$mode,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return wa::TTNNOperandsWorkaroundsFactory::createUpsampleOpOperandsWorkarounds();
      }
    }];

    let hasVerifier = 1;
}

def TTNN_QuantizeOp : TTNN_Op<"quantize"> {
    let summary = "Quantize operation.";
    let description = [{
      Applies quantization to the input tensor.

      Inputs:
        - `input` AnyRankedTensor: The input tensor to be quantized. Must have floating-point element type.
        - `scale` AnyRankedTensor: The scale factor (or factors for per-axis quantization). Must be either a scalar (for per-tensor quantization) or a 1D tensor with size matching the dimension of the specified axis (for per-axis quantization).
        - `zero_point` AnyRankedTensor: The zero point value (or values for per-axis quantization). Must be in range of the quantized storage type.
        - `axis` Optional<i32>: The axis along which quantization is applied. Must be in range [0, rank) where rank is the rank of the input tensor.
        - `output_dtype` Optional<TTCore_DataTypeAttr>: The data type of the output tensor.
        - `memory_config` Optional<TTNN_MemoryConfigAttr>: The memory configuration for the output tensor.

      ```
      // For per-tensor quantization:
      output[i] = round(input[i] / scale) + zero_point
      // For per-axis quantization:
      output[i0, i1, ..., ia, ..., in] = round(input[i0, i1, ..., ia, ..., in] / scale[ia]) + zero_point[ia]
      ```
      Example:
      ```mlir
      %input = ttir.empty() : () -> tensor<64x128xf32>
      %output = ttir.empty() : () -> tensor<64x128x!quant.uniform<i32:f32, 0.1>>
      %quantized = "ttir.quantize"(%input, %output) : (tensor<64x128xf32>, tensor<64x128x!quant.uniform<i32:f32, 0.1>>) -> tensor<64x128x!quant.uniform<i32:f32, 0.1>>
      ```
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$scale,
                         AnyRankedTensor:$zero_point,
                         OptionalAttr<I32Attr>:$axis,
                         OptionalAttr<TTCore_DataTypeAttr>:$output_dtype,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_DequantizeOp : TTNN_Op<"dequantize"> {
    let summary = "Dequantize operation.";
    let description = [{
      Applies dequantization to the input tensor.

      Inputs:
        - `input` AnyRankedTensor: The input tensor to be dequantized. Must have quantized element type.
        - `scale` AnyRankedTensor: The scale factor (or factors for per-axis quantization).
        - `zero_point` AnyRankedTensor: The zero point value (or values for per-axis quantization). Must be in range of the quantized storage type.
        - `axis` Optional<i32>: The axis along which quantization is applied. Must be in range [0, rank) where rank is the rank of the input tensor.
        - `output_dtype` Optional<TTCore_DataTypeAttr>: The data type of the output tensor.
        - `memory_config` Optional<TTNN_MemoryConfigAttr>: The memory configuration for the output tensor.
      ```
      // For per-tensor dequantization:
      output[i] = (input[i] - zero_point) * scale
      // For per-axis dequantization:
      output[i0, i1, ..., ia, ..., in] = (input[i0, i1, ..., ia, ..., in] - zero_point[ia]) * scale[ia]
      ```
      Example:
      ```mlir
      %input = ttir.empty() : () -> tensor<64x128x!quant.uniform<i32:f32, 0.1>>
      %output = ttir.empty() : () -> tensor<64x128xf32>
      %dequantized = "ttnn.dequantize"(%input, %output) : (tensor<64x128x!quant.uniform<i32:f32, 0.1>, tensor<64x128xf32>) -> tensor<64x128xf32>
      ```
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$scale,
                         AnyRankedTensor:$zero_point,
                         OptionalAttr<I32Attr>:$axis,
                         OptionalAttr<TTCore_DataTypeAttr>:$output_dtype,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_RequantizeOp : TTNN_Op<"requantize"> {
    let summary = "Requantize operation.";
    let description = [{
      Applies requantization to the input tensor.

      Inputs:
        - `input` AnyRankedTensor: The input tensor to be requantized. Must have quantized element type.
        - `in_scale` AnyRankedTensor: The input scale factor (or factors for per-axis quantization). Must be either a scalar (for per-tensor quantization) or a 1D tensor with size matching the dimension of the specified axis (for per-axis quantization).
        - `in_zero_point` AnyRankedTensor: The input zero point value (or values for per-axis quantization). Must be in range of the quantized storage type.
        - `out_scale` AnyRankedTensor: The output scale factor (or factors for per-axis quantization). Must be either a scalar (for per-tensor quantization) or a 1D tensor with size matching the dimension of the specified axis (for per-axis quantization).
        - `out_zero_point` AnyRankedTensor: The output zero point value (or values for per-axis quantization). Must be in range of the quantized storage type.
        - `axis` Optional<i32>: The axis along which quantization is applied. Must be in range [0, rank) where rank is the rank of the input tensor.
        - `output_dtype` Optional<TTCore_DataTypeAttr>: The data type of the output tensor.
        - `memory_config` Optional<TTNN_MemoryConfigAttr>: The memory configuration for the output tensor.
      ```
      // For per-tensor requantization:
      output[i] = round((input[i] - input_zero_point) * (input_scale / output_scale)) + output_zero_point
      // For per-axis requantization:
      output[i0, i1, ..., ia, ..., in] = round((input[i0, i1, ..., ia, ..., in] - in_zero_point[ia]) * (in_scale[ia] / out_scale[ia])) + out_zero_point[ia]
      ```
      Example:
      ```mlir
      %input = ttir.empty() : () -> tensor<64x128x!quant.uniform<i32:f32, 0.1>>
      %output = ttir.empty() : () -> tensor<64x128x!quant.uniform<i32:f32, 0.2>>
      %requantized = "ttnn.requantize"(%input, %output) : (tensor<64x128x!quant.uniform<i32:f32, 0.1>, tensor<64x128x!quant.uniform<i32:f32, 0.2>>) -> tensor<64x128x!quant.uniform<i32:f32, 0.2>>
      ```
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$in_scale,
                         AnyRankedTensor:$in_zero_point,
                         AnyRankedTensor:$out_scale,
                         AnyRankedTensor:$out_zero_point,
                         OptionalAttr<I32Attr>:$axis,
                         OptionalAttr<TTCore_DataTypeAttr>:$output_dtype,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_PointToPointOp: TTNN_Op<"point_to_point"> {
  let summary = "Point To Point operation.";
  let description = [{
    Performs point-to-point communication by copying a tensor shard from one device to another
    within a multi-device mesh. This operation is typically used for explicit data movement in
    distributed tensor computations, where a specific device (send_coord) sends its local tensor
    data to a target device (receive_coord).

    If `optional_output_tensor` is not provided, a new output tensor will be allocated automatically
    at the receiver. If provided, the data will be written into the specified output tensor.

    The operation returns a multi-device tensor whose buffer layout follows the mesh configuration.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                         DenseI64ArrayAttr:$sender_coord,
                         DenseI64ArrayAttr:$receiver_coord,
                         Arg<Optional<AnyRankedTensor>, "optional output tensor", [MemWrite]>:$optional_output_tensor);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def TTNN_WriteTensorOp : TTNN_InplaceOp<"write_tensor"> {
  let summary = "Write tensor op.";
  let description = [{
    Copies host_tensor data into device_tensor through cq_id.
    Memory copy is done in place, thus no output is returned.
    Inputs:
      - `host_tensor` AnyRankedTensor: The host tensor to copy.
      - `device_tensor` AnyRankedTensor: The device tensor to copy into.
      - `blocking` bool: Whether the copy should be executed synchronously.
      - `cq_id` i32: The command queue to copy the tensor with. Must be 0 or 1.
  }];

  let arguments = (ins AnyRankedTensor:$host_tensor,
                       Arg<AnyRankedTensor, "device tensor", [MemWrite]>:$device_tensor,
                       DefaultValuedAttr<BoolAttr, "false">:$blocking,
                       DefaultValuedAttr<UI32Attr, "0">:$cq_id);

  let hasVerifier = 1;
}

def TTNN_BeginTraceCaptureOp : TTNN_MemoryEffectOp<"begin_trace_capture"> {
  let summary = "Begin trace capture.";
  let description = [{
    Begins trace capture. Returns a scalar tensor containing the trace id.
    Inputs:
      - `device` TTNN_Device: The device to capture the trace on.
      - `cq_id` ui32: The command queue to capture the trace with. Must be 0 or 1.
    Outputs:
      - `trace_id` AnyRankedTensor: The scalar trace id tensor containing the trace id.
  }];

  let arguments = (ins TTNN_Device:$device,
                       DefaultValuedAttr<UI32Attr, "0">:$cq_id);

  let results = (outs AnyRankedTensor:$trace_id);

  let hasVerifier = 1;
}

def TTNN_EndTraceCaptureOp : TTNN_MemoryEffectOp<"end_trace_capture"> {
  let summary = "End trace capture.";
  let description = [{
    Ends trace capture for the given trace id. Consumes a scalar tensor containing the trace id.
    Has no output, but will have memory effects on the trace region of the device, modelled by
    trace resource in the compiler.
    Inputs:
      - `device` TTNN_Device: The device to end the trace capture on.
      - `trace_id` AnyRankedTensor: The trace id tensor to end the capture for. Must be a scalar.
      - `cq_id` ui32: The command queue to end the capture with. Must be 0 or 1.
  }];


  let arguments = (ins TTNN_Device:$device,
                       AnyRankedTensor:$trace_id,
                       DefaultValuedAttr<UI32Attr, "0">:$cq_id);

  let hasVerifier = 1;
}

def TTNN_ExecuteTraceOp : TTNN_MemoryEffectOp<"execute_trace"> {
  let summary = "Execute trace.";
  let description = [{
    Executes the captured trace. Consumes a scalar tensor containing the trace id.
    Has no output, but will have read/write memory effects on the cached trace input/output tensors
    created when capturing the trace.
    Inputs:
      - `device` TTNN_Device: The device where the trace was captured.
      - `trace_id` AnyRankedTensor: The trace id tensor to execute. Must be a scalar.
      - `cq_id` ui32: The command queue to execute the trace with. Must be 0 or 1.
      - `blocking` bool: Whether the trace should be executed synchronously.
  }];

  let arguments = (ins TTNN_Device:$device,
                       AnyRankedTensor:$trace_id,
                       DefaultValuedAttr<UI32Attr, "0">:$cq_id,
                       DefaultValuedAttr<BoolAttr, "false">:$blocking);

  let hasVerifier = 1;
}

def TTNN_CaptureOrExecuteTraceOp : TTNN_MemoryEffectOp<"capture_or_execute_trace"> {
  let summary = "Capture or execute trace.";
  let description = [{
    Captures or executes the trace. Will have read/write memory effects on the cached trace data.
    If the trace data exists (meaning the trace was captured previously), it will be executed with
    the execute_callee function. Otherwise, the trace will be captured with the capture_callee function.

    Inputs:
      - `device` TTNN_Device: The device where the trace was captured.
      - `capture_callee` FlatSymbolRefAttr: The symbol of the capture trace function.
      - `execute_callee` FlatSymbolRefAttr: The symbol of the execute trace function.
      - `inputs` Variadic<AnyRankedTensor>: The input tensors to the trace function.
    Outputs:
      - `results` Variadic<AnyRankedTensor>: The output tensors from the trace function.
  }];

  let arguments = (ins TTNN_Device:$device,
                       FlatSymbolRefAttr:$capture_callee,
                       FlatSymbolRefAttr:$execute_callee,
                       Variadic<AnyRankedTensor>:$inputs);

  let results = (outs Variadic<AnyRankedTensor>:$results);

  let hasVerifier = 1;
}

def TTNN_GenericOp : TTNN_Op<"generic", [TTNN_MemoryConfigOpInterface, MemoryEffects<[MemRead, MemWrite]>, TTCore_NonCacheableTrait]> {
    let summary = "Generic operation.";
    let description = [{
      Generic operation capable of running a program with custom kernels. Each kernel is described with a
      symbol reference to its function in EmitC dialect plus compile and runtime arguments. Generic operation
      is supplied with concatenated input and output `ios` tensors.

      Inputs:
        - `inputs_and_outputs` Variadic<AnyRankedTensor>: The input and output tensors.
        - `program` ProgramAttr: Program descriptor that includes a description of each kernels, array of CBs and array of semaphores.
    }];

    let arguments = (ins Variadic<AnyRankedTensor>:$inputs_and_outputs,
                         TTNN_GenericProgramAttr:$program,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let hasVerifier = 1;
}

def TTNN_D2MSubgraphOp : TTNN_MemoryEffectOp<"d2m_subgraph",
  [AttrSizedOperandSegments]> {
  let summary = "Dispatch D2M compiled subgraph.";
  let description = [{
    References a D2M-compiled subgraph function containing ttnn.generic ops.
    The function is a private function in the same module.

    Before TTNNMaterializeD2M runs, the referenced function contains a TTNN subgraph
    to be compiled via D2M.

    After TTNNMaterializeD2M runs, the referenced function contains:
    - ttnn.generic ops (the compiled subgraph)
    - Kernel functions are generated at module scope

    TTNNCollaspeD2M will inline the D2M function body at the call site.

    Example:
    ```mlir
    // Before D2M compilation
    %result = ttnn.d2m_subgraph @d2m_subgraph
        ins(%a : tensor<...>)
        outs(%out : tensor<...>) : tensor<...>

    func.func private @d2m_subgraph(...) -> tensor<...> {
      // ttnn subgraph
    }

    // After D2M compilation
    %result = ttnn.d2m_subgraph @d2m_subgraph
        ins(%a : tensor<...>)
        outs(%out : tensor<...>) : tensor<...>

    func.func private @d2m_subgraph(...) -> tensor<...> {
      ttnn.generic ...
      // more generic ops if subgraph didn't fully fuse
    }
    func.func private @kernel0() { ... }
    ... other kernel functions at module scope
    ```
  }];

  let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                       Variadic<AnyRankedTensor>:$outputs,
                       SymbolRefAttr:$d2m_func);

  let results = (outs Variadic<AnyRankedTensor>:$results);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $d2m_func attr-dict `\n`
    `ins` `(` $inputs `:` type($inputs) `)` `\n`
    `outs` `(` $outputs `:` type($outputs) `)` (`:` type($results)^)?
  }];

  let extraClassDeclaration = [{
    // Returns the D2M function referenced by this op (looks up in parent module).
    func::FuncOp getD2MMainFunc();
  }];
}

def TTNN_RotaryEmbeddingLlamaOp : TTNN_Op<"rotary_embedding_llama", [TTNN_MemoryConfigOpInterface, TTNN_ComputeKernelConfigOpInterface]> {
    let summary = "Rotary embedding llama operation.";
    let description = [{
      Applies rotary embedding to the input tensor using precomputed cosine and sine caches along with a transformation matrix.

      The operation supports both prefill and decode modes:
      - Prefill mode: Uses interleaved memory layout
      - Decode mode: Uses height-sharded memory layout

      Example:
      ```mlir
      %result = ttnn.rotary_embedding_llama(%input, %cos_cache, %sin_cache, %trans_mat)
        {is_decode_mode = false, memory_config = #ttnn.memory_config<interleaved>}
        : tensor<1x32x128xbf16>, tensor<1x32x128xbf16>, tensor<1x32x128xbf16>,
        tensor<1x1x32x32xbf16> -> tensor<1x32x128xbf16>
      ```
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$cos_cache,
                         AnyRankedTensor:$sin_cache,
                         AnyRankedTensor:$trans_mat,
                         DefaultValuedAttr<BoolAttr, "false">:$is_decode_mode,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                         OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_RotaryEmbeddingOp : TTNN_Op<"rotary_embedding", [TTNN_MemoryConfigOpInterface, TTNN_ComputeKernelConfigOpInterface]> {
  let summary = "Rotary embedding op in TTNN dialect.";
  let description = [{
    Applies rotary embedding to the input tensor using precomputed cosine and sine caches.
    Formula used:
      x_rotated = x * cos + rotate_half(x) * sin
      where rotate_half(x) swaps the first and second halves of the last dimension of x.

    Example:
      ```mlir
      %result = ttnn.rotary_embedding(%input, %cos_cache, %sin_cache)
        {memory_config = #ttnn.memory_config<interleaved>}
        : tensor<1x32x1024x64xf16>, tensor<1x1x1024x64xf16>, tensor<1x1x1024x64xf16>
        -> tensor<1x32x1024x64xf16>
      ```
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$cos_cache,
                       AnyRankedTensor:$sin_cache,
                       OptionalAttr<UI32Attr>:$token_index,
                       OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                       OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config);

  let results = (outs AnyRankedTensor:$result);

  let extraClassDeclaration = [{
    wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
      return wa::TTNNOperandsWorkaroundsFactory::createRotaryEmbeddingOpOperandsWorkarounds(*this);
    }
  }];

  let hasVerifier = 1;
}

def TTNN_NLPConcatHeadsOp : TTNN_Op<"nlp_concat_heads", [TTNN_MemoryConfigOpInterface]> {
  let summary = "nlp_concat_heads op in TTNN dialect.";
  let description = [{
    "This op targets specific case of concatenate heads operation where input tensor
    [B, num_heads, S, head_dim] is permuted and reshaped into [B, 1, S, num_heads * head_dim]."
  }];

  let arguments = (ins AnyRankedTensor:$input,
                   OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

def TTNN_DumpTensorOp : TTNN_InplaceOp<"dump_tensor"> {
  let summary = "Saves a tensor to disk in the TTNN binary format";
  let description = [{
    Saves a tensor to disk in the TTNN binary format. Files must use the `.tensorbin` extension.

    Inputs:
      - `file_path` StrAttr: Path of the file where tensor should be dumped. Must end with `.tensorbin` extension.
      - `input` AnyRankedTensor: Tensor to serialize.
  }];

  let arguments = (ins StrAttr:$file_path,
                       AnyRankedTensor:$input);

  let hasVerifier = 1;
}

def NLPCreateQKVHeadsDecodeOp : TTNN_Op<"nlp_create_qkv_heads_decode", [TTNN_MemoryConfigOpInterface]> {
  let summary = "nlp_create_qkv_heads_decode op in TTNN dialect.";
  let description = [{
    Shuffles [1, S=1, B, head_dim * (num_heads + 2*num_kv_heads)] fused qkv matrix into Q, K, and V heads with shape [S, B, num_heads, head_dim] for Q and [S, B, num_kv_heads, head_dim] for K and V, where num_heads and num_kv_heads will be padded to nearest 32.
      - Input must be sharded, B=32 and S=1.
      - overlap_qk_coregrid is a boolean flag that determines whether the output Q and K heads are on same core grid. If true, then Q, K, and V heads are on the same core grid. If false, the Q and K heads are on non-overlapping core-grid useful for processing Q and K in parallel.
      - Batch offset is used to fuse batch slicing. If provided slice size must also be provided in which batch dim of QKV output will be slice_size.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       Optional<AnyRankedTensor>:$batch_offset,
                       UI32Attr:$num_heads,
                       OptionalAttr<UI32Attr>:$num_kv_heads,
                       OptionalAttr<BoolAttr>:$overlap_qk_coregrid,
                       OptionalAttr<UI32Attr>:$slice_size,
                       OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

  let results = (outs AnyRankedTensor:$query,
                      AnyRankedTensor:$key,
                      AnyRankedTensor:$value);

  let hasVerifier = 1;
}

def TTNN_LoadTensorOp : TTNN_Op<"load_tensor"> {
  let summary = "Loads a tensor from disk";
  let description = [{
    Loads a tensor from disk, optionally placing it directly on a device.

    Inputs:
      - `file_path` StrAttr: Path of the file of the serialized tensor. Must end with `.tensorbin` extension.
      - `device` Optional<TTNN_Device>: Device where tensor should be deserialized. It has to be provided iff the serialized tensor is a device tensor.
    Outputs:
      - `result` AnyRankedTensor: Deserialized tensor from the `file_path`.
  }];

  let arguments = (ins StrAttr:$file_path,
                       Optional<TTNN_Device>:$device);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}


def TTNN_ScaledDotProductAttentionDecodeOp : TTNN_Op<"scaled_dot_product_attention_decode", [TTNN_MemoryConfigOpInterface, AttrSizedOperandSegments]> {
    let summary = "A version of scaled dot product attention specifically for decode.";
    let description = [{
      A version of scaled dot product attention specifically for decode.
        The implementation is Flash-Decode and it currently only supports MQA on decoding single token.

        Args:
            input_tensor_q (AnyRankedTensor): The input tensor [1 x batch x num_heads x head_size]. Note that there is no sequence length dimension as this op is intended for processing a single query token.
            input_tensor_k (AnyRankedTensor): The input tensor [batch x num_kv_heads x   seq_len x head_size].
            input_tensor_v (AnyRankedTensor): The input tensor [b x num_kv_heads x   seq_len x head_size].
            is_causal (bool, optional): Whether the attention is causal. Defaults to `true`.
            attention_mask (AnyRankedTensor, optional): The attention mask [batch x 1 x query_seq_len x kv_seq_len].
            cur_pos_tensor (AnyRankedTensor): [batch] Tensor of integers of length batch.
            attention_sink (AnyRankedTensor, optional): The attention sink [num_heads, 32] (must be a single tile wide).
            scale (float, optional): Defaults to `None`.
            memory_config (MemoryConfigAttr, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            AnyRankedTensor: The output tensor [1 x b x pnh x dh].
    }];

    let arguments = (ins AnyRankedTensor:$query,
                         AnyRankedTensor:$key,
                         AnyRankedTensor:$value,
                         DefaultValuedAttr<BoolAttr, "true">:$is_causal,
                         Optional<AnyRankedTensor>:$attention_mask,
                         Optional<AnyRankedTensor>:$cur_pos_tensor,
                         Optional<AnyRankedTensor>:$attention_sink,
                         OptionalAttr<F32Attr>:$scale,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                         OptionalAttr<TTNN_SDPAProgramConfigAttr>:$program_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_PagedScaledDotProductAttentionDecodeOp : TTNN_Op<"paged_scaled_dot_product_attention_decode", [TTNN_MemoryConfigOpInterface, AttrSizedOperandSegments]> {
  let summary = "Paged scaled dot product attention decode operation.";
  let description = [{
    Paged scaled dot product attention decode operation.
  }];

  let arguments = (ins AnyRankedTensor:$query,
                       AnyRankedTensor:$key,
                       AnyRankedTensor:$value,
                       AnyRankedTensor:$page_table,
                       DefaultValuedAttr<BoolAttr, "true">:$is_causal,
                       Optional<AnyRankedTensor>:$attention_mask,
                       Optional<AnyRankedTensor>:$cur_pos_tensor,
                       Optional<AnyRankedTensor>:$attention_sink,
                       OptionalAttr<F32Attr>:$scale,
                       OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

  let results = (outs AnyRankedTensor:$result);

  let extraClassDeclaration = [{
    wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
      return wa::TTNNOperandsWorkaroundsFactory::createPagedScaledDotProductAttentionDecodeOpOperandsWorkarounds(getOperation());
    }
  }];

  let hasVerifier = 1;
}

def TTNN_ScaledDotProductAttentionOp : TTNN_Op<"scaled_dot_product_attention", [TTNN_MemoryConfigOpInterface]> {
    let summary = "Scaled dot product attention operation.";
    let description = [{
      Scaled dot product attention.
      The implementation is FlashAttention-2."

      Args:
          query (AnyRankedTensor): The query tensor.          [batch x num_heads x query_seq_len x head_size]
          key (AnyRankedTensor): The key tensor.              [batch x num_kv_heads x kv_seq_len x head_size]
          value (AnyRankedTensor): The value tensor.          [batch x num_kv_heads x kv_seq_len x head_size]
          attention_mask (AnyRankedTensor, optional): Defaults to `None`. [batch x 1 x query_seq_len x kv_seq_len]. Head broadcasting is implied.
          is_causal (bool): Whether the attention is causal. Defaults to `true`.
          scale (float, optional): Defaults to `None`.
          sliding_window_size (uint, optional): Defaults to `None`. Size of sliding window for attention. If provided && is_causal, only attends to the last `sliding_window_size` tokens. If provided && !is_causal, attends to a window of size `sliding_window_size` centered at the current position.
          memory_config (MemoryConfigAttr, optional): Memory configuration for the operation. Defaults to `None`.


      Returns:
          AnyRankedTensor: The output tensor [batch x num_heads x query_seq_len x head_size].
    }];

    let arguments = (ins AnyRankedTensor:$query,
                         AnyRankedTensor:$key,
                         AnyRankedTensor:$value,
                         Optional<AnyRankedTensor>:$attention_mask,
                         DefaultValuedAttr<BoolAttr, "true">:$is_causal,
                         OptionalAttr<F32Attr>:$scale,
                         OptionalAttr<UI32Attr>:$sliding_window_size,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}


def TTNN_GlobalAvgPool2dOp: TTNN_Op<"global_avg_pool2d", [
    TTNN_MemoryConfigOpInterface,
    TTNN_DtypeOpInterface
]>{
  let summary = "A global average pooling 2d operation";
  let description = [{
    The `global_avg_pool2d` operation applies global average pooling over the spatial dimensions
    (height and width) of a 4D input tensor. In essence, it should be realised as the sum-reduce style operation
    under the hood, for performance reasons (since we include all elements, there is no need for kernel allocation).
    It reduces spatial dimensions to 1.

    Example:
    ```mlir
    %device = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    %result = "ttnn.global_avg_pool2d"(%input) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>,
                                                           dtype = #ttcore.supportedDataTypes<bf16>}>
              : (tensor<1x128x128x32xbf16>) -> tensor<1x1x1x32xbf16>
    ```

    Inputs:
    - `input`: 4D tensor with shape [N, H, W, C] where N is batch size, H is height, W is width, and C is channels

    Attributes:
    - `memory_config` (optional): Memory configuration specifying where the operation should be performed
    - `dtype` (optional): Data type for the output tensor (e.g., bf16, f32)

    Outputs:
    - `result`: 4D tensor with shape [N, 1, 1, C] containing the global average pooled values

    Note: The operation reduces spatial dimensions (H, W) to (1, 1) by computing the average across
    all spatial locations for each channel independently.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                       OptionalAttr<TTCore_DataTypeAttr>:$dtype);

  let results = (outs AnyRankedTensor:$result);

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$input),
    [{
      build($_builder, $_state, resultType, input,
            /*memory_config=*/nullptr, /*dtype=*/nullptr);
    }]>
  ];

  let hasVerifier = 1;
}

def TTNN_AssignOp : TTNN_Op<"assign", [TTNN_MemoryConfigOpInterface, TTNN_DtypeOpInterface]> {
    let summary = "Assign Tensor";
    let description = [{
      Returns a new tensor which is a new copy of input tensor.
      Alternatively, copies input tensor ``input`` to ``optional_output_tensor``
      if their shapes and memory layouts match, and returns input_b tensor.
      Input tensors can be of any data type.
      Output tensor will be of same data type as Input tensor.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         TTNN_MemoryConfigAttr:$memory_config,
                         OptionalAttr<TTCore_DataTypeAttr>:$dtype);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_DistributeTensorOp : TTNN_Op<"distribute_tensor"> {
  let summary = "Distribute tensor across mesh devices";
  let description = [{
    Distributes a host-side tensor across multiple devices in a mesh
    according to the specified mapping configuration.

    This operation takes a single-device tensor and creates a multi-device
    tensor distributed according to the MeshMapperConfig configuration.

    // Note: In the ttnn API, the distribute_tensor op takes an input tensor and a TensorToMesh object as its arguments.
    // To create a TensorToMesh object, the API create_mesh_mapper is used, which requires both MeshMapperConfig and MeshDevice.
    // Instead of modeling create_mesh_mapper as a separate op in the IR, these two strongly coupled APIs are combined into one distribute_tensor op
    // to keep the IR simple and less cluttered. If a clear need arises in the future, splitting them into separate ops can be considered.

    Example:
    ```mlir
    %result = "ttnn.distribute_tensor"(%input, %device) <{
      mapper_config = #ttnn.mesh_mapper_config<
        placements = [#ttnn.placement<replicate>, #ttnn.placement<shard, 1>],
        mesh_shape_override = #ttnn.mesh_shape<2x2>
      >,
      cq_id = 0 : ui32
    }> : (tensor<32x64xbf16>, !ttnn.device) -> tensor<32x64xbf16>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    TTNN_MeshMapperConfigAttr:$mapper_config,
    TTNN_Device:$mesh_device,
    OptionalAttr<UI32Attr>:$cq_id
  );

  let results = (outs AnyRankedTensor:$result); // Distributed tensor
}

def TTNN_AggregateTensorOp : TTNN_Op<"aggregate_tensor"> {
  let summary = "Aggregate distributed tensor back to host";
  let description = [{
    Aggregates a multi-device tensor back into a single host-side tensor
    according to the specified composer configuration.

    This operation takes a distributed tensor and creates a single-device
    host tensor aggregated according to the MeshComposerConfig configuration.

    // Note: In the ttnn API, the aggregate_tensor op takes an input tensor and a MeshToTensor object as its arguments.
    // To create a MeshToTensor object, the API create_mesh_composer is used, which requires both MeshComposerConfig and MeshDevice.
    // Instead of modeling create_mesh_composer as a separate op in the IR, these two strongly coupled APIs are combined into one aggregate_tensor op
    // to keep the IR simple and less cluttered. If a clear need arises in the future, splitting them into separate ops can be considered.

    Example:
    ```mlir
    %result = "ttnn.aggregate_tensor"(%input, %device) <{
      composer_config = #ttnn.mesh_composer_config<
        dims = [1],
        mesh_shape_override = #ttnn.mesh_shape<2x2>
      >
    }> : (tensor<32x64xbf16>, !ttnn.device) -> tensor<32x64xbf16>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    TTNN_MeshComposerConfigAttr:$composer_config,
    TTNN_Device:$mesh_device
  );

  let results = (outs AnyRankedTensor:$result);
}

def TTNN_TopKOp : TTNN_Op<"topk", [TTNN_MemoryConfigOpInterface]> {
  let summary = "Top-K selection operation.";
  let description = [{
    Returns the `k` largest or `k` smallest elements of the `input_tensor` along a given dimension `dim`.
    If `dim` is not provided, the last dimension of the input_tensor is used.
    If `largest` is True, the `k` largest elements are returned. Otherwise, the `k` smallest elements are returned.
    The boolean option `sorted` if True, will make sure that the returned `k` elements are sorted.
    the `memory_config` parameter is optional which represents the memory configuration for the operation.
  }];

  let arguments = (ins AnyRankedTensor:$input_tensor,
                       I32Attr:$k,
                       DefaultValuedAttr<I32Attr, "-1">:$dim,
                       DefaultValuedAttr<BoolAttr, "true">:$largest,
                       DefaultValuedAttr<BoolAttr, "false">:$sorted,
                       OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

  let results = (outs AnyRankedTensor:$values,
                      AnyRankedTensor:$indices);

  let hasVerifier = 1;
}
#endif
