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
                         TTCore_DataTypeAttr:$dtype);
    let results = (outs AnyRankedTensor:$result);

    let hasFolder = 1;
    let hasCanonicalizeMethod = 1;
    let hasVerifier = 1;
}

def TTNN_ToDTypeOp : TTNN_Op<"to_dtype"> {
    let summary = "ToDType op.";
    let description = [{
      This op converts the data type of the input tensor based on the given data type on the host.

      Args:
        - :attr:`input`: the ttnn.Tensor
        - :attr:`dtype`: `ttnn` data type.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         TTCore_DataTypeAttr:$dtype);
    let results = (outs AnyRankedTensor:$result);

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

def TTNN_PowOp : TTNN_ElementwiseBinaryCompositeOp<"pow"> {
    let summary = "Eltwise power OP.";
    let description = [{
      Performs element-wise exponentiation of lhs tensor by rhs tensor and produces a
      result tensor. Tensors must be of same shape.

      Example:
      ```
        %result = "ttnn.pow"(%lhs, %rhs) : (tensor<6xf64>, tensor<6xf64>) -> tensor<6xf64>

        %lhs: [-2.0, -0.0, -36.0, 5.0, 3.0, 10000.0]
        %rhs: [2.0, 2.0, 1.1, 2.0, -1.0, 10.0]
        %result: [4.0, 0.0, -nan, 25.0, 0.333333343, inf]
      ```
    }];
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

class TTNN_ReductionOp<string mnemonic, list<Trait> traits = []> : TTNN_Op<mnemonic, traits> {
  let summary = "Reduction op.";
  let description = [{
    Reduction op.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       BoolAttr:$keep_dim,
                       OptionalAttr<I32ArrayAttr>:$dim_arg);

  let results = (outs AnyRankedTensor:$result);

  let extraClassDeclaration = [{
    wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
      return
        wa::TTNNOperandsWorkaroundsFactory::createReductionOpOperandsWorkarounds(*this);
    }
  }];

  let hasVerifier = 1;
}

def TTNN_SumOp : TTNN_ReductionOp<"sum"> {
    let summary = "Sum reduction op.";
    let description = [{
      Sum reduction op.
    }];
}

def TTNN_MeanOp : TTNN_ReductionOp<"mean"> {
  let summary = "Mean reduction op.";
  let description = [{
    Mean reduction op.
  }];
}

def TTNN_MaxOp : TTNN_ReductionOp<"max"> {
  let summary = "Max reduction op.";
  let description = [{
    Max reduction op.
  }];
}

def TTNN_MinOp : TTNN_ReductionOp<"min"> {
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
      mlir::Type elementType = getInput().getType().getElementType();
      bool allDimensions = !getDimArg();
      return
        wa::TTNNOperandsWorkaroundsFactory::createReduceProdOpOperandsWorkarounds(
                                                    elementType, allDimensions);
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
  let summary = "Moreh cummulative sum op.";
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

def TTNN_SoftmaxOp : TTNN_Op<"softmax"> {

    let summary = "Softmax op.";
    let description = [{
      Softmax operation.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         SI32Attr: $dimension,
                         DefaultValuedAttr<BoolAttr, "false">:$numericStable);

    let results = (outs AnyRankedTensor:$result);

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

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        ::mlir::Operation::operand_range inputs = getInputs();
        int64_t numOperands = getOperands().size();
        int32_t dim = getDim();
        return
          wa::TTNNOperandsWorkaroundsFactory::createConcatOpOperandsWorkarounds(
                                                        inputs, numOperands, dim);
      }
    }];

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

def TTNN_RepeatOp : TTNN_Op<"repeat"> {
    let summary = "Repeat op.";
    let description = [{
      Returns a new tensor filled with repetition of input tensor according to number of times specified in repeat_dims.

      Parameters:
        - `input_tensor` (ttnn.Tensor): the input tensor.
        - `repeat_dims` (number): The number of repetitions for each element.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         TTNN_ShapeAttr:$repeat_dims);

    let results = (outs AnyRankedTensor:$result);

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
        ::mlir::ArrayAttr step = getStep();
        return wa::TTNNOperandsWorkaroundsFactory::
            createSliceStaticOpOperandsWorkarounds(step);
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
        ::mlir::ArrayAttr step = getStepAttr();
        return wa::TTNNOperandsWorkaroundsFactory::
            createSliceDynamicOpOperandsWorkarounds(step);
      }
    }];

    let hasVerifier = 1;
}

def TTNN_LinearOp : TTNN_Op<"linear"> {
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
                         DefaultValuedAttr<BoolAttr, "false">:$transpose_b);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}


// ANCHOR: adding_an_op_matmul_ttnn
def TTNN_MatmulOp : TTNN_Op<"matmul"> {
    let arguments = (ins AnyRankedTensor:$a,
                         AnyRankedTensor:$b,
                         DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
                         DefaultValuedAttr<BoolAttr, "false">:$transpose_b,
                         OptionalAttr<AnyAttrOf<[
                            TTNN_MatmulMultiCoreReuseProgramConfigAttr,
                            TTNN_MatmulMultiCoreReuseMultiCastProgramConfigAttr,
                            TTNN_MatmulMultiCoreReuseMultiCast1DProgramConfigAttr,
                            TTNN_MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr
                         ]>>:$matmul_program_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}
// ANCHOR_END: adding_an_op_matmul_ttnn

def TTNN_PrepareConv2dWeightsOp : TTNN_Op<"prepare_conv2d_weights"> {
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
                         OptionalAttr<TTNN_Conv2dConfigAttr>:$conv2d_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_PrepareConv2dBiasOp : TTNN_Op<"prepare_conv2d_bias"> {
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
                         OptionalAttr<TTNN_Conv2dConfigAttr>:$conv2d_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_Conv2dOp : TTNN_Op<"conv2d", [TTNN_DtypeOpInterface]> {
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
                         OptionalAttr<TTNN_DeviceComputeKernelConfig>:$compute_config);

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

def TTNN_ConvTranspose2dOp : TTNN_Op<"conv_transpose2d", [TTNN_DtypeOpInterface]> {
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
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

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
                         BoolAttr:$in_place_halo,
                         DefaultValuedAttr<BoolAttr, "true">:$count_include_pad);

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
                         BoolAttr:$in_place_halo);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return wa::TTNNOperandsWorkaroundsFactory::createPool2DOpOperandsWorkarounds();
      }
    }];

    let hasVerifier = 1;
}

def TTNN_BatchNormOp : TTNN_Op<"batch_norm", [AttrSizedOperandSegments]> {
    let summary = "Batch normalization op.";
    let description = [{
      Batch normalization operation over each channel on input tensor.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         Optional<AnyRankedTensor>:$running_mean,
                         Optional<AnyRankedTensor>:$running_var,
                         DefaultValuedAttr<BoolAttr, "false">:$training,
                         DefaultValuedAttr<F32Attr, "1e-05">:$epsilon,
                         DefaultValuedAttr<F32Attr, "0.1">:$momentum,
                         Optional<AnyRankedTensor>:$weight,
                         Optional<AnyRankedTensor>:$bias,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_RMSNormOp : TTNN_Op<"rms_norm", [AttrSizedOperandSegments, TTNN_MemoryConfigOpInterface]> {
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
                         F32Attr:$min,
                         F32Attr:$max,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let builders =
    [
      OpBuilder<(ins "Type": $resultType, "Value": $input, "::llvm::APFloat": $min, "::llvm::APFloat": $max),
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

def TTNN_EmptyOp : TTNN_CreationOp<"empty"> {
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

def TTNN_RandOp : TTNN_CreationOp<"rand"> {
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

def TTNN_DeallocateOp : TTNN_Op<"deallocate"> {
    let summary = "Deallocate op.";
    let description = [{
      Tensor Deallocate operation
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         DefaultValuedAttr<BoolAttr, "false">:$force);
}

def TTNN_ScatterOp: TTNN_ElementwiseBinaryCompositeOp<"scatter"> {
    let summary = "Scatter op.";
    let description = [{
      Embeds the values of the 'update' tensor into 'input' at the given index and puts the value in the 'output' tensor.
      }];
}

def TTNN_AllGatherOp: TTNN_Op<"all_gather"> {
    let summary = "All gather op.";
    let description = [{
        Tensor All Gather operation
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         TTNN_Device:$device,
                         SI32Attr:$all_gather_dim,
                         UI32Attr:$cluster_axis,
                         DefaultValuedAttr<UI32Attr, "1">:$num_links);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let hasFolder = 1;
}

def TTNN_ReduceScatterOp: TTNN_Op<"reduce_scatter"> {
    let summary = "Reduce scatter op.";
    let description = [{
        Tensor Reduce Scatter operation
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         TTNN_Device:$device,
                         TTCore_ReduceTypeAttr:$reduce_type,
                         SI32Attr:$scatter_dim,
                         UI32Attr:$cluster_axis,
                         DefaultValuedAttr<UI32Attr, "1">:$num_links);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let hasFolder = 1;
}

def TTNN_AllReduceOp: TTNN_Op<"all_reduce"> {
    let summary = "All reduce op.";
    let description = [{
        Tensor All Reduce operation
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         TTNN_Device:$device,
                         TTCore_ReduceTypeAttr:$reduce_type,
                         UI32Attr:$cluster_axis,
                         DefaultValuedAttr<UI32Attr, "1">:$num_links);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let hasFolder = 1;
}

def TTNN_CollectivePermuteOp: TTNN_Op<"collective_permute"> {
    let summary = "Collective permute op.";
    let description = [{
      Collective permute op. This operation ingests a multi-device tensor spread across multi-devices and will shuffle the data according to source_target_pairs [['src', 'dest']].

      Example:
        For a 1x2 mesh, the following will take the device shard living in device 0 and move it to device 1. The device shard living in device 1 will move to device 0.
        %source_target_pairs: [[0, 1], [1, 0]]

        In the case of missing 'dest', the device shard living on that device will contain values of 0. For example, device shard living in device 0 will contain 0 values.
        %source_target_pairs: [[0, 1]]
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         TTNN_Device:$device,
                         I64ElementsAttr:$source_target_pairs);

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

    If `accum_tensor` is not provided, a new output tensor will be allocated automatically
    at the receiver. If provided, the data will be written into the specified accum tensor.

    The operation returns a multi-device tensor whose buffer layout follows the mesh configuration.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                         DenseI64ArrayAttr:$send_coord,
                         DenseI64ArrayAttr:$receive_coord,
                         Optional<AnyRankedTensor>:$accum_tensor);
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

def TTNN_GenericOp : TTNN_Op<"generic", [TTNN_MemoryConfigOpInterface, MemoryEffects<[MemRead, MemWrite]>]> {
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
                         TTNN_ProgramAttr:$program,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

    let hasVerifier = 1;
}

def TTNN_RotaryEmbeddingLlamaOp : TTNN_Op<"rotary_embedding_llama", [TTNN_MemoryConfigOpInterface]> {
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

def TTNN_DumpTensorOp : TTNN_InplaceOp<"dump_tensor"> {
  let summary = "Saves a tensor to disk in the TT-NN binary format";
  let description = [{
    Saves a tensor to disk in the TT-NN binary format. Files must use the `.tensorbin` extension.

    Inputs:
      - `file_path` StrAttr: Path of the file where tensor should be dumped. Must end with `.tensorbin` extension.
      - `input` AnyRankedTensor: Tensor to serialize.
  }];

  let arguments = (ins StrAttr:$file_path,
                       AnyRankedTensor:$input);

  let hasVerifier = 1;
}

def TTNN_LoadTensorOp : TTNN_Op<"load_tensor"> {
  let summary = "Loads a tensor from disk";
  let description = [{
    Loads a tensor from disk, optionally placing it directly on a device.

    Inputs:
      - `file_path` StrAttr: Path of the file of the serialized tensor. Must end with `.tensorbin` extension.
      - `device` Optional<TTNN_Device>: Device where tensor should be desirialized. It has to be provided iff the serialized tensor is a deivce tensor.
    Outputs:
      - `result` AnyRankedTensor: Desirialized tensor from the `file_path`.
  }];

  let arguments = (ins StrAttr:$file_path,
                       Optional<TTNN_Device>:$device);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

#endif
