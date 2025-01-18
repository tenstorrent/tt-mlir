// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPS_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPS_TD

include "ttmlir/Dialect/TT/IR/TTOpsTypes.td"
include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.td"
include "ttmlir/Dialect/TTNN/IR/TTNNBase.td"
include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.td"
include "ttmlir/Dialect/TTNN/IR/TTNNOpsEnums.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"

def TTNN_GetDeviceOp : TTNN_Op<"get_device"> {
    let summary = "Get Device op.";
    let description = [{
      This op returns the current runtime device.
    }];

    let arguments = (ins OptionalAttr<TTNN_MeshShapeAttr>:$mesh_shape);
    let results = (outs TT_Device:$device);
}

def TTNN_ToMemoryConfigOp : TTNN_Op<"to_memory_config"> {
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
                         OptionalAttr<TT_DataTypeAttr>:$dtype,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                         Optional<TT_Device>:$device);
    let results = (outs AnyRankedTensor:$result);

    let hasCanonicalizeMethod = 1;
}

def TTNN_TypecastOp : TTNN_Op<"typecast"> {
    let summary = "Typecast op.";
    let description = [{
      This op converts the data type of the input tensor based on the given data type.
      It handles:
        - conversions of data types.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         TT_DataTypeAttr:$dtype);
    let results = (outs AnyRankedTensor:$result);
}

def TTNN_ToDeviceOp : TTNN_Op<"to_device"> {
    let summary = "ToDevice op.";
    let description = [{
      This op sends the input tensor to the given device with the given memory config.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         TT_Device:$device,
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


class TTNN_NamedDPSOp<string mnemonic, list<Trait> traits = []> :
    TTNN_Op<mnemonic, !listconcat(traits, [DestinationStyleOpInterface])> {
    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }
    }];
}

class TTNN_ElementwiseOp<string mnemonic, list<Trait> traits = []> :
    TTNN_NamedDPSOp<mnemonic, !listconcat(traits, [AttrSizedOperandSegments])> {

    let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                         Variadic<AnyRankedTensor>:$outputs);
    let results = (outs Variadic<AnyRankedTensor>:$results);
}

class TTNN_ElementwiseUnaryOp<string mnemonic, list<Trait> traits = []> :
    TTNN_ElementwiseOp<mnemonic, traits> {
    let summary = "Eltwise unary op.";
    let description = [{
      Eltwise unary op.
    }];

    let builders =
    [
      OpBuilder<(ins "Value": $in, "Value": $out),
      [{
        build($_builder, $_state, {out.getType()}, in, out);
      }]>
    ];
}

class TTNN_ElementwiseBinaryOp<string mnemonic, list<Trait> traits = []> :
    TTNN_ElementwiseOp<mnemonic, traits> {
    let summary = "Eltwise binary op.";
    let description = [{
      Eltwise binary op.
    }];

    let builders =
    [
      OpBuilder<(ins "Value": $lhs, "Value": $rhs, "Value": $out),
      [{
        build($_builder, $_state, {out.getType()}, {lhs, rhs}, out);
      }]>
    ];
}

class TTNN_ElementwiseTernaryOp<string mnemonic, list<Trait> traits = []> :
    TTNN_ElementwiseOp<mnemonic, traits> {
    let summary = "Eltwise ternary op.";
    let description = [{
      Eltwise ternary op.
    }];

    let builders =
    [
      OpBuilder<(ins "Value": $first, "Value": $second, "Value": $third, "Value": $out),
      [{
        build($_builder, $_state, {out.getType()}, {first, second, third}, out);
      }]>
    ];
}

def TTNN_WhereOp : TTNN_ElementwiseTernaryOp<"where"> {
    let summary = "Eltwise where.";
    let description = [{
      Eltwise where operation.
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

def TTNN_TanhOp: TTNN_ElementwiseUnaryOp<"tanh"> {
    let summary = "Eltwise tanh op.";
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

def TTNN_ReluOp : TTNN_ElementwiseUnaryOp<"relu",
      [DeclareOpInterfaceMethods<TTNN_OpModelInterface, ["getOpConstraints"]>]
      > {
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
    TTNN_ElementwiseUnaryOp<mnemonic, traits> {
    let summary = "Eltwise unary op with the float parameter.";
    let description = [{
      Eltwise unary op with the float parameter.
    }];

    let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                         Variadic<AnyRankedTensor>:$outputs,
                         F32Attr:$parameter);

    let builders =
    [
      OpBuilder<(ins "Value": $in, "Value": $out, "FloatAttr":$parameter),
      [{
        build($_builder, $_state, {out.getType()}, {in}, {out}, parameter);
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

def TTNN_AddOp : TTNN_ElementwiseBinaryOp<"add",
      [DeclareOpInterfaceMethods<TTNN_OpModelInterface, ["getOpConstraints"]>]
      > {

    let summary = "Eltwise add.";
    let description = [{
      Eltwise add operation.
    }];
}

def TTNN_DivOp : TTNN_ElementwiseBinaryOp<"div"> {
    let summary = "Eltwise divide.";
    let description = [{
      Eltwise divide operation.
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

def TTNN_BitwiseAndOp : TTNN_ElementwiseBinaryOp<"bitwise_and"> {
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

def TTNN_BitwiseOrOp : TTNN_ElementwiseBinaryOp<"bitwise_or"> {
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

def TTNN_BitwiseXorOp : TTNN_ElementwiseBinaryOp<"bitwise_xor"> {
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

def TTNN_MaximumOp :  TTNN_ElementwiseBinaryOp<"maximum"> {
    let summary = "Eltwise maximum OP.";
    let description = [{
      Calculates maximum of input tensors' values element-wise and stores result in output tensor.

      Example:
        %lhs: [[3, 2, 7], [1, 4, 4]]
        %rhs: [[1, 4, 2], [1, 2, 3]]
        "ttnn.maximum"(%lhs, %rhs, %out) -> %out: [[3, 4, 7], [1, 4, 4]]
    }];
}

def TTNN_MinimumOp :  TTNN_ElementwiseBinaryOp<"minimum"> {
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

def TTNN_MultiplyOp : TTNN_ElementwiseBinaryOp<"multiply"> {
    let summary = "Eltwise multiply.";
    let description = [{
      Eltwise multiply operation.
    }];
}

def TTNN_SubtractOp : TTNN_ElementwiseBinaryOp<"subtract"> {
    let summary = "Eltwise subtract.";
    let description = [{
      Eltwise subtract operation.
    }];
}

def TTNN_RemainderOp : TTNN_ElementwiseBinaryOp<"remainder"> {
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

class TTNN_ReductionOp<string mnemonic, list<Trait> traits = []> : TTNN_Op<mnemonic, traits> {
    let summary = "Reduction op.";
    let description = [{
      Reduction op.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         BoolAttr:$keep_dim,
                         OptionalAttr<I32ArrayAttr>:$dim_arg);

    let results = (outs AnyRankedTensor:$result);

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

def TTNN_ProdOp : TTNN_ReductionOp<"prod"> {
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
}

def TTNN_EmbeddingOp : TTNN_NamedDPSOp<"embedding"> {
    let summary = "Embedding op.";
    let description = [{
      Embedding operation.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$weight,
                         AnyRankedTensor:$output);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
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

def TTNN_EmbeddingBackwardOp : TTNN_NamedDPSOp<"embedding_bw"> {
    let summary = "Embedding backward op.";
    let description = [{
      Embedding backward operation. Generates the gradient of the embedding operation with respect to the input.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$weight,
                         AnyRankedTensor:$in_gradient,
                         OptionalAttr<TT_DataTypeAttr>:$dtype,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                         AnyRankedTensor:$output);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return wa::TTNNOperandsWorkaroundsFactory::createEmbeddingBackwardOpOperandsWorkarounds();
      }
    }];

    let hasVerifier = 1;
}

def TTNN_SoftmaxOp : TTNN_Op<"softmax",
      [DeclareOpInterfaceMethods<TTNN_OpModelInterface, ["getOpConstraints"]>]
      > {

    let summary = "Softmax op.";
    let description = [{
      Softmax operation.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         SI32Attr: $dimension);

    let results = (outs AnyRankedTensor:$result);

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

def TTNN_ConcatOp : TTNN_NamedDPSOp<"concat"> {
    let summary = "Concat op.";
    let description = [{
      Concat tensors along a given dimension.
    }];

    let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                         AnyRankedTensor:$output,
                         SI32Attr:$dim);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTNN_ReshapeOp : TTNN_Op<"reshape"> {
    let summary = "Reshape op.";
    let description = [{
      Reshape tensor.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         I32ArrayAttr:$shape);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_RepeatOp : TTNN_Op<"repeat"> {
    let summary = "Repeat op.";
    let description = [{
      Repeat the input tensor according to number of times specified in repeat_dimensions.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         I32ArrayAttr:$shape);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_SliceOp: TTNN_NamedDPSOp<"slice"> {
    let summary = "Slice op.";
    let description = [{
      Extract a portion of a tensor based on the specified start (`begins`), stop (`ends`), and step
      indices for each dimension.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         I32ArrayAttr:$begins,
                         I32ArrayAttr:$ends,
                         I32ArrayAttr:$step);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTNN_LinearOp : TTNN_NamedDPSOp<"linear"> {
    let summary = "Linear transformation of inputs.";

    let description = [{
      Produces the matmul of tensors `a` and `b` with optional addition with `bias`.

      Example:
        // %a = [[1., 2.]], [2., 1.]]
        // %b = [[0., 1.], [1., 0.]]
        // %bias = [[1.]]
        "ttnn.linear"(%a, %b, %bias, %result) : (tensor<2x2xf16>, tensor<2x2xf16>, tensor<1xf16>, tensor<2x2xf16>) -> tensor<2x2xf16>
        // %result = [[3., 2.], [2., 3.]]
    }];

    let arguments = (ins AnyRankedTensor:$a,
                         AnyRankedTensor:$b,
                         Optional<AnyRankedTensor>:$bias,
                         AnyRankedTensor:$output);
    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}


// ANCHOR: adding_an_op_matmul_ttnn
def TTNN_MatmulOp : TTNN_NamedDPSOp<"matmul",
      [DeclareOpInterfaceMethods<TTNN_OpModelInterface, ["getOpConstraints"]>]
      > {
    let arguments = (ins AnyRankedTensor:$a,
                         AnyRankedTensor:$b,
                         AnyRankedTensor:$output);
    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}
// ANCHOR_END: adding_an_op_matmul_ttnn

def TTNN_Conv2dOp : TTNN_NamedDPSOp<"conv2d"> {
    let summary = "Conv2d operation.";
    let description = [{
      Applies a 2D convolution over an input image composed of several input planes.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$weight,
                         Optional<AnyRankedTensor>:$bias,
                         AnyRankedTensor:$output,
                         TT_Device:$device,
                         I32Attr:$in_channels,
                         I32Attr:$out_channels,
                         I32Attr:$batch_size,
                         I32Attr:$input_height,
                         I32Attr:$input_width,
                         I32Attr:$kernel_height,
                         I32Attr:$kernel_width,
                         I32Attr:$stride_height,
                         I32Attr:$stride_width,
                         I32Attr:$padding_height,
                         I32Attr:$padding_width,
                         I32Attr:$dilation_height,
                         I32Attr:$dilation_width,
                         I32Attr:$groups);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTNN_ConvTranspose2dOp : TTNN_NamedDPSOp<"conv_transpose2d"> {
    let summary = "ConvTranspose2d operation.";
    let description = [{
      Applies a 2D transposed convolution operator over an input image composed of several input planes.

      Inputs:
      - `input` AnyRankedTensor: NHWC format (batch_size x height x width x channels)
      - `weight` AnyRankedTensor: OIHW format (output_channels x input_channels x height x width)
      - `bias` Optional<AnyRankedTensor>: (1 x 1 x 1 x output_channels)
      - `output` AnyRankedTensor: (1 x 1 x (batch_size * height * width) x channels)

      Attributes:
      - `in_channels` i32: The number of input channels.
      - `out_channels` i32: The number of output channels.
      - `batch_size` i32: The batch size.
      - `input_height` i32: The input height.
      - `input_width` i32: The input width.
      - `kernel_size` array<i32>: The kernel size.
      - `stride` array<i32>: Controls the stride for the cross-correlation.
      - `padding` array<i32>: Controls the amount of implicit zero padding on both sides for dilation * (kernel_size - 1) - padding number of points.
      - `output_padding` array<i32>: Controls the additional size added to one side of the output shape.
      - `dilation` array<i32>: Controls the spacing between the kernel points
      - `groups` i32: Controls the connections between inputs and outputs. Must be divisible by input and output channels.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$weight,
                         Optional<AnyRankedTensor>:$bias,
                         AnyRankedTensor:$output,
                         TT_Device:$device,
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
                         I32Attr:$groups);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTNN_MaxPool2dOp : TTNN_NamedDPSOp<"max_pool2d"> {
    let summary = "Applies a 2D max pooling over an input signal composed of several input planes.";
    let description = [{
      Applies a 2D max pooling over an input signal composed of several input planes.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         TT_Device:$device,
                         SI32Attr:$batch_size,
                         SI32Attr:$input_height,
                         SI32Attr:$input_width,
                         SI32Attr:$channels,
                         SI32Attr:$kernel_height,
                         SI32Attr:$kernel_width,
                         SI32Attr:$stride_height,
                         SI32Attr:$stride_width,
                         SI32Attr:$dilation_height,
                         SI32Attr:$dilation_width,
                         BoolAttr:$ceil_mode,
                         SI32Attr:$padding_height,
                         SI32Attr:$padding_width);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
      wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
        return wa::TTNNOperandsWorkaroundsFactory::createMaxPool2DOpOperandsWorkarounds();
      }
    }];

    let hasVerifier = 1;
}

def TTNN_ClampOp : TTNN_Op<"clamp"> {
    let summary = "Clamp op.";
    let description = [{
      Clamp tensor values to a specified range.

      Example:
        min: 2.000000+00
        input: [[0, 1, 2, 3, 4, 5, 6, 7]]
        max: 5.000000+00

        "ttnn.clamp"(%arg0) <{max = 2.000000e+00 : f32, min = 5.000000e+00 : f32}>
        -> %out = [[2, 2, 2, 3, 4, 5, 5, 5]]
    }];

    let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                         F32Attr:$min,
                         F32Attr:$max);

    let results = (outs Variadic<AnyRankedTensor>:$result);

    let hasVerifier = 1;
}

def TTNN_EmptyOp : TTNN_Op<"empty"> {
    let summary = "Empty op.";
    let description = [{
      Tensor empty operation
    }];

    let arguments = (ins TTNN_ShapeAttr:$shape,
                         TT_DataTypeAttr:$dtype,
                         TTNN_LayoutAttr:$layout,
                         TT_Device:$device,
                         TTNN_MemoryConfigAttr:$memory_config);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_ArangeOp : TTNN_Op<"arange"> {
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

  let arguments = (ins I64Attr:$start,
                       I64Attr:$end,
                       I64Attr:$step,
                       OptionalAttr<TT_DataTypeAttr>:$dtype,
                       Optional<TT_Device>:$device,
                       OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def TTNN_OnesOp : TTNN_Op<"ones"> {
  let summary = "Creates a tensor filled with ones.";
  let description = [{
    Tensor operation to create a tensor filled with ones.

    Given a ShapeAttr `shape`, produces a tensor with the same shape, filled with ones.

    Example:
      %0 = "ttnn.ones"() <{shape = array<i32:64, 28, 28>}> : () -> tensor<64x28x28xbf16>
      // %0: [[[1, 1, 1, ..., 1], [1, 1, 1, ..., 1], ..., [1, 1, 1, ..., 1]]]
  }];

  let arguments = (ins TTNN_ShapeAttr:$shape,
                       OptionalAttr<TT_DataTypeAttr>:$dtype,
                       OptionalAttr<TTNN_LayoutAttr>:$layout,
                       Optional<TT_Device>:$device,
                       OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

  let results = (outs AnyRankedTensor:$result);
}

def TTNN_FullOp : TTNN_Op<"full"> {
    let summary = "Full op.";
    let description = [{
      Tensor full operation
    }];

    let arguments = (ins TT_Device:$device, F32Attr:$fillValue);
    let results = (outs AnyRankedTensor:$result);
}

def TTNN_AllocOp : TTNN_Op<"alloc"> {
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

def TTNN_ScatterOp: TTNN_ElementwiseBinaryOp<"scatter"> {
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
                         TT_Device:$device,
                         SI32Attr:$dim,
                         DefaultValuedAttr<SI32Attr, "1">:$num_links);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_ReduceScatterOp: TTNN_Op<"reduce_scatter"> {
    let summary = "Reduce scatter op.";
    let description = [{
        Tensor Reduce Scatter operation
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         TT_Device:$device,
                         SI32Attr:$scatter_split_dim,
                         TT_ReduceTypeAttr:$math_op,
                         DefaultValuedAttr<SI32Attr, "1">:$num_links);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_AllReduceOp: TTNN_Op<"all_reduce"> {
    let summary = "All reduce op.";
    let description = [{
        Tensor All Reduce operation
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         TT_Device:$device,
                         SI32Attr:$scatter_dim,
                         SI32Attr:$scatter_num,
                         TT_ReduceTypeAttr:$math_op,
                         DefaultValuedAttr<SI32Attr, "1">:$num_links);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_MeshShardOp: TTNN_Op<"mesh_shard"> {
    let summary = "Mesh shard op.";
    let description = [{
        Tensor Mesh Shard operation
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         TT_Device:$device,
                         TT_MeshShardDirectionAttr:$shard_direction,
                         TT_MeshShardTypeAttr:$shard_type,
                         TT_GridAttr:$shard_shape);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_PermuteOp : TTNN_Op<"permute"> {
    let summary = "Permute operation.";
    let description = [{
      Permute input tensor dimensions.

      Attributes:
        - `permutation` array<i64>: The permutation of the input tensor dimensions.

      Example:
      %a = tensor.empty() : () -> tensor<2x3x4xi32>
      %0 = "ttir.permute"(%a) {permutation = array<i64: 1, 2, 0>} : (tensor<2x3x4xi32>) -> tensor<3x4x2xi32>
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         DenseI64ArrayAttr:$permutation,
                         OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                         DefaultValuedOptionalAttr<F32Attr, "0.0f">:$pad_value);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

#endif
