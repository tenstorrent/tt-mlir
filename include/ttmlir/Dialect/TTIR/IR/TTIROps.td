// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIROPS_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIROPS_TD

include "ttmlir/Dialect/TT/IR/TTOpsTypes.td"
include "ttmlir/Dialect/TTIR/IR/TTIRBase.td"
include "ttmlir/Dialect/TTIR/IR/TTIROpsAttrs.td"
include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"

class TTIR_DPSOp<string mnemonic, list<Trait> traits = []> :
    TTIR_Op<mnemonic, !listconcat([TTIROpInterface, DestinationStyleOpInterface], traits)> {
    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }
    }];
}

//===----------------------------------------------------------------------===//
// TTIR top level, bufferizable, ops
//===----------------------------------------------------------------------===//

class TTIR_BufferizableOp<string mnemonic, list<Trait> traits = []> :
    TTIR_DPSOp<mnemonic, !listconcat(
      [ DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                           , "bufferizesToMemoryWrite"
                                                           , "bufferize"
                                                           , "getAliasingValues"
                                                           , "getBufferType"
                                                           ]>
      , DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
      ], traits)> {
}

def TTIR_GenericOp : TTIR_BufferizableOp<"generic", [AttrSizedOperandSegments, NoTerminator]> {
    let summary = "Generically dispatch work to a grid of cores.";
    let description = [{
      This generic op carries a region that represents the work each core does. The region is
      expected to have the same signature as the op itself with respect to input and output
      operands. The op is expected to be lowered to a backend specific form by a consuming backend.
      This op is heavily inspired by the linalg.generic op so it can be useful to refer to
      linalg.generic documentation for more details.

      ```llvm
      %5 = "ttir.generic"(%1, %2, %3, %4) <{
        grid = #tt.grid<1x1>,                        // The grid range of cores to dispatch work to.
        indexing_maps = [#map, #map, #map],          // Affine maps for indexing into the input/output tensors. See linalg.generic
        iterator_types = [#parallel, #parallel],     // Iterator types for the input/output tensors. See linalg.generic
        operandSegmentSizes = array<i32: 2, 1, 1>,   // Sizes of the operand segments, i.e. 2 inputs, 1 cb and 1 output.
        operand_cb_mapping = array<i64: -1, 0, -1>,  // Mapping of input & output operands to cbs. -1 means no mapping.
                                                     // Mapped operands correspond to buffers in streaming mode.
                                                     // Non-mapped operands correspond to buffers in alias mode.
      ({
      ^bb0(%arg2: tensor<64x128xf32, #tt.buffer<memref<64x128xf32, #l1_>, alias>>,
           %arg3: tensor<64x128xf32, #tt.buffer<memref<64x128xf32, #l1_>, stream>>,
           %arg4: tensor<64x128xf32, #tt.buffer<memref<64x128xf32, #l1_>, alias>>):
          // Region body, would contain some computation that represents the work each core does.
      }) : (tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
      ```
    }];

    let arguments = (ins Variadic<AnyRankedTensorOrMemRef>:$inputs,
                         Variadic<AnyRankedTensorOrMemRef>:$cbs,
                         Variadic<AnyRankedTensorOrMemRef>:$outputs,
                         TT_GridAttr:$grid,
                         AffineMapArrayAttr:$indexing_maps,
                         TT_IteratorTypeArrayAttr:$iterator_types,
                         DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$operand_cb_mapping); // index of input operand and index of cb go together
    let results = (outs Variadic<AnyRankedTensor>:$results);
    let regions = (region AnyRegion:$region);
    let hasVerifier = 1;

    let extraClassDeclaration = [{
      // For a given block argument index, return the corresponding operand of the surrounding generic op.
      // This is needed because extra CB operands may be present in between the inputs and outputs.
      Value getMatchingOperand(size_t blockArgIndex) {
        assert(blockArgIndex < getInputs().size() + getOutputs().size() &&
               "blockArgIndex should be within the range of inputs and outputs");
        return blockArgIndex < getInputs().size() ?
          getOperand(blockArgIndex) : getOperand(blockArgIndex + getCbs().size());
      }

      MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }
    }];
}

def TTIR_ToLayoutOp : TTIR_BufferizableOp<"to_layout"> {
    let summary = "Layout op.";
    let description = [{
      ToLayout operation, transition tensors from one layout to another.  Some examples include:
        - Transitioning between different memory spaces, e.g. DRAM to L1.
        - Transitioning between different data types, e.g. f32 to f16.
        - Transitioning between different tile sizes, e.g. 1x16 to 32x32
        - Transitioning between different tensor sharding
        - Some combination of the above

      ```llvm
      #layout = #tt.metal_layout<8192x128x1, undef, <1x1>, memref<64x128xf32, #system>>
      #layout1 = #tt.metal_layout<8192x128x1, undef, <1x1>, memref<64x128xf32, #l1_>>
      %1 = "ttir.to_layout"(%arg0, %0) : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
      ```
    }];

    let arguments = (ins AnyRankedTensorOrMemRef:$input,
                         AnyRankedTensorOrMemRef:$output);
    let results = (outs Variadic<AnyRankedTensor>:$results);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }

      struct CompoundComponents {
        bool isLayoutChange = false;
        bool isGridChange = false;
        bool isFormatChange = false;
        bool isMemorySpaceChange = false;
      };

      // Returns booleans indicating if the op changes layout, grid, format, memory space or memory layout.
      CompoundComponents compoundComponents();
    }];

    let hasVerifier = 1;

    let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// TTIR top level ops
//===----------------------------------------------------------------------===//

def TTIR_GetDimensionSizeOp : TTIR_Op<"get_dimension_size"> {
  let summary = "GetDimensionSize op.";
  let description = [{
      Produces the size of the given `dimension` of the `operand`.

      Example:
        %operand: [[3, 2, 7], [1, 4, 4]]
        "ttir.get_dimension_size"(%operand, value = dense<0>, %out) -> %out: [[3]]
  }];

  let arguments = (ins AnyRankedTensor:$operand,
                       I32Attr:$dimension);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;

  let hasFolder = 1;
}

def TTIR_AllocOp : TTIR_Op<"alloc"> {
    let summary = "Alloc op.";
    let description = [{
      Tensor Alloc operation
    }];

    let arguments = (ins I64Attr:$address, I64Attr:$size, TT_MemorySpaceAttr:$memory_space);
    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_DotGeneralOp : TTIR_Op<"dot_general"> {
    let summary = "Dot general operation.";
    let description = [{
      Flexible tensor operation that generalizes matrix multiplication by allowing user to specify which
      dimensions of two tensors to contract. Matrix multiplication is a special case of this operation,
      where the contraction happens along the last axis of the first tensor and the second-to-last axis of the second tensor.
      From StableHLO DotGeneral Op https://openxla.org/stablehlo/spec#dot_general
    }];

    let arguments = (ins AnyRankedTensor:$lhs,
                         AnyRankedTensor:$rhs,
                         DenseI64ArrayAttr:$batch_dims_lhs,
                         DenseI64ArrayAttr:$contract_dims_lhs,
                         DenseI64ArrayAttr:$batch_dims_rhs,
                         DenseI64ArrayAttr:$contract_dims_rhs);

    let results = (outs AnyRankedTensor:$result);
}


def TTIR_DeallocOp : TTIR_Op<"dealloc"> {
    let summary = "Dealloc op.";
    let description = [{
      Tensor Dealloc operation
    }];

    let arguments = (ins AnyRankedTensor:$result);
}

//===----------------------------------------------------------------------===//
// TTIR top level named ops
//===----------------------------------------------------------------------===//

class TTIR_NamedOp<string mnemonic, list<Trait> traits = []> :
    TTIR_DPSOp<mnemonic, !listconcat([Pure], traits)> {
}

class TTIR_ElementwiseOp<string mnemonic, list<Trait> traits = []> :
    TTIR_NamedOp<mnemonic, !listconcat([AttrSizedOperandSegments, TTIR_Broadcastable], traits)> {

    let description = [{
      Base class for elementwise operations. Elementwise operations can take inputs with different shape,
      as long as the input tensors are broadcastable to the same shape.
    }];

    let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                         Variadic<AnyRankedTensor>:$outputs);
    let results = (outs Variadic<AnyRankedTensor>:$results);
}

class TTIR_ElementwiseTernaryOp<string mnemonic, list<Trait> traits = []> :
    TTIR_ElementwiseOp<mnemonic, !listconcat([FourOperands], traits)> {
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

def TTIR_WhereOp: TTIR_ElementwiseTernaryOp<"where", [TTIR_PartiallyBroadcastable]> {
    let summary = "Eltwise where op.";
    let description = [{
      Eltwise where operation.
    }];
}

class TTIR_ElementwiseUnaryOp<string mnemonic, list<Trait> traits = []> :
    TTIR_ElementwiseOp<mnemonic, !listconcat([TwoOperands], traits)> {
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

def TTIR_AbsOp: TTIR_ElementwiseUnaryOp<"abs", [TTIR_Idempotence]> {
    let summary = "Eltwise absolute op.";
    let description = [{
      Eltwise absolute operation.
    }];
}

def TTIR_CbrtOp: TTIR_ElementwiseUnaryOp<"cbrt"> {
    let summary = "Eltwise cubic root op.";
    let description = [{
      Eltwise cubic root operation.
    }];
}

def TTIR_CeilOp: TTIR_ElementwiseUnaryOp<"ceil", [TTIR_Idempotence]> {
    let summary = "Eltwise ceil op.";
    let description = [{
      Eltwise ceil operation.
    }];
}

def TTIR_CosOp: TTIR_ElementwiseUnaryOp<"cos"> {
    let summary = "Eltwise cosine op.";
    let description = [{
      Eltwise cosine operation.
    }];
}

def TTIR_FloorOp: TTIR_ElementwiseUnaryOp<"floor", [TTIR_Idempotence]> {
    let summary = "Eltwise floor op.";
    let description = [{
      Eltwise floor operation.
    }];
}

def TTIR_GeluOp: TTIR_ElementwiseUnaryOp<"gelu"> {
  let summary = "Eltwise GELU op.";
  let description = [{
    Eltwise GELU operation.
  }];
}

def TTIR_IsFiniteOp: TTIR_ElementwiseUnaryOp<"isfinite"> {
    let summary = "Eltwise isfinite op.";
    let description = [{
      Eltwise isfinite operation.
    }];
}

def TTIR_LogicalNotOp: TTIR_ElementwiseUnaryOp<"logical_not"> {
    let summary = "Eltwise logical not op.";
    let description = [{
      Eltwise logical not operation.
    }];
}

def TTIR_BitwiseNotOp : TTIR_ElementwiseUnaryOp<"bitwise_not", [TTIR_Involution]> {
    let summary = "Eltwise bitwise NOT.";
    let description = [{
        Performs element-wise NOT of tensor `operand` and produces a `result` tensor.

        Example:
            // Bitwise operation with with integer tensors
            // %operand: [[1, 2], [3, 4]]
            %result = "ttir.bitwise_not"(%operand) : (tensor<2x2xi32>) -> tensor<2x2xi32>
            // %result: [[-2, -3], [-4, -5]]
    }];
}

def TTIR_NegOp: TTIR_ElementwiseUnaryOp<"neg", [TTIR_Involution]> {
    let summary = "Eltwise negate op.";
    let description = [{
      Eltwise negate operation.
    }];
}

def TTIR_TanOp: TTIR_ElementwiseUnaryOp<"tan"> {
    let summary = "Eltwise tan op.";
    let description = [{
      Eltwise tan operation.
    }];
}

def TTIR_TanhOp: TTIR_ElementwiseUnaryOp<"tanh"> {
    let summary = "Eltwise tanh op.";
    let description = [{
      Eltwise tanh operation.
    }];
}

// TODO (azecevic): What should we do with 0.0 case?
// 1/0.0 = inf, 1/inf = 0.0, but TTNN isn't IEEE754 compliant.
// https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Handling_Special_Value/special_values.md
def TTIR_ReciprocalOp : TTIR_ElementwiseUnaryOp<"reciprocal", [TTIR_Involution]> {
    let summary = "Eltwise reciprocal.";
    let description = [{
      Eltwise reciprocal operation.
    }];
}

def TTIR_ReluOp : TTIR_ElementwiseUnaryOp<"relu", [TTIR_Idempotence]> {
    let summary = "Eltwise ReLU.";
    let description = [{
      Eltwise ReLU operation.
    }];
}

def TTIR_RsqrtOp : TTIR_ElementwiseUnaryOp<"rsqrt"> {
    let summary = "Eltwise reciprocal square root.";
    let description = [{
      Eltwise reciprocal square root operation.
    }];
}

def TTIR_SigmoidOp: TTIR_ElementwiseUnaryOp<"sigmoid"> {
    let summary = "Eltwise sigmoid.";
    let description = [{
      Eltwise sigmoid operation.
    }];
}

def TTIR_SignOp: TTIR_ElementwiseUnaryOp<"sign", [TTIR_Idempotence]> {
    let summary = "Eltwise sign operation.";
    let description = [{
      Returns the sign of the `operand` element-wise and produces a `result`
      tensor.

      Example:
        %a: [[3, -2, 0], [1, -4, 4]]
        "ttir.sign"(%a, %out) -> %out: [[1, -1, 0], [1, -1, 1]]
    }];
}

def TTIR_SinOp: TTIR_ElementwiseUnaryOp<"sin"> {
    let summary = "Eltwise sine.";
    let description = [{
      Eltwise sine operation.
    }];
}

def TTIR_SqrtOp : TTIR_ElementwiseUnaryOp<"sqrt"> {
    let summary = "Eltwise square root.";
    let description = [{
      Eltwise square root operation.
    }];
}

def TTIR_TypecastOp: TTIR_ElementwiseUnaryOp<"typecast"> {
    let summary = "Eltwise cast op.";
    let description = [{
      Eltwise cast operation.
    }];

    let hasFolder = 1;
}

def TTIR_LogOp: TTIR_ElementwiseUnaryOp<"log"> {
    let summary = "Eltwise logarithm op.";
    let description = [{
      Eltwise logarithm operation. Calculates log(x) for all elements x in input tensor.
    }];
}

def TTIR_Log1pOp: TTIR_ElementwiseUnaryOp<"log1p"> {
    let summary = "Eltwise log1p operation.";
    let description = [{
        Performs element-wise logarithm plus one operation on `operand` tensor and
        puts the result in the output tensor.

        Example:
          %a: [0.0, -0.999, 7.0, 6.38905621, 15.0]
          "ttir.logp1"(%a, %out) -> %out: [0.0, -6.90776825, 2.07944155, 2.0, 2.77258873]
      }];
}

def TTIR_Expm1Op: TTIR_ElementwiseUnaryOp<"expm1"> {
  let description = [{
    Performs element-wise exponential minus one operation on `operand` tensor
    and stores the result in the output tensor.

    Example:
        %a: [[0, 1], [0, 0]]
        "ttir.exmp1"(%a, %out) -> %out: [[0, 1.71828], [0, 0]]
  }];
}

class TTIR_ElementwiseUnaryWithFloatParameterOp<string mnemonic, list<Trait> traits = []> :
    TTIR_ElementwiseUnaryOp<mnemonic, traits> {
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

def TTIR_LeakyReluOp : TTIR_ElementwiseUnaryWithFloatParameterOp<"leaky_relu"> {
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

class TTIR_ElementwiseBinaryOp<string mnemonic, list<Trait> traits = []> :
    TTIR_ElementwiseOp<mnemonic, !listconcat([ThreeOperands], traits)> {
    let summary = "Eltwise binary op.";
    let description = [{
      Eltwise binary op.
    }];

    let builders =
    [
      OpBuilder<(ins "Value": $lhs, "Value": $rhs, "Value": $out),
      [{
        build($_builder, $_state, {out.getType()}, {lhs, rhs}, out);
      }]>,
      OpBuilder<(ins "Type":$outputType, "Value": $lhs, "Value": $rhs, "Value": $out),
      [{
        build($_builder, $_state, {outputType}, {lhs, rhs}, out);
      }]>
    ];
}

// TODO (azecevic): NaN != NaN, otherwise eq(x, x) == 1.
def TTIR_EqualOp : TTIR_ElementwiseBinaryOp<"eq"> {
    let summary = "Eltwise equal to.";
    let description = [{
      Eltwise equal to operation.
    }];
}

// TODO (azecevic): NaN != NaN, otherwise ne(x, x) == 0.
def TTIR_NotEqualOp : TTIR_ElementwiseBinaryOp<"ne"> {
    let summary = "Eltwise not equal to.";
    let description = [{
      Eltwise not equal to operation.
    }];
}

// TODO (azecevic): NaN != NaN, otherwise ge(x, x) == 1.
def TTIR_GreaterEqualOp : TTIR_ElementwiseBinaryOp<"ge"> {
    let summary = "Eltwise greater than or equal to.";
    let description = [{
      Eltwise greater than or equal to operation.
    }];
}

// TODO (azecevic): NaN != NaN, otherwise gt(x, x) == 0.
def TTIR_GreaterThanOp : TTIR_ElementwiseBinaryOp<"gt"> {
    let summary = "Eltwise greater than.";
    let description = [{
      Eltwise greater than operation.
    }];
}

// TODO (azecevic): NaN != NaN, otherwise le(x, x) == 1.
def TTIR_LessEqualOp : TTIR_ElementwiseBinaryOp<"le"> {
    let summary = "Eltwise less than or equal to.";
    let description = [{
      Eltwise less than or equal to operation.
    }];
}

// TODO (azecevic): NaN != NaN, otherwise lt(x, x) == 0.
def TTIR_LessThanOp : TTIR_ElementwiseBinaryOp<"lt"> {
    let summary = "Eltwise less than.";
    let description = [{
      Eltwise less than operation.
    }];
}

def TTIR_LogicalAndOp : TTIR_ElementwiseBinaryOp<"logical_and", [TTIR_BinaryIdempotence]> {
    let summary = "Eltwise logical and.";
    let description = [{
      Eltwise logical and operation.
    }];
}

def TTIR_LogicalOrOp : TTIR_ElementwiseBinaryOp<"logical_or", [TTIR_BinaryIdempotence]> {
    let summary = "Eltwise logical or.";
    let description = [{
      Eltwise logical or operation.
    }];
}

def TTIR_LogicalXorOp : TTIR_ElementwiseBinaryOp<"logical_xor"> {
    let summary = "Eltwise logical xor.";
    let description = [{
      Eltwise logical xor operation.
    }];
}

def TTIR_BitwiseAndOp : TTIR_ElementwiseBinaryOp<"bitwise_and", [TTIR_BinaryIdempotence]> {
    let summary = "Eltwise bitwise AND.";
    let description = [{
        Performs element-wise bitwise AND of two tensors `lhs` and `rhs`
        and produces a `result` tensor.

        Example:
            // %lhs: [[1, 2], [3, 4]]
            // %rhs: [[5, 6], [7, 8]]
            %result = "ttir.bitwise_and"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
            // %result: [[1, 2], [3, 0]]
    }];
}

def TTIR_BitwiseOrOp : TTIR_ElementwiseBinaryOp<"bitwise_or", [TTIR_BinaryIdempotence]> {
    let summary = "Eltwise bitwise OR.";
    let description = [{
        Performs element-wise bitwise OR of two tensors `lhs` and `rhs`
        and produces a `result` tensor.

        Example:
            // %lhs: [[1, 2], [3, 4]]
            // %rhs: [[5, 6], [7, 8]]
            %result = "ttir.bitwise_or"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
            // %result: [[5, 6], [7, 12]]
    }];
}

def TTIR_BitwiseXorOp : TTIR_ElementwiseBinaryOp<"bitwise_xor"> {
    let summary = "Eltwise bitwise XOR.";
    let description = [{
        Performs element-wise bitwise XOR of two tensors `lhs` and `rhs`
        and produces a `result` tensor.

        Example:
          // %lhs: [[1, 2], [3, 4]]
          // %rhs: [[5, 6], [7, 8]]
          %result = "ttir.bitwise_xor"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
          // %result: [[4, 4], [4, 12]]
    }];

    let hasCanonicalizer = 1;
}

def TTIR_MinimumOp :  TTIR_ElementwiseBinaryOp<"minimum", [TTIR_BinaryIdempotence, TTIR_PartiallyBroadcastable]> {
    let summary = "Eltwise minimum OP.";
    let description = [{
      Calculates minimum of input tensors' values element-wise and stores result
      in output tensor.

      Example:
        %lhs: [[3, 2, 7], [1, 4, 4]]
        %rhs: [[1, 4, 2], [1, 2, 3]]
        "ttir.minimum"(%lhs, %rhs, %out) -> %out: [[1, 2, 2], [1, 2, 3]]
    }];
}

def TTIR_SubtractOp : TTIR_ElementwiseBinaryOp<"subtract", [TTIR_PartiallyBroadcastable]> {
    let summary = "Eltwise subtract.";
    let description = [{
      Eltwise subtract operation.
    }];
}

def TTIR_RemainderOp : TTIR_ElementwiseBinaryOp<"remainder", [TTIR_PartiallyBroadcastable]> {
    let summary = "Eltwise remainder.";
    let description = [{
      Performs element-wise remainder of dividend lhs and divisor rhs tensors and produces a
      result tensor.

      Example:

      // %lhs: [17, -17, 17, -17]
      // %rhs: [3, 3, -3, -3]
      %result = "ttir.remainder"(%lhs, %rhs) : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
      // %result: [2, -2, 2, -2]
    }];
}

// TODO We need to add a broadcastable trait to this op.
// See issue https://github.com/tenstorrent/tt-mlir/issues/2015
def TTIR_PowerOp : TTIR_ElementwiseBinaryOp<"power"> {
    let summary = "Eltwise power OP.";
    let description = [{
      Performs element-wise exponentiation of lhs tensor by rhs tensor and produces a
      result tensor. Tensors must be of same shape.

      Example:
      ```
        %result = "ttir.power"(%lhs, %rhs) : (tensor<6xf64>, tensor<6xf64>) -> tensor<6xf64>

        %lhs: [-2.0, -0.0, -36.0, 5.0, 3.0, 10000.0]
        %rhs: [2.0, 2.0, 1.1, 2.0, -1.0, 10.0]
        %result: [4.0, 0.0, -nan, 25.0, 0.333333343, inf]
      ```
    }];
}

class TTIR_ReductionOp<string mnemonic, list<Trait> traits = []> :
    TTIR_NamedOp<mnemonic, !listconcat([TTIR_GenericRegionOpInterface], traits)> {

    let summary = "Reduction op.";
    let description = [{
      Reduction op.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         BoolAttr:$keep_dim,
                         OptionalAttr<I32ArrayAttr>:$dim_arg);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }

      void buildGenericRegion(::mlir::OpBuilder &opBuilder, ::mlir::Block* block);

      // Returns the indexing maps and iterator types for the reduction op.
      // Indexing maps are identity maps with dropped dimensions corresponding to the
      // reduction dimensions. Iterator types are parallel for non-reduction dimensions
      // and reduction for reduction dimensions.
      std::pair<::mlir::ArrayAttr, ::mlir::ArrayAttr> getIndexingMaps(Builder &builder) {
        auto rank = mlir::cast<RankedTensorType>(getInput().getType()).getRank();
        SmallVector<AffineMap> indexingMaps(getNumOperands(),
                                            builder.getMultiDimIdentityMap(rank));
        SmallVector<Attribute> iteratorTypes(
            rank, builder.getAttr<IteratorTypeAttr>(IteratorType::Parallel));

        auto reduceDims = getDimArgAttr();
        auto resultIndexingMap = indexingMaps.back();
        for (auto reduceDim : reduceDims) {
          int64_t reduceDimInt = mlir::cast<IntegerAttr>(reduceDim).getInt();
          if (reduceDimInt < 0) {
            reduceDimInt += rank;
          }
          assert(reduceDimInt >= 0 && reduceDimInt < rank);
          resultIndexingMap.dropResult(reduceDimInt);
          iteratorTypes[reduceDimInt] =
              builder.getAttr<IteratorTypeAttr>(IteratorType::Reduction);
        }

        return {builder.getAffineMapArrayAttr(indexingMaps),
                builder.getArrayAttr(iteratorTypes)};}
    }];

    let hasVerifier = 1;
}

def TTIR_ArgMaxOp : TTIR_ReductionOp<"argmax"> {
  let summary = "Argmax reduction op.";
  let description = [{
    Determine the indices of the maximum values along a specified dimension of a tensor or over all elements in a tensor.

    Parameters:
      - `input`: The input tensor.
      - `dim`: Specifies the dimension along which the argmax is applied.
      - `keep_dim`: Whether to keep the reduce dimension or not.

    ### Example IR Usage:
    ```mlir
    // Input tensor of shape (128, 28, 28)
    %input = ... : tensor<128x28x28xbf16>

    %empty = tensor.empty() : tensor<128x28xi32>
    %argmax = "ttir.argmax"(%arg0, %0) <{dim_arg = [0: i32, 2 : i32], keep_dim = false}> : (tensor<128x28x28xbf16>, tensor<128x28xi32>) -> tensor<128x28xi32>
    ```

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
}

def TTIR_SumOp : TTIR_ReductionOp<"sum"> {
    let summary = "Sum reduction op.";
    let description = [{
      Sum reduction op.
    }];
}

def TTIR_MeanOp : TTIR_ReductionOp<"mean"> {
  let summary = "Mean reduction op.";
  let description = [{
    Mean reduction op.
  }];
}

def TTIR_MaxOp : TTIR_ReductionOp<"max"> {
  let summary = "Max reduction op.";
  let description = [{
    Max reduction op.
  }];
}

def TTIR_MinOp : TTIR_ReductionOp<"min"> {
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

def TTIR_ReduceAndOp : TTIR_ReductionOp<"reduce_and"> {
    let summary = "And reduction op.";
    let description = [{
      Reduces a given tensor using logical and operator along the given dimension(s).

      Example:
        input: [[True,  False, True, False],
                [True,  True,  True, True],
                [False, False, True, True],
                [False, True,  True, False]]

        // Reduction along dim 0
        output: [False, False, True, False]

        // Reduction along dim 1
        output: [False, True, False, False]

        // Reduction for both dimensions (entire tensor)
        output: [False]
    }];
}

def TTIR_ReduceOrOp : TTIR_ReductionOp<"reduce_or"> {
    let summary = "Or reduction op.";
    let description = [{
      Reduces a given tensor using logical or operator along the given dimension(s).

      Example:
        input: [[True,  False, False, False],
                [True,  True,  False, True],
                [False, False, False, True],
                [False, False, False, False]]

        // Reduction along dim 0
        output: [True, True, False, True]

        // Reduction along dim 1
        output: [True, True, True, False]

        // Reduction for both dimensions (entire tensor)
        output: [True]
    }];
}

def TTIR_ProdOp : TTIR_ReductionOp<"prod"> {
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

def TTIR_EmbeddingOp : TTIR_NamedOp<"embedding"> {
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
    }];

    let hasVerifier = 1;
}

def TTIR_EmbeddingBackwardOp : TTIR_NamedOp<"embedding_backward"> {
    let summary = "Embedding backward op.";
    let description = [{
      Embedding backward operation. Generates the gradient of the embedding operation with respect to the input.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$weight,
                         AnyRankedTensor:$in_gradient,
                         AnyRankedTensor:$output);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTIR_CumSumOp : TTIR_NamedOp<"cumsum"> {
  let summary = "Cummulative sum op.";
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
                       AnyRankedTensor:$output);

  let results = (outs AnyRankedTensor:$result);

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
  }];

  let hasVerifier = 1;
}

def TTIR_SoftmaxOp : TTIR_NamedOp<"softmax"> {
    let summary = "Softmax operation.";
    let description = [{
      Softmax operation.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         SI32Attr:$dimension);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTIR_TransposeOp : TTIR_NamedOp<"transpose"> {
    let summary = "Transpose op.";
    let description = [{
      Transpose tensor along two given dimensions.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         SI32Attr:$dim0,
                         SI32Attr:$dim1);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;

    let hasCanonicalizer = 1;
}

def TTIR_ConcatOp : TTIR_NamedOp<"concat"> {
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

def TTIR_RepeatOp : TTIR_NamedOp<"repeat"> {
  let summary = "Repeat operation.";
  let description = [{
    The `repeat` operation creates a new tensor by replicating the input tensor's elements
    along specified dimensions. The number of repetitions for each dimension is defined by
    the `repeats` attribute, which must have the same rank as the input tensor.

    Parameters:
      - `input`: The input tensor.
      - `repeats`: Specifies the number of times to repeat this tensor along each dimension.

    ### Example IR Usage:
    ```mlir
    // Input tensor of shape (2, 3)
    %input = ... : tensor<2x3xf32>

    // Repeat each dimension twice
    %empty = tensor.empty() : tensor<4x6xf32>
    %repeated = "repeat"(%input, %empty) {repeat_dimensions = array<i64: 2, 2>} : (tensor<2x3xf32>, tensor<4x6xf32>) -> tensor<4x6xf32>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$output,
                       DenseI64ArrayAttr:$repeat_dimensions);

  let results = (outs AnyRankedTensor:$result);

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
  }];

  let hasVerifier = 1;
}

def TTIR_RepeatInterleaveOp : TTIR_NamedOp<"repeat_interleave"> {
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
                         AnyRankedTensor:$output,
                         UI32Attr:$repeats,
                         SI32Attr:$dim);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTIR_UpdateCacheOp : TTIR_NamedOp<"update_cache"> {
  let summary = "Update static cache tensor.";
  let description = [{
      Updates the `cache` tensor in-place with values from `input` at `update_index` and `batch_offset`.
  }];

  let arguments = (ins AnyRankedTensor:$cache,
                       AnyRankedTensor:$input,
                       AnyRankedTensor:$update_index,
                       I32Attr:$batch_offset);

  let results = (outs AnyRankedTensor:$result);

  let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getCacheMutable(); }
  }];

  let hasVerifier = 1;
}

def TTIR_FillCacheOp : TTIR_NamedOp<"fill_cache"> {
  let summary = "Fill static cache tensor.";
  let description = [{
      Fills the `cache` tensor in-place with values from `input` at `batch_offset`.
  }];

  let arguments = (ins AnyRankedTensor:$cache,
                       AnyRankedTensor:$input,
                       I32Attr:$batch_offset);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getCacheMutable(); }
  }];
}

def TTIR_BroadcastOp : TTIR_NamedOp<"broadcast"> {
    let summary = "Broadcast operation.";
    let description = [{
      Broadcasts each dimension of the input tensor based on the broadcast dimensions.

      Example:
      // %arg0: tensor<1x1x32xf32>
      %0 = tensor.empty() : tensor<1x16x32xf32>
      %1 = "ttir.broadcast"(%arg0, %0) <{broadcast_dimensions = array<i32: 1, 16, 1>}> : (tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>

      Note: Currently, when generating a TTNN executable, the broadcast and repeat operations share the same semantics due to the lack of tensor view support in TTNN.
      As a result, the broadcast operation is lowered to a repeat operation in the TTNN compilation pipeline.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         DenseI64ArrayAttr:$broadcast_dimensions);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;

    let hasFolder = 1;
}

def TTIR_Conv2dOp : TTIR_NamedOp<"conv2d"> {
    let summary = "Conv2d operation.";
    let description = [{
     Applies a 2D convolution over an input image composed of several input planes.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$weight,
                         Optional<AnyRankedTensor>:$bias,
                         AnyRankedTensor:$output,
                         SI32Attr:$stride_height,
                         SI32Attr:$stride_width,
                         SI32Attr:$dilation_height,
                         SI32Attr:$dilation_width,
                         SI32Attr:$groups,
                         SI32Attr:$padding_left,
                         SI32Attr:$padding_right,
                         SI32Attr:$padding_top,
                         SI32Attr:$padding_bottom);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTIR_ConvTranspose2dOp : TTIR_NamedOp<"conv_transpose2d"> {
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
          - H_out = (H_in - 1) * stride[0] - (padding_top + padding_bottom) + dilation[0] * (K_H - 1) + output_padding[0] + 1
          - W_out = (W_in - 1) * stride[1] - (padding_left + padding_right) + dilation[1] * (K_W - 1) + output_padding[1] + 1

      Attributes:
        - `stride` (i32 | array<2xi32>): Controls the stride for the cross-correlation.
        - `padding` (i32 | array<2xi32> | array<4xi32>): Controls the amount of implicit zero padding on both sides for dilation * (kernel_size - 1) - padding number of points.
        - `output_padding` (i32 | array<2xi32>): Controls the additional size added to one side of the output shape.
        - `dilation` (i32 | array<2xi32>): Controls the spacing between the kernel points
        - `groups` i32: Controls the connections between inputs and outputs. Must be divisible by input and output channels.

      Example:
        %input = tensor.empty() : () -> tensor<3x8x8x256xbf16>
        %weight = tensor.empty() : () -> tensor<256x256x3x3xbf16>
        %bias = tensor.empty() : () -> tensor<1x1x1x256xbf16>
        %output = tensor.empty() : () -> tensor<1x10x10x256xbf16>
        %0 = "ttir.conv_transpose2d"(%input, %weight, %bias, %output)
          <{
            stride = 1: i32,
            padding = 0: i32,
            output_padding = 0: i32,
            dilation = 1: i32,
            groups = 1: i32
          > : (tensor<3x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<3x10x10x256xbf16>) -> tensor<3x10x10x256xbf16>
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$weight,
                         Optional<AnyRankedTensor>:$bias,
                         AnyRankedTensor:$output,
                         AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$stride,
                         AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$padding,
                         AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$output_padding,
                         AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$dilation,
                         I32Attr:$groups);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTIR_ConvolutionOp : TTIR_NamedOp<"convolution"> {
  let summary = "Generalized convolution op.";
  let description = [{
    Applies a convolution of the rhs with the lhs.

    This operation captures convolutions of all dimensionality as well
    as deconvolution/conv transpose.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weight,
    Optional<AnyRankedTensor>:$bias,
    AnyRankedTensor:$output,

    DenseI64ArrayAttr:$window_strides,
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$input_dilation,
    DenseI64ArrayAttr:$weight_dilation,
    DenseBoolArrayAttr:$window_reversal,
    TTIR_ConvolutionLayoutAttr:$convolution_layout,
    ConfinedAttr<I64Attr, [IntPositive]>:$feature_group_count,
    ConfinedAttr<I64Attr, [IntPositive]>:$batch_group_count
  );

  let results = (outs AnyRankedTensor);
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
  }];
}

def TTIR_GatherOp: TTIR_NamedOp<"gather"> {
    let summary = "Gather operation.";
    let description = [{
      Gathers slices from operand tensor from offsets specified in start_indices and produces a result tensor.
      From StableHLO Gather Op: https://openxla.org/stablehlo/spec#gather
    }];
    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$start_indices,
                         AnyRankedTensor:$output,
                         DenseI64ArrayAttr:$offset_dims,
                         DenseI64ArrayAttr:$collapsed_slice_dims,
                         DenseI64ArrayAttr:$operand_batching_dims,
                         DenseI64ArrayAttr:$start_indices_batching_dims,
                         DenseI64ArrayAttr:$start_index_map,
                         SI64Attr:$index_vector_dim,
                         DenseI64ArrayAttr:$slice_sizes,
                         BoolAttr:$indices_are_sorted);
    let results = (outs AnyRankedTensor:$result);
    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];
}

def TTIR_PoolingOp : TTIR_NamedOp<"pooling", [AttrSizedOperandSegments]> {
  let summary = "General pooling op";
  let description = [{
    General pooling op
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    Variadic<AnyRankedTensor>:$outputs,
    TTIR_PoolingMethodAttr:$pooling_method,
    DenseI64ArrayAttr:$window_dimensions,
    DenseI64ArrayAttr:$window_strides,
    DenseI64ArrayAttr:$base_dilations,
    DenseI64ArrayAttr:$window_dilations,
    DenseI64ArrayAttr:$padding
  );

  let results = (outs Variadic<AnyRankedTensor>);

  let hasVerifier = 1;
}

def TTIR_MaxPool2dOp : TTIR_NamedOp<"max_pool2d"> {
    let summary = "Applies a 2D max pooling over an input signal composed of several input planes.";
    let description = [{
      Applies a 2D max pooling over an input signal composed of several input planes.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         SI32Attr:$kernel_height,
                         SI32Attr:$kernel_width,
                         SI32Attr:$stride_height,
                         SI32Attr:$stride_width,
                         SI32Attr:$dilation_height,
                         SI32Attr:$dilation_width,
                         BoolAttr:$ceil_mode,
                         SI32Attr:$padding_left,
                         SI32Attr:$padding_right,
                         SI32Attr:$padding_top,
                         SI32Attr:$padding_bottom);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTIR_ReshapeOp: TTIR_NamedOp<"reshape"> {
    let summary = "Reshape op.";
    let description = [{
      Reshape tensor.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         I32ArrayAttr:$shape);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;

    let hasFolder = 1;
}

def TTIR_PadOp: TTIR_Op<"pad"> {
    let summary = "Pad op.";
    let description = [{
      Pad input tensor with tuple of padding values.

      The `padding` attribute must be a sequence of integers that is twice the size as the rank of the input.
      Each pair of integers in the padding attribute represents the amount of padding to add to the low and high of that dimension.
      I.e: an input tensor of shape <1x30x30x64xf32> with padding attribute <0, 0, 1, 1, 1, 1, 0, 0> will return a tensor of shape <1x32x32x64xf32>,
      and so will a padding attribute of <0, 0, 0, 2, 0, 2, 0, 0>.

      Example:
      input: [[1, 2, 3],
              [4, 5, 6]]

      // padding dimensions:
      padding: (1, 0, 1, 1) // format: (dim0_low, dim0_high, dim1_low, dim1_high)

      // padding value:
      // value: 0

      // padded output:
      output: [[0, 0, 0, 0, 0],
               [0, 1, 2, 3, 0],
               [0, 4, 5, 6, 0]]
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         DenseI32ArrayAttr:$padding,
                         F32Attr:$value);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_SliceOp: TTIR_NamedOp<"slice"> {
    let summary = "Slice op.";
    let description = [{
      Extract a sub-tensor (slice) from the input tensor across one or more dimensions.
      The `begins`, `ends`, and `step` attributes specify the start, stop, and step indices
      for each dimension of the tensor.
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

def TTIR_SelectOp: TTIR_NamedOp<"select"> {
    let summary = "Select op.";
    let description = [{
      Extracts a sub-tensor (slice) from the input tensor along a specified dimension in few steps defined by the
      `begin`, `length`, and `stride` attributes.
      The `begin` specifies the start index for the selected dimension of the tensor.
      The `length` specifies the number of elements to extract from the input tensor along the selected dimension.
      The `stride` specifies the step size for the start index. The default value is 0. 0 means no stride.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         SI32Attr:$dim,
                         SI32Attr:$begin,
                         SI32Attr:$length,
                         DefaultValuedOptionalAttr<SI32Attr, "0">:$stride);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

// ANCHOR: decomposing_an_op_index_ttir
def TTIR_IndexOp: TTIR_NamedOp<"index"> {
    let summary = "Index op.";
    let description = [{
      Extract a sub-tensor (slice) from the input tensor along a specified dimension.
      The `begin`, `end`, and `step` attributes define the start, stop, and step indices for the
      selected dimension (`dim`) of the tensor.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         I32Attr:$dim,
                         I32Attr:$begin,
                         I32Attr:$end,
                         I32Attr:$step);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}
// ANCHOR_END: decomposing_an_op_index_ttir

def TTIR_SqueezeOp : TTIR_NamedOp<"squeeze"> {
    let summary = "Squeeze op.";
    let description = [{
      Squeeze tensor.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         SI32Attr:$dim);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTIR_UnsqueezeOp : TTIR_NamedOp<"unsqueeze"> {
    let summary = "Unsqueeze op.";
    let description = [{
      Unsqueeze tensor.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         SI32Attr:$dim);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTIR_ClampOp : TTIR_NamedOp<"clamp"> {
    let summary = "Clamp op.";
    let description = [{
      Clamp tensor values to a specified range.

      Example:
        min: 2.000000+00
        input: [[0, 1, 2, 3, 4, 5, 6, 7]]
        max: 5.000000+00

        "ttir.clamp"(%arg0) <{max = 2.000000e+00 : f32, min = 5.000000e+00 : f32}>
        -> %out = [[2, 2, 2, 3, 4, 5, 5, 5]]
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         F32Attr:$min,
                         F32Attr:$max);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_ArangeOp : TTIR_Op<"arange"> {
  let summary = "Arange operation.";
  let description = [{
    Tensor arange operation.

    Produces a tensor with values from `start` to `end` (exclusive) with a step size of `step`, along the dimension specified by `arange_dimension`.

    Examples:
      %0 = "ttir.arange"() {start = 0 : i64, end = 5 : i64 step = 1 : i64, arange_dimension = 0 : i64} : () -> tensor<5xi64>
      // %0: [0, 1, 2, 3, 4]

      %1 = "ttir.arange"() {start = 0 : i64, end = 10 : i64, step = 2 : i64, arange_dimension = 0 : i64} : () -> tensor<5xf32>
      // %1: [0.0, 2.0, 4.0, 6.0, 8.0]

      %2 = "ttir.arange"() {start = 0 : i64, end = 5 : i64, step = 1 : i64, arange_dimension = 0 : i64} : () -> tensor<5x3xi64>
      // %2: [
              [0, 0, 0],
              [1, 1, 1],
              [2, 2, 2],
              [3, 3, 3],
              [4, 4, 4]
             ]

      %3 = "ttir.arange"() {start = 0 : i64, end = 3 : i64, step = 1 : i64, arange_dimension = 1 : i64} : () -> tensor<5x3xi64>
      // %3: [
              [0, 1, 2],
              [0, 1, 2],
              [0, 1, 2],
              [0, 1, 2],
              [0, 1, 2]
             ]
  }];

  let arguments = (ins SI64Attr:$start,
                       SI64Attr:$end,
                       SI64Attr:$step,
                       I64Attr:$arange_dimension);

  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def TTIR_ZerosOp : TTIR_Op<"zeros"> {
  let summary = "Creates a tensor filled with zeros.";
  let description = [{
    Tensor operation to create a tensor filled with zeros.

    Given a `shape`, produces a tensor with the shape, filled with zeros.

    Example:
      %0 = "ttir.zeros"() <{shape = array<i32:64, 28, 28>}> : () -> tensor<64x28x28xbf16>
      // %0: [[[0, 0, 0, ..., 0], [0, 0, 0, ..., 0], ..., [0, 0, 0, ..., 0]]]
  }];

  let arguments = (ins DenseI32ArrayAttr:$shape);

  let results = (outs AnyRankedTensor:$result);
}

def TTIR_OnesOp : TTIR_Op<"ones"> {
  let summary = "Creates a tensor filled with ones.";
  let description = [{
    Tensor operation to create a tensor filled with ones.

    Given a `shape`, produces a tensor with the shape, filled with ones.

    Example:
      %0 = "ttir.ones"() <{shape = array<i32:64, 28, 28>}> : () -> tensor<64x28x28xbf16>
      // %0: [[[1, 1, 1, ..., 1], [1, 1, 1, ..., 1], ..., [1, 1, 1, ..., 1]]]
  }];

  let arguments = (ins DenseI32ArrayAttr:$shape);

  let results = (outs AnyRankedTensor:$result);
}

def TTIR_ReverseOp : TTIR_NamedOp<"reverse", [AllShapesMatch<["input", "result"]>]> {
  let summary = "Reverse operation.";

  let description = [{
      Reverses the order of elements in the `operand` along the specified
      `dimensions` and produces a `result` tensor.

      Examples:
        // %operand = [[1, 2], [3, 4], [5, 6]]
        %result = "ttir.reverse"(%operand) {
          dimensions = array<i64: 1>
        } : (tensor<3x2xi32>) -> tensor<3x2xi32>
        // %result: [[2, 1], [4, 3], [6, 5]]

        // %operand = [[1, 2], [3, 4], [5, 6]]
        %result = "ttir.reverse"(%operand) {
          dimensions = array<i64: 1, 0>
        } : (tensor<3x2xi64>) -> tensor<3x2xi64>
        // %result: [[6, 5], [4, 3], [2, 1]]
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         DenseI64ArrayAttr:$dimensions);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;

    let hasCanonicalizer = 1;
}

def TTIR_ConstantOp : TTIR_Op<"constant", [ConstantLike,
                                           AllShapesMatch<["value", "result"]>]> {
    let summary = "Constant op.";
    let description = [{
      Produces tensor filled with given constant value.

      Examples:
        %0 = "ttir.constant"() {value = dense<0> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
        // %0: [[0, 0, 0], [0, 0, 0]]
        %1 = "ttir.constant"() {value = dense<[0.2, 1.3]> : tensor<2xf32>} : () -> tensor<2xf32>
        // %1: [0.2, 1.3]
    }];

    let arguments = (ins ElementsAttr:$value);

    let results = (outs AnyRankedTensor:$result);

    let hasFolder = 1;
}

def TTIR_FillOp : TTIR_NamedOp<"fill", [AllShapesMatch<["value", "result"]>]> {
    let summary = "Fill operation.";
    let description = [{
      Produces tensor filled with given fill value.

      Examples:
        %0 = tensor.empty() : () -> tensor<2x3xi32>
        %1 = "ttir.fill"(%0) {value = dense<0> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
        %2 = tensor.empty() : () -> tensor<2xf32>
        %3 = "ttir.fill"(%2) {value = dense<[0.2, 1.3]> : tensor<2xf32>} : () -> tensor<2xf32>
    }];

    let arguments = (ins AnyRankedTensor:$output,
                         ElementsAttr:$value);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];
}

def TTIR_LinearOp : TTIR_NamedOp<"linear"> {
    let summary = "Linear transformation of inputs.";
    let description = [{
      Produces the matmul of tensors `a` and `b` with optional addition with `bias`.

      Example:
        %a = tensor.empty() : () -> tensor<10x64x32xbf16>
        %b = tensor.empty() : () -> tensor<32x128xbf16>
        %bias = tensor.empty() : () -> tensor<128xbf16>
        %output = tensor.empty() : () -> tensor<10x64x128xbf16>
        %0 = "ttir.linear"(%a, %b, %bias, %output) : (tensor<10x64x32xbf16>, tensor<32x128xbf16>, tensor<128xbf16>, tensor<10x64x128xbf16>) -> tensor<10x64x128xbf16>
    }];

    let arguments = (ins AnyRankedTensor:$a,
                         AnyRankedTensor:$b,
                         Optional<AnyRankedTensor>:$bias,
                         AnyRankedTensor:$output,
                         DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
                         DefaultValuedAttr<BoolAttr, "false">:$transpose_b);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;

    let hasCanonicalizer = 1;
}

// ANCHOR: adding_an_op_matmul_ttir
def TTIR_MatmulOp : TTIR_NamedOp<"matmul"> {
    let summary = "Matrix multiply operation.";
    let description = [{
      Matrix multiply operation.
    }];

    let arguments = (ins AnyRankedTensor:$a,
                         AnyRankedTensor:$b,
                         AnyRankedTensor:$output,
                         DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
                         DefaultValuedAttr<BoolAttr, "false">:$transpose_b);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;

    let hasCanonicalizer = 1;
}
// ANCHOR_END: adding_an_op_matmul_ttir

def TTIR_PermuteOp : TTIR_NamedOp<"permute"> {
    let summary = "Permute operation.";
    let description = [{
      Permute input tensor dimensions.

      Attributes:
        - `permutation` array<i64>: The permutation of the input tensor dimensions.

      Example:
      %a = tensor.empty() : () -> tensor<2x3x4xi32>
      %output = tensor.empty() : () -> tensor<3x4x2xi32>
      %0 = "ttir.permute"(%a, %output) {permutation = array<i64: 1, 2, 0>} : (tensor<2x3x4xi32>, tensor<3x4x2xi32>) -> tensor<3x4x2xi32>
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         DenseI64ArrayAttr:$permutation);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;

    let hasCanonicalizer = 1;
}

def TTIR_Upsample2dOp : TTIR_NamedOp<"upsample2d"> {
    let summary = "Upsample 2D operation.";

    let description = [{
      Upsample 2D operation. Input tensor is assumed to be in NHWC format.

      Attributes:
      - `scale_factor` (si32 | array<i32>): The scale factor for upsampling in H and W dimensions respectively.
      - `mode` (str): The upsampling algorithm. Currently only "nearest" and "bilinear" are supported. Default is "nearest".

      Example:
        %a = tensor.empty() : () -> tensor<10x64x32x3xbf16>
        %output = tensor.empty() : () -> tensor<10x128x128x3xbf16>
        %0 = "ttir.upsample2d"(%a, %output) <{scale_factor = array<i32: 2, 4>}> : (tensor<10x64x32x3xbf16>, tensor<10x128x128x3xbf16>) -> tensor<10x64x128x3xbf16>
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         AnyAttrOf<[SI32Attr, DenseI32ArrayAttr]>:$scale_factor,
                         DefaultValuedAttr<StrAttr, "\"nearest\"">:$mode);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TTIR top level generic ops
//===----------------------------------------------------------------------===//

class TTIR_GenericElementwiseUnaryOp<string mnemonic, list<Trait> traits = []> :
    TTIR_ElementwiseUnaryOp<mnemonic, !listconcat([TTIR_GenericRegionOpInterface], traits)> {

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

      void buildGenericRegion(::mlir::OpBuilder &opBuilder, ::mlir::Block* block);

      std::pair<::mlir::ArrayAttr, ::mlir::ArrayAttr> getIndexingMaps(Builder &builder) {
        assert(sameRank(getOperation()->getOperands()) &&
               "Input and output operand must have the same rank");

        auto rank = mlir::cast<RankedTensorType>(getOperation()->getOperand(0).getType()).getRank();

        SmallVector<AffineMap> indexingMaps(2, builder.getMultiDimIdentityMap(rank));
        SmallVector<Attribute> iteratorTypes(
            rank, builder.getAttr<IteratorTypeAttr>(IteratorType::Parallel));

        return {builder.getAffineMapArrayAttr(indexingMaps),
                builder.getArrayAttr(iteratorTypes)};
      }
    }];
}

def TTIR_ExpOp: TTIR_GenericElementwiseUnaryOp<"exp"> {
    let summary = "Eltwise exponential op.";
    let description = [{
      Eltwise exponential operation. Calculates e^x for all elements x in input tensor.
    }];
}

class TTIR_GenericElementwiseBinaryOp<string mnemonic, list<Trait> traits = []> :
    TTIR_ElementwiseBinaryOp<mnemonic, !listconcat([TTIR_GenericRegionOpInterface], traits)> {

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

      void buildGenericRegion(::mlir::OpBuilder &opBuilder, ::mlir::Block* block);

      std::pair<::mlir::ArrayAttr, ::mlir::ArrayAttr> getIndexingMaps(Builder &builder) {
        assert(sameRank(getOperation()->getOperands()) &&
               "For now all operands must have the same rank");
        auto rank = mlir::cast<RankedTensorType>(getOperation()->getOperand(0).getType()).getRank();
        SmallVector<AffineMap> indexingMaps(getOperation()->getNumOperands(),
                                            builder.getMultiDimIdentityMap(rank));
        SmallVector<Attribute> iteratorTypes(
            rank, builder.getAttr<IteratorTypeAttr>(IteratorType::Parallel));
        return {builder.getAffineMapArrayAttr(indexingMaps),
                builder.getArrayAttr(iteratorTypes)};
      }
    }];
}

def TTIR_AddOp : TTIR_GenericElementwiseBinaryOp<"add", [TTIR_FullyBroadcastable]> {
    let summary = "Eltwise add.";
    let description = [{
      Eltwise add operation.
    }];
}

def TTIR_MultiplyOp : TTIR_GenericElementwiseBinaryOp<"multiply", [TTIR_PartiallyBroadcastable]> {
    let summary = "Eltwise multiply.";
    let description = [{
      Eltwise multiply operation.
    }];
}

def TTIR_DivOp : TTIR_GenericElementwiseBinaryOp<"div", [TTIR_PartiallyBroadcastable]> {
    let summary = "Eltwise divide.";
    let description = [{
      Eltwise divide operation.
    }];
}

def TTIR_MaximumOp : TTIR_GenericElementwiseBinaryOp<"maximum"> {
    let summary = "Eltwise maximum.";
    let description = [{
      Calculates maximum of input tensors' values element-wise and stores result in output tensor.

      Example:
        %lhs: [[3, 2, 7], [1, 4, 4]]
        %rhs: [[1, 4, 2], [1, 2, 3]]
        "ttir.maximum"(%lhs, %rhs, %out) -> %out: [[3, 4, 7], [1, 4, 4]]
    }];
}

//===----------------------------------------------------------------------===//

def TTIR_ScatterOp: TTIR_NamedOp<"scatter"> {
  let summary = "Scatter operation";
  let description = [{
    Produces a 'result' tensor which are equal to `input` tensor except that
    several slices specified by `scatter_indices` are updated with the values
    `updates`.
    }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$scatter_indices,
                       AnyRankedTensor:$update,
                       AnyRankedTensor:$output,
                       DenseI32ArrayAttr:$update_window_dims,
                       DenseI32ArrayAttr:$inserted_window_dims,
                       DenseI32ArrayAttr:$input_batching_dims,
                       DenseI32ArrayAttr:$scatter_indices_batching_dims,
                       DenseI32ArrayAttr:$scatter_dims_to_operand_dims,
                       I32Attr:$index_vector_dim,
                       BoolAttr:$indices_are_sorted,
                       BoolAttr:$unique_indices);

  let regions = (region SizedRegion<1>:$update_computation);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
  }];
}

//===----------------------------------------------------------------------===//
// TTIR region ops (ops that may appear inside of ttir.generic region)
//    DEPRECATED: Generic region ops are moving to TTIRGenericRegionOps.td
//===----------------------------------------------------------------------===//

def TTIR_KernelOp : TTIR_NamedOp<"kernel", [AttrSizedOperandSegments]> {
    let summary = "Kernel call.";
    let description = [{
      A generic kernel call operation. This operation is used to pattern match by some consuming backend.
    }];

    let arguments = (ins FlatSymbolRefAttr:$op,
                         FlatSymbolRefAttr:$kind,
                         Variadic<AnyRankedTensorOrMemRef>:$inputs,
                         Variadic<AnyRankedTensorOrMemRef>:$outputs);
    let results = (outs Variadic<AnyRankedTensorOrMemRef>:$results);
}

//===----------------------------------------------------------------------===//
// TTIR ccl ops
//===----------------------------------------------------------------------===//

def TTIR_AllGatherOp : TTIR_NamedOp<"all_gather"> {
    let summary = "All gather operation.";
    let description = [{
      All gather op.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         SI32Attr:$all_gather_dim,
                         UI32Attr:$cluster_axis);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTIR_AllReduceOp : TTIR_NamedOp<"all_reduce"> {
    let summary = "AllReduce operation.";
    let description = [{
      AllReduce op.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         TT_ReduceTypeAttr:$reduce_type,
                         UI32Attr:$cluster_axis);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTIR_MeshShardOp : TTIR_NamedOp<"mesh_shard"> {
    let summary = "Mesh shard operation.";
    let description = [{
      MeshShard op shards the inputs (FullToShard) or concatnates the outputs (ShardToFull) for ccl ops.

      shard_direction attribute determines whether to shard or concat.

      shard_type attribute determines how to shard or concat.
        manual: no sharding
        replicate: all devices have identical data
        maximal: only one device contains full data
        devices: shard_shape/shard_dims determine particular sharding

      shard_dims attribute determines row and column sharding dimension of input tensor

      For example, on 2x4 mesh hardware, following op shards arg0 to 8 slices, row divided by 2
      and col divided by 4.

        %1 = "ttir.mesh_shard"(%arg0, %0) <
          {... shard_direction = #tt.shard_direction<full_to_shard>,
               shard_shape = array<i64: 2, 4>,
               shard_dims = array<i64: 0, 1>,
               shard_type = #tt.shard_type<devices>}> :  (tensor<8192x784xf32>, ...) -> tensor<4096x196xf32>

      On the other hand, this op concatnates %4 to single tensor by concatnating
      one of the top row tensor with one of the bottom row tensor.

        %6 = "ttir.mesh_shard"(%4, %5) <
          {..., shard_direction = #tt.shard_direction<shard_to_full>,
                shard_shape = array<i64: 2, 1>,
                shard_dims = arrray<i64: 1, -1>,
                shard_type = #tt.shard_type<devices>}> : (tensor<4096x16384xf32>, ...) -> tensor<8192x16384xf32>
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         TT_MeshShardTypeAttr:$shard_type,
                         TT_MeshShardDirectionAttr:$shard_direction,
                         DenseI64ArrayAttr:$shard_shape,
                         DenseI64ArrayAttr:$shard_dims
    );

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

#endif
