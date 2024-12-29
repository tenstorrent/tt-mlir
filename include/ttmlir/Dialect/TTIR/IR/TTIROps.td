// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIROPS_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIROPS_TD

include "ttmlir/Dialect/TT/IR/TTOpsTypes.td"
include "ttmlir/Dialect/TTIR/IR/TTIRBase.td"
include "ttmlir/Dialect/TTIR/IR/TTIROpsAttrs.td"
include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.td"
include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"

class TTIR_DPSOp<string mnemonic, list<Trait> traits = []> :
    TTIR_Op<mnemonic, !listconcat(traits, [TTIROpInterface, DestinationStyleOpInterface])> {
    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }
    }];
}

//===----------------------------------------------------------------------===//
// TTIR top level ops
//===----------------------------------------------------------------------===//

def TTIR_GenericOp : TTIR_DPSOp<"generic", [AttrSizedOperandSegments]> {
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

    let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                         Variadic<AnyRankedTensor>:$cbs,
                         Variadic<AnyRankedTensor>:$outputs,
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

  let hasFolder = 1;
  let hasVerifier = 1;
}

def TTIR_ToLayoutOp : TTIR_Op<"to_layout", [DestinationStyleOpInterface, TTIROpInterface]> {
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

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output);
    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }

      struct CompoundComponents {
        bool isLayoutChange;
        bool isGridChange;
        bool isFormatChange;
        bool isMemorySpaceChange;
        bool isMemoryLayoutChange;
      };

      // Returns booleans indicating if the op changes layout, grid, format, memory space or memory layout.
      CompoundComponents compoundComponents();
    }];

    let hasVerifier = 1;
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

def TwoOperands : ParamNativeOpTrait<"NOperands", "2">;
def ThreeOperands : ParamNativeOpTrait<"NOperands", "3">;
def FourOperands : ParamNativeOpTrait<"NOperands", "4">;

class TTIR_ElementwiseOp<string mnemonic, list<Trait> traits = []> :
    TTIR_DPSOp<mnemonic, !listconcat(traits, [AttrSizedOperandSegments, TTIR_Broadcastable])> {

    let description = [{
      Base class for elementwise operations. Elementwise operations can take inputs with different shape,
      as long as the input tensors are broadcastable to the same shape.
    }];

    let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                         Variadic<AnyRankedTensor>:$outputs);
    let results = (outs Variadic<AnyRankedTensor>:$results);
}

class TTIR_ElementwiseTernaryOp<string mnemonic, list<Trait> traits = []> :
    TTIR_ElementwiseOp<mnemonic, !listconcat(traits, [FourOperands])> {
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

def TTIR_WhereOp: TTIR_ElementwiseTernaryOp<"where"> {
    let summary = "Eltwise where op.";
    let description = [{
      Eltwise where operation.
    }];
}

class TTIR_ElementwiseUnaryOp<string mnemonic, list<Trait> traits = []> :
    TTIR_ElementwiseOp<mnemonic, !listconcat(traits, [TwoOperands])> {
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

def TTIR_AbsOp: TTIR_ElementwiseUnaryOp<"abs"> {
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

def TTIR_CeilOp: TTIR_ElementwiseUnaryOp<"ceil"> {
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

def TTIR_FloorOp: TTIR_ElementwiseUnaryOp<"floor"> {
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

def TTIR_BitwiseNotOp : TTIR_ElementwiseUnaryOp<"bitwise_not"> {
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

def TTIR_NegOp: TTIR_ElementwiseUnaryOp<"neg"> {
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

def TTIR_ReciprocalOp : TTIR_ElementwiseUnaryOp<"reciprocal"> {
    let summary = "Eltwise reciprocal.";
    let description = [{
      Eltwise reciprocal operation.
    }];
}

def TTIR_ReluOp : TTIR_ElementwiseUnaryOp<"relu"> {
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

def TTIR_SignOp: TTIR_ElementwiseUnaryOp<"sign"> {
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
    TTIR_ElementwiseOp<mnemonic, !listconcat(traits, [ThreeOperands])> {
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

def TTIR_EqualOp : TTIR_ElementwiseBinaryOp<"eq"> {
    let summary = "Eltwise equal to.";
    let description = [{
      Eltwise equal to operation.
    }];
}

def TTIR_NotEqualOp : TTIR_ElementwiseBinaryOp<"ne"> {
    let summary = "Eltwise not equal to.";
    let description = [{
      Eltwise not equal to operation.
    }];
}

def TTIR_GreaterEqualOp : TTIR_ElementwiseBinaryOp<"ge"> {
    let summary = "Eltwise greater than or equal to.";
    let description = [{
      Eltwise greater than or equal to operation.
    }];
}

def TTIR_GreaterThanOp : TTIR_ElementwiseBinaryOp<"gt"> {
    let summary = "Eltwise greater than.";
    let description = [{
      Eltwise greater than operation.
    }];
}

def TTIR_LessEqualOp : TTIR_ElementwiseBinaryOp<"le"> {
    let summary = "Eltwise less than or equal to.";
    let description = [{
      Eltwise less than or equal to operation.
    }];
}

def TTIR_LessThanOp : TTIR_ElementwiseBinaryOp<"lt"> {
    let summary = "Eltwise less than.";
    let description = [{
      Eltwise less than operation.
    }];
}

def TTIR_LogicalAndOp : TTIR_ElementwiseBinaryOp<"logical_and"> {
    let summary = "Eltwise logical and.";
    let description = [{
      Eltwise logical and operation.
    }];
}

def TTIR_LogicalOrOp : TTIR_ElementwiseBinaryOp<"logical_or"> {
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

def TTIR_BitwiseAndOp : TTIR_ElementwiseBinaryOp<"bitwise_and"> {
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

def TTIR_BitwiseOrOp : TTIR_ElementwiseBinaryOp<"bitwise_or"> {
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
}

def TTIR_MinimumOp :  TTIR_ElementwiseBinaryOp<"minimum"> {
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

def TTIR_SubtractOp : TTIR_ElementwiseBinaryOp<"subtract"> {
    let summary = "Eltwise subtract.";
    let description = [{
      Eltwise subtract operation.
    }];
}

def TTIR_RemainderOp : TTIR_ElementwiseBinaryOp<"remainder"> {
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

class TTIR_ReductionOp<string mnemonic, list<Trait> traits = []> :
    TTIR_DPSOp<mnemonic, !listconcat(traits, [TTIR_GenericRegionOpInterface])> {

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

def TTIR_EmbeddingOp : TTIR_DPSOp<"embedding"> {
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

def TTIR_EmbeddingBackwardOp : TTIR_DPSOp<"embedding_backward"> {
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

def TTIR_SoftmaxOp : TTIR_DPSOp<"softmax"> {
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

def TTIR_TransposeOp : TTIR_DPSOp<"transpose"> {
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
}

def TTIR_ConcatOp : TTIR_DPSOp<"concat"> {
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

def TTIR_UpdateCacheOp : TTIR_DPSOp<"update_cache"> {
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

def TTIR_FillCacheOp : TTIR_DPSOp<"fill_cache"> {
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

def TTIR_BroadcastOp : TTIR_DPSOp<"broadcast"> {
    let summary = "Broadcast operation.";
    let description = [{
      Broadcast op.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         I64ArrayAttr:$dimension);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];
}

def TTIR_Conv2dOp : TTIR_DPSOp<"conv2d"> {
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

def TTIR_ConvolutionOp : TTIR_DPSOp<"convolution"> {
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

def TTIR_GatherOp: TTIR_DPSOp<"gather"> {
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

def TTIR_PoolingOp : TTIR_DPSOp<"pooling", [AttrSizedOperandSegments]> {
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

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }
    SmallVector<uint32_t> getIndicesOfSpatialDims();
    uint32_t countSpatialDims();
  }];

  let hasVerifier = 1;
}

def TTIR_MaxPool2dOp : TTIR_DPSOp<"max_pool2d"> {
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
                         SI32Attr:$padding_bottom,
                         DefaultValuedAttr<I64Attr, "0">:$channel_last);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTIR_ReshapeOp: TTIR_DPSOp<"reshape"> {
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

def TTIR_SliceOp: TTIR_DPSOp<"slice"> {
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

def TTIR_SelectOp: TTIR_DPSOp<"select"> {
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
def TTIR_IndexOp: TTIR_DPSOp<"index"> {
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

def TTIR_SqueezeOp : TTIR_DPSOp<"squeeze"> {
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

def TTIR_UnsqueezeOp : TTIR_DPSOp<"unsqueeze"> {
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

def TTIR_ClampOp : TTIR_DPSOp<"clamp"> {
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

def TTIR_ReverseOp : TTIR_DPSOp<"reverse", [AllShapesMatch<["input", "result"]>]> {
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

def TTIR_FillOp : TTIR_DPSOp<"fill", [AllShapesMatch<["value", "result"]>]> {
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

def TTIR_LinearOp : TTIR_DPSOp<"linear"> {
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
                         AnyRankedTensor:$output);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

// ANCHOR: adding_an_op_matmul_ttir
def TTIR_MatmulOp : TTIR_DPSOp<"matmul"> {
    let summary = "Matrix multiply operation.";
    let description = [{
      Matrix multiply operation.
    }];

    let arguments = (ins AnyRankedTensor:$a,
                         AnyRankedTensor:$b,
                         AnyRankedTensor:$output);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}
// ANCHOR_END: adding_an_op_matmul_ttir

def TTIR_PermuteOp : TTIR_DPSOp<"permute"> {
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

    let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// TTIR top level generic ops
//===----------------------------------------------------------------------===//

class TTIR_GenericElementwiseUnaryOp<string mnemonic, list<Trait> traits = []> :
    TTIR_ElementwiseUnaryOp<mnemonic, !listconcat(traits, [TTIR_GenericRegionOpInterface])> {

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
    TTIR_ElementwiseBinaryOp<mnemonic, !listconcat(traits, [TTIR_GenericRegionOpInterface])> {

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

def TTIR_AddOp : TTIR_GenericElementwiseBinaryOp<"add"> {
    let summary = "Eltwise add.";
    let description = [{
      Eltwise add operation.
    }];
}

def TTIR_MultiplyOp : TTIR_GenericElementwiseBinaryOp<"multiply"> {
    let summary = "Eltwise multiply.";
    let description = [{
      Eltwise multiply operation.
    }];
}

def TTIR_DivOp : TTIR_GenericElementwiseBinaryOp<"div"> {
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

def TTIR_ScatterOp: TTIR_DPSOp<"scatter"> {
  let summary = "Scatter operation";
  let description = [{
    Produces a 'result' tensor which are equal to `input` tensor except that
    several slices specified by `scatter_indices` are updated with the values
    `updates`.
    }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$scatter_indices,
                       AnyRankedTensor:$update,
                       DenseI32ArrayAttr:$update_window_dims,
                       DenseI32ArrayAttr:$inserted_window_dims,
                       DenseI32ArrayAttr:$input_batching_dims,
                       DenseI32ArrayAttr:$scatter_indices_batching_dims,
                       DenseI32ArrayAttr:$scatter_dims_to_operand_dims,
                       I32Attr:$index_vector_dim,
                       BoolAttr:$indices_are_sorted,
                       BoolAttr:$unique_indices,
                       AnyRankedTensor:$output);

  let regions = (region SizedRegion<1>:$update_computation);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
  }];
}

//===----------------------------------------------------------------------===//
// TTIR region ops (ops that may appear inside of ttir.generic region)
//===----------------------------------------------------------------------===//

def AnyRankedTensorOrMemRef: AnyTypeOf<[AnyRankedTensor, AnyNon0RankedMemRef]>;

def TTIR_KernelOp : TTIR_DPSOp<"kernel", [AttrSizedOperandSegments]> {
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

def TTIR_YieldOp : TTIR_Op<"yield", [Pure, ReturnLike, Terminator]> {
    let summary = "Yield op.";
    let description = [{
      Yield operation, this is required by MLIR to mark the end of a dispatch region.
    }];

    let arguments = (ins Variadic<AnyRankedTensorOrMemRef>:$values);
}

//===----------------------------------------------------------------------===//
// TTIR ccl ops
//===----------------------------------------------------------------------===//

def TTIR_AllGatherOp : TTIR_DPSOp<"all_gather"> {
    let summary = "All gather operation.";
    let description = [{
      All gather op.
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

def TTIR_AllReduceOp : TTIR_DPSOp<"all_reduce"> {
    let summary = "AllReduce operation.";
    let description = [{
      AllReduce op.
    }];

    let arguments = (ins
      Variadic<AnyRankedTensor>:$inputs,
      AnyRankedTensor:$output,
      I64ElementsAttr:$replica_groups,
      SI32Attr:$dim,
      OptionalAttr<SI32Attr>:$channel_handle,
      UnitAttr:$use_global_device_ids,
      TT_ReduceTypeAttr:$reduce_type
    );

    let results = (outs Variadic<AnyRankedTensor>:$results);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTIR_MeshShardOp : TTIR_DPSOp<"mesh_shard"> {
    let summary = "Mesh shard operation.";
    let description = [{
      MeshShard op shards the inputs (FullToShard) or concatnates the outputs (ShardToFull) for ccl ops.

      shard_direction attribute determines whether to shard or concat.

      shard_type attribute determines how to shard or concat.
        manual: no sharding
        replicate: all devices have identical data
        maximal: only one device contains full data
        devices: shard_shape determines sharded dimensions

      For example, on 2x4 mesh hardware, following op shards arg0 to 8 slices, row divided by 2
      and col divided by 4.

        %1 = "ttir.mesh_shard"(%arg0, %0) <
          {... shard_direction = #tt.shard_direction<full_to_shard>,
               shard_shape = #tt.grid<2x4>,
               shard_type = #tt.shard_type<devices>}> :  (tensor<8192x784xf32>, ...) -> tensor<4096x196xf32>

      On the other hand, this op concatnates %4 to single tensor by concatnating
      one of the top row tensor with one of the bottom row tensor.

        %6 = "ttir.mesh_shard"(%4, %5) <
          {..., shard_direction = #tt.shard_direction<shard_to_full>,
                shard_shape = #tt.grid<2x1>,
                shard_type = #tt.shard_type<devices>}> : (tensor<4096x16384xf32>, ...) -> tensor<8192x16384xf32>
    }];

    let arguments = (ins
      AnyRankedTensor:$input,
      AnyRankedTensor:$output,
      TT_MeshShardTypeAttr:$shard_type,
      TT_MeshShardDirectionAttr:$shard_direction,
      TT_GridAttr:$shard_shape
    );

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

#endif
