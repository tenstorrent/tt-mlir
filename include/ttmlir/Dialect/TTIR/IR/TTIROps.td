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
                         TT_OperandConstraintArrayAttr:$operand_constraints,
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
      #layout = #tt.layout<8192x128x1, undef, <1x1>, memref<64x128xf32, #system>>
      #layout1 = #tt.layout<8192x128x1, undef, <1x1>, memref<64x128xf32, #l1_>>
      %1 = "ttir.to_layout"(%arg0, %0) : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
      ```
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output);
    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
      ArrayAttr getOperandConstraints() {
        return nullptr;
        // TODO return below, but we need a way to properly create an ArrayAttr:
        // return {OperandConstraint::Any, OperandConstraint::Any};
      }

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

class TTIR_ElementwiseOp<string mnemonic, list<Trait> traits = []> :
    TTIR_DPSOp<mnemonic, !listconcat(traits, [AttrSizedOperandSegments, TTIR_ElementwiseOpInterface])> {

    let description = [{
      Base class for elementwise operations. Elementwise operations can take inputs with different shape,
      as long as the input tensors are broadcastable to the same shape.
    }];

    let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                         Variadic<AnyRankedTensor>:$outputs,
                         TT_OperandConstraintArrayAttr:$operand_constraints);
    let results = (outs Variadic<AnyRankedTensor>:$results);
}

class TTIR_ElementwiseUnaryOp<string mnemonic, list<Trait> traits = []> :
    TTIR_ElementwiseOp<mnemonic, traits> {
    let summary = "Eltwise unary op.";
    let description = [{
      Eltwise unary op.
    }];

    let builders =
    [
      OpBuilder<(ins "Value": $in, "Value": $out, "ArrayAttr": $operand_constraints),
      [{
        build($_builder, $_state, {out.getType()}, in, out, operand_constraints);
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

def TTIR_LogicalNotOp: TTIR_ElementwiseUnaryOp<"logical_not"> {
    let summary = "Eltwise logical not op.";
    let description = [{
      Eltwise logical not operation.
    }];
}

def TTIR_NegOp: TTIR_ElementwiseUnaryOp<"neg"> {
    let summary = "Eltwise negate op.";
    let description = [{
      Eltwise negate operation.
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

class TTIR_ElementwiseBinaryOp<string mnemonic, list<Trait> traits = []> :
    TTIR_ElementwiseOp<mnemonic, traits> {
    let summary = "Eltwise binary op.";
    let description = [{
      Eltwise binary op.
    }];

    let builders =
    [
      OpBuilder<(ins "Value": $lhs, "Value": $rhs, "Value": $out, "ArrayAttr": $operand_constraints),
      [{
        build($_builder, $_state, {out.getType()}, {lhs, rhs}, out, operand_constraints);
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

def TTIR_MaximumOp :  TTIR_ElementwiseBinaryOp<"maximum"> {
    let summary = "Eltwise maximum OP.";
    let description = [{
      Calculates maximum of input tensors' values element-wise and stores result in output tensor.

      Example:
        %lhs: [[3, 2, 7], [1, 4, 4]]
        %rhs: [[1, 4, 2], [1, 2, 3]]
        "ttir.maximum"(%lhs, %rhs, %out) -> %out: [[3, 4, 7], [1, 4, 4]]
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

class TTIR_ReductionOp<string mnemonic, list<Trait> traits = []> :
    TTIR_DPSOp<mnemonic, !listconcat(traits, [TTIR_GenericRegionOpInterface])> {

    let summary = "Reduction op.";
    let description = [{
      Reduction op.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         BoolAttr:$keep_dim,
                         OptionalAttr<I32ArrayAttr>:$dim_arg,
                         TT_OperandConstraintArrayAttr:$operand_constraints);

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
                         AnyRankedTensor:$output,
                         TT_OperandConstraintArrayAttr:$operand_constraints);

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
                         SI32Attr:$dimension,
                         TT_OperandConstraintArrayAttr:$operand_constraints);

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
                         SI32Attr:$dim1,
                         TT_OperandConstraintArrayAttr:$operand_constraints);

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
                         SI32Attr:$dim,
                         TT_OperandConstraintArrayAttr:$operand_constraints);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTIR_BroadcastOp : TTIR_DPSOp<"broadcast"> {
    let summary = "Broadcast operation.";
    let description = [{
      Broadcast op.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         I64ArrayAttr:$dimension,
                         TT_OperandConstraintArrayAttr:$operand_constraints);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];
}

// CCL ops
def TTIR_AllGatherOp : TTIR_DPSOp<"all_gather"> {
    let summary = "All gather operation.";
    let description = [{
      All gather op.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         SI32Attr:$dim,
                         TT_OperandConstraintArrayAttr:$operand_constraints);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
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
                         SI32Attr:$padding_bottom,
                         TT_OperandConstraintArrayAttr:$operand_constraints);

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
    // Default value: one for each of the spatial dimension.
    DefaultValuedOptionalAttr<DenseI64ArrayAttr, "SmallVector<int64_t>(getConvolutionLayout().getInputSpatialDimensions().size(), 1)">:$window_strides,
    // Default value: two zeros for each of the spatial dimension.
    DefaultValuedOptionalAttr<I64ElementsAttr, "SmallVector<int64_t>(getConvolutionLayout().getInputSpatialDimensions().size()*2, 0)">:$padding,
    // Default value: one for each of the spatial dimension.
    DefaultValuedOptionalAttr<DenseI64ArrayAttr, "SmallVector<int64_t>(getConvolutionLayout().getInputSpatialDimensions().size(), 1)">:$input_dilation,
    // Default value: one for each of the spatial dimension.
    DefaultValuedOptionalAttr<DenseI64ArrayAttr, "SmallVector<int64_t>(getConvolutionLayout().getInputSpatialDimensions().size(), 1)">:$weight_dilation,
    // Default value: false for each of the spatial dimension.
    DefaultValuedOptionalAttr<DenseBoolArrayAttr, "SmallVector<bool>(getConvolutionLayout().getInputSpatialDimensions().size(), false)">:$window_reversal,
    TTIR_ConvolutionLayoutAttr:$convolution_layout,
    ConfinedAttr<I64Attr, [IntPositive]>:$feature_group_count,
    ConfinedAttr<I64Attr, [IntPositive]>:$batch_group_count,
    TT_OperandConstraintArrayAttr:$operand_constraints
  );

  let results = (outs AnyRankedTensor);
  let hasVerifier = 1;

  let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];
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
                         TT_OperandConstraintArrayAttr:$operand_constraints,
                         OptionalAttr<SI32Attr>:$original_height,
                         OptionalAttr<SI32Attr>:$original_width);

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
                         I32ArrayAttr:$shape,
                         TT_OperandConstraintArrayAttr:$operand_constraints);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
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
                         I32ArrayAttr:$step,
                         TT_OperandConstraintArrayAttr:$operand_constraints);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

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
                         I32Attr:$step,
                         TT_OperandConstraintArrayAttr:$operand_constraints);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTIR_SqueezeOp : TTIR_DPSOp<"squeeze"> {
    let summary = "Squeeze op.";
    let description = [{
      Squeeze tensor.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         SI32Attr:$dim,
                         TT_OperandConstraintArrayAttr:$operand_constraints);

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
                         SI32Attr:$dim,
                         TT_OperandConstraintArrayAttr:$operand_constraints);

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
                         ElementsAttr:$value,
                         TT_OperandConstraintArrayAttr:$operand_constraints);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];
}

// ANCHOR: adding_an_op_matmul_ttir
def TTIR_MatmulOp : TTIR_DPSOp<"matmul"> {
    let summary = "Matrix multiply operation.";
    let description = [{
      Matrix multiply operation.
    }];

    let arguments = (ins AnyRankedTensor:$a,
                         AnyRankedTensor:$b,
                         AnyRankedTensor:$output,
                         TT_OperandConstraintArrayAttr:$operand_constraints);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}
// ANCHOR_END: adding_an_op_matmul_ttir

//===----------------------------------------------------------------------===//
// TTIR top level generic ops
//===----------------------------------------------------------------------===//

class TTIR_GenericElementwiseUnaryOp<string mnemonic, list<Trait> traits = []> :
    TTIR_ElementwiseUnaryOp<mnemonic, !listconcat(traits, [TTIR_GenericRegionOpInterface])> {

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

      void buildGenericRegion(::mlir::OpBuilder &opBuilder, ::mlir::Block* block);

      std::pair<::mlir::ArrayAttr, ::mlir::ArrayAttr> getIndexingMaps(Builder &builder) {
        assert(getNumOperands() == 2 && "Input and output operand must have the same rank");
        assert(sameRank(getOperands()) &&
               "Elementwise unary op must have only one input and one output operand.");

        auto rank = mlir::cast<RankedTensorType>(getOperand(0).getType()).getRank();

        SmallVector<AffineMap> indexingMaps(2, builder.getMultiDimIdentityMap(rank));
        SmallVector<Attribute> iteratorTypes(
            rank, builder.getAttr<IteratorTypeAttr>(IteratorType::Parallel));

        return {builder.getAffineMapArrayAttr(indexingMaps),
                builder.getArrayAttr(iteratorTypes)};
      }

      static bool sameRank(mlir::OperandRange operands) {
        if (operands.empty()) {
          return true;
        }
        auto rank = mlir::cast<RankedTensorType>(operands[0].getType()).getRank();
        for (auto operand : operands) {
          if (mlir::cast<RankedTensorType>(operand.getType()).getRank() != rank) {
            return false;
          }
        }
        return true;
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
        assert(sameRank(getOperands()) &&
               "For now all operands must have the same rank");
        auto rank = mlir::cast<RankedTensorType>(getOperand(0).getType()).getRank();
        SmallVector<AffineMap> indexingMaps(getNumOperands(),
                                            builder.getMultiDimIdentityMap(rank));
        SmallVector<Attribute> iteratorTypes(
            rank, builder.getAttr<IteratorTypeAttr>(IteratorType::Parallel));
        return {builder.getAffineMapArrayAttr(indexingMaps),
                builder.getArrayAttr(iteratorTypes)};
      }

      static bool sameRank(mlir::OperandRange operands) {
        if (operands.empty()) {
          return true;
        }
        auto rank = mlir::cast<RankedTensorType>(operands[0].getType()).getRank();
        for (auto operand : operands) {
          if (mlir::cast<RankedTensorType>(operand.getType()).getRank() != rank) {
            return false;
          }
        }
        return true;
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
                         Variadic<AnyRankedTensorOrMemRef>:$outputs,
                         TT_OperandConstraintArrayAttr:$operand_constraints);
    let results = (outs Variadic<AnyRankedTensorOrMemRef>:$results);
}

def TTIR_YieldOp : TTIR_Op<"yield", [Pure, ReturnLike, Terminator]> {
    let summary = "Yield op.";
    let description = [{
      Yield operation, this is required by MLIR to mark the end of a dispatch region.
    }];

    let arguments = (ins Variadic<AnyRankedTensorOrMemRef>:$values);
}

#endif
