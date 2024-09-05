// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIROPS_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIROPS_TD

include "ttmlir/Dialect/TT/IR/TTOpsTypes.td"
include "ttmlir/Dialect/TTIR/IR/TTIRBase.td"
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
      expected to have the same signature as the op itself. The op is expected to be lowered
      to a backend specific form by a consuming backend. This op is heavily inspired by the
      linalg.generic op so it can be useful to refer to linalg.generic documentation for more
      details.

      ```llvm
      %5 = "ttir.generic"(%1, %3, %4, %2) <{
        grid = #tt.grid<1x1>,                        // The grid range of cores to dispatch work to.
        indexing_maps = [#map, #map, #map],          // Affine maps for indexing into the input/output tensors. See linalg.generic
        iterator_types = [#parallel, #parallel],     // Iterator types for the input/output tensors. See linalg.generic
        operandSegmentSizes = array<i32: 2, 1, 1>,   // Sizes of the operand segments, i.e. 2 inputs, 1 output and 1 cb.
        operand_cb_mapping = array<i64: -1, 0, -1>,  // Mapping of input & output operands to cbs. -1 means no mapping.
                                                     // Mapped operands correspond to buffers in streaming mode.
                                                     // Non-mapped operands correspond to buffers in alias mode.
      ({
      ^bb0(%arg2: tensor<64x128xf32, #tt.buffer<memref<64x128xf32, #l1_>, alias>>,
           %arg3: tensor<64x128xf32, #tt.buffer<memref<64x128xf32, #l1_>, stream>>,
           %arg4: tensor<64x128xf32, #tt.buffer<memref<64x128xf32, #l1_>, alias>>):
          // Region body, would contain some computation that represents the work each core does.
      }) : (tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
      ```
    }];

    let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                         Variadic<AnyRankedTensor>:$outputs,
                         Variadic<AnyRankedTensor>:$cbs,
                         TT_GridAttr:$grid,
                         AffineMapArrayAttr:$indexing_maps,
                         TT_IteratorTypeArrayAttr:$iterator_types,
                         TT_OperandConstraintArrayAttr:$operand_constraints,
                         DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$operand_cb_mapping); // index of input operand and index of cb go together
    let results = (outs Variadic<AnyRankedTensor>:$results);
    let regions = (region AnyRegion:$region);
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
      // Returns a tuple of booleans indicating if the op changes layout, grid, format, or memory space.
      std::tuple<bool, bool, bool, bool> compoundComponents();
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
//  A named op is one that is not generic and has a specific name. For example
//  "add", "multiply", "matmul", "concat", etc. They do not carry a region.
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

def TTIR_DivOp : TTIR_GenericElementwiseBinaryOp<"div"> {
    let summary = "Eltwise divide.";
    let description = [{
      Eltwise divide operation.
    }];
}

def TTIR_SubtractOp : TTIR_ElementwiseBinaryOp<"subtract"> {
    let summary = "Eltwise subtract.";
    let description = [{
      Eltwise subtract operation.
    }];
}

def TTIR_MultiplyOp : TTIR_GenericElementwiseBinaryOp<"multiply"> {
    let summary = "Eltwise multiply.";
    let description = [{
      Eltwise multiply operation.
    }];
}

def TTIR_GreaterEqualOp :  TTIR_ElementwiseBinaryOp<"ge"> {
    let summary = "Eltwise greater than or equal to.";
    let description = [{
      Eltwise greater than or equal to operation.
    }];
}

def TTIR_SqrtOp : TTIR_ElementwiseUnaryOp<"sqrt"> {
    let summary = "Eltwise square root.";
    let description = [{
      Eltwise square root operation.
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

def TTIR_SigmoidOp: TTIR_ElementwiseUnaryOp<"sigmoid"> {
    let summary = "Eltwise sigmoid.";
    let description = [{
      Eltwise sigmoid operation.
    }];
}

def TTIR_ExpOp: TTIR_ElementwiseUnaryOp<"exp"> {
    let summary = "Eltwise exponential op.";
    let description = [{
      Eltwise exponential operation.
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
