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
    TTIR_Op<mnemonic, [TTIROpInterface, DestinationStyleOpInterface] # traits> {
    let extraClassDeclaration = [{
      // base implementation that will detect getOutputMutable() or getOutputsMutable()
      MutableOperandRange getDpsInitsMutable() { return ttir::getDpsOutputs(this); }
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

    let extraClassDefinition = [{
      bool $cppClass::bufferizesToMemoryRead(
          mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
        // If the operand is an input, it is a bufferized to a memory read.
        return isDpsInput(&operand);
      }

      bool $cppClass::bufferizesToMemoryWrite(
          mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
        // If the operand is an output, it is a bufferized to a memory write.
        return isDpsInit(&operand);
      }

      bufferization::AliasingValueList $cppClass::getAliasingValues(
          OpOperand &, const bufferization::AnalysisState &) {
        bufferization::AliasingValueList result;
        return result;
      }

      void $cppClass::getEffects(
          SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
              &effects) {
        ttir::getDpsEffects(*this, effects);
      }
    }];
}

def TTIR_GenericOp : TTIR_BufferizableOp<"generic",
  [ AttrSizedOperandSegments
  , NoTerminator
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames", "getAsmBlockNames"]>
  ]> {
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
        threads = [#ttir.thread<compute>],           // Thread types for the regions.
        operandSegmentSizes = array<i32: 2, 1>       // Sizes of the operand segments, i.e. 2 inputs and 1 output.
      }> ({
      ^bb0(%arg2: memref<64x128xf32, #l1_>,
           %arg3: memref<64x128xf32, #l1_>,
           %arg4: memref<64x128xf32, #l1_>):
          // Region body, would contain some computation that represents the work each core does.
      }) : (tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
      ```
    }];

    let arguments = (ins Variadic<AnyRankedTensorOrMemRef>:$inputs,
                         Variadic<AnyRankedTensorOrMemRef>:$outputs,
                         TT_GridAttr:$grid,
                         I64ArrayAttr:$block_factors,
                         AffineMapArrayAttr:$indexing_maps,
                         TT_IteratorTypeArrayAttr:$iterator_types,
                         TTIR_ThreadArrayAttr:$threads);
    let results = (outs Variadic<AnyRankedTensor>:$results);
    let regions = (region VariadicRegion<AnyRegion>:$regions);
    let hasVerifier = 1;

    let assemblyFormat = [{ attr-dict `\n`
    ` ` ` ` ` ` ` ` `ins` `(` $inputs `:` type($inputs) `)` `\n`
    ` ` ` ` ` ` ` ` `outs` `(` $outputs  `:` type($outputs) `)` ` `  $regions (`:`  type($results)^ )?
    }];

    let builders =
    [
      OpBuilder<(ins "ValueRange": $inputs,
                     "ValueRange": $outputs,
                     "ArrayAttr": $indexingMaps,
                     "ArrayAttr": $iteratorTypes,
                     CArg<"ThreadType", "ThreadType::Compute">: $singleThreadType,
                     CArg<"GridAttr", "nullptr">: $grid,
                     CArg<"ArrayRef<int64_t>", "{}">: $blockFactors),
      [{
        assert(!indexingMaps.empty() && "expected non-empty indexing maps");
        assert(outputs.size() == 1 && "expected single output");
        if (!grid) {
          RankedTensorType tensorType = mlir::dyn_cast<RankedTensorType>(outputs[0].getType());
          if (tensorType) {
            grid = mlir::cast<tt::MetalLayoutAttr>(tensorType.getEncoding()).getGrid();
          } else {
            MemRefType memrefType = mlir::cast<MemRefType>(outputs[0].getType());
            DeviceLayoutInterface layout = mlir::cast<DeviceLayoutInterface>(memrefType.getLayout());
            grid = $_builder.getAttr<tt::GridAttr>(layout.getGridShape(memrefType));
          }
        }
        auto threads = $_builder.getArrayAttr($_builder.getAttr<ThreadAttr>(singleThreadType));
        auto blockFactorsAttr = $_builder.getI64ArrayAttr(blockFactors.empty() ?
                                                          SmallVector<int64_t>(iteratorTypes.size(), 1) :
                                                          blockFactors);
        build($_builder, $_state, TypeRange(outputs), inputs, outputs, grid, blockFactorsAttr, indexingMaps, iteratorTypes, threads, 1);
      }]>,
      OpBuilder<(ins "ValueRange": $inputs,
                     "ValueRange": $outputs,
                     "ArrayAttr": $indexingMaps,
                     "ArrayAttr": $iteratorTypes,
                     "llvm::function_ref<void(OpBuilder&, Location, ValueRange)>": $singleThreadRegionBuilder,
                     CArg<"ThreadType", "ThreadType::Compute">: $singleThreadType,
                     CArg<"GridAttr", "nullptr">: $grid,
                     CArg<"ArrayRef<int64_t>", "{}">: $blockFactors),
      [{
        build($_builder, $_state, inputs, outputs, indexingMaps, iteratorTypes, singleThreadType, grid, blockFactors);
        llvm::SmallVector<Type> blockTypes = llvm::map_to_vector(TypeRange($_state.operands), [&](Type t) -> Type {
          mlir::RankedTensorType tensorType = mlir::cast<RankedTensorType>(t);
          return mlir::cast<tt::MetalLayoutAttr>(tensorType.getEncoding()).getMemref();
        });
        Region& region = *$_state.regions.front().get();
        llvm::SmallVector<mlir::Location> locs($_state.operands.size(), $_state.location);
        OpBuilder::InsertionGuard guard($_builder);
        Block* block = $_builder.createBlock(&region, region.end(), blockTypes, locs);
        singleThreadRegionBuilder($_builder, $_state.location, block->getArguments());
      }]>,
      OpBuilder<(ins "ValueRange": $inputs,
                     "ValueRange": $outputs,
                     "llvm::function_ref<void(OpBuilder&, Location, ValueRange)>": $singleThreadRegionBuilder,
                     CArg<"ThreadType", "ThreadType::Compute">: $singleThreadType,
                     CArg<"GridAttr", "nullptr">: $grid,
                     CArg<"ArrayRef<int64_t>", "{}">: $blockFactors),
      [{
        assert(outputs.size() == 1 && "expected single output");
        RankedTensorType tensorType = mlir::cast<RankedTensorType>(outputs[0].getType());
        MetalLayoutAttr layout = mlir::cast<tt::MetalLayoutAttr>(tensorType.getEncoding());
        auto [indexingMaps, iteratorTypes] = buildParallelAffineMapsAndIteratorTypes($_builder, inputs.size() + outputs.size(), layout.getRank());
        build($_builder, $_state, inputs, outputs, indexingMaps, iteratorTypes, singleThreadRegionBuilder, singleThreadType, grid, blockFactors);
      }]>,
    ];

    let extraClassDeclaration = [{
      static std::pair<ArrayAttr, ArrayAttr> buildParallelAffineMapsAndIteratorTypes(mlir::OpBuilder &builder, size_t arity, size_t rank) {
        auto map = builder.getMultiDimIdentityMap(rank);
        Attribute parallel = builder.getAttr<tt::IteratorTypeAttr>(tt::IteratorType::Parallel);
        return std::make_pair(builder.getAffineMapArrayAttr(SmallVector<AffineMap>(arity, map)),
                              builder.getArrayAttr(SmallVector<Attribute>(rank, parallel)));
      }
      MutableOperandRange getDpsInitsMutable() { return ttir::getDpsOutputs(this); }
      unsigned getNumLoops();
      unsigned getNumDims();
      SmallVector<int64_t> getLoopBounds();
      SmallVector<int64_t> getParticipatingLoopDims(int64_t operandIndex);
      SmallVector<int64_t> getNonParticipatingLoopDims(int64_t operandIndex);
      AffineMap getIndexingMap(int64_t operandIndex);
      SmallVector<AffineMap> getIndexingMapsValue();
      SmallVector<IteratorType> getIteratorTypesValue();
      SmallVector<int64_t> getBlockFactorsValue();
      SmallVector<SmallVector<int64_t>> getOperandGridShapes();
      SmallVector<SmallVector<int64_t>> getOperandShardShapes(bool convertTileToScalar = false);
      ThreadType getRegionThreadType(unsigned regionIndex) {
        return mlir::cast<ThreadAttr>(getThreads()[regionIndex]).getThreadType();
      }
      bool isComputeOnlyForm() {
        auto threads = getThreads();
        return threads.size() == 1 && mlir::cast<ThreadAttr>(threads[0]).getThreadType() == ThreadType::Compute;
      }
      bool isAffineMapForm() { return !getIndexingMaps().empty(); }
      bool isLoweredLoopForm() { return !isAffineMapForm(); }
      bool isExternalSymbolForm() { return getRegions().empty(); }
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
                         AnyRankedTensorOrMemRef:$output,
                         OptionalAttr<TT_MetalLayoutAttr>:$layout);
    let results = (outs Variadic<AnyRankedTensor>:$results);

    let assemblyFormat = [{ $input `,` $output `:` type($input) `into` type($output) (`hostInfo` `=` $layout^)? attr-dict (`->` type($results)^)? }];

    let builders =
    [
      OpBuilder<(ins "Value": $input, "Value": $output, CArg<"MetalLayoutAttr", "nullptr">: $layout),
      [{
        build($_builder, $_state, {output.getType()}, input, output, layout);
      }]>,
    ];

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return ttir::getDpsOutputs(this); }

      struct CompoundComponents {
        bool isLayoutChange = false;
        bool isGridChange = false;
        bool isFormatChange = false;
        bool isMemorySpaceChange = false;

        bool isCompound() {
          return ((isLayoutChange ? 1 : 0) +
                  (isGridChange ? 1 : 0) +
                  (isFormatChange ? 1 : 0) +
                  (isMemorySpaceChange ? 1 : 0)) > 1;
        }
      };

      // Returns booleans indicating if the op changes layout, grid, format, memory space or memory layout.
      CompoundComponents compoundComponents();
      bool isCompound() { return compoundComponents().isCompound(); }
      MetalLayoutAttr getInputLayout();
      MetalLayoutAttr getOutputLayout();
    }];

    let hasVerifier = 1;

    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

def TTIR_StreamLayoutOp : TTIR_Op<"stream_layout",
  [ Pure
  , DeclareOpInterfaceMethods<OpAsmOpInterface , ["getAsmResultNames"]>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                       , "bufferizesToMemoryWrite"
                                                       , "bufferize"
                                                       , "getAliasingValues"
                                                       , "getBufferType"
                                                       ]>
  , TTIR_ViewOpInterface
  ]> {
    let summary = "Stream Layout op.";
    let description = [{
      StreamLayout operation used to form a stream between remote and local memory spaces. Note that this op has
      no side-effects, it's purely representational. The primary use cases include, to enable streaming a large
      tensor out of dram via a small L1 buffer and also as a means for forming reduce or gather multicast
      operations. A stream definition includes:

      - The tensor to be streamed.
      - The storage buffer to be used for streaming.
      - A result, which is also able to take a view over the input, i.e. same semantics as the ViewLayout op.

      Additional constraints:
      - It is not capable of changing the data type nor the memory space of the tensor.

      ```llvm
      %input = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>
      %storage = memref.alloc() {alignment = 64 : i64} : memref<2x4x1x1x!tt.tile<32x32, f32>, #l1_>
      %stream = "ttir.stream_layout"(%input, %storage) : (memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>, memref<2x4x1x1x!tt.tile<32x32, f32>, #l1_>) -> memref<2x4x4x6x!tt.tile<32x32, f32>, #tt.view<map(4)>, #l1_>
      ```
    }];

    let arguments = (ins AnyRankedTensorOrMemRef:$input,
                         AnyRankedTensorOrMemRef:$storage);
    let results = (outs AnyRankedTensorOrMemRef:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getStorageMutable(); }
    }];

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def TTIR_ViewLayoutOp : TTIR_Op<"view_layout",
  [ Pure
  , TTIROpInterface
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                       , "bufferizesToMemoryWrite"
                                                       , "bufferize"
                                                       , "getAliasingValues"
                                                       , "getBufferType"
                                                       ]>
  , TTIR_ViewOpInterface
  ]> {
    let summary = "View Layout op.";
    let description = [{
      ViewLayout operation, used to take a view of one layout into another.  Note that this op is purely representational
      and doesn't have any side-effects. Its primary usecase is to allow reinterpreting the layout of a tensor without actually
      moving the data. Consumers of this op are expected to compose the layout with the underlying backing layout.

      Additional notes/constraints:
      - It is not capable of changing the data type nor the memory space of the tensor.
      - If reinterpretLayout is true, the layout view change can include a data type cast, but note this does not actually change the format of the data in memory.
      - All ViewLayout ops can trivially be converted to ToLayout ops.

      ```llvm
      #layout = #tt.metal_layout<8192x128x1, undef, <1x1>, memref<64x128xf32, #system>>
      #layout1 = #tt.metal_layout<8192x128x1, undef, <1x1>, memref<64x128xf32, #l1_>>
      %1 = "ttir.view_layout"(%arg0, %0) : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
      ```
    }];

    let arguments = (ins AnyRankedTensorOrMemRef:$input, DefaultValuedAttr<BoolAttr, "false">:$reinterpretLayout);
    let results = (outs AnyRankedTensorOrMemRef:$result);

    let hasVerifier = 1;
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
    TTIR_DPSOp<mnemonic, [Pure] # traits>;

class TTIR_ElementwiseOp<string mnemonic, list<Trait> traits = []> :
    TTIR_NamedOp<mnemonic, [TTIR_Broadcastable] # traits> {

    let description = [{
      Base class for elementwise operations. Elementwise operations can take inputs with different shape,
      as long as the input tensors are broadcastable to the same shape.
    }];
}

class TTIR_ElementwiseTernaryOp<string mnemonic, list<Trait> traits = []> :
    TTIR_ElementwiseOp<mnemonic, [FourOperands, TTIR_ElementwiseTernary] # traits> {
    let summary = "Eltwise ternary op.";
    let description = [{
      Eltwise ternary op.
    }];

    let arguments = (ins AnyRankedTensor:$first,
                         AnyRankedTensor:$second,
                         AnyRankedTensor:$third,
                         AnyRankedTensor:$output);

    let results = (outs AnyRankedTensor:$result);

    let builders =
    [
      OpBuilder<(ins "Value": $first, "Value": $second, "Value": $third, "Value": $output),
      [{
        build($_builder, $_state, output.getType(), first, second, third, output);
      }]>,
    ];
}

def TTIR_WhereOp: TTIR_ElementwiseTernaryOp<"where", [TTIR_PartiallyBroadcastable]> {
    let summary = "Eltwise where op.";
    let description = [{
      Eltwise where operation.
    }];
}

class TTIR_ElementwiseUnaryOp<string mnemonic, list<Trait> traits = []> :
    TTIR_ElementwiseOp<mnemonic, [TwoOperands, TTIR_ElementwiseUnary] # traits> {
    let summary = "Eltwise unary op.";
    let description = [{
      Eltwise unary op.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output);

    let results = (outs AnyRankedTensor:$result);

    let builders =
    [
      OpBuilder<(ins "Value": $input, "Value": $output),
      [{
        build($_builder, $_state, output.getType(), input, output);
      }]>,
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

def TTIR_AtanOp: TTIR_ElementwiseUnaryOp<"atan"> {
    let summary = "Eltwise arctangent op.";
    let description = [{
      Performs an elementwise arctangent (`atan`) operation on the input tensor.
      This operation computes the inverse tangent of each element, returning
      values in the range [-π/2, π/2]. Supports floating-point tensor types.

      Example:

      ```mlir
      %input = tensor<4xf32> {1.0, 0.5, 0.0, -1.0}
      %empty = tensor<4xf32>
      %result = "ttir.atan"(%input, %empty) : (tensor<4xf32>) -> tensor<4xf32>
      ```

      Given the input `[1.0, 0.5, 0.0, -1.0]`, the result would be approximately:
      `[0.785, 0.464, 0.0, -0.785]` (values in radians).
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
    let hasCanonicalizeMethod = 1;
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

def TTIR_ExpOp: TTIR_ElementwiseUnaryOp<"exp"> {
    let summary = "Eltwise exponential op.";
    let description = [{
      Eltwise exponential operation. Calculates e^x for all elements x in input tensor.
    }];
}

class TTIR_ElementwiseUnaryWithFloatParameterOp<string mnemonic, list<Trait> traits = []> :
    TTIR_ElementwiseUnaryOp<mnemonic, traits> {
    let summary = "Eltwise unary op with the float parameter.";
    let description = [{
      Eltwise unary op with the float parameter.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         F32Attr:$parameter);

    let builders =
    [
      OpBuilder<(ins "Value": $input, "Value": $output, "FloatAttr": $parameter),
      [{
        build($_builder, $_state, output.getType(), input, output, parameter);
      }]>,
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
    TTIR_ElementwiseOp<mnemonic, [ThreeOperands, TTIR_ElementwiseBinary] #  traits> {
    let summary = "Eltwise binary op.";
    let description = [{
      Eltwise binary op.
    }];

    let arguments = (ins AnyRankedTensor:$lhs,
                         AnyRankedTensor:$rhs,
                         AnyRankedTensor:$output);

    let results = (outs AnyRankedTensor:$result);

    let builders =
    [
      OpBuilder<(ins "Value": $lhs, "Value": $rhs, "Value": $output),
      [{
        build($_builder, $_state, output.getType(), lhs, rhs, output);
      }]>,
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

// TODO We need to add a broadcastable trait to this op.
// See issue https://github.com/tenstorrent/tt-mlir/issues/2015
def TTIR_PowOp : TTIR_ElementwiseBinaryOp<"pow"> {
    let summary = "Eltwise power OP.";
    let description = [{
      Performs element-wise exponentiation of lhs tensor by rhs tensor and produces a
      result tensor. Tensors must be of same shape.

      Example:
      ```
        %result = "ttir.pow"(%lhs, %rhs) : (tensor<6xf64>, tensor<6xf64>) -> tensor<6xf64>

        %lhs: [-2.0, -0.0, -36.0, 5.0, 3.0, 10000.0]
        %rhs: [2.0, 2.0, 1.1, 2.0, -1.0, 10.0]
        %result: [4.0, 0.0, -nan, 25.0, 0.333333343, inf]
      ```
    }];
}

def TTIR_Atan2Op :  TTIR_ElementwiseBinaryOp<"atan2"> {
    let summary = "Eltwise atan2 OP.";
    let description = [{
      Performs element-wise atan2 operation on lhs and rhs tensor and produces a result
      tensor.

      Example:
      ```
        // %lhs: [0.0, 1.0, -1.0]
        // %rhs: [1.0, 0.0, 0.0]
        %result = "ttir.atan2"(%lhs, %rhs) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
        // %result: [0.0, 1.57079637, -1.57079637] // [0.0, pi/2, -pi/2]
      ```
    }];
}

def TTIR_AddOp : TTIR_ElementwiseBinaryOp<"add", [TTIR_FullyBroadcastable]> {
    let summary = "Eltwise add.";
    let description = [{
      Eltwise add operation.
    }];
}

def TTIR_MultiplyOp : TTIR_ElementwiseBinaryOp<"multiply", [TTIR_PartiallyBroadcastable]> {
    let summary = "Eltwise multiply.";
    let description = [{
      Eltwise multiply operation.
    }];
}

def TTIR_DivOp : TTIR_ElementwiseBinaryOp<"div"> {
    let summary = "Eltwise divide.";
    let description = [{
      Eltwise divide operation.
    }];
}

def TTIR_MaximumOp : TTIR_ElementwiseBinaryOp<"maximum"> {
    let summary = "Eltwise maximum.";
    let description = [{
      Calculates maximum of input tensors' values element-wise and stores result in output tensor.

      Example:
        %lhs: [[3, 2, 7], [1, 4, 4]]
        %rhs: [[1, 4, 2], [1, 2, 3]]
        "ttir.maximum"(%lhs, %rhs, %out) -> %out: [[3, 4, 7], [1, 4, 4]]
    }];
}

def TTIR_BatchNormOp : TTIR_DPSOp<"batch_norm"> {
  let summary = "BatchNormInference operation";
  let description = [{
    Normalizes the `operand` tensor across all dimensions except for the
    `feature_index` dimension and puts the result in the `output` tensor.

    Example:
    ```mlir
    "ttir.batch_norm"(%operand, %scale, %offset, %mean, %variance, epsilon = dense<f32>, feature_index = dense<i64>, %out) {
    } : (tensor<2x2x2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, dense<f32>, dense<i64>, tensor<2x2x2xf64>) -> tensor<2x2x2xf64>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$operand,
                       AnyRankedTensor:$scale,
                       AnyRankedTensor:$offset,
                       AnyRankedTensor:$mean,
                       AnyRankedTensor:$variance,
                       AnyRankedTensor:$output,
                       F32Attr:$epsilon,
                       I32Attr:$dimension,
                       BoolAttr:$training);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

class TTIR_ReductionOp<string mnemonic, list<Trait> traits = []> :
    TTIR_NamedOp<mnemonic, traits> {

    let summary = "Reduction op.";
    let description = [{
      Reduction op.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         BoolAttr:$keep_dim,
                         OptionalAttr<I32ArrayAttr>:$dim_arg);

    let results = (outs AnyRankedTensor:$result);

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

    %empty = ttir.empty() : tensor<128x28xi32>
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

  let hasVerifier = 1;

  let hasFolder = 1;
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

    let hasVerifier = 1;
}

def TTIR_TransposeOp : TTIR_NamedOp<"transpose", [TTIR_TensorManipulation]> {
    let summary = "Transpose op.";
    let description = [{
      Transpose tensor along two given dimensions.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         SI32Attr:$dim0,
                         SI32Attr:$dim1);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let hasFolder = 1;
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
    %empty = ttir.empty() : tensor<4x6xf32>
    %repeated = "repeat"(%input, %empty) {repeat_dimensions = array<i64: 2, 2>} : (tensor<2x3xf32>, tensor<4x6xf32>) -> tensor<4x6xf32>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$output,
                       DenseI64ArrayAttr:$repeat_dimensions);

  let results = (outs AnyRankedTensor:$result);

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
      %0 = ttir.empty() : tensor<1x16x32xf32>
      %1 = "ttir.broadcast"(%arg0, %0) <{broadcast_dimensions = array<i32: 1, 16, 1>}> : (tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>

      Note: Currently, when generating a TTNN executable, the broadcast and repeat operations share the same semantics due to the lack of tensor view support in TTNN.
      As a result, the broadcast operation is lowered to a repeat operation in the TTNN compilation pipeline.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         DenseI64ArrayAttr:$broadcast_dimensions);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let hasFolder = 1;
}

def TTIR_Conv2dOp : TTIR_NamedOp<"conv2d"> {
    let summary = "Conv2d operation.";
    let description = [{
      Applies a 2D convolution over an input image composed of several input planes.

      Inputs:
      - `input` (AnyRankedTensor): expected in the following format (N, H_in, W_in, C) where:
        - N is the batch size
        - H_in is the height of the input planes
        - W_in is the width of the input planes
        - C is the number of channels
      - `weight` (AnyRankedTensor): expected in the following format (O, C/G, K_H, K_W) where:
        - C is the number of input channels
        - O is the number of output channels
        - G is the number of groups
        - K_H is the height of the kernel
        - K_W is the width of the kernel
      - `bias` Optional<AnyRankedTensor>: expected in the following format (1, 1, 1, O).
      - `output` AnyRankedTensor()): expected in the following format (N, H_out, W_out, O) where:
        - `H_out = (H_in + pT + pB - dH * (K_H - 1) - 1) / sH + 1`
        - `W_out = (W_in + pL + pR - dW * (K_W - 1) - 1) / sW + 1`

      Attributes:
      - `stride` (i32 | array<2xi32>):
        - i32: Same stride for height and width dimensions (sH = sW = value).
        - array<2xi32>: [sH, sW] where sH is stride for height and sW is stride for width.
      - `padding` (i32 | array<2xi32> | array<4xi32>):
        - i32: Same padding for all sides (pT = pL = pB = pR = value).
        - array<2xi32>: [pH, pW] where pH is padding for height (top/bottom) and pW is padding for width (left/right).
        - array<4xi32>: [pT, pL, pB, pR] for top, left, bottom, and right padding respectively.
      - `dilation` (i32 | array<2xi32>): Spacing between kernel elements.
        - i32: Same dilation for height and width dimensions (dH = dW = value).
        - array<2xi32>: [dH, dW] where dH is dilation for height and dW is dilation for width.
      - `groups` (i32): Number of blocked connections from input channels to output channels. Input and output channels must both be divisible by groups.

      Example:
        %0 = "ttir.conv2d"(%input, %weight, %bias, %output)
          <{
            stride = 1: i32,
            padding = 0: i32,
            dilation = 1: i32,
            groups = 1: i32
          > : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>

        %1 = "ttir.conv2d"(%input, %weight, %bias, %output)
          <{
              stride = array<i32: 2, 1>,        // Height stride=2, width stride=1
              padding = array<i32: 1, 1, 2, 0>, // Top=1, left=1, bottom=2, right=0
              dilation = array<i32: 1, 2>,      // Height dilation=1, width dilation=2
              groups = 2: i32                   // Group convolution with 2 groups
          }> : (tensor<1x32x32x64xbf16>, tensor<64x32x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x16x28x64xbf16>) -> tensor<1x16x28x64xbf16>
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$weight,
                         Optional<AnyRankedTensor>:$bias,
                         AnyRankedTensor:$output,
                         AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$stride,
                         AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$padding,
                         AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$dilation,
                         I32Attr:$groups,
                         DefaultValuedAttr<TTIR_FlattenedCompatInfoAttr, "nullptr">:$flattened_compat_info);

    let extraClassDeclaration = [{
      // Get number of output channels
      int64_t getOutputChannelSize();
      bool isBiasCompatible(llvm::ArrayRef<int64_t> biasDims);
      MutableOperandRange getDpsInitsMutable() { return ttir::getDpsOutputs(this); }
    }];

    let results = (outs AnyRankedTensor:$result);

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
      - `weight` (AnyRankedTensor): expected in the following format (O, C/G, K_H, K_W) where:
        - C is the number of input channels
        - O is the number of output channels
        - G is the number of groups
        - K_H is the height of the kernel
        - K_W is the width of the kernel
      - `bias` Optional<AnyRankedTensor>: expected in the following format (1, 1, 1, O).
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
        %input = ttir.empty() : () -> tensor<3x8x8x256xbf16>
        %weight = ttir.empty() : () -> tensor<256x256x3x3xbf16>
        %bias = ttir.empty() : () -> tensor<1x1x1x256xbf16>
        %output = ttir.empty() : () -> tensor<1x10x10x256xbf16>
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

def TTIR_AvgPool2dOp : TTIR_NamedOp<"avg_pool2d"> {
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
                       DefaultValuedAttr<TTIR_FlattenedCompatInfoAttr, "nullptr">:$flattened_compat_info);

  let results = (outs AnyRankedTensor:$result);

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
                         SI32Attr:$padding_bottom,
                         DefaultValuedAttr<TTIR_FlattenedCompatInfoAttr, "nullptr">:$flattened_compat_info);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_ReshapeOp: TTIR_NamedOp<"reshape", [TTIR_TensorManipulation]> {
    let summary = "Reshape op.";
    let description = [{
      Reshape tensor.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         I32ArrayAttr:$shape);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let hasFolder = 1;
}

def TTIR_PadOp: TTIR_NamedOp<"pad"> {
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
                         AnyRankedTensor:$output,
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

    let hasVerifier = 1;
}

def TTIR_ClampScalarOp : TTIR_NamedOp<"clamp_scalar"> {
    let summary = "Clamp op.";
    let description = [{
      Clamp tensor values to a specified range.

      Example:
        min: 2.000000+00
        input: [[0, 1, 2, 3, 4, 5, 6, 7]]
        max: 5.000000+00

        "ttir.clamp_scalar"(%arg0) <{max = 2.000000e+00 : f32, min = 5.000000e+00 : f32}>
        -> %out = [[2, 2, 2, 3, 4, 5, 5, 5]]
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         F32Attr:$min,
                         F32Attr:$max);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_ClampTensorOp : TTIR_NamedOp<"clamp_tensor"> {
  let summary = "Clamp op.";
  let description = [{
    Clamp tensor values to a specified range using min/max as tensor.

    Example:
      min:   [[2, 2, 2, 3, 3, 3, 0, 0]]
      input: [[0, 1, 2, 3, 4, 5, 6, 7]]
      max:   [[5, 5, 5, 9, 9, 9, 6, 6]]

      "ttir.clamp_tensor"(%input, %min, %max)
      %out:  [[2, 2, 2, 3, 4, 5, 6, 6]]
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$min,
                       AnyRankedTensor:$max,
                       AnyRankedTensor:$output);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
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

    let hasVerifier = 1;

    let hasCanonicalizer = 1;
}

def TTIR_LinearOp : TTIR_NamedOp<"linear"> {
    let summary = "Linear transformation of inputs.";
    let description = [{
      Produces the matmul of tensors `a` and `b` with optional addition with `bias`.

      Example:
        %a = ttir.empty() : () -> tensor<10x64x32xbf16>
        %b = ttir.empty() : () -> tensor<32x128xbf16>
        %bias = ttir.empty() : () -> tensor<128xbf16>
        %output = ttir.empty() : () -> tensor<10x64x128xbf16>
        %0 = "ttir.linear"(%a, %b, %bias, %output) : (tensor<10x64x32xbf16>, tensor<32x128xbf16>, tensor<128xbf16>, tensor<10x64x128xbf16>) -> tensor<10x64x128xbf16>
    }];

    let arguments = (ins AnyRankedTensor:$a,
                         AnyRankedTensor:$b,
                         Optional<AnyRankedTensor>:$bias,
                         AnyRankedTensor:$output,
                         DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
                         DefaultValuedAttr<BoolAttr, "false">:$transpose_b);

    let results = (outs AnyRankedTensor:$result);

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

    let hasVerifier = 1;

    let hasCanonicalizer = 1;
}
// ANCHOR_END: adding_an_op_matmul_ttir

def TTIR_PermuteOp : TTIR_NamedOp<"permute", [TTIR_TensorManipulation]> {
    let summary = "Permute operation.";
    let description = [{
      Permute input tensor dimensions.

      Attributes:
        - `permutation` array<i64>: The permutation of the input tensor dimensions.

      Example:
      %a = ttir.empty() : () -> tensor<2x3x4xi32>
      %output = ttir.empty() : () -> tensor<3x4x2xi32>
      %0 = "ttir.permute"(%a, %output) {permutation = array<i64: 1, 2, 0>} : (tensor<2x3x4xi32>, tensor<3x4x2xi32>) -> tensor<3x4x2xi32>
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         DenseI64ArrayAttr:$permutation);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let hasFolder = 1;
}

def TTIR_Upsample2dOp : TTIR_NamedOp<"upsample2d"> {
    let summary = "Upsample 2D operation.";

    let description = [{
      Upsample 2D operation. Input tensor is assumed to be in NHWC format.

      Attributes:
      - `scale_factor` (si32 | array<i32>): The scale factor for upsampling in H and W dimensions respectively.
      - `mode` (str): The upsampling algorithm. Currently only "nearest" and "bilinear" are supported. Default is "nearest".

      Example:
        %a = ttir.empty() : () -> tensor<10x64x32x3xbf16>
        %output = ttir.empty() : () -> tensor<10x128x128x3xbf16>
        %0 = "ttir.upsample2d"(%a, %output) <{scale_factor = array<i32: 2, 4>}> : (tensor<10x64x32x3xbf16>, tensor<10x128x128x3xbf16>) -> tensor<10x64x128x3xbf16>
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         AnyAttrOf<[SI32Attr, DenseI32ArrayAttr]>:$scale_factor,
                         DefaultValuedAttr<StrAttr, "\"nearest\"">:$mode);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

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

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TTIR creation ops
//===----------------------------------------------------------------------===//

class TTIR_CreationOp<string mnemonic, list<Trait> traits = []> :
    TTIR_Op<mnemonic, [Pure, TT_CreationOpTrait] # traits>;


def TTIR_ArangeOp : TTIR_CreationOp<"arange"> {
  let summary = "Arange operation.";
  let description = [{
    Tensor arange operation.

    Produces a tensor with values from `start` to `end` (exclusive) with a step size of `step`, along the dimension specified by `arange_dimension`.

    Examples:
      %0 = "ttir.arange"() {start = 0 : si64, end = 5 : si64 step = 1 : i64, arange_dimension = 0 : i64} : () -> tensor<5xi64>
      // %0: [0, 1, 2, 3, 4]

      %1 = "ttir.arange"() {start = 0 : si64, end = 10 : si64, step = 2 : si64, arange_dimension = 0 : i64} : () -> tensor<5xf32>
      // %1: [0.0, 2.0, 4.0, 6.0, 8.0]

      %2 = "ttir.arange"() {start = 0 : si64, end = 5 : si64, step = 1 : si64, arange_dimension = 0 : i64} : () -> tensor<5x3xi64>
      // %2: [
              [0, 0, 0],
              [1, 1, 1],
              [2, 2, 2],
              [3, 3, 3],
              [4, 4, 4]
             ]

      %3 = "ttir.arange"() {start = 0 : si64, end = 3 : si64, step = 1 : si64, arange_dimension = 1 : i64} : () -> tensor<5x3xi64>
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

def TTIR_FullOp : TTIR_CreationOp<"full"> {
  let summary = "Creates a tensor filled with the specified value";
  let description = [{
    Tensor operation to create a tensor filled with a specified value.

    Given a `shape` and a `fill_value`, produces a tensor with the shape, filled with the specified value.

    Example:
      %0 = "ttir.full"() <{shape = array<i32: 64, 32, 32>, fill_value = 7 : i32}> : () -> tensor<64x32x32xi32>
      // %0: [[[7, 7, 7, ..., 7], [7, 7, 7, ..., 7], ..., [7, 7, 7, ..., 7]]]
  }];

  let arguments = (ins DenseI32ArrayAttr:$shape,
                       AnyAttrOf<[F32Attr, I32Attr]>:$fill_value);

  let results = (outs AnyRankedTensor:$result);

  let builders =
  [
    OpBuilder<(ins "Type": $resultType, "Attribute": $fillValue),
    [{
      RankedTensorType tensorType = cast<RankedTensorType>(resultType);
      build($_builder, $_state, tensorType, llvm::to_vector_of<int32_t>(tensorType.getShape()), fillValue);
    }]>
  ];

  let hasVerifier = 1;
}

class TTIR_NamedFullOp<string mnemonic, list<Trait> traits = []> :
  TTIR_CreationOp<mnemonic, traits> {
  let arguments = (ins DenseI32ArrayAttr:$shape);

  let results = (outs AnyRankedTensor:$result);
}

def TTIR_ZerosOp : TTIR_NamedFullOp<"zeros"> {
  let summary = "Creates a tensor filled with zeros.";
  let description = [{
    Tensor operation to create a tensor filled with zeros.

    Given a `shape`, produces a tensor with the shape, filled with zeros.

    Example:
      %0 = "ttir.zeros"() <{shape = array<i32:64, 28, 28>}> : () -> tensor<64x28x28xbf16>
      // %0: [[[0, 0, 0, ..., 0], [0, 0, 0, ..., 0], ..., [0, 0, 0, ..., 0]]]
  }];
}

def TTIR_OnesOp : TTIR_NamedFullOp<"ones"> {
  let summary = "Creates a tensor filled with ones.";
  let description = [{
    Tensor operation to create a tensor filled with ones.

    Given a `shape`, produces a tensor with the shape, filled with ones.

    Example:
      %0 = "ttir.ones"() <{shape = array<i32:64, 28, 28>}> : () -> tensor<64x28x28xbf16>
      // %0: [[[1, 1, 1, ..., 1], [1, 1, 1, ..., 1], ..., [1, 1, 1, ..., 1]]]
  }];
}

def TTIR_EmptyOp : TTIR_CreationOp<"empty",
  [DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                      , "bufferizesToMemoryWrite"
                                                      , "bufferizesToAllocation"
                                                      , "bufferize"
                                                      , "getAliasingValues"
                                                      , "getBufferType"
                                                      ]>
  ]> {
    let summary = "Empty op.";
    let description = [{
      Tensor empty operation
    }];

    let results = (outs AnyRankedTensor:$result);

    let builders =
    [
      OpBuilder<(ins "ArrayRef<int64_t>": $shape, "Type": $elementType, "Attribute": $encoding),
      [{
        build($_builder, $_state, RankedTensorType::get(shape, elementType, encoding));
      }]>,
      OpBuilder<(ins "ArrayRef<int64_t>": $shape, "Type": $elementType),
      [{
        build($_builder, $_state, shape, elementType, nullptr);
      }]>
    ];

    let assemblyFormat = "`(` `)` attr-dict `:` type($result)";
}

def TTIR_ConstantOp : TTIR_CreationOp<"constant",
  [ ConstantLike
  , AllShapesMatch<["value", "result"]>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                       , "bufferizesToMemoryWrite"
                                                       , "bufferize"
                                                       , "getAliasingValues"
                                                       , "getBufferType"
                                                       ]>
  ]> {
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

    let hasCanonicalizer = 1;

    let hasVerifier = 1;
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

    let hasVerifier = 1;
}

def TTIR_ReduceScatterOp : TTIR_NamedOp<"reduce_scatter"> {
    let summary = "Reduce scatter operation.";
    let description = [{
      Reduce scatter op.
    }];
    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         TT_ReduceTypeAttr:$reduce_type,
                         SI32Attr:$scatter_dim,
                         UI32Attr:$cluster_axis);

    let results = (outs AnyRankedTensor:$result);
    let hasVerifier = 1;
}

def TTIR_CollectivePermuteOp : TTIR_NamedOp<"collective_permute"> {
    let summary = "Collective permute operation.";
    let description = [{
      Collective permute op. This operation ingests a multi-device tensor spread across multi-devices and will shuffle the data according to source_target_pairs [['src', 'dest']].

      Example:
        For a 1x2 mesh, the following will take the device shard living in device 0 and move it to device 1. The device shard living in device 1 will move to device 0.
        %source_target_pairs: [[0, 1], [1, 0]]

        In the case of missing 'dest', the device shard living on that device will contain values of 0. For example, device shard living in device 0 will contain 0 values.
        %source_target_pairs: [[0, 1]]
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         I64ElementsAttr:$source_target_pairs);

    let results = (outs AnyRankedTensor:$result);
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

    let hasVerifier = 1;
}

def TTIR_QuantizeOp : TTIR_NamedOp<"quantize"> {
  let summary = "Quantize operation.";
  let description = [{
      The Quantize operation converts a tensor into a quantized tensor using the `quant.uniform` type from the MLIR Quant dialect.
      This type encapsulates the scale and zero-point metadata directly within the tensor type.
      The output tensor will be of type 'quant.uniform', where each element is computed as:
      ```
      output[i] = (input[i] / scale) + zero_point
      ```

      Example:
      ```mlir
      %input = ttir.empty() : () -> tensor<64x128xf32>
      %output = ttir.empty() : () -> tensor<64x128x!quant.uniform<i32:f32, 0.1>>
      %quantized = "ttir.quantize"(%input, %output) : (tensor<64x128xf32>, tensor<64x128x!quant.uniform<i32:f32, 0.1>>) -> tensor<64x128x!quant.uniform<i32:f32, 0.1>>
      ```
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$output);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

def TTIR_DequantizeOp : TTIR_NamedOp<"dequantize"> {
  let summary = "Dequantize operation.";
  let description = [{
    The Dequantize operation converts a quantized tensor back into a floating-point tensor using the `quant.uniform` type from the MLIR Quant dialect.
    The input tensor is expected to be of type `quant.uniform.`
    The output tensor will be a floating-point tensor, where each element is computed as:
    ```
    output[i] = (input[i] - zero_point) * scale
    ```
    Example:
    ```mlir
    %input = ttir.empty() : () -> tensor<64x128x!quant.uniform<i32:f32, 0.1>>
    %output = ttir.empty() : () -> tensor<64x128xf32>
    %dequantized = "ttir.dequantize"(%input, %output) : (tensor<64x128x!quant.uniform<i32:f32, 0.1>, tensor<64x128xf32>) -> tensor<64x128xf32>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$output);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

def TTIR_RequantizeOp : TTIR_NamedOp<"requantize"> {
  let summary = "Requantize operation.";
  let description = [{
    The Requantize operation converts a quantized tensor from one scale and zero-point to another, using the `quant.uniform` type from the MLIR Quant dialect.
    The input tensor is expected to be of type `quant.uniform.`
    The output tensor will also be of type `quant.uniform.`
    Each element in the output tensor is computed as:
    ```
    output[i] = round((input[i] - input_zero_point) * (input_scale / output_scale)) + output_zero_point
    ```
    Example:
    ```mlir
    %input = ttir.empty() : () -> tensor<64x128x!quant.uniform<i32:f32, 0.1>>
    %output = ttir.empty() : () -> tensor<64x128x!quant.uniform<i32:f32, 0.2>>
    %requantized = "ttir.requantize"(%input, %output) : (tensor<64x128x!quant.uniform<i32:f32, 0.1>, tensor<64x128x!quant.uniform<i32:f32, 0.2>>) -> tensor<64x128x!quant.uniform<i32:f32, 0.2>>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$output);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

#endif
