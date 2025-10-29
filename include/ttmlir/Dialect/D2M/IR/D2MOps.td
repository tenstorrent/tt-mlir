// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_D2M_D2MOPS_TD
#define TTMLIR_TTMLIR_DIALECT_D2M_D2MOPS_TD

include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td"
include "ttmlir/Dialect/D2M/IR/D2MBase.td"
include "ttmlir/Dialect/D2M/IR/D2MOpsTypes.td"
include "ttmlir/Dialect/D2M/IR/D2MOpsAttrs.td"
include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/IR/OpAsmInterface.td"

// Minimal scaffolding op: ViewLayout

def D2M_ViewLayoutOp : D2M_Op<"view_layout", [
    Pure,
    D2M_ViewOpInterface,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<BufferizableOpInterface, [
      "bufferizesToMemoryRead",
      "bufferizesToMemoryWrite",
      "bufferize",
      "getAliasingValues",
      "getBufferType"
    ]>
  ]> {
  let summary = "View Layout op (D2M subset)";
  let description = [{
    Create a representational view of a tensor/memref with a different layout.
    This is a no-op for codegen; consumers are expected to compose layouts.
  }];

  let arguments = (ins AnyRankedTensorOrMemRef:$input, DefaultValuedAttr<BoolAttr, "false">:$reinterpretLayout);
  let results = (outs AnyRankedTensorOrMemRef:$result);

  let builders = [
    OpBuilder<(ins "Value":$input, "ArrayRef<int64_t>":$reblockedShape,
                   CArg<"bool", "false">:$reinterpretLayout)>,
  ];

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Layout transition and streaming ops (D2M variants)
//===----------------------------------------------------------------------===//

def D2M_ToLayoutOp : D2M_Op<"to_layout", [
    DeclareOpInterfaceMethods<BufferizableOpInterface, [
      "bufferizesToMemoryRead",
      "bufferizesToMemoryWrite",
      "bufferize",
      "getAliasingValues",
      "getBufferType"
    ]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
    let summary = "Layout op.";
    let description = [{
      ToLayout operation, transition tensors from one layout to another.  Some examples include:
        - Transitioning between different memory spaces, e.g. DRAM to L1.
        - Transitioning between different data types, e.g. f32 to f16.
        - Transitioning between different tile sizes, e.g. 1x16 to 32x32
        - Transitioning between different tensor sharding
        - Some combination of the above

      ```llvm
      #layout = #ttcore.metal_layout<8192x128x1, undef, <1x1>, memref<64x128xf32, #system>>
      #layout1 = #ttcore.metal_layout<8192x128x1, undef, <1x1>, memref<64x128xf32, #l1_>>
      %1 = "d2m.to_layout"(%arg0, %0) : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
      ```
    }];

    let arguments = (ins AnyRankedTensorOrMemRef:$input,
                         AnyRankedTensorOrMemRef:$output,
                         OptionalAttr<TTCore_MetalLayoutAttr>:$layout);
    let results = (outs Variadic<AnyRankedTensor>:$results);

    let assemblyFormat = [{ $input `,` $output `:` type($input) `into` type($output) (`hostInfo` `=` $layout^)? attr-dict (`->` type($results)^)? }];


    let builders =
    [
      OpBuilder<(ins "Value": $input, "Value": $output, CArg<"ttcore::MetalLayoutAttr", "nullptr">: $layout),
      [{
        build($_builder, $_state, {output.getType()}, input, output, layout);
      }]>,
    ];

    let extraClassDeclaration = [{
      struct CompoundComponents {
        bool isLayoutChange = false;
        bool isGridChange = false;
        bool isFormatChange = false;
        bool isMemorySpaceChange = false;

        bool isCompound() {
          return isLayoutChange + isFormatChange + (isGridChange || isMemorySpaceChange) > 1;
        }
      };

      // Returns booleans indicating if the op changes layout, grid, format, memory space or memory layout.
      CompoundComponents compoundComponents();
      bool isCompound() { return compoundComponents().isCompound(); }
      ttcore::MetalLayoutAttr getOrCreateInputLayout();
      ttcore::MetalLayoutAttr getOrCreateOutputLayout();

      bool isHostToDevice();
      bool isDeviceToHost();
    }];

    let hasVerifier = 1;
    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

def D2M_StreamLayoutOp : D2M_Op<"stream_layout", [
    Pure,
    D2M_ViewOpInterface,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<BufferizableOpInterface, [
      "bufferizesToMemoryRead",
      "bufferizesToMemoryWrite",
      "bufferize",
      "getAliasingValues",
      "getBufferType"
    ]>
  ]> {
  let hasCanonicalizer = 1;
  let summary = "Stream layout (D2M)";
  let description = [{
    Represent a streaming relationship between a source tensor/memref and a
    storage buffer, producing a view result.
  }];

  let arguments = (ins AnyRankedTensorOrMemRef:$input,
                       AnyRankedTensorOrMemRef:$storage);
  let results = (outs AnyRankedTensorOrMemRef:$result);

  let hasVerifier = 1;
}

def D2M_GenericOp : D2M_Op<"generic",
  [ AttrSizedOperandSegments
  , NoTerminator
  , DestinationStyleOpInterface
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames", "getAsmBlockNames"]>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [
      "bufferizesToMemoryRead",
      "bufferizesToMemoryWrite",
      "bufferize",
      "getAliasingValues",
      "getBufferType",
      "isWritable"
    ]>
  , DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
    let summary = "Generically dispatch work to a grid of cores (D2M).";
    let description = [{
      Same semantics as D2M generic; carries regions for compute/datamovement
      to be consumed by the metal path.
    }];

    let arguments = (ins Variadic<AnyRankedTensorOrMemRef>:$inputs,
                         Variadic<AnyRankedTensorOrMemRef>:$outputs,
                         TTCore_GridAttr:$grid,
                         I64ArrayAttr:$block_factors,
                         AffineMapArrayAttr:$indexing_maps,
                         TTCore_IteratorTypeArrayAttr:$iterator_types,
                         D2M_ThreadArrayAttr:$threads);
    let results = (outs Variadic<AnyRankedTensor>:$results);
    let regions = (region VariadicRegion<AnyRegion>:$regions);

    let assemblyFormat = [{ attr-dict `\n`
    ` ` ` ` ` ` ` ` `ins` `(` $inputs `:` type($inputs) `)` `\n`
    ` ` ` ` ` ` ` ` `outs` `(` $outputs  `:` type($outputs) `)` ` `  $regions (`:`  type($results)^ )?
    }];

    let hasVerifier = 1;
    let hasCanonicalizer = 1;

    let builders = [
      OpBuilder<(ins "ValueRange": $inputs,
                     "ValueRange": $outputs,
                     "ArrayAttr": $indexingMaps,
                     "ArrayAttr": $iteratorTypes,
                     CArg<"ThreadType", "ThreadType::Compute">: $singleThreadType,
                     CArg<"ttcore::GridAttr", "nullptr">: $grid,
                     CArg<"ArrayRef<int64_t>", "{}">: $blockFactors)>,
      OpBuilder<(ins "ValueRange": $inputs,
                     "ValueRange": $outputs,
                     "ArrayAttr": $indexingMaps,
                     "ArrayAttr": $iteratorTypes,
                     "llvm::function_ref<void(OpBuilder&, Location, ValueRange)>": $singleThreadRegionBuilder,
                     CArg<"ThreadType", "ThreadType::Compute">: $singleThreadType,
                     CArg<"ttcore::GridAttr", "nullptr">: $grid,
                     CArg<"ArrayRef<int64_t>", "{}">: $blockFactors)>,
      OpBuilder<(ins "ValueRange": $inputs,
                     "ValueRange": $outputs,
                     "llvm::function_ref<void(OpBuilder&, Location, ValueRange)>": $singleThreadRegionBuilder,
                     CArg<"ThreadType", "ThreadType::Compute">: $singleThreadType,
                     CArg<"ttcore::GridAttr", "nullptr">: $grid,
                     CArg<"ArrayRef<int64_t>", "{}">: $blockFactors)>,
    ];

    let extraClassDeclaration = [{
      // Required by DestinationStyleOpInterface
      MutableOperandRange getDpsInitsMutable() {
        return getOutputsMutable();
      }
      static std::pair<ArrayAttr, ArrayAttr> buildParallelAffineMapsAndIteratorTypes(mlir::OpBuilder &builder, size_t arity, size_t rank) {
        auto map = builder.getMultiDimIdentityMap(rank);
        Attribute parallel = builder.getAttr<ttcore::IteratorTypeAttr>(ttcore::IteratorType::Parallel);
        return std::make_pair(builder.getAffineMapArrayAttr(SmallVector<AffineMap>(arity, map)),
                              builder.getArrayAttr(SmallVector<Attribute>(rank, parallel)));
      }
      ThreadType getRegionThreadType(unsigned regionIndex) {
        auto arr = getThreadsAttr().getValue();
        return mlir::cast<ThreadAttr>(arr[regionIndex]).getThreadType();
      }
      SmallVector<ttcore::IteratorType> getIteratorTypesValue();
      bool isComputeOnlyForm() {
        auto threads = getThreads();
        return threads.size() == 1 && mlir::cast<ThreadAttr>(threads[0]).getThreadType() == ThreadType::Compute;
      }
      bool isDMAOnlyForm() {
        return llvm::none_of(getThreads(), [](Attribute attr) {
          return mlir::cast<d2m::ThreadAttr>(attr).getThreadType() ==
                d2m::ThreadType::Compute;
        });
      }
      bool isAffineMapForm() { return !getIndexingMaps().empty(); }
      bool isLoweredLoopForm() { return !isAffineMapForm(); }
      bool isExternalSymbolForm() { return getRegions().empty(); }
      bool hasExplicitBlockFactors() { return getIndexingMaps().empty() && !getBlockFactors().empty(); }
      SmallVector<SmallVector<int64_t>> getOperandGridShapes();
      SmallVector<int64_t> getLoopBounds();
      SmallVector<AffineMap> getIndexingMapsValue();
      SmallVector<int64_t> getBlockFactorsValue();
      SmallVector<int64_t> getFullBlockFactors();
      SmallVector<int64_t> getExplicitBlockFactors(int64_t operandIndex);
      ttcore::DeviceAttr getDevice();
      SmallVector<int64_t> getParticipatingLoopDims(int64_t operandIndex);
      SmallVector<int64_t> getNonParticipatingLoopDims(int64_t operandIndex);
      SmallVector<SmallVector<int64_t>> getOperandShardShapes(bool convertTileToScalar = false);
      unsigned getNumLoops();
      unsigned getNumDims();
      std::optional<AffineMap> getIndexingMap(int64_t operandIndex);
      bool hasCompatibleBlocking(GenericOp gOp);
      bool hasSkipOpEltwiseFusionTrait();
      bool hasReduction();
      bool hasMultiUseInputOperand();
      bool isAllParallel();
    }];
}

// Simple creation op needed by LowerToLayout splitting

def D2M_EmptyOp : D2M_Op<"empty", [
    MemoryEffects<[MemAlloc]>,
    DeclareOpInterfaceMethods<BufferizableOpInterface, [
      "bufferizesToMemoryRead",
      "bufferizesToMemoryWrite",
      "bufferize",
      "getAliasingValues",
      "getBufferType"
    ]>
  ]> {
  let summary = "Empty tensor allocation operation (D2M).";
  let description = [{
    Create an uninitialized tensor with the specified shape, element type and encoding.
  }];

  let results = (outs AnyRankedTensor:$result);

  let builders = [
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

def D2M_FullOp : D2M_Op<"full", [
    MemoryEffects<[MemAlloc]>,
    DeclareOpInterfaceMethods<BufferizableOpInterface, [
      "bufferizesToMemoryRead",
      "bufferizesToMemoryWrite",
      "bufferize",
      "getAliasingValues",
      "getBufferType"
    ]>
  ]> {
  let summary = "Creates a tensor filled with the specified value (D2M).";
  let description = [{
    Tensor operation to create a tensor filled with a specified value.
    Given a `shape` and a `fill_value`, produces a tensor with the shape, filled with the specified value.
  }];

  let arguments = (ins DenseI32ArrayAttr:$shape,
                       AnyAttrOf<[F32Attr, I32Attr]>:$fill_value);

  let results = (outs AnyRankedTensor:$result);

  let builders = [
    OpBuilder<(ins "Type": $resultType, "Attribute": $fillValue),
    [{
      RankedTensorType tensorType = mlir::cast<RankedTensorType>(resultType);
      build($_builder, $_state, tensorType, llvm::to_vector_of<int32_t>(tensorType.getShape()), fillValue);
    }]>
  ];

  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def D2M_MeshShardOp: D2M_Op<"mesh_shard", [
    Pure,
    DeclareOpInterfaceMethods<BufferizableOpInterface, [
      "bufferizesToMemoryRead",
      "bufferizesToMemoryWrite",
      "bufferize",
      "getAliasingValues",
      "getBufferType"
    ]>
  ]> {
  let summary = "Mesh shard operation (D2M).";
  let description = [{
    MeshShard op shards the inputs (FullToShard) or concatenates the outputs (ShardToFull) for ccl ops.
  }];

  let arguments = (ins AnyRankedTensorOrMemRef:$input,
                       TTCore_MeshShardTypeAttr:$shard_type,
                       TTCore_MeshShardDirectionAttr:$shard_direction,
                       DenseI64ArrayAttr:$shard_shape,
                       DenseI64ArrayAttr:$shard_dims);

  let results = (outs AnyRankedTensorOrMemRef:$result);

  let extraClassDefinition = [{
    bool $cppClass::bufferizesToMemoryRead(
        mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
      return true;
    }
    bool $cppClass::bufferizesToMemoryWrite(
        mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
      return false;
    }
    bufferization::AliasingValueList $cppClass::getAliasingValues(
        OpOperand &, const bufferization::AnalysisState &) {
      bufferization::AliasingValueList result;
      return result;
    }
  }];

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

#endif // TTMLIR_TTMLIR_DIALECT_D2M_D2MOPS_TD
