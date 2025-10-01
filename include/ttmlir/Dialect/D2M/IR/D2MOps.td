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
                     CArg<"ArrayRef<int64_t>", "{}">: $blockFactors),
      [{
        assert(!indexingMaps.empty() && "expected non-empty indexing maps");
        assert(outputs.size() == 1 && "expected single output");
        if (!grid) {
          if (RankedTensorType tensorType = mlir::dyn_cast<RankedTensorType>(outputs[0].getType()); tensorType) {
             grid = $_builder.getAttr<ttcore::GridAttr>(mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding()).getGridShape(tensorType));
          } else {
            MemRefType memrefType = mlir::cast<MemRefType>(outputs[0].getType());
            ttcore::DeviceLayoutInterface layout = mlir::cast<ttcore::DeviceLayoutInterface>(memrefType.getLayout());
            grid = $_builder.getAttr<ttcore::GridAttr>(layout.getGridShape(memrefType));
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
                     CArg<"ttcore::GridAttr", "nullptr">: $grid,
                     CArg<"ArrayRef<int64_t>", "{}">: $blockFactors),
      [{
        build($_builder, $_state, inputs, outputs, indexingMaps, iteratorTypes, singleThreadType, grid, blockFactors);
        llvm::SmallVector<Type> blockTypes = llvm::map_to_vector(TypeRange($_state.operands), [&](Type t) -> Type {
          mlir::RankedTensorType tensorType = mlir::cast<RankedTensorType>(t);
          auto layout = mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
          auto shardShape = layout.getShardShape(tensorType);
          return mlir::RankedTensorType::get(shardShape, tensorType.getElementType());
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
                     CArg<"ttcore::GridAttr", "nullptr">: $grid,
                     CArg<"ArrayRef<int64_t>", "{}">: $blockFactors),
      [{
        assert(outputs.size() == 1 && "expected single output");
        RankedTensorType tensorType = mlir::cast<RankedTensorType>(outputs[0].getType());
        ttcore::MetalLayoutAttr maybeLayout = mlir::dyn_cast_or_null<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
        const size_t rank = (maybeLayout != nullptr) ? maybeLayout.getShardShape(tensorType).size() : tensorType.getShape().size();
        auto [indexingMaps, iteratorTypes] = buildParallelAffineMapsAndIteratorTypes($_builder,
            inputs.size() + outputs.size(), rank);
        build($_builder, $_state, inputs, outputs, indexingMaps, iteratorTypes, singleThreadRegionBuilder, singleThreadType, grid);
      }]>,
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
      SmallVector<SmallVector<int64_t>> getOperandGridShapes();
      SmallVector<int64_t> getLoopBounds();
      SmallVector<AffineMap> getIndexingMapsValue();
      SmallVector<int64_t> getBlockFactorsValue();
      ttcore::DeviceAttr getDevice();
      SmallVector<int64_t> getParticipatingLoopDims(int64_t operandIndex);
      SmallVector<int64_t> getNonParticipatingLoopDims(int64_t operandIndex);
      SmallVector<SmallVector<int64_t>> getOperandShardShapes(bool convertTileToScalar = false);
      unsigned getNumLoops();
      unsigned getNumDims();
      AffineMap getIndexingMap(int64_t operandIndex);
      bool isAllParallel();
    }];
}

// Simple creation op needed by LowerToLayout splitting

def D2M_EmptyOp : D2M_Op<"empty", [
    Pure,
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

#endif // TTMLIR_TTMLIR_DIALECT_D2M_D2MOPS_TD
