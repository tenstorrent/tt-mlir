// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIRGENERICREGIONOPS_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIRGENERICREGIONOPS_TD

include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td"
include "ttmlir/Dialect/TTIR/IR/TTIROpsTypes.td"
include "ttmlir/Dialect/TTIR/IR/TTIRBase.td"
include "ttmlir/Dialect/TTIR/IR/TTIROpsAttrs.td"
include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.td"

include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"

//===----------------------------------------------------------------------===//
// Generic Region Op Traits and Classes
//===----------------------------------------------------------------------===//

def IsDeviceL1MemorySpace : CPred<"::llvm::cast<::mlir::tt::ttcore::MemorySpaceAttr>(::llvm::cast<::mlir::MemRefType>($_self).getMemorySpace()).getValue() == ::mlir::tt::ttcore::MemorySpace::DeviceL1">;

def IsDeviceDRAMMemorySpace : CPred<"::llvm::cast<::mlir::tt::ttcore::MemorySpaceAttr>(::llvm::cast<::mlir::MemRefType>($_self).getMemorySpace()).getValue() == ::mlir::tt::ttcore::MemorySpace::DeviceDRAM">;

def IsDeviceMemorySpace : Or<[IsDeviceL1MemorySpace, IsDeviceDRAMMemorySpace]>;

def IsRegisterDstMemorySpace : CPred<"::llvm::cast<::mlir::tt::ttcore::MemorySpaceAttr>(::llvm::cast<::mlir::MemRefType>($_self).getMemorySpace()).getValue() == ::mlir::tt::ttcore::MemorySpace::RegisterDst">;

def DeviceL1MemRef : Type<
  And<[IsDeviceL1MemorySpace, HasRankGreaterOrEqualPred<1>]>,
  "device l1 memoryspace memref type",
  "::mlir::MemRefType">;

def DeviceMemRef : Type<
  And<[IsDeviceMemorySpace, HasRankGreaterOrEqualPred<1>]>,
  "device memoryspace memref type",
  "::mlir::MemRefType">;

def RegisterDstMemRef : Type<
  And<[IsRegisterDstMemorySpace, HasRankGreaterOrEqualPred<1>]>,
  "register memoryspace memref type",
  "::mlir::MemRefType">;

def DeviceOrRegisterMemRef : AnyTypeOf<[DeviceMemRef, RegisterDstMemRef]>;

class TTIR_GenericRegionOp<string mnemonic, list<Trait> traits = []> :
    TTIR_Op<mnemonic, [TTIR_GenericParent] # traits> {}

class TTIR_GenericRegionComputeOp<string mnemonic, list<Trait> traits = []> :
    TTIR_GenericRegionOp<mnemonic, [TTIR_GenericRegionComputeOpTrait, TTIR_OperandLoadRegisterOpInterface] # traits> {}

class TTIR_GenericRegionComputeUnaryDstOp<string mnemonic, list<Trait> traits = []> :
    TTIR_GenericRegionComputeOp<mnemonic, traits> {

  let extraClassDeclaration = [{
    mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister() {
      return {0};
    }
  }];
}

class TTIR_GenericRegionComputeBinaryDstOp<string mnemonic, list<Trait> traits = []> :
    TTIR_GenericRegionComputeOp<mnemonic, traits> {

  let extraClassDeclaration = [{
    mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister() {
      return {0, 1};
    }
  }];
}

class TTIR_GenericRegionDatamovementOp<string mnemonic, list<Trait> traits = []> :
    TTIR_GenericRegionOp<mnemonic, [TTIR_GenericRegionDatamovementOpTrait] # traits> {}

//===----------------------------------------------------------------------===//
// TTIR Generic Region Math Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def TTIR_TileAddOp : TTIR_GenericRegionComputeOp<"tile_add"> {
    let summary = "TTIR Tile Add Op";
    let description = [{
        The `tile_add` operation adds two tiles element-wise.
    }];

    let arguments = (ins TTCore_Tile:$lhs,
                         TTCore_Tile:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TileSubOp : TTIR_GenericRegionComputeOp<"tile_sub"> {
    let summary = "TTIR Tile Sub Op";
    let description = [{
        The `tile_sub` operation subtracts two tiles element-wise.
    }];

    let arguments = (ins TTCore_Tile:$lhs,
                         TTCore_Tile:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TileMulOp : TTIR_GenericRegionComputeOp<"tile_mul"> {
    let summary = "TTIR Tile Mul Op";
    let description = [{
        The `tile_mul` operation multiplies two tiles element-wise.
    }];

    let arguments = (ins TTCore_Tile:$lhs,
                         TTCore_Tile:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TileDivOp : TTIR_GenericRegionComputeBinaryDstOp<"tile_div"> {
    let summary = "TTIR Tile Div Op";
    let description = [{
        The `tile_div` operation divides two tiles element-wise.
    }];

    let arguments = (ins TTCore_Tile:$lhs,
                         TTCore_Tile:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TileRecipOp : TTIR_GenericRegionComputeUnaryDstOp<"tile_recip"> {
    let summary = "TTIR Tile Recip Op";
    let description = [{
        The `tile_recip` operation computes the reciprocal of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TilePowOp : TTIR_GenericRegionComputeBinaryDstOp<"tile_pow"> {
    let summary = "TTIR Tile Pow Op";
    let description = [{
        The `tile_pow` operation raises two tiles element-wise.
    }];

    let arguments = (ins TTCore_Tile:$lhs,
                         TTCore_Tile:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TileExpOp : TTIR_GenericRegionComputeUnaryDstOp<"tile_exp"> {
    let summary = "TTIR Tile Exp Op";
    let description = [{
        The `tile_exp` operation computes the exponential of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TileLogOp : TTIR_GenericRegionComputeUnaryDstOp<"tile_log"> {
    let summary = "TTIR Tile Log Op";
    let description = [{
        The `tile_log` operation computes the natural logarithm of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TileNegativeOp : TTIR_GenericRegionComputeUnaryDstOp<"tile_negative"> {
    let summary = "TTIR Tile Negative Op";
    let description = [{
        The `tile_negative` operation computes the negative of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TileCosOp : TTIR_GenericRegionComputeUnaryDstOp<"tile_cos"> {
    let summary = "TTIR Tile Cos Op";
    let description = [{
        The `tile_cos` operation computes the cosine function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TileTanOp : TTIR_GenericRegionComputeUnaryDstOp<"tile_tan"> {
    let summary = "TTIR Tile Tan Op";
    let description = [{
        The `tile_tan` operation computes the tangent function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TileSqrtOp : TTIR_GenericRegionComputeUnaryDstOp<"tile_sqrt"> {
    let summary = "TTIR Tile Sqrt Op";
    let description = [{
        The `tile_sqrt` operation computes the sqrt function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TileRsqrtOp : TTIR_GenericRegionComputeUnaryDstOp<"tile_rsqrt"> {
    let summary = "TTIR Tile Rsqrt Op";
    let description = [{
        The `tile_rsqrt` operation computes the reciprocal sqrt function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TileSinOp : TTIR_GenericRegionComputeUnaryDstOp<"tile_sin"> {
    let summary = "TTIR Tile Sin Op";
    let description = [{
        The `tile_sin` operation computes the sine function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TileSigmoidOp : TTIR_GenericRegionComputeUnaryDstOp<"tile_sigmoid"> {
    let summary = "TTIR Tile Sigmoid Op";
    let description = [{
        The `tile_sigmoid` operation computes the sigmoid of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TileCeilOp : TTIR_GenericRegionComputeUnaryDstOp<"tile_ceil"> {
    let summary = "TTIR Tile Ceil Op";
    let description = [{
        The `tile_ceil` operation computes the ceiling function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TileFloorOp : TTIR_GenericRegionComputeUnaryDstOp<"tile_floor"> {
    let summary = "TTIR Tile Floor Op";
    let description = [{
        The `tile_floor` operation computes the floor function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TileAbsOp : TTIR_GenericRegionComputeUnaryDstOp<"tile_abs"> {
    let summary = "TTIR Tile Abs Op";
    let description = [{
        The `tile_abs` operation computes the absolute value of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TileLogicalNotOp : TTIR_GenericRegionComputeUnaryDstOp<"tile_logical_not"> {
    let summary = "TTIR Tile Logical Not Op";
    let description = [{
        The `tile_logical_not` operation computes the logical negation of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TileMaximumOp : TTIR_GenericRegionComputeBinaryDstOp<"tile_maximum">{
    let summary = "TTIR Tile Maximum Op";
    let description = [{
        The `tile_maximum` operation calculates the maximum of two tensors element-wise.
    }];

    let arguments = (ins TTCore_Tile:$lhs, TTCore_Tile:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TileReduceSumOp : TTIR_GenericRegionComputeOp<"tile_reduce_sum">{
    let summary = "TTIR Tile Reduce Sum Op";
    let description = [{
        The `tile_reduce_sum` operation computes the weighted sum of all elements in the input tile over the specified reduction dim(s).
    }];

    let arguments = (ins TTCore_Tile:$input,
                         TTCore_Tile:$weight,
                         TTIR_ReduceDimAttr:$reduce_dim);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TileReduceMaxOp : TTIR_GenericRegionComputeOp<"tile_reduce_max"> {
    let summary = "TTIR Tile Reduce Max Op";
    let description = [{
        The `tile_reduce_max` operation computes the max of all elements in the input tile over the specified reduction dim(s).
    }];

    let arguments = (ins TTCore_Tile:$a,
                         TTCore_Tile:$b,
                         TTIR_ReduceDimAttr:$reduce_dim);
    let results = (outs TTCore_Tile:$result);
}

def TTIR_TileMatmulOp : TTIR_GenericRegionComputeOp<"tile_matmul"> {
  let summary = "TTIR Tile Matmul Op";
  let description = [{
        The `tile_matmul` operation computes the matrix multiplication of A and B input tiles and element-wise adds C tile: result <- a @ b + c.
    }];

  let arguments = (ins TTCore_Tile:$a, TTCore_Tile:$b, TTCore_Tile:$c);
  let results = (outs TTCore_Tile:$result);

  let extraClassDeclaration = [{
    mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister() {
      return {2};
    }
  }];
}

def TTIR_TileMatmulBlockOp : TTIR_GenericRegionComputeOp<"tile_matmul_block",
  [DestinationStyleOpInterface, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "TTIR Tile Matmul Block Op";
    let description = [{
        The `tile_matmul_block` operation computes the matrix multiplication of two input blocks.
    }];

    let arguments = (ins DeviceL1MemRef:$a,
                         DeviceL1MemRef:$b,
                         DeviceL1MemRef:$output);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTIR_TileTilizeBlockOp : TTIR_GenericRegionComputeOp<"tile_tilize_block",
  [DestinationStyleOpInterface, MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "TTIR Tile Tilize Block Op";
    let description = [{
        The `tile_tilize_block` operation tilizes the input row major memref block and outputs the memref containing the tilized data.
    }];

    let arguments = (ins DeviceL1MemRef:$input,
                         DeviceL1MemRef:$output);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTIR_TileUntilizeBlockOp : TTIR_GenericRegionComputeOp<"tile_untilize_block",
  [DestinationStyleOpInterface, MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "TTIR Tile Untilize Block Op";
    let description = [{
        The `tile_untilize_block` operation untilizes the input tilized memref block and outputs the memref contianing the row major data.
    }];

    let arguments = (ins DeviceL1MemRef:$input,
                         DeviceL1MemRef:$output);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTIR_TileTypecastOp : TTIR_GenericRegionComputeUnaryDstOp<"tile_typecast"> {
    let summary = "TTIR Tile Typecast Op";
    let description = [{
        The `tile_typecast` operation casts the input tile to the desired dataformat.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

//===----------------------------------------------------------------------===//
// TTIR Destination Control Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def TTIR_AcquireDstOp : TTIR_GenericRegionComputeOp<"acquire_dst",
  [ MemoryEffects<[MemRead, MemWrite]>
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
    let summary = "Acquire Destination Register op.";
    let description = [{
      This op returns a memref with memory space dest that models the destination register
      resource on tensorrent hardware. Example IR:

      ```mlir
      %dst = ttir.acquire_dst() : memref<2x4x!tt.tile<32x32, f32>, #tt.memory_space<dst>>
      ```
    }];

    let results = (outs RegisterDstMemRef:$result);

    let assemblyFormat = [{ `(` `)` attr-dict `:` type($result) }];
}

//===----------------------------------------------------------------------===//
// TTIR Generic Region Datamovement Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def TTIR_DMAOp : TTIR_GenericRegionDatamovementOp<"dma",
  [ AttrSizedOperandSegments
  , DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                       , "bufferizesToMemoryWrite"
                                                       , "bufferize"
                                                       , "getAliasingValues"
                                                       , "getBufferType"
                                                       ]>
  ]> {
    let summary = "TTIR DMA Op";
    let description = [{
      DMA operation, moves data from src to dst where src and dst are memrefs that may or may not belong to
      different memory spaces. This op comes in a few flavors and is capable of roughly expressing everything
      that underlying Noc hardware can do. The op can be used to express a wide range of data movement operations:

      - Local to local
      ```mlir
      %tx = ttir.dma %src, %dst : (memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
      ```

      - Local to remote dram
      ```mlir
      %tx = ttir.dma %src, %dst : (memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, memref<1x1x6x8x!ttcore.tile<32x32, f32>, $ttcore.shard<...>, #dram>) -> !ttir.mem_tx
      ```

      - Remote dram to local
      ```mlir
      %tx = ttir.dma %src, %dst : (memref<1x1x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<...>, #dram>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
      ```

      - Local to mcast, e.g. starting at offset core[1, 2] with mcast shape [4, 4] (src and dst have the same SSA value, implies NoC doesn't loopback)
      ```mlir
      %tx = ttir.dma %foo, %foo, core[%c1, %c2] mcast[%c4, %c4] : (memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
      ```

      - Local to mcast w/ loopback (same as above but src and dst are different SSA values)
      ```mlir
      %tx = ttir.dma %src, %dst, core[%c1, %c2] mcast[%c4, %c4] : (memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
      ```

      - A lower level form, is "fully indexed" DMA, where the src and dst memrefs have thier ranks fully indexed the additional numElems argument
        is used to specify the number of elements to transfer. As opposed to the forms above that are not fully indexed which implicitly means to
        transfer the leftover block.  That is given memref<2x2x3x4x!ttcore.tile<>>, if we indexed by [i, j], since we only indexed a rank of 2, the remaining
        3x4 block is transferred.  In the fully indexed form, to express the same thing we would need to specify the number of elements to transfer:
      ```mlir
      %tx = ttir.dma %src[%i, %j, %k, %l], %dst[%i, %j, %k, %l], 12 : (memref<2x2x3x4x!ttcore.tile<32x32, f32>, #dram>, memref<2x2x3x4x!ttcore.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
      ```

      - Conversely, a higher level form is affine form, where the src and dst indices are implied by an affine map shared with the parent generic op.
        Given some affine map, as demonstrated below, the dma op will be evaluated at each point in the affine map's iteration space.
      ```mlir
      #map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
      %tx = ttir.dma %stream<#map1>, %cb0 : (memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<rank(4)>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
      ```

      Some constraints:
      - src and dst must have the same element type.
      - src and dst cannot both be remote
    }];

    let arguments = (ins AnyRankedTensorOrMemRef:$src, OptionalAttr<AffineMapAttr>:$srcAffineMap, Variadic<Index>:$srcIndices,
                         AnyRankedTensorOrMemRef:$dst, OptionalAttr<AffineMapAttr>:$dstAffineMap, Variadic<Index>:$dstIndices,
                         OptionalAttr<I64Attr>:$optNumElems, Variadic<Index>:$mcastStartIndex, Variadic<Index>:$mcastShape);
    let results = (outs TTIR_MemTx:$result);

    let builders =
    [
      // %tx = ttir.dma %src, %dst
      OpBuilder<(ins "Value": $src, "Value": $dst),
      [{
        build($_builder, $_state, src, nullptr, dst);
      }]>,
      // %tx = ttir.dma %src<#map>, %dst
      OpBuilder<(ins "Value": $src, "AffineMapAttr": $srcAffineMap, "Value": $dst),
      [{
        build($_builder, $_state, src, srcAffineMap, dst, ValueRange(), ValueRange());
      }]>,
      // %tx = ttir.dma %src<#map>, %dst core[%c0, %c1] mcast[%c2, %c3]
      OpBuilder<(ins "Value": $src, "AffineMapAttr": $srcAffineMap, "Value": $dst, "ValueRange": $mcastStartIndex, "ValueRange": $mcastShape),
      [{
        build($_builder, $_state, $_builder.getType<MemTxType>(), src, srcAffineMap, ValueRange(), dst, nullptr, ValueRange(), nullptr, mcastStartIndex, mcastShape);
      }]>,
      // %tx = ttir.dma %src, %dst<#map>
      OpBuilder<(ins "Value": $src, "Value": $dst, "AffineMapAttr": $dstAffineMap),
      [{
        build($_builder, $_state, $_builder.getType<MemTxType>(), src, nullptr, ValueRange(), dst, dstAffineMap, ValueRange(), nullptr, ValueRange(), ValueRange());
      }]>,
      // %tx = ttir.dma %src[%c0, %c1], %dst
      OpBuilder<(ins "Value": $src, "ValueRange": $srcIndices, "Value": $dst),
      [{
        build($_builder, $_state, src, srcIndices, dst, ValueRange());
      }]>,
      // %tx = ttir.dma %src[%c0, %c1], %dst core[%c2, %c3] mcast[%c4, %c5]
      OpBuilder<(ins "Value": $src, "ValueRange": $srcIndices, "Value": $dst, "ValueRange": $mcastStartIndex, "ValueRange": $mcastShape),
      [{
        build($_builder, $_state, $_builder.getType<MemTxType>(), src, nullptr, srcIndices, dst, nullptr, ValueRange(), nullptr, mcastStartIndex, mcastShape);
      }]>,
      // %tx = ttir.dma %src, %dst[%c0, %c1]
      OpBuilder<(ins "Value": $src, "Value": $dst, "ValueRange": $dstIndices),
      [{
        build($_builder, $_state, src, ValueRange(), dst, dstIndices);
      }]>,
      // %tx = ttir.dma %src[%c0, %c1], %dst[%c2, %c3]
      OpBuilder<(ins "Value": $src, "ValueRange": $srcIndices, "Value": $dst, "ValueRange": $dstIndices),
      [{
        build($_builder, $_state, src, srcIndices, dst, dstIndices, ValueRange(), ValueRange());
      }]>,
      // %tx = ttir.dma %src[%c0, %c1], %dst[%c2, %c3] core[%c4, %c5] mcast[%c6, %c7]
      OpBuilder<(ins "Value": $src, "ValueRange": $srcIndices, "Value": $dst, "ValueRange": $dstIndices, "ValueRange": $mcastStartIndex, "ValueRange": $mcastShape),
      [{
        build($_builder, $_state, $_builder.getType<MemTxType>(), src, nullptr, srcIndices, dst, nullptr, dstIndices, nullptr, mcastStartIndex, mcastShape);
      }]>,
      // %tx = ttir.dma %src[%c0, %c1], %dst[%c2, %c3], <4>
      OpBuilder<(ins "Value": $src, "ValueRange": $srcIndices, "Value": $dst, "ValueRange": $dstIndices, "size_t": $numElems),
      [{
        build($_builder, $_state, src, srcIndices, dst, dstIndices, ValueRange(), ValueRange(), numElems);
      }]>,
      // %tx = ttir.dma %src[%c0, %c1], %dst[%c2, %c3] core[%c4, %c5] mcast[%c6, %c7], <4>
      OpBuilder<(ins "Value": $src, "ValueRange": $srcIndices, "Value": $dst, "ValueRange": $dstIndices, "ValueRange": $mcastStartIndex, "ValueRange": $mcastShape, "size_t": $numElems),
      [{
        build($_builder, $_state, $_builder.getType<MemTxType>(), src, nullptr, srcIndices, dst, nullptr, dstIndices, $_builder.getI64IntegerAttr(numElems), mcastStartIndex, mcastShape);
      }]>,
    ];

    let assemblyFormat = [{ $src (`<` $srcAffineMap^ `>`)? (`[` $srcIndices^ `]`)? `,` $dst (`<` $dstAffineMap^ `>`)? (`[` $dstIndices^ `]`)? (`core` `[` $mcastStartIndex^ `]` `mcast` `[` $mcastShape `]`)? (`,` `<` $optNumElems^ `>`)? attr-dict `:` `(` type($src) `,` type($dst) `)` `->` type($result)}];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      MemRefType getSrcMemRefType() { return cast<MemRefType>(getSrc().getType()); }
      MemRefType getDstMemRefType() { return cast<MemRefType>(getDst().getType()); }
      int64_t getNumElems();
      size_t getSizeBytes();
      bool isSrcLocal() {
        Block *block = getSrc().getParentBlock();
        Block::BlockArgListType blockArgs = block->getArguments();
        return std::find(blockArgs.begin(), blockArgs.end(), getSrc()) !=
               blockArgs.end();
      }
      bool isSrcRemote() { return !isSrcLocal(); }
      bool isDstLocal() {
        Block *block = getDst().getParentBlock();
        Block::BlockArgListType blockArgs = block->getArguments();
        return std::find(blockArgs.begin(), blockArgs.end(), getDst()) !=
               blockArgs.end();
      }
      bool isDstRemote() { return !isDstLocal(); }
      bool isMcast() { return !getMcastShape().empty(); }
      bool isAffine() { return (getSrcAffineMap() || getDstAffineMap()) && getSrcIndices().empty() && getDstIndices().empty(); }
      bool isLowered() {
        return bool(getOptNumElems());
      }
    }];
}

def TTIR_NullTxOp : TTIR_GenericRegionDatamovementOp<"null_tx", [Pure]> {
    let summary = "Create a null transaction.";
    let description = [{
      Utility op to create a null transaction.  This is required for creating a sentinel
      starting transaction for a DMA nested inside of a loop nest.
    }];

    let results = (outs TTIR_MemTx:$result);

    let builders =
    [
      OpBuilder<(ins),
      [{
        build($_builder, $_state, $_builder.getType<MemTxType>());
      }]>
    ];

    let assemblyFormat = [{ attr-dict }];
}

def TTIR_DMAWaitOp : TTIR_GenericRegionDatamovementOp<"dma_wait", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "TTIR DMA wait Op";
    let description = [{
      Waits for the producer DMA memory transaction to complete.
    }];

    let arguments = (ins TTIR_MemTx:$mem_tx);

    let assemblyFormat = [{ $mem_tx attr-dict }];
}

//===----------------------------------------------------------------------===//
// TTIR Generic Region Semaphore Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

class TTIR_SemaphoreUpdateOp<string mnemonic> : TTIR_GenericRegionOp<mnemonic,
  [ AttrSizedOperandSegments
  , MemoryEffects<[MemRead, MemWrite]>
  ]> {
    let summary = "TTIR Semaphore Set or Inc Op";
    let description = [{
      Set or increment the semaphore value atomically. This op comes in a few flavors:

      - Set or increment this local core's semaphore value
      ```mlir
      ttir.semaphore_set %sem0, %c1
      ttir.semaphore_inc %sem0, %c1
      ```

      - Update a remote core's semaphore value
      ```mlir
      ttir.semaphore_inc %sem0, %c1 core[%c2, %c2]
      ```

      - Update a remote mcast region of cores' semaphore value
      ```mlir
      ttir.semaphore_set %sem0, %c1 core[%c2, %c2] mcast[%c4, %c4]
      ```
    }];

    let arguments = (ins TTIR_Semaphore:$semaphore, Index:$value,
                         Variadic<Index>:$dstCoreIndex, Variadic<Index>:$mcastShape);

    let assemblyFormat = [{ $semaphore `,` $value (`,` `core` `[` $dstCoreIndex^ `]`)? (`mcast` `[` $mcastShape^ `]`)? attr-dict }];

    let builders =
    [
      OpBuilder<(ins "Value":$semaphore, "Value":$value),
      [{
        build($_builder, $_state, semaphore, value, ValueRange(), ValueRange());
      }]>,
      OpBuilder<(ins "Value":$semaphore, "Value":$value, "ValueRange":$dstCoreIndex),
      [{
        build($_builder, $_state, semaphore, value, dstCoreIndex, ValueRange());
      }]>,
    ];
}

def TTIR_SemaphoreSetOp : TTIR_SemaphoreUpdateOp<"semaphore_set"> {}

def TTIR_SemaphoreIncOp : TTIR_SemaphoreUpdateOp<"semaphore_inc"> {}

def TTIR_SemaphoreWaitOp : TTIR_GenericRegionOp<"semaphore_wait", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "TTIR Semaphore Set Op.";
    let description = [{
      Wait for the semaphore value to reach the specified value. Optionall supply a reset value as a shorthand syntax.

      ```mlir
      ttir.semaphore_wait %sem1, %c1 reset %c0
      // is equivalent to
      ttir.semaphore_wait %sem1, %c1
      ttir.semaphore_set %sem1, %c0
      ```
    }];

    let arguments = (ins TTIR_Semaphore:$semaphore, Index:$value, Optional<Index>:$resetValue);

    let assemblyFormat = [{ $semaphore `,` $value (`reset` $resetValue^)? attr-dict }];

    let builders =
    [
      OpBuilder<(ins "Value":$semaphore, "Value":$value),
      [{
        build($_builder, $_state, semaphore, value, nullptr);
      }]>,
    ];
}

//===----------------------------------------------------------------------===//
// TTIR Generic Region Control Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def TTIR_YieldOp : TTIR_GenericRegionOp<"yield", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "Yield op.";
    let description = [{
      Yield operation, return control flow to another thread. Corresponds to a circular buffer push.
    }];

    let arguments = (ins Variadic<AnyRankedTensorOrMemRef>:$values);

    let assemblyFormat = [{ $values attr-dict `:` `(` type($values) `)` }];

    let hasVerifier = 1;
}

def TTIR_AwaitOp : TTIR_GenericRegionOp<"await", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "Await op.";
    let description = [{
      Await operation, await control flow from another thread. Corresponds to a circular buffer wait.
    }];

    let arguments = (ins Variadic<DeviceL1MemRef>:$values);

    let assemblyFormat = [{ $values attr-dict `:` `(` type($values) `)` }];

    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TTIR Generic Region Indexing Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

class TTIR_IndexOp<string mnemonic, list<Trait> traits = []> : TTIR_GenericRegionOp<mnemonic,
  traits #
  [ Pure
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  , DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  ]> {
    let arguments = (ins ConfinedAttr<I64Attr, [IntMinValue<0>]>:$dim);
    let results = (outs Index:$result);
    let assemblyFormat = [{ `(` $dim `)` attr-dict `:` type($result) }];
    let hasFolder = true;

    let builders =
    [
      OpBuilder<(ins "int64_t":$dim),
      [{
        build($_builder, $_state, $_builder.getIndexType(), $_builder.getI64IntegerAttr(dim));
      }]>
    ];
}

def TTIR_IterIndexOp : TTIR_IndexOp<"iter_index"> {
    let summary = "Iter Index op.";
    let description = [{
      Return the index of the current element in the iteration for the given generic op dimension.
    }];
}

def TTIR_CoreIndexOp : TTIR_IndexOp<"core_index"> {
    let summary = "Core Index op.";
    let description = [{
      Return the index of this core's coordinate inside the generic op's grid dimension.
    }];
}

//===----------------------------------------------------------------------===//
// TTIR Remote Access ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def TTIR_GetGlobalOperandOp : TTIR_GenericRegionOp<"get_global_operand", [Pure]> {
    let summary = "Get global operand op.";
    let description = [{
      Access the global, aka parent generic op, operand at the specified index.

      The following forms are all equivalent, but the latter forms are required
      when moving generic regions to top level func ops in the module.
      ```mlir
      ttir.generic (%arg0, %arg1, %arg2) {
        ^datamovement(%cb0, %cb1, %cb2)
          ttir.dma %arg1, %cb1 // Capture %arg1 from parent scope
      }
      ```

      And:
      ```mlir
      ttir.generic (%arg0, %arg1, %arg2) {
        ^datamovement(%cb0, %cb1, %cb2)
          %operand_arg1 = ttir.get_global_operand 1
          ttir.dma %operand_arg1, %cb0
      }
      ```

      And:
      ```mlir
      func.func @main(...) {
        ttir.generic (%arg0, %arg1, %arg2) { kernel_symbols = [@dm0] }
      }

      func.func private @dm0(%cb0, %cb1, %cb2) {
        %operand_arg1 = ttir.get_global_operand 1
        ttir.dma %operand_arg1, %cb0
      }
      ```
    }];

    let arguments = (ins ConfinedAttr<I64Attr, [IntMinValue<0>]>:$operand_index);
    let results = (outs DeviceMemRef:$result);

    let assemblyFormat = [{ `(` $operand_index `)` attr-dict `:` type($result) }];
}

#endif // TTMLIR_TTMLIR_DIALECT_TTIR_TTIRGENERICREGIONOPS_TD
