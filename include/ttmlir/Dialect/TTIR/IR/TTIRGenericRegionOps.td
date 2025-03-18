// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIRGENERICREGIONOPS_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIRGENERICREGIONOPS_TD

include "ttmlir/Dialect/TT/IR/TTOpsTypes.td"
include "ttmlir/Dialect/TTIR/IR/TTIROpsTypes.td"
include "ttmlir/Dialect/TTIR/IR/TTIRBase.td"
include "ttmlir/Dialect/TTIR/IR/TTIROpsAttrs.td"
include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.td"

include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

class TTIR_GenericRegionOp<string mnemonic, list<Trait> traits = [TTIR_GenericRegionOpTrait, TTIR_GenericParent]> :
    TTIR_Op<mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// TTIR Generic Region Math Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def TTIR_TileAddOp : TTIR_GenericRegionOp<"tile_add">{
    let summary = "TTIR Tile Add Op";
    let description = [{
        The `tile_add` operation adds two tiles element-wise.
    }];

    let arguments = (ins TT_Tile:$lhs,
                         TT_Tile:$rhs);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileSubOp : TTIR_GenericRegionOp<"tile_sub">{
    let summary = "TTIR Tile Sub Op";
    let description = [{
        The `tile_sub` operation subtracts two tiles element-wise.
    }];

    let arguments = (ins TT_Tile:$lhs,
                         TT_Tile:$rhs);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileMulOp : TTIR_GenericRegionOp<"tile_mul">{
    let summary = "TTIR Tile Mul Op";
    let description = [{
        The `tile_mul` operation multiplies two tiles element-wise.
    }];

    let arguments = (ins TT_Tile:$lhs,
                         TT_Tile:$rhs);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileRecipOp : TTIR_GenericRegionOp<"tile_recip">{
    let summary = "TTIR Tile Recip Op";
    let description = [{
        The `tile_recip` operation computes the reciprocal of each element in the input tile.
    }];

    let arguments = (ins TT_Tile:$input);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileExpOp : TTIR_GenericRegionOp<"tile_exp">{
    let summary = "TTIR Tile Exp Op";
    let description = [{
        The `tile_exp` operation computes the exponential of each element in the input tile.
    }];

    let arguments = (ins TT_Tile:$input);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileLogOp : TTIR_GenericRegionOp<"tile_log">{
    let summary = "TTIR Tile Log Op";
    let description = [{
        The `tile_log` operation computes the natural logarithm of each element in the input tile.
    }];

    let arguments = (ins TT_Tile:$input);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileReduceSumOp : TTIR_GenericRegionOp<"tile_reduce_sum">{
    let summary = "TTIR Tile Reduce Sum Op";
    let description = [{
        The `tile_reduce_sum` operation computes the weighted sum of all elements in the input tile over the specified reduction dim(s).
    }];

    let arguments = (ins TT_Tile:$input,
                         TT_Tile:$weight,
                         TTIR_ReduceDimAttr:$reduce_dim);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileReduceMaxOp : TTIR_GenericRegionOp<"tile_reduce_max">{
    let summary = "TTIR Tile Reduce Max Op";
    let description = [{
        The `tile_reduce_max` operation computes the max of all elements in the input tile over the specified reduction dim(s).
    }];

    let arguments = (ins TT_Tile:$a,
                         TT_Tile:$b,
                         TTIR_ReduceDimAttr:$reduce_dim);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileMatmulBlockOp : TTIR_GenericRegionOp<"tile_matmul_block",
  [DestinationStyleOpInterface, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "TTIR Tile Matmul Block Op";
    let description = [{
        The `tile_matmul_block` operation computes the matrix multiplication of two input blocks.
    }];

    let arguments = (ins AnyNon0RankedMemRef:$a,
                         AnyNon0RankedMemRef:$b,
                         AnyNon0RankedMemRef:$output);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTIR_TileTilizeBlockOp : TTIR_GenericRegionOp<"tile_tilize_block",
  [DestinationStyleOpInterface, MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "TTIR Tile Tilize Block Op";
    let description = [{
        The `tile_tilize_block` operation tilizes the input row major memref block and outputs the memref containing the tilized data.
    }];

    let arguments = (ins AnyNon0RankedMemRef:$input,
                         AnyNon0RankedMemRef:$output);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTIR_TileUntilizeBlockOp : TTIR_GenericRegionOp<"tile_untilize_block",
  [DestinationStyleOpInterface, MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "TTIR Tile Untilize Block Op";
    let description = [{
        The `tile_untilize_block` operation untilizes the input tilized memref block and outputs the memref contianing the row major data.
    }];

    let arguments = (ins AnyNon0RankedMemRef:$input,
                         AnyNon0RankedMemRef:$output);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// TTIR Generic Region Datamovement Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def TTIR_DMAOp : TTIR_GenericRegionOp<"dma",
  [ AttrSizedOperandSegments
  , DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
    let summary = "TTIR DMA Op";
    let description = [{
      DMA operation, moves data from src to dst where src and dst are memrefs that may or may not belong to
      different memory spaces. This op comes in a few flavors and is capable of roughly expressing everything
      that underlying Noc hardware can do. The op can be used to express a wide range of data movement operations:

      - Local to local
      ```mlir
      %tx = ttir.dma %src, %dst : (memref<6x8x!tt.tile<32x32, f32>, #l1_>, memref<6x8x!tt.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
      ```

      - Local to remote dram
      ```mlir
      %tx = ttir.dma %src, %dst : (memref<6x8x!tt.tile<32x32, f32>, #l1_>, memref<6x8x!tt.tile<32x32, f32>, #dram>) -> !ttir.mem_tx
      ```

      - Remote dram to local
      ```mlir
      %tx = ttir.dma %src, %dst : (memref<6x8x!tt.tile<32x32, f32>, #dram>, memref<6x8x!tt.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
      ```

      - Local to remote L1, e.g. core[1, 2]
      ```mlir
      %tx = ttir.dma %src, %dst, core[%c1, %c2] : (memref<6x8x!tt.tile<32x32, f32>, #dram>, memref<6x8x!tt.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
      ```

      - Local to mcast, e.g. starting at offset core[1, 2] with mcast shape [4, 4] (src and dst have the same SSA value, implies NoC doesn't loopback)
      ```mlir
      %tx = ttir.dma %foo, %foo, core[%c1, %c2] mcast[%c4, %c4] : (memref<6x8x!tt.tile<32x32, f32>, #l1_>, memref<6x8x!tt.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
      ```

      - Local to mcast w/ loopback (same as above but src and dst are different SSA values)
      ```mlir
      %tx = ttir.dma %src, %dst, core[%c1, %c2] mcast[%c4, %c4] : (memref<6x8x!tt.tile<32x32, f32>, #l1_>, memref<6x8x!tt.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
      ```

      - A lower level form, is "fully indexed" DMA, where the src and dst memrefs have thier ranks fully indexed the additional numElems argument
        is used to specify the number of elements to transfer. As opposed to the forms above that are not fully indexed which implicitly means to
        transfer the leftover block.  That is given memref<2x2x3x4x!tt.tile<>>, if we indexed by [i, j], since we only indexed a rank of 2, the remaining
        3x4 block is transferred.  In the fully indexed form, to express the same thing we would need to specify the number of elements to transfer:
      ```mlir
      %tx = ttir.dma %src[%i, %j, %k, %l], %dst[%i, %j, %k, %l], 12 : (memref<2x2x3x4x!tt.tile<32x32, f32>, #dram>, memref<2x2x3x4x!tt.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
      ```

      - Conversely, a higher level form is affine form, where the src and dst indices are implied by an affine map shared with the parent generic op.
        Given some affine map, as demonstrated below, the dma op will be evaluated at each point in the affine map's iteration space.
      ```mlir
      #map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
      %tx = ttir.dma %stream #map1, %cb0 : (memref<4x4x6x8x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 32768 + d3 * 4096)>, #l1_>, memref<6x8x!tt.tile<32x32, f32>, #l1_>) -> !ttir.mem
      ```
_tx

      Some constraints:
      - src and dst must have the same element type.
      - src and dst cannot both be remote
    }];

    let arguments = (ins AnyNon0RankedMemRef:$src, OptionalAttr<AffineMapAttr>:$srcAffineMap, Variadic<Index>:$srcIndices,
                         AnyNon0RankedMemRef:$dst, OptionalAttr<AffineMapAttr>:$dstAffineMap, Variadic<Index>:$dstIndices,
                         OptionalAttr<I64Attr>:$optNumElems, Variadic<Index>:$dstCoreIndex, Variadic<Index>:$mcastShape);
    let results = (outs TTIR_MemTx:$result);

    let assemblyFormat = [{ $src ($srcAffineMap^)? (`[` $srcIndices^ `]`)? `,` $dst ($dstAffineMap^)? (`[` $dstIndices^ `]`)? (`,` $optNumElems^)? (`,` `core` `[` $dstCoreIndex^ `]`)? (`mcast` `[` $mcastShape^ `]`)? attr-dict `:` `(` type($src) `,` type($dst) `)` `->` type($result)}];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      MemRefType getSrcMemRefType() { return cast<MemRefType>(getSrc().getType()); }
      MemRefType getDstMemRefType() { return cast<MemRefType>(getDst().getType()); }
      int64_t getNumElems();
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
      bool isAffine() { return getSrcAffineMap() || getDstAffineMap(); }
      bool isFullyIndexed() {
        return static_cast<int64_t>(getSrcIndices().size()) == getSrcMemRefType().getRank() &&
               static_cast<int64_t>(getDstIndices().size()) == getDstMemRefType().getRank() &&
               getOptNumElems();
      }
    }];
}

def TTIR_DMAWaitOp : TTIR_GenericRegionOp<"dma_wait", [MemoryEffects<[MemRead, MemWrite]>]> {
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
      ttir.semaphore_set %sem0, %c1 core[%c2, %c2]
      ttir.semaphore_inc %sem0, %c1 core[%c2, %c2]
      ```

      - Update a remote mcast region of cores' semaphore value
      ```mlir
      ttir.semaphore_set %sem0, %c1 core[%c2, %c2] mcast[%c4, %c4]
      ttir.semaphore_inc %sem0, %c1 core[%c2, %c2] mcast[%c4, %c4]
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

    let arguments = (ins Variadic<AnyNon0RankedMemRef>:$values);

    let assemblyFormat = [{ $values attr-dict `:` `(` type($values) `)` }];

    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TTIR Generic Region Indexing Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

class TTIR_IndexOp<string mnemonic> : TTIR_GenericRegionOp<mnemonic,
  [ Pure
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
    let arguments = (ins ConfinedAttr<I64Attr, [IntMinValue<0>]>:$dim);
    let results = (outs Index:$result);
    let assemblyFormat = [{ `(` $dim `)` attr-dict `:` type($result) }];
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

#endif // TTMLIR_TTMLIR_DIALECT_TTIR_TTIRGENERICREGIONOPS_TD
