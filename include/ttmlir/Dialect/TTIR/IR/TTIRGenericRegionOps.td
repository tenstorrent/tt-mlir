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

include "mlir/Interfaces/ControlFlowInterfaces.td"

class TTIR_GenericRegionOp<string mnemonic, list<Trait> traits = [TTIR_GenericRegionOpTrait, TTIR_GenericParent]> :
    TTIR_Op<mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// TTIR Generic Region Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def AnyRankedTensorOrMemRef: AnyTypeOf<[AnyRankedTensor, AnyNon0RankedMemRef]>;

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

def TTIR_TileMaximumOp : TTIR_GenericRegionOp<"tile_maximum">{
    let summary = "TTIR Tile Maximum Op";
    let description = [{
        The `tile_maximum` operation calculates the maximum of two tensors element-wise.
    }];

    let arguments = (ins TT_Tile:$lhs, TT_Tile:$rhs);
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
  [DestinationStyleOpInterface, MemoryEffects<[MemRead, MemWrite]>]> {
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

def TTIR_DMAOp : TTIR_GenericRegionOp<"dma", 
  [ AttrSizedOperandSegments
  , MemoryEffects<[MemRead, MemWrite]> // Custom memory effects??
  ]> {
    let summary = "TTIR DMA Op";
    let description = [{
    }];

    let arguments = (ins AnyNon0RankedMemRef:$src, Variadic<Index>:$srcIndices, Variadic<Index>:$srcStride,
                         AnyNon0RankedMemRef:$dst, Variadic<Index>:$dstIndices, Variadic<Index>:$dstStride,
                         Index:$numElements, Variadic<Index>:$mcastOffset, Variadic<Index>:$mcastShape);
    let results = (outs TTIR_MemTx:$result);

    let assemblyFormat = [{ $src `[` $srcIndices `]` `[` $srcStride `]` `,` $dst `[` $dstIndices `]` `[` $dstStride `]` `,` $numElements (`,` `mcast` `[` $mcastOffset^ `]` `[` $mcastShape `]`)? attr-dict `:` `(` type($src) `,` type($dst) `)` `->` type($result)}];
}

def TTIR_DMAWaitOp : TTIR_GenericRegionOp<"dma_wait", [MemoryEffects<[MemRead, MemWrite]>]> { // Custom memory effects??
    let summary = "TTIR DMA wait Op";
    let description = [{
    }];

    let arguments = (ins TTIR_MemTx:$mem_tx);

    let assemblyFormat = [{ $mem_tx attr-dict }];
}

def TTIR_IterIndexOp : TTIR_Op<"iter_index", [HasParent<"GenericOp">, Pure]> {
    let summary = "Iter Index op.";
    let description = [{
      Return the index of the current element in the iteration for the given generic op dimension.
    }];

    let arguments = (ins ConfinedAttr<I64Attr, [IntMinValue<0>]>:$dim);
    let results = (outs Index:$result);
    let assemblyFormat = [{ `(` $dim `)` attr-dict `:` type($result) }];
}

def TTIR_GridIndexOp : TTIR_Op<"grid_index", [HasParent<"GenericOp">, Pure]> {
    let summary = "Grid Index op.";
    let description = [{
      Return the index of the this grid coordinate for the given grid dimension.
    }];

    let arguments = (ins ConfinedAttr<I64Attr, [IntMinValue<0>]>:$dim);
    let results = (outs Index:$result);
    let assemblyFormat = [{ `(` $dim `)` attr-dict `:` type($result) }];
}

def TTIR_YieldOp : TTIR_Op<"yield", [HasParent<"GenericOp">, MemoryEffects<[MemRead, MemWrite]>, TTIR_GenericRegionOpTrait]> {
    let summary = "Yield op.";
    let description = [{
      Yield operation, return control flow to another thread
    }];

    let arguments = (ins Variadic<AnyNon0RankedMemRef>:$values);

    let assemblyFormat = [{ $values attr-dict `:` `(` type($values) `)` }];

    let hasVerifier = 1;
}

def TTIR_AwaitOp : TTIR_Op<"await", [HasParent<"GenericOp">, MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "Await op.";
    let description = [{
      Await operation, return control flow to another thread
    }];

    let arguments = (ins Variadic<AnyNon0RankedMemRef>:$values);

    let assemblyFormat = [{ $values attr-dict `:` `(` type($values) `)` }];

    let hasVerifier = 1;
}

def TTIR_SynchronizeOp : TTIR_Op<"synchronize", [HasParent<"GenericOp">, MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "Synchronize op.";
    let description = [{
    }];

    let assemblyFormat = [{ attr-dict }];
}

#endif // TTMLIR_TTMLIR_DIALECT_TTIR_TTIRGENERICREGIONOPS_TD
