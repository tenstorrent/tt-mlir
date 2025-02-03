// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIRTILEOPS_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIRTILEOPS_TD

include "ttmlir/Dialect/TT/IR/TTOpsTypes.td"
include "ttmlir/Dialect/TTIR/IR/TTIRBase.td"
include "ttmlir/Dialect/TTIR/IR/TTIROpsAttrs.td"
include "ttmlir/Dialect/TTIR/IR/TTIROpsTypes.td"
include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.td"

class TTIR_TileOp<string mnemonic, list<Trait> traits = []> :
    TTIR_Op<mnemonic, !listconcat([TTIR_GenericParent], traits)> {
}

//===----------------------------------------------------------------------===//
// TTIR Tile Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def TTIR_TileAddOp : TTIR_TileOp<"tile_add">{
    let summary = "TTIR Tile Add Op";
    let description = [{
        The `tile_add` operation adds two tensors element-wise.
    }];

    let arguments = (ins TT_Tile:$lhs, TT_Tile:$rhs);
    let results = (outs TT_Tile:$result); // questionable: should tile ops actually return anything? tensix math ops will leave results in dest regs, up to packer to get result to l1
}

def TTIR_TileSubOp : TTIR_TileOp<"tile_sub">{
    let summary = "TTIR Tile Sub Op";
    let description = [{
        The `tile_sub` operation subtracts two tensors element-wise.
    }];

    let arguments = (ins TT_Tile:$lhs, TT_Tile:$rhs);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileMulOp : TTIR_TileOp<"tile_mul">{
    let summary = "TTIR Tile Mul Op";
    let description = [{
        The `tile_mul` operation multiplies two tensors element-wise.
    }];

    let arguments = (ins TT_Tile:$lhs, TT_Tile:$rhs);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileMaximumOp : TTIR_TileOp<"tile_maximum">{
    let summary = "TTIR Tile Maximum Op";
    let description = [{
        The `tile_maximum` operation calculates the maximum of two tensors element-wise.
    }];

    let arguments = (ins TT_Tile:$lhs, TT_Tile:$rhs);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileRecipOp : TTIR_TileOp<"tile_recip">{
    let summary = "TTIR Tile Recip Op";
    let description = [{
        The `tile_recip` operation computes the reciprocal of each element in the input tensor.
    }];

    let arguments = (ins TT_Tile:$input);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileExpOp : TTIR_TileOp<"tile_exp">{
    let summary = "TTIR Tile Exp Op";
    let description = [{
        The `tile_exp` operation computes the exponential of each element in the input tensor.
    }];

    let arguments = (ins TT_Tile:$input);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileLogOp : TTIR_TileOp<"tile_log">{
    let summary = "TTIR Tile Log Op";
    let description = [{
        The `tile_log` operation computes the natural logarithm of each element in the input tensor.
    }];

    let arguments = (ins TT_Tile:$input);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileReduceSumOp : TTIR_TileOp<"tile_reduce_sum">{
    let summary = "TTIR Tile Reduce Sum Op";
    let description = [{
        The `tile_reduce_sum` operation computes the sum of all elements in the input tensor.
    }];

    let arguments = (ins TT_Tile:$a, TT_Tile:$b);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileMatmulBlockOp : TTIR_TileOp<"tile_matmul_block">{
    let summary = "TTIR Tile Matmul Block Op";
    let description = [{
        The `tile_matmul_block` operation computes the matrix multiplication of two input blocks.
    }];

    let arguments = (ins AnyNon0RankedMemRef:$a, AnyNon0RankedMemRef:$b);
    let results = (outs AnyNon0RankedMemRef:$result);

    let hasVerifier = 1;
}

def TTIR_DMAOp : TTIR_TileOp<"dma", 
  [ AttrSizedOperandSegments
  , MemoryEffects<[MemRead, MemWrite]> // Custom memory effects??
  ]> {
    let summary = "TTIR DMA Op";
    let description = [{
    }];

    let arguments = (ins AnyNon0RankedMemRef:$src, Variadic<Index>:$srcIndices, Variadic<Index>:$srcStride,
                         AnyNon0RankedMemRef:$dst, Variadic<Index>:$dstIndices, Variadic<Index>:$dstStride,
                         Index:$numElements);
    let results = (outs TTIR_MemTx:$result);
}

def TTIR_DMAWaitOp : TTIR_TileOp<"dma_wait", [MemoryEffects<[MemRead, MemWrite]>]> { // Custom memory effects??
    let summary = "TTIR DMA wait Op";
    let description = [{
    }];

    let arguments = (ins TTIR_MemTx:$mem_tx);
}

#endif // TTMLIR_TTMLIR_DIALECT_TTIR_TTIRTILEOPS_TD
