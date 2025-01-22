// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIRTILEOPS_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIRTILEOPS_TD

include "ttmlir/Dialect/TT/IR/TTOpsTypes.td"
include "ttmlir/Dialect/TTIR/IR/TTIRBase.td"
include "ttmlir/Dialect/TTIR/IR/TTIROpsAttrs.td"
include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.td"

//===----------------------------------------------------------------------===//
// TTIR Tile Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def TTIR_TileAddOp : TTIR_Op<"tile_add", [TTIR_GenericParent]>{
    let summary = "TTIR Tile Add Op";
    let description = [{
        The `tile_add` operation adds two tensors element-wise.
    }];

    let arguments = (ins TT_Tile:$lhs, TT_Tile:$rhs);
    let results = (outs TT_Tile:$result); // questionable: should tile ops actually return anything? tensix math ops will leave results in dest regs, up to packer to get result to l1
}

def TTIR_TileSubOp : TTIR_Op<"tile_sub", [TTIR_GenericParent]>{
    let summary = "TTIR Tile Sub Op";
    let description = [{
        The `tile_sub` operation subtracts two tensors element-wise.
    }];

    let arguments = (ins TT_Tile:$lhs, TT_Tile:$rhs);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileMulOp : TTIR_Op<"tile_mul", [TTIR_GenericParent]>{
    let summary = "TTIR Tile Mul Op";
    let description = [{
        The `tile_mul` operation multiplies two tensors element-wise.
    }];

    let arguments = (ins TT_Tile:$lhs, TT_Tile:$rhs);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileRecipOp : TTIR_Op<"tile_recip", [TTIR_GenericParent]>{
    let summary = "TTIR Tile Recip Op";
    let description = [{
        The `tile_recip` operation computes the reciprocal of each element in the input tensor.
    }];

    let arguments = (ins TT_Tile:$input);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileExpOp : TTIR_Op<"tile_exp", [TTIR_GenericParent]>{
    let summary = "TTIR Tile Exp Op";
    let description = [{
        The `tile_exp` operation computes the exponential of each element in the input tensor.
    }];

    let arguments = (ins TT_Tile:$input);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileLogOp : TTIR_Op<"tile_log", [TTIR_GenericParent]>{
    let summary = "TTIR Tile Log Op";
    let description = [{
        The `tile_log` operation computes the natural logarithm of each element in the input tensor.
    }];

    let arguments = (ins TT_Tile:$input);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileReduceSumOp : TTIR_Op<"tile_reduce_sum", [TTIR_GenericParent]>{
    let summary = "TTIR Tile Reduce Sum Op";
    let description = [{
        The `tile_reduce_sum` operation computes the sum of all elements in the input tensor.
    }];

    let arguments = (ins TT_Tile:$a, TT_Tile:$b);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileMatmulBlockOp : TTIR_Op<"tile_matmul_block", [TTIR_GenericParent]>{
    let summary = "TTIR Tile Matmul Block Op";
    let description = [{
        The `tile_matmul_block` operation computes the matrix multiplication of two input blocks.
    }];

    let arguments = (ins AnyNon0RankedMemRef:$a, AnyNon0RankedMemRef:$b);
    let results = (outs AnyNon0RankedMemRef:$result);

    let hasVerifier = 1;
}

#endif // TTMLIR_TTMLIR_DIALECT_TTIR_TTIRTILEOPS_TD
