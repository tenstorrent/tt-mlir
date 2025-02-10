// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIRGENERICREGIONOPS_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIRGENERICREGIONOPS_TD

include "ttmlir/Dialect/TT/IR/TTOpsTypes.td"
include "ttmlir/Dialect/TTIR/IR/TTIRBase.td"
include "ttmlir/Dialect/TTIR/IR/TTIROpsAttrs.td"
include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"

class TTIR_GenericRegionOp<string mnemonic, list<Trait> traits = [TTIR_GenericRegionOpTrait]> :
    TTIR_Op<mnemonic, !listconcat([HasParent<"GenericOp">], traits)> {
}

//===----------------------------------------------------------------------===//
// TTIR Generic Region Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def AnyRankedTensorOrMemRef: AnyTypeOf<[AnyRankedTensor, AnyNon0RankedMemRef]>;

def TTIR_YieldOp : TTIR_Op<"yield", [Pure, ReturnLike, Terminator, TTIR_GenericRegionOpTrait, ParentOneOf<["GenericOp", "ScatterOp"]>]> {
    let summary = "Yield op.";
    let description = [{
      Yield operation, this is required by MLIR to mark the end of a dispatch region.
    }];

    let arguments = (ins Variadic<AnyRankedTensorOrMemRef>:$values);
}

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
                         TTIR_ReduceDim:$dim_arg);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileReduceMaxOp : TTIR_GenericRegionOp<"tile_reduce_max">{
    let summary = "TTIR Tile Reduce Max Op";
    let description = [{
        The `tile_reduce_max` operation computes the max of all elements in the input tile over the specified reduction dim(s).
    }];

    let arguments = (ins TT_Tile:$input, 
                         TTIR_ReduceDim:$dim_arg);
    let results = (outs TT_Tile:$result);
}

def TTIR_TileMatmulBlockOp : TTIR_GenericRegionOp<"tile_matmul_block">{
    let summary = "TTIR Tile Matmul Block Op";
    let description = [{
        The `tile_matmul_block` operation computes the matrix multiplication of two input blocks.
    }];

    let arguments = (ins AnyNon0RankedMemRef:$a, 
                         AnyNon0RankedMemRef:$b);
    let results = (outs AnyNon0RankedMemRef:$result);

    let hasVerifier = 1;
}

#endif // TTMLIR_TTMLIR_DIALECT_TTIR_TTIRGENERICREGIONOPS_TD
