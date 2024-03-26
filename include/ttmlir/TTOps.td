// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.     ■ Too many errors emitted, stopping now
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_TTOPS_TD
#define TTMLIR_TTMLIR_TTOPS_TD

include "ttmlir/TTOpsTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

def TT_TensorTilizeOp : TT_Op<"tensor_tilize", [DestinationStyleOpInterface, 
                                  TypesMatchWith<"result type matches type of dest",
                                   "dest", "result",
                                   "$_self">]> {
    let summary = "Tensor tilize op.";
    let description = [{
      Tensor tilize operation
    }];

    let arguments = (ins AnyRankedTensor:$source,
                     AnyRankedTensor:$dest,
                     Optional<AnyType>:$padding_value);
    let results = (outs AnyRankedTensor:$result);

    let assemblyFormat = [{
      $source
      (`padding_value` `(` $padding_value^ `:` type($padding_value) `)`)?
      `into` $dest attr-dict `:` type($source) `->` type($dest)
    }];

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getDestMutable(); }
    }];
}

def TT_TensorUntilizeOp : TT_Op<"tensor_untilize", [DestinationStyleOpInterface, 
                                  TypesMatchWith<"result type matches type of dest",
                                   "dest", "result",
                                   "$_self">]> {
    let summary = "Tensor untilize op.";
    let description = [{
      Tensor untilize operation
    }];

    let arguments = (ins AnyRankedTensor:$source,
                     AnyRankedTensor:$dest);
    let results = (outs AnyRankedTensor:$result);

    let assemblyFormat = [{
      $source `into` $dest attr-dict `:` type($source) `->` type($dest)
    }];

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getDestMutable(); }
    }];
}

def TT_DispatchOp : TT_Op<"dispatch", [DestinationStyleOpInterface, AttrSizedOperandSegments]> {
    let summary = "Dispatch op.";
    let description = [{
      Dispatch operation
    }];

    let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                         Variadic<AnyRankedTensor>:$outputs,
                         AffineMapAttr:$paralellization);
    let results = (outs Variadic<AnyRankedTensor>:$results);
    let regions = (region AnyRegion:$region);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }
    }];
}

def TT_YieldOp : TT_Op<"yield", [Pure, ReturnLike, Terminator]> {
    let summary = "Yield op.";
    let description = [{
      Yield operation
    }];

    let arguments = (ins Variadic<AnyType>:$values);
}

// TODO: Make these Tensix dialect
def TT_PackOp : TT_Op<"pack"> {
    let summary = "Pack op.";
    let description = [{
      Pack operation
    }];

    let arguments = (ins AnyFloat:$source);
    let results = (outs TT_Tile:$result);

    let assemblyFormat = [{
      $source attr-dict `:` type($source) `->` type($result)
    }];
}

def TT_UnpackOp : TT_Op<"unpack"> {
    let summary = "Unpack op.";
    let description = [{
      Unpack operation
    }];

    let arguments = (ins TT_Tile:$source);
    let results = (outs AnyFloat:$result);

    let assemblyFormat = [{
      $source attr-dict `:` type($source) `->` type($result)
    }];
}

#endif
