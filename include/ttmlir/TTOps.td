// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.     ■ Too many errors emitted, stopping now
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_TTOPS_TD
#define TTMLIR_TTMLIR_TTOPS_TD

include "ttmlir/TTOpsTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/IR/CommonTypeConstraints.td"

def TT_TilizeOp : TT_Op<"tilize", [DestinationStyleOpInterface, 
                                  TypesMatchWith<"result type matches type of dest",
                                   "dest", "result",
                                   "$_self">]> {
    let summary = "Tilize op.";
    let description = [{
      Tilize operation
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

def TT_UntilizeOp : TT_Op<"untilize", [DestinationStyleOpInterface, 
                                  TypesMatchWith<"result type matches type of dest",
                                   "dest", "result",
                                   "$_self">]> {
    let summary = "Untilize op.";
    let description = [{
      Untilize operation
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

#endif
