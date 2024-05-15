// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TENSIX_TENSIXOPS_TD
#define TTMLIR_TTMLIR_DIALECT_TENSIX_TENSIXOPS_TD

include "ttmlir/Dialect/Tensix/IR/TensixBase.td"
include "ttmlir/Dialect/TT/IR/TTOpsTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

def Tensix_PackOp : Tensix_Op<"pack"> {
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

def Tensix_UnpackOp : Tensix_Op<"unpack"> {
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
