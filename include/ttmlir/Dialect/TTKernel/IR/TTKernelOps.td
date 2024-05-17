// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELOPS_TD
#define TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELOPS_TD

include "ttmlir/Dialect/TTKernel/IR/TTKernelBase.td"
include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.td"
include "ttmlir/Dialect/TT/IR/TTOpsTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

def AnyMemRefOrCB: AnyTypeOf<[AnyNon0RankedMemRef, TTKernel_CB]>;

def TTKernel_BuiltinOp : TTKernel_Op<"builtin"> {
    let summary = "Builtin call.";
    let description = [{
      Kernel operation
    }];

    let arguments = (ins FlatSymbolRefAttr:$op,
                         FlatSymbolRefAttr:$kind,
                         Variadic<AnyMemRefOrCB>:$args);

    let hasVerifier = 1;
}

def TTKernel_PackOp : TTKernel_Op<"pack"> {
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

def TTKernel_UnpackOp : TTKernel_Op<"unpack"> {
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

def TTKernel_CBPushBackOp : TTKernel_Op<"cb_push_back"> {
    let summary = "CBPushBack call.";
    let description = [{
      CBPushBack operation
    }];

    let arguments = (ins TTKernel_CB:$cb);

    let hasVerifier = 1;
}

def TTKernel_CBPopFrontOp : TTKernel_Op<"cb_pop_front"> {
    let summary = "CBPopFront call.";
    let description = [{
      CBPopFront operation
    }];

    let arguments = (ins TTKernel_CB:$cb);

    let hasVerifier = 1;
}

def TTKernel_CBReserveBackOp : TTKernel_Op<"cb_reserve_back"> {
    let summary = "CBReserveBack call.";
    let description = [{
      CBReserveBack operation
    }];

    let arguments = (ins TTKernel_CB:$cb);

    let hasVerifier = 1;
}

def TTKernel_CBWaitFrontOp : TTKernel_Op<"cb_wait_front"> {
    let summary = "CBWaitFront call.";
    let description = [{
      CBWaitFront operation
    }];

    let arguments = (ins TTKernel_CB:$cb);

    let hasVerifier = 1;
}

def TTKernel_ReturnOp : TTKernel_Op<"return", [Pure, ReturnLike, Terminator]> {
    let summary = "Return op.";
    let description = [{
      Return operation
    }];

    let hasVerifier = 1;
}

def TTKernel_UnreachableOp : TTKernel_Op<"unreachable", [Pure, ReturnLike, Terminator]> {
    let summary = "Unreachable op.";
    let description = [{
      Unreachable operation
    }];
}

#endif
