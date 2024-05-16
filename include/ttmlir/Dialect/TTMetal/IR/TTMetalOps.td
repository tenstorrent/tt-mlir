// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPS_TD
#define TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPS_TD

include "ttmlir/Dialect/TT/IR/TTOpsTypes.td"
include "ttmlir/Dialect/TTMetal/IR/TTMetalBase.td"
include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.td"
include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"

class AtLeastRegion<int numBlocks> : Region<
  CPred<"$_self.getBlocks().size() >= " # numBlocks>,
  "region with " # numBlocks # " blocks">;

def TTMetal_DispatchOp : TTMetal_Op<"dispatch", [DestinationStyleOpInterface, AttrSizedOperandSegments]> {
    let summary = "Dispatch op.";
    let description = [{
      Dispatch operation
    }];

    let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                         Variadic<AnyRankedTensor>:$outputs,
                         TTMetal_CoreRangeArrayAttr:$core_ranges,
                         TTMetal_ThreadTypeArrayAttr:$threadTypes,
                         ArrayAttr:$operand_cb_port_mapping);
    let results = (outs Variadic<AnyRankedTensor>:$results);
    let regions = (region VariadicRegion<AnyRegion>:$regions);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }
    }];

    let hasVerifier = 1;
}

def TTMetal_HostWriteOp : TTMetal_Op<"host_write", [DestinationStyleOpInterface]> {
    let summary = "Host write op.";
    let description = [{
      Host write operation
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output);
    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTMetal_HostReadOp : TTMetal_Op<"host_read", [DestinationStyleOpInterface]> {
    let summary = "Host read op.";
    let description = [{
      Host read operation
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output);
    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTMetal_AllocOp : TTMetal_Op<"alloc"> {
    let summary = "Alloc op.";
    let description = [{
      Tensor Alloc operation
    }];

    let arguments = (ins I64Attr:$address, I64Attr:$size, TT_MemorySpaceAttr:$memory_space);
    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTMetal_DeallocOp : TTMetal_Op<"dealloc"> {
    let summary = "Dealloc op.";
    let description = [{
      Tensor Dealloc operation
    }];

    let arguments = (ins AnyRankedTensor:$result);
}

//===----------------------------------------------------------------------===//
// TTMetal ops below are only legal inside a DispatchOp region
//===----------------------------------------------------------------------===//

def AnyMemRefOrCB: AnyTypeOf<[AnyNon0RankedMemRef, TTMetal_CB]>;

def TTMetal_KernelOp : TTMetal_Op<"kernel"> {
    let summary = "Kernel call.";
    let description = [{
      Kernel operation
    }];

    let arguments = (ins FlatSymbolRefAttr:$op,
                         FlatSymbolRefAttr:$kind,
                         Variadic<AnyMemRefOrCB>:$args);

    let hasVerifier = 1;
}

def TTMetal_CBPushBackOp : TTMetal_Op<"cb_push_back"> {
    let summary = "CBPushBack call.";
    let description = [{
      CBPushBack operation
    }];

    let arguments = (ins TTMetal_CB:$cb);

    let hasVerifier = 1;
}

def TTMetal_CBPopFrontOp : TTMetal_Op<"cb_pop_front"> {
    let summary = "CBPopFront call.";
    let description = [{
      CBPopFront operation
    }];

    let arguments = (ins TTMetal_CB:$cb);

    let hasVerifier = 1;
}

def TTMetal_CBReserveBackOp : TTMetal_Op<"cb_reserve_back"> {
    let summary = "CBReserveBack call.";
    let description = [{
      CBReserveBack operation
    }];

    let arguments = (ins TTMetal_CB:$cb);

    let hasVerifier = 1;
}

def TTMetal_CBWaitFrontOp : TTMetal_Op<"cb_wait_front"> {
    let summary = "CBWaitFront call.";
    let description = [{
      CBWaitFront operation
    }];

    let arguments = (ins TTMetal_CB:$cb);

    let hasVerifier = 1;
}

def TTMetal_ReturnOp : TTMetal_Op<"return", [Pure, ReturnLike, Terminator]> {
    let summary = "Return op.";
    let description = [{
      Return operation
    }];

    let hasVerifier = 1;
}

def TTMetal_UnreachableOp : TTMetal_Op<"unreachable", [Pure, ReturnLike, Terminator]> {
    let summary = "Unreachable op.";
    let description = [{
      Unreachable operation
    }];
}

#endif
