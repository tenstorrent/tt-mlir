// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPS_TD
#define TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPS_TD

include "ttmlir/Dialect/TT/IR/TTOpsTypes.td"
include "ttmlir/Dialect/TTMetal/IR/TTMetalBase.td"
include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.td"
include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.td"
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

def TTMetal_EnqueueProgramOp : TTMetal_Op<"enqueue_program", [DestinationStyleOpInterface, AttrSizedOperandSegments]> {
    let summary = "Enqueue program op.";
    let description = [{
      Enqueue program operation
    }];

    let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                         Variadic<AnyRankedTensor>:$outputs,
                         TTMetal_CoreRangeArrayAttr:$core_ranges,
                         TTKernel_KernelConfigArrayAttr:$kernelConfigs);
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

    let arguments = (ins AnyRankedTensor:$output, ElementsAttr:$value);
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

    let arguments = (ins AnyRankedTensor:$input);
}

#endif
