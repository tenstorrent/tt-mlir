// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPS_TD
#define TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPS_TD

include "ttmlir/Dialect/TT/TTOpsTypes.td"
include "ttmlir/Dialect/TTMetal/TTMetalDialect.td"
include "ttmlir/Dialect/TTMetal/TTMetalOpsTypes.td"
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
                         TT_GridAttr:$grid,
                         AffineMapArrayAttr:$indexing_maps,
                         TT_IteratorTypeArrayAttr:$iterator_types,
                         TT_OperandConstraintArrayAttr:$operand_constraints);
    let results = (outs Variadic<AnyRankedTensor>:$results);
    let regions = (region AtLeastRegion<1>:$region);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }
    }];
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

def AnyRankedTensorOrMemRef: AnyTypeOf<[AnyRankedTensor, AnyNon0RankedMemRef]>;

def TTMetal_KernelOp : TTMetal_Op<"kernel", [DestinationStyleOpInterface, AttrSizedOperandSegments]> {
    let summary = "Kernel call.";
    let description = [{
      Kernel operation
    }];

    let arguments = (ins FlatSymbolRefAttr:$op,
                         FlatSymbolRefAttr:$kind,
                         Variadic<AnyRankedTensorOrMemRef>:$inputs,
                         Variadic<AnyRankedTensorOrMemRef>:$outputs);
    let results = (outs Variadic<AnyRankedTensorOrMemRef>:$results);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }
    }];
}

def TTMetal_YieldOp : TTMetal_Op<"yield", [Pure, ReturnLike, Terminator]> {
    let summary = "Yield op.";
    let description = [{
      Yield operation
    }];

    let arguments = (ins Variadic<AnyRankedTensorOrMemRef>:$values);
}

#endif
