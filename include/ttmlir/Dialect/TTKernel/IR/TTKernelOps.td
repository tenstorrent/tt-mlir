// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
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

//===----------------------------------------------------------------------===//
// TTKernel Register operations
//===----------------------------------------------------------------------===//

def TTKernel_PackSetDataTypeOp : TTKernel_Op<"pack_set_data_type"> {
    let summary = "Pack set DataType op.";
    let description = [{
      Pack set DataType operation
    }];

    let arguments = (ins TT_DataType:$data_type);
}

def TTKernel_UnpackSetDataTypeOp : TTKernel_Op<"unpack_set_data_type"> {
    let summary = "Unpack set DataType op.";
    let description = [{
      Unpack set DataType operation
    }];

    let arguments = (ins TT_DataType:$data_type_a, TT_DataType:$data_type_b);
}

def TTKernel_PackOp : TTKernel_Op<"pack"> {
    let summary = "Pack op.";
    let description = [{
      Pack operation
    }];

    let arguments = (ins I32:$dst_index, TTKernel_CB:$out_cb, I32:$out_index);
}

def TTKernel_UnpackAOp : TTKernel_Op<"unpack_a"> {
    let summary = "UnpackA op.";
    let description = [{
      UnpackA operation
    }];

    let arguments = (ins TTKernel_CB:$cb, I32:$src_index);
}

def TTKernel_UnpackABOp : TTKernel_Op<"unpack_ab"> {
    let summary = "UnpackAB op.";
    let description = [{
      UnpackAB operation
    }];

    let arguments = (ins TTKernel_CB:$cb_a, I32:$src_a_index, TTKernel_CB:$cb_b, I32:$src_b_index);
}

def TTKernel_AcquireDstOp : TTKernel_Op<"acquire_dst"> {
    let summary = "Aquire dest call.";
    let description = [{
      Aquire dest operation
    }];

    //TODO: Add MemRefAttr
    //let arguments = (ins MemRefAttr:$view);
    //let hasVerifier = 1;
}

def TTKernel_ReleaseDstOp : TTKernel_Op<"release_dst"> {
    let summary = "Release dest call.";
    let description = [{
      Release dest operation
    }];
}

//===----------------------------------------------------------------------===//
// TTKernel FPU operations
//===----------------------------------------------------------------------===//

def TTKernel_AddOp : TTKernel_Op<"add"> {
    let summary = "Add operation";
    let description = [{
      Add operation
    }];

    let arguments = (ins I32:$dst_index);
}

def TTKernel_SubOp : TTKernel_Op<"sub"> {
    let summary = "Sub operation";
    let description = [{
      Sub operation
    }];

    let arguments = (ins I32:$dst_index);
}

def TTKernel_MulOp : TTKernel_Op<"mul"> {
    let summary = "Mul operation";
    let description = [{
      Mul operation
    }];

    let arguments = (ins I32:$dst_index);
}

def TTKernel_MatmulOp : TTKernel_Op<"matmul"> {
    let summary = "Matmul operation";
    let description = [{
      Matmul operation
    }];

    let arguments = (ins I32:$dst_index);
}

//===----------------------------------------------------------------------===//
// TTKernel CB operations
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// TTKernel Misc operations
//===----------------------------------------------------------------------===//

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
