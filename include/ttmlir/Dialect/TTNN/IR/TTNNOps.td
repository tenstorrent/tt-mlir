// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPS_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPS_TD

include "ttmlir/Dialect/TT/IR/TTOpsTypes.td"
include "ttmlir/Dialect/TTNN/IR/TTNNBase.td"
include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"

def TTNN_OpenDeviceOp : TTNN_Op<"open_device"> {
    let summary = "Open Device op.";
    let description = [{
    }];

    let arguments = (ins);
    let results = (outs TT_Device:$device);
}

def TTNN_CloseDeviceOp : TTNN_Op<"close_device"> {
    let summary = "Close Device op.";
    let description = [{
    }];

    let arguments = (ins TT_Device:$device);
}

def TTNN_ToMemoryConfigOp : TTNN_Op<"to_memory_config", [DestinationStyleOpInterface]> {
    let summary = "ToMemoryConfig op.";
    let description = [{
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output);
    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

class TTNN_NamedDPSOp<string mnemonic, list<Trait> traits = []> :
    TTNN_Op<mnemonic, !listconcat(traits, [DestinationStyleOpInterface])> {
    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }
    }];
}

class TTNN_ElementwiseOp<string mnemonic, list<Trait> traits = []> :
    TTNN_NamedDPSOp<mnemonic, !listconcat(traits, [Elementwise, AttrSizedOperandSegments])> {

    let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                         Variadic<AnyRankedTensor>:$outputs);
    let results = (outs Variadic<AnyRankedTensor>:$results);
}

class TTNN_ElementwiseBinaryOp<string mnemonic, list<Trait> traits = []> :
    TTNN_ElementwiseOp<mnemonic, traits> {
    let summary = "Eltwise binary op.";
    let description = [{
      Eltwise binary op.
    }];
}

def TTNN_AddOp : TTNN_ElementwiseBinaryOp<"add"> {
    let summary = "Eltwise add.";
    let description = [{
      Eltwise add operation.
    }];
}

def TTNN_SubtractOp : TTNN_ElementwiseBinaryOp<"subtract"> {
    let summary = "Eltwise subtract.";
    let description = [{
      Eltwise subtract operation.
    }];
}

def TTNN_MultiplyOp : TTNN_ElementwiseBinaryOp<"multiply"> {
    let summary = "Eltwise multiply.";
    let description = [{
      Eltwise multiply operation.
    }];
}

def TTNN_GreaterEqualOp : TTNN_ElementwiseBinaryOp<"ge"> {
    let summary = "Eltwise greater than or equal to.";
    let description = [{
      Eltwise greater than or equal to operation.
    }];
}

class TTNN_ReductionOp<string mnemonic, list<Trait> traits = []> : TTNN_NamedDPSOp<mnemonic, traits> {
    let summary = "Reduction op.";
    let description = [{
      Reduction op.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         BoolAttr:$keep_dim,
                         OptionalAttr<I32ArrayAttr>:$dim_arg);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];
}

def TTNN_SumOp : TTNN_ReductionOp<"sum"> {
    let summary = "Sum reduction op.";
    let description = [{
      Sum reduction op.
    }];
}

def TTNN_MeanOp : TTNN_ReductionOp<"mean"> {
  let summary = "Mean reduction op.";
  let description = [{
    Mean reduction op.
  }];
}

def TTNN_ReluOp : TTNN_ElementwiseOp<"relu"> {
    let summary = "Eltwise ReLU.";
    let description = [{
      Eltwise ReLU operation.
    }];
}

def TTNN_SoftmaxOp : TTNN_NamedDPSOp<"softmax"> {
    let summary = "Softmax op.";
    let description = [{
      Softmax operation.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         SI32Attr: $dimension);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTNN_TransposeOp : TTNN_NamedDPSOp<"transpose"> {
    let summary = "Transpose op.";
    let description = [{
      Transpose tensor along two given dimensions.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         SI32Attr:$dimension1,
                         SI32Attr:$dimension2);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

// ANCHOR: adding_an_op_matmul_ttnn
def TTNN_MatmulOp : TTNN_NamedDPSOp<"matmul"> {
    let arguments = (ins AnyRankedTensor:$a,
                         AnyRankedTensor:$b,
                         AnyRankedTensor:$output);
    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}
// ANCHOR_END: adding_an_op_matmul_ttnn

def TTNN_Conv2dOp : TTNN_NamedDPSOp<"conv2d"> {
    let summary = "Conv2d operation.";
    let description = [{
      Conv2d operation.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$weight,
                         AnyRankedTensor:$bias,
                         AnyRankedTensor:$output,
                         I32Attr:$in_channels,
                         I32Attr:$out_channels,
                         I32Attr:$batch_size,
                         I32Attr:$input_height,
                         I32Attr:$input_width,
                         I32Attr:$kernel_height,
                         I32Attr:$kernel_width,
                         I32Attr:$stride_height,
                         I32Attr:$stride_width,
                         I32Attr:$padding_height,
                         I32Attr:$padding_width,
                         I32Attr:$dilation_vertical,
                         I32Attr:$dilation_horizontal,
                         I32Attr:$groups);

    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTNN_FullOp : TTNN_Op<"full"> {
    let summary = "Full op.";
    let description = [{
      Tensor full operation
    }];

    let arguments = (ins TT_Device:$device, F32Attr:$fillValue);
    let results = (outs AnyRankedTensor:$result);
}

def TTNN_AllocOp : TTNN_Op<"alloc"> {
    let summary = "Alloc op.";
    let description = [{
      Tensor Alloc operation
    }];

    let arguments = (ins I64Attr:$address, I64Attr:$size, TT_MemorySpaceAttr:$memory_space);
    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTNN_DeallocOp : TTNN_Op<"dealloc"> {
    let summary = "Dealloc op.";
    let description = [{
      Tensor Dealloc operation
    }];

    let arguments = (ins AnyRankedTensor:$input);
}

#endif
