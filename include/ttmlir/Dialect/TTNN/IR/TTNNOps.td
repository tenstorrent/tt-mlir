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

def TTNN_GenericOp : TTNN_Op<"generic", [AttrSizedOperandSegments]> {
    let summary = "Generic op modelling any kind of user-written work a group of cores executes.";

    let description = [{
      Generic op form where the key properties of the computation are specified as attributes.

      It contains a region that hooks to a backend call which executes the described work.

      ```mlir
      %4 = "ttnn.generic"(%1, %2, %3) < { attributes-specifying-underlying-computation } > ({
        ^bb0(%arg2: tensor<?xf32, #l1_layout>, %arg3: tensor<?xf32, #l1_layout>, %arg4: tensor<?xf32, #l1_layout>):
          // Region body, would contain a backend hook to some computation that represents the work each core does.
          // In ttnn terms it would pass the user-written compute and data movement kernels, as well as parameters for
          // setting up entire ttnn execution flow (check out 
          // https://tenstorrent.github.io/tt-metal/latest/ttnn/ttnn/adding_new_ttnn_operation.html) to the backend hook.
      }) : (tensor<?xf32, #l1_layout>, tensor<?xf32, #l1_layout>, tensor<?xf32, #l1_layout>) -> tensor<?xf32, #l1_layout>
      ```
    }];

    let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                         Variadic<AnyRankedTensor>:$outputs,
                         TT_GridAttr:$grid,
                         AffineMapArrayAttr:$indexing_maps,
                         TT_IteratorTypeArrayAttr:$iterator_types,
                         TT_OperandConstraintArrayAttr:$operand_constraints);

    let results = (outs Variadic<AnyRankedTensor>:$results);
    
    let regions = (region AnyRegion:$region);
}

//===----------------------------------------------------------------------===//
// TTNN region ops (ops that may appear inside of ttnn.generic region)
//===----------------------------------------------------------------------===//

def AnyRankedTensorOrMemRef: AnyTypeOf<[AnyRankedTensor, AnyNon0RankedMemRef]>;

def TTNN_KernelOp : TTNN_Op<"kernel", [DestinationStyleOpInterface, AttrSizedOperandSegments]> {
    let summary = "Kernel call.";
    let description = [{
      A generic kernel call operation. This operation is used to pattern match by some consuming backend.
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

def TTNN_YieldOp : TTNN_Op<"yield", [Pure, ReturnLike, Terminator]> {
    let summary = "Yield op.";
    
    let description = [{
      Yield operation, this is required by MLIR to mark the end of a dispatch region.
    }];

    let arguments = (ins Variadic<AnyRankedTensorOrMemRef>:$values);
}

#endif
