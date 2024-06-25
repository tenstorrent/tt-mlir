// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIROPS_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIROPS_TD

include "ttmlir/Dialect/TT/IR/TTOpsTypes.td"
include "ttmlir/Dialect/TTIR/IR/TTIRBase.td"
include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.td"
include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"

class TTIR_DPSOp<string mnemonic, list<Trait> traits = []> :
    TTIR_Op<mnemonic, !listconcat(traits, [TTIROpInterface, DestinationStyleOpInterface, AttrSizedOperandSegments])> {
    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }
    }];
}

//===----------------------------------------------------------------------===//
// TTIR top level ops
//===----------------------------------------------------------------------===//

def TTIR_GenericOp : TTIR_DPSOp<"generic"> {
    let summary = "Generically dispatch work to a grid of cores.";
    let description = [{
      This generic op carries a region that represents the work each core does. The region is
      expected to have the same signature as the op itself. The op is expected to be lowered
      to a backend specific form by a consuming backend. This op is heavily inspired by the
      linalg.generic op so it can be useful to refer to linalg.generic documentation for more
      details.

      %5 = "ttir.generic"(%1, %3, %4) <{
        grid = #tt.grid<1x1>,                     // The grid range of cores to dispatch work to.
        indexing_maps = [#map, #map, #map],       // Affine maps for indexing into the input/output tensors. See linalg.generic
        iterator_types = [#parallel, #parallel],  // Iterator types for the input/output tensors. See linalg.generic
        operandSegmentSizes = array<i32: 2, 1>,   // Sizes of the operand segments, i.e. 2 inputs and 1 output.
      ({
      ^bb0(%arg2: memref<64x128xf32, #l1_>, %arg3: memref<64x128xf32, #l1_>, %arg4: memref<64x128xf32, #l1_>):
          // Region body, would contain some computation that represents the work each core does.
      }) : (tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
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

def TTIR_LayoutOp : TTIR_Op<"layout", [DestinationStyleOpInterface]> {
    let summary = "Layout op.";
    let description = [{
      Layout operation, transition tensors from one layout to another.  Some examples include:
        - Transitioning between different memory spaces, e.g. DRAM to L1.
        - Transitioning between different data types, e.g. f32 to f16.
        - Transitioning between different tile sizes, e.g. 1x16 to 32x32
        - Transitioning between different tensor sharding
        - Some combination of the above

      #layout = #tt.layout<8192x128x1, undef, <1x1>, memref<64x128xf32, #system>>
      #layout1 = #tt.layout<8192x128x1, undef, <1x1>, memref<64x128xf32, #l1_>>
      %1 = "ttir.layout"(%arg0, %0) : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output);
    let results = (outs AnyRankedTensor:$result);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def TTIR_AllocOp : TTIR_Op<"alloc"> {
    let summary = "Alloc op.";
    let description = [{
      Tensor Alloc operation
    }];

    let arguments = (ins I64Attr:$address, I64Attr:$size, TT_MemorySpaceAttr:$memory_space);
    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_DeallocOp : TTIR_Op<"dealloc"> {
    let summary = "Dealloc op.";
    let description = [{
      Tensor Dealloc operation
    }];

    let arguments = (ins AnyRankedTensor:$result);
}

//===----------------------------------------------------------------------===//
// TTIR top level named ops
//  A named op is one that is not generic and has a specific name. For example
//  "add", "multiply", "matmul", "concat", etc. They do not carry a region.
//===----------------------------------------------------------------------===//

class TTIR_ElementwiseOp<string mnemonic, list<Trait> traits = []> :
    TTIR_DPSOp<mnemonic, !listconcat(traits, [Elementwise])> {

    let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                         Variadic<AnyRankedTensor>:$outputs,
                         TT_OperandConstraintArrayAttr:$operand_constraints);
    let results = (outs Variadic<AnyRankedTensor>:$results);
}

class TTIR_ElementwiseBinaryOp<string mnemonic, list<Trait> traits = []> :
    TTIR_ElementwiseOp<mnemonic, traits> {
    let summary = "Eltwise binary op.";
    let description = [{
      Eltwise binary op.
    }];
}

def TTIR_AddOp : TTIR_ElementwiseBinaryOp<"add"> {
    let summary = "Eltwise add.";
    let description = [{
      Eltwise add operation.
    }];
}

def TTIR_MultiplyOp : TTIR_ElementwiseBinaryOp<"multiply"> {
    let summary = "Eltwise multiply.";
    let description = [{
      Eltwise multiply operation.
    }];
}

//===----------------------------------------------------------------------===//
// TTIR region ops (ops that may appear inside of ttir.generic region)
//===----------------------------------------------------------------------===//

def AnyRankedTensorOrMemRef: AnyTypeOf<[AnyRankedTensor, AnyNon0RankedMemRef]>;

def TTIR_KernelOp : TTIR_Op<"kernel", [DestinationStyleOpInterface, AttrSizedOperandSegments]> {
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

def TTIR_YieldOp : TTIR_Op<"yield", [Pure, ReturnLike, Terminator]> {
    let summary = "Yield op.";
    let description = [{
      Yield operation, this is required by MLIR to mark the end of a dispatch region.
    }];

    let arguments = (ins Variadic<AnyRankedTensorOrMemRef>:$values);
}

#endif
