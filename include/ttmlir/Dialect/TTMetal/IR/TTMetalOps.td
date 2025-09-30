// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPS_TD
#define TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPS_TD

include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td"
include "ttmlir/Dialect/TTMetal/IR/TTMetalBase.td"
include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsAttrs.td"
include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.td"
include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"

class AtLeastRegion<int numBlocks> : Region<
  CPred<"$_self.getBlocks().size() >= " # numBlocks>,
  "region with " # numBlocks # " blocks">;

def TTMetal_EnqueueProgramOp : TTMetal_Op<"enqueue_program", [MemoryEffects<[MemRead, MemWrite]>, AttrSizedOperandSegments]> {
    let summary = "Enqueue program op.";
    let description = [{
      Enqueue program operation
    }];

    let arguments = (ins Variadic<AnyNon0RankedMemRef>:$buffers,
                         Variadic<AnyNon0RankedMemRef>:$cbs,
                         DenseI64ArrayAttr:$cb_ports,
                         TTMetal_KernelConfigArrayAttr:$kernelConfigs);
    let hasVerifier = 1;
}

def TTMetal_EnqueueWriteBufferOp : TTMetal_Op<"enqueue_write_buffer", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "Enqueue write buffer op.";
    let description = [{
      Enqueue write buffer operation
    }];

    let arguments = (ins AnyNon0RankedMemRef:$input, AnyNon0RankedMemRef:$output);
    let hasVerifier = 1;
}

def TTMetal_EnqueueReadBufferOp : TTMetal_Op<"enqueue_read_buffer", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "Enqueue read buffer op.";
    let description = [{
      Enqueue read buffer operation
    }];

    let arguments = (ins AnyNon0RankedMemRef:$input,
                         AnyNon0RankedMemRef:$output);
    let hasVerifier = 1;
}

def TTMetal_CreateBufferOp : TTMetal_Op<"create_buffer", [MemoryEffects<[MemAlloc]>]> {
    let summary = "Create buffer op.";
    let description = [{
      Create buffer operation
    }];

    let arguments = (ins I64Attr:$address);
    let results = (outs AnyNon0RankedMemRef:$result);
}

def TTMetal_DeallocateBufferOp : TTMetal_Op<"deallocate_buffer", [MemoryEffects<[MemFree]>]> {
    let summary = "Deallocate buffer op.";
    let description = [{
      Deallocate buffer operation
    }];

    let arguments = (ins AnyNon0RankedMemRef:$input);
}

def TTMetal_MeshShardOp : TTMetal_Op<"mesh_shard", [MemoryEffects<[MemRead, MemAlloc]>]> {
    let summary = "Nd sharding or (partial) concat op";
    let description = [{
      Nd sharding or (partial) concat op in D2M runtime.
      ShardToFull: Nd sharding in host memory.
      FullToshard: (partial) concat in host memory.
    }];

    let arguments = (ins AnyNon0RankedMemRef:$input,
                         TTCore_MeshShardTypeAttr:$shard_type,
                         TTCore_MeshShardDirectionAttr:$shard_direction,
                         DenseI64ArrayAttr:$shard_shape,
                         DenseI64ArrayAttr:$shard_dims);

    let results = (outs AnyNon0RankedMemRef:$result);
}

def TTMetal_FinishOp : TTMetal_Op<"finish", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "Finish op for command queue.";
    let description = [{
      Global barrier op, used to wait for all commands on queue to finish.
    }];
}

#endif
