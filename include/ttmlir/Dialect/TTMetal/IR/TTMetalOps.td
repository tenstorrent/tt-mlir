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
include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.td"

class AtLeastRegion<int numBlocks> : Region<
  CPred<"$_self.getBlocks().size() >= " # numBlocks>,
  "region with " # numBlocks # " blocks">;

def TTMetal_EnqueueProgramOp : TTMetal_Op<"enqueue_program", [MemoryEffects<[MemRead, MemWrite]>, AttrSizedOperandSegments]> {
    let summary = "Enqueue program op.";
    let description = [{
      Enqueue program operation.

      `dfbs` / `dfb_ids` carry the same shape as `cbs` / `cb_ports` for
      Quasar Dataflow Buffers, allowing one program to enqueue both
      vocabularies (CBs on WH/BH, DFBs on Quasar). For the MVP a program
      will only use one or the other.
    }];

    let arguments = (ins Variadic<AnyType>:$args,
                         Variadic<AnyNon0RankedMemRef>:$cbs,
                         DenseI64ArrayAttr:$cb_ports,
                         Variadic<AnyNon0RankedMemRef>:$dfbs,
                         DenseI64ArrayAttr:$dfb_ids,
                         TTMetal_KernelConfigArrayAttr:$kernelConfigs,
                         OptionalAttr<TTCore_FabricConnectionConfigAttr>:$fabricConnectionConfig);
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
      Create buffer operation.

      When this buffer uses a virtual grid, `virtualGridInverseMapping` stores the
      inverse affine map (physical to virtual grid coordinates) and
      `virtualGridForwardMapping` stores the forward affine map (virtual to
      physical grid coordinates).  Both are propagated from d2m.empty through
      bufferization.
    }];

    let arguments = (ins I64Attr:$address,
                         OptionalAttr<AffineMapAttr>:$virtualGridInverseMapping,
                         OptionalAttr<AffineMapAttr>:$virtualGridForwardMapping);
    let results = (outs AnyNon0RankedMemRef:$result);
}

def TTMetal_CreateDataflowBufferOp : TTMetal_Op<"create_dataflow_buffer", [MemoryEffects<[MemAlloc]>]> {
    let summary = "Create dataflow buffer (Quasar DFB) op.";
    let description = [{
      Allocate a Quasar Dataflow Buffer. Host-side analogue of
      `experimental::dfb::CreateDataflowBuffer(program, core_spec, config)`.
      Returns a logical DFB id (UI32) that subsequent
      `ttmetal.bind_dfb_to_kernels` ops reference.
    }];

    let arguments = (ins TTMetal_CoreRangeAttr:$core_range,
                         TTMetal_DataflowBufferConfigAttr:$config);
    let results = (outs UI32:$dfb_id);
}

def TTMetal_BindDFBToKernelsOp : TTMetal_Op<"bind_dfb_to_kernels"> {
    let summary = "Bind a Quasar dataflow buffer to its producer and consumer kernels.";
    let description = [{
      Host-side analogue of `experimental::dfb::BindDataflowBufferToProducer
      ConsumerKernels(program, dfb_id, producer_kernel, consumer_kernel)`.

      For 1P-nC consumer fan-out, all N compute threads share a single
      consumer kernel symbol (identical binary); slot assignment across
      the N hardware threads is handled by the DFB runtime against the
      `num_consumers` field of the bound config.
    }];

    let arguments = (ins UI32:$dfb_id,
                         FlatSymbolRefAttr:$producer_kernel,
                         FlatSymbolRefAttr:$consumer_kernel);
}

def TTMetal_CreateGlobalSemaphoreOp : TTMetal_Op<"create_global_semaphore", [MemoryEffects<[MemWrite]>]> {
    let summary = "Create global semaphore op.";
    let description = [{
      Create global semaphore operation
    }];

    let arguments = (ins I64Attr:$address, OptionalAttr<UI32Attr>:$initial_value, TTMetal_CoreRangeAttr:$core_range);
    let results = (outs TTMetal_GlobalSemaphore:$result);
}

def TTMetal_ResetGlobalSemaphoreOp : TTMetal_Op<"reset_global_semaphore", [MemoryEffects<[MemWrite]>]> {
    let summary = "Reset global semaphore op.";
    let description = [{
      Reset global semaphore operation
    }];

    let arguments = (ins TTMetal_GlobalSemaphore:$semaphore, UI32Attr:$value);
}

def TTMetal_CreateLocalSemaphoreOp : TTMetal_Op<"create_local_semaphore", [MemoryEffects<[MemAlloc]>]> {
    let summary = "Create local semaphore op.";
    let description = [{
      Create local semaphore operation
    }];

    let arguments = (ins OptionalAttr<UI32Attr>:$initial_value);
    let results = (outs TTMetal_LocalSemaphore:$result);
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
