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

def TTKernel_TileRegsAcquireOp : TTKernel_Op<"tile_regs_acquire"> {
    let summary = "tile_regs_acquire";
    let description = [{
      Acquire an exclusive lock on the DST register for the MATH thread.
      This register is an array of 16 tiles of 32x32 elements each.
      This is a blocking function, i.e. this function will wait until the lock is acquired.
    }];
}

def TTKernel_TileRegsCommitOp : TTKernel_Op<"tile_regs_commit"> {
    let summary = "tile_regs_commit";
    let description = [{
      Release lock on DST register by MATH thread. The lock had to be previously acquired with tile_regs_acquire.
    }];
}

def TTKernel_TileRegsWaitOp : TTKernel_Op<"tile_regs_wait"> {
    let summary = "tile_regs_wait";
    let description = [{
      Acquire an exclusive lock on the DST register for the PACK thread.
      It waits for the MATH thread to commit the DST register.
      This is a blocking function, i.e. this function will wait until the lock is acquired.
    }];
}

def TTKernel_TileRegsReleaseOp : TTKernel_Op<"tile_regs_release"> {
    let summary = "tile_regs_release";
    let description = [{
      Release lock on DST register by PACK thread. The lock had to be previously acquired with tile_regs_wait.
    }];
}

def TTKernel_PackTileOp : TTKernel_Op<"pack_tile"> {
    let summary = "PackTile op.";
    let description = [{
      Copies a single tile from the DST register buffer at a specified index to a
      specified CB at a given index. For the out_tile_index to be valid for this
      call, cb_reserve_back(n) has to be called first to reserve at least some
      number n > 0 of tiles in the output CB. out_tile_index = 0 then references
      the first tile in the reserved section of the CB, up to index n - 1, which will
      then be visible to the consumer in the same order after a cb_push_back call.
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.

      Each subsequent pack call will increment the write pointer in the cb by single
      tile size. The pointer is then again set to a valid position with space for n
      reserved tiles by another cb_reserve_back call.

      Operates in tandem with functions cb_reserve_back and cb_push_back.

      A typical use case is first the producer ensures that there is a number of
      tiles available in the buffer via cb_reserve_back, then the producer uses
      the pack_tile call to copy a tile from one of DST slots to a slot in
      reserved space and finally cb_push_back is called to announce visibility of
      the reserved section of the circular buffer to the consumer.
    }];

    let arguments = (ins I32:$dst_index, TTKernel_CB:$out_cb, I32:$out_index);
}

def TTKernel_CopyTileInitOp : TTKernel_Op<"copy_tile_init"> {
    let summary = "Perform the init for copy tile. This does not reconfigure the unpacker data types.";
    let description = [{
      Must be called before copy_tile.
    }];

    let arguments = (ins TTKernel_CB:$cb0);
}

def TTKernel_CopyTileOp : TTKernel_Op<"copy_tile"> {
    let summary = "Copy tile from specified CB to DST.";
    let description = [{
      Copies a single tile from the specified input CB and writes the result to
      DST at a specified index. The function will employ unpacker to first unpack into SRC
      registers and then perform move into DST registers, at a specified index.
      For the in_tile_index to be valid for this call, cb_wait_front(n) had to be
      previously called to ensure that at least some number n>0 of tiles are available
      in the input CB. The CB index 0 then references the first tile in the received section of the CB,
      up to index n-1 (in a FIFO order). The DST register buffer must be in acquired state via
      tile_regs_acquire call. This call is blocking and is only available on the compute
      engine.
    }];

    let arguments = (ins TTKernel_CB:$cb0, I32:$tile_index_cb, I32:$tile_index_dst);
}

//===----------------------------------------------------------------------===//
// TTKernel FPU operations
//===----------------------------------------------------------------------===//

def TTKernel_BinaryOpInitCommonOp : TTKernel_Op<"binary_op_init_common"> {
    let summary = "Init function for all binary ops";
    let description = [{
      Followed by the specific init required with an opcode (binrary_op_specific_init).
    }];

    let arguments = (ins TTKernel_CB:$in0_cb, TTKernel_CB:$in1_cb, TTKernel_CB:$out_cb);
}

def TTKernel_AddTilesInitOp : TTKernel_Op<"add_tiles_init"> {
    let summary = "Short init function";
    let description = [{
      Must be run before add_tiles.
    }];

    let arguments = (ins TTKernel_CB:$in0_cb, TTKernel_CB:$in1_cb); // FIXME: , BOOL:$acc_to_dst);
}

def TTKernel_AddTilesOp : TTKernel_Op<"add_tiles"> {
    let summary = "Add operation";
    let description = [{
      Performs element-wise addition C=A+B of tiles in two CBs at given indices
      and writes the result to the DST register at index dst_tile_index. The DST
      register buffer must be in acquired state via *tile_regs_acquire* call. This call
      is blocking and is only available on the compute engine.
    }];

    let arguments = (ins TTKernel_CB:$in0_cb, TTKernel_CB:$in1_cb, I32:$in0_tile_index, I32:$in1_tile_index, I32:$dst_index);
}

def TTKernel_MulTilesInitOp : TTKernel_Op<"mul_tiles_init"> {
    let summary = "Short init function";
    let description = [{
      Must be run before mul_tiles.
    }];

    let arguments = (ins TTKernel_CB:$in0_cb, TTKernel_CB:$in1_cb);
}

def TTKernel_MulTilesInitFOp : TTKernel_Op<"mul_tiles_init_f"> {
    let summary = "Short init function. Init for math only.";
    let description = [{
      Must be run before mul_tiles.
    }];
}

def TTKernel_MulTilesOp : TTKernel_Op<"mul_tiles"> {
    let summary = "Mul operation";
    let description = [{
      Performs element-wise multiplication C=A*B of tiles in two CBs at given
      indices and writes the result to the DST register at index dst_tile_index.
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins TTKernel_CB:$in0_cb, TTKernel_CB:$in1_cb, I32:$in0_tile_index, I32:$in1_tile_index, I32:$dst_index);
}

def TTKernel_UnaryOpInitCommonOp : TTKernel_Op<"unary_op_init_common"> {
    let summary = "Initialization function for unary operations.";
    let description = [{
      This operation initializes all necessary components for unary operations,
      including unpacking, packing, and math configurations.
    }];

    let arguments = (ins TTKernel_CB:$icb, TTKernel_CB:$ocb);
}

def TTKernel_ExpTileInitOp : TTKernel_Op<"exp_tile_init"> {
    let summary = "Short init function which configures compute unit for execution of exp_tile.";
    let description = [{
      Must be run before exp_tile.
    }];
}

def TTKernel_ExpTileOp : TTKernel_Op<"exp_tile"> {
    let summary = "Exp operation";
    let description = [{
      Performs element-wise computation of exponential on each element of a tile
      in DST register at index tile_index. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call. This call is blocking and is only
      available on the compute engine.
    }];

    let arguments = (ins I32:$tile_index);
}

def TTKernel_RecipTileInitOp : TTKernel_Op<"recip_tile_init"> {
    let summary = "Init function for recip_tile operation. Refer to documentation for any init function.";
    let description = [{
      Must be called before recip_tile function.
    }];
}

def TTKernel_RecipTileOp : TTKernel_Op<"recip_tile"> {
    let summary = "Recip tile in the DST at specified index.";
    let description = [{
      Performs element-wise computation of the reciprocal on each element of a tile
      in DST register at index tile_index. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call. This call is blocking and is only
      available on the compute engine.
      Only works for Float32, Float16_b, Bfp8_b data formats for full accuracy.
    }];

    let arguments = (ins I32:$tile_index);
}

def TTKernel_ReduceInitOp : TTKernel_Op<"reduce_init"> {
    let summary = "Init function";
    let description = [{
      Must be run before reduce_tile.
    }];

    let arguments = (ins TTKernel_CB:$in_cb,
                         TTKernel_CB:$scaling_cb,
                         TTKernel_CB:$out_cb,
                         TTKernel_ReduceTypeAttr:$reduce_type,
                         TTKernel_ReduceDimAttr:$reduce_dim);
}

def TTKernel_ReduceTileOp : TTKernel_Op<"reduce_tile"> {
    let summary = "Reduce operation";
    let description = [{
      Performs a reduction operation *B = reduce(A)* using reduce_func for
      dimension reduction on a tile in the CB at a given index and writes the
      result to the DST register at index *dst_tile_index*. Reduction can be
      either of type *Reduce::R*, *Reduce::C* or *Reduce::RC*, identifying the
      dimension(s) to be reduced in size to 1. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call.
      The templates takes reduce_type which can be ReduceFunc::Sum, ReduceFunc::Max
      and reduce_dim which can be Reduce::R, Reduce::C, Reduce::RC.
      They can also be specified by defines REDUCE_OP and REDUCE_DIM.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins TTKernel_CB:$in_cb,
                         TTKernel_CB:$scaling_cb,
                         I32:$in_tile_index,
                         I32:$scaling_tile_index,
                         I32:$dst_index,
                         TTKernel_ReduceTypeAttr:$reduce_type,
                         TTKernel_ReduceDimAttr:$reduce_dim);
}

//===----------------------------------------------------------------------===//
// TTKernel SFPU operations
//===----------------------------------------------------------------------===//

def TTKernel_MaxTilesInitOp : TTKernel_Op<"max_tile_init"> {
    let summary = "Short init function";
    let description = [{
      Must be run before max_tile.
    }];

    let arguments = (ins);
}

def TTKernel_MaxTilesOp : TTKernel_Op<"max_tile"> {
    let summary = "Max operation";
    let description = [{
      Performs element-wise computation of maximum operation
      DST[dst0_index] <- max(DST[dst0_index], DST[dst1_index])
      on DST register operands. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call.
    }];

    let arguments = (ins I32:$dst0_index, I32:$dst1_index);
}

//===----------------------------------------------------------------------===//
// TTKernel CB operations
//===----------------------------------------------------------------------===//

def TTKernel_CBPushBackOp : TTKernel_Op<"cb_push_back"> {
    let summary = "CBPushBack call.";
    let description = [{
      CBPushBack operation
    }];

    let arguments = (ins TTKernel_CB:$cb, I32:$numPages);

    let hasVerifier = 1;
}

def TTKernel_CBPopFrontOp : TTKernel_Op<"cb_pop_front"> {
    let summary = "CBPopFront call.";
    let description = [{
      CBPopFront operation
    }];

    let arguments = (ins TTKernel_CB:$cb, I32:$numPages);

    let hasVerifier = 1;
}

def TTKernel_CBReserveBackOp : TTKernel_Op<"cb_reserve_back"> {
    let summary = "CBReserveBack call.";
    let description = [{
      CBReserveBack operation
    }];

    let arguments = (ins TTKernel_CB:$cb, I32:$numPages);

    let hasVerifier = 1;
}

def TTKernel_CBWaitFrontOp : TTKernel_Op<"cb_wait_front"> {
    let summary = "CBWaitFront call.";
    let description = [{
      CBWaitFront operation
    }];

    let arguments = (ins TTKernel_CB:$cb, I32:$numPages);

    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TTKernel Tile operations
//===----------------------------------------------------------------------===//

def TTKernel_TilizeInitOp : TTKernel_Op<"tilize_init"> {
    let summary = "TilizeInitOp call.";
    let description = [{
      TilizeInitOp operation
    }];

    let arguments = (ins TTKernel_CB:$cbIn, I32:$numTiles, TTKernel_CB:$cbOut);

    let hasVerifier = 1;
}

def TTKernel_UntilizeInitOp : TTKernel_Op<"untilize_init"> {
    let summary = "UntilizeInitOp call.";
    let description = [{
      UntilizeInitOp operation
    }];

    let arguments = (ins TTKernel_CB:$cbIn, TTKernel_CB:$cbOut);

    let hasVerifier = 1;
}

def TTKernel_TilizeBlockOp : TTKernel_Op<"tilize_block"> {
    let summary = "TilizeBlockOp call.";
    let description = [{
      TilizeBlockOp operation
    }];

    let arguments = (ins TTKernel_CB:$cbIn, I32:$numTiles, TTKernel_CB:$cbOut);

    let hasVerifier = 1;
}

def TTKernel_UntilizeBlockOp : TTKernel_Op<"untilize_block"> {
    let summary = "UntilizeBlockOp call.";
    let description = [{
      UntilizeBlockOp operation
    }];

    let arguments = (ins TTKernel_CB:$cbIn, I32:$numTiles, TTKernel_CB:$cbOut);

    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TTKernel NOC operations
//===----------------------------------------------------------------------===//

def TTKernel_GetNocAddrXYOp : TTKernel_Op<"get_noc_addr_xy"> {
    let summary = "GetNocAddrXY";
    let description = [{
      GetNocAddr api including core coordinates
    }];

    let arguments = (ins I32:$x, I32:$y, I32:$l1Address);
    let results = (outs TTKernel_NocAddr:$nocAddr);
}

def TTKernel_GetNocAddrFromBankIDOp: TTKernel_Op<"get_noc_addr_from_bank_id"> {
    let summary = "GetNocAddrFromBankID";
    let description = [{
      GetNocAddrFromBankID api
    }];

    let arguments = (ins I32:$bank_id, I32:$bankAddressOffset);
    let results = (outs TTKernel_NocAddr:$nocAddr);
}

def TTKernel_GetNocAddrOp : TTKernel_Op<"get_noc_addr"> {
    let summary = "GetNocAddr";
    let description = [{
      GetNocAddr
    }];

    let arguments = (ins I32:$l1Address);
    let results = (outs TTKernel_NocAddr:$nocAddr);
}

def TTKernel_NocAsyncReadOp : TTKernel_Op<"noc_async_read"> {
    let summary = "NocAsyncRead";
    let description = [{
      NocAsyncRead
    }];

    let arguments = (ins TTKernel_NocAddr:$srcNocAddr, I32:$dstLocalL1Addr, I32:$size);
}

def TTKernel_NocAsyncReadOnePacketSetStateOp : TTKernel_Op<"noc_async_read_one_packet_set_state"> {
    let summary = "NocAsyncReadOnePacketSetState";
    let description = [{
      NocAsyncReadOnePacketSetState
    }];

    let arguments = (ins TTKernel_NocAddr:$srcNocAddr, I32:$size);
}

def TTKernel_NocAsyncReadOnePacketWithStateOp : TTKernel_Op<"noc_async_read_one_packet_with_state"> {
    let summary = "NocAsyncReadOnePacketWithState";
    let description = [{
      NocAsyncReadOnePacketWithState
    }];

    let arguments = (ins TTKernel_NocAddr:$srcNocAddr, AnyTypeOf<[I32, TTKernel_L1Addr]>:$dstLocalL1Addr);
}

def TTKernel_NocAsyncReadBarrierOp : TTKernel_Op<"noc_async_read_barrier"> {
    let summary = "NocAsyncReadBarrier";
    let description = [{
      NocAsyncReadBarrier
    }];
}

def TTKernel_NocAsyncWriteOp : TTKernel_Op<"noc_async_write"> {
    let summary = "NocAsyncWrite";
    let description = [{
      NocAsyncWrite
    }];

    let arguments = (ins I32:$srcLocalL1Addr, TTKernel_NocAddr:$dstNocAddr, I32:$size);
}

def TTKernel_NocAsyncWriteBarrierOp : TTKernel_Op<"noc_async_write_barrier"> {
    let summary = "NocAsyncWriteBarrier";
    let description = [{
      NocAsyncWriteBarrier
    }];
}

def TTKernel_GetSemaphoreOp : TTKernel_Op<"get_semaphore"> {
    let summary = "GetSemaphoreOp";
    let description = [{
      Get L1 addr of the semaphore with specified semaphore id
    }];

    let arguments = (ins I32:$semaphore_id);
    let results = (outs TTKernel_L1Addr:$sem_addr);
}

def TTKernel_NocSemaphoreIncOp : TTKernel_Op<"noc_semaphore_inc"> {
    let summary = "NocSemaphoreInc";
    let description = [{
      The Tensix core executing this function call initiates an atomic increment
      (with 32-bit wrap) of a remote Tensix core L1 memory address. This L1 memory
      address is used as a semaphore of size 4 Bytes, as a synchronization
      mechanism.
    }];

    let arguments = (ins TTKernel_NocAddr:$addr, I32:$incr, I32:$noc_id);
}

def TTKernel_NocSemaphoreSetOp : TTKernel_Op<"noc_semaphore_set"> {
    let summary = "NocSemaphoreSet";
    let description = [{
      Sets the value of a local L1 memory address on the Tensix core executing
      this function to a specific value. This L1 memory address is used as a
      semaphore of size 4 Bytes, as a synchronization mechanism. Also, see
      *noc_semaphore_wait*.
    }];

    let arguments = (ins TTKernel_L1AddrPtr:$sem_addr, I32:$val);
}

def TTKernel_NocSemaphoreWaitOp : TTKernel_Op<"noc_semaphore_wait"> {
    let summary = "NocSemaphoreWait";
    let description = [{
      A blocking call that waits until the value of a local L1 memory address on
      the Tensix core executing this function becomes equal to a target value.
      This L1 memory address is used as a semaphore of size 4 Bytes, as a
      synchronization mechanism. Also, see *noc_semaphore_set*.
    }];

    let arguments = (ins TTKernel_L1AddrPtr:$sem_addr, I32:$val);
}

def TTKernel_NocSemaphoreWaitMinOp : TTKernel_Op<"noc_semaphore_wait_min"> {
    let summary = "NocSemaphoreWaitMin";
    let description = [{
      A blocking call that waits until the value of a local L1 memory address on
      the Tensix core executing this function becomes equal or greater than a target value.
      This L1 memory address is used as a semaphore of size 4 Bytes, as a
      synchronization mechanism. Also, see *noc_semaphore_set*.
    }];

    let arguments = (ins TTKernel_L1AddrPtr:$sem_addr, I32:$val);
}

def TTKernel_NocSemaphoreSetMulticastOp : TTKernel_Op<"noc_semaphore_set_multicast"> {
    let summary = "NocSemaphoreSetMulticast";
    let description = [{
      Initiates an asynchronous write from a source address in L1 memory on the
      Tensix core executing this function call to a rectangular destination grid.
      The destinations are specified using a uint64_t encoding referencing an
      on-chip grid of nodes located at NOC coordinate range
      (x_start,y_start,x_end,y_end) and a local address created using
      *get_noc_multicast_addr* function. The size of data that is sent is 4 Bytes.
      This is usually used to set a semaphore value at the destination nodes, as a
      way of a synchronization mechanism. The same as *noc_async_write_multicast*
      with preset size of 4 Bytes.
      With this API, the multicast sender cannot be part of the multicast
      destinations. If the multicast sender has to be in the multicast
      destinations (i.e. must perform a local L1 write), the other API variant
      *noc_semaphore_set_multicast_loopback_src* can be used.
    }];

    let arguments = (ins TTKernel_L1Addr:$src_local_l1_addr,
                         TTKernel_NocAddr:$dst_noc_addr_multicast,
                         I32:$num_dests,
                         BoolAttr:$linked,
                         BoolAttr:$multicast_path_reserve);
}

def TTKernel_NocSemaphoreSetMulticastLoopbackOp : TTKernel_Op<"noc_semaphore_set_multicast_loopback_src"> {
    let summary = "NocSemaphoreSetMulticastLoopback";
    let description = [{
      Initiates an asynchronous write from a source address in L1 memory on the
      Tensix core executing this function call to a rectangular destination grid.
      The destinations are specified using a uint64_t encoding referencing an
      on-chip grid of nodes located at NOC coordinate range
      (x_start,y_start,x_end,y_end) and a local address created using
      *get_noc_multicast_addr* function. The size of data that is sent is 4 Bytes.
      This is usually used to set a semaphore value at the destination nodes, as a
      way of a synchronization mechanism. The same as *noc_async_write_multicast*
      with preset size of 4 Bytes.
      Note: With this API, sending data only to the source node (when num_dests
      is 1) may result in unexpected behaviour. For some parameters, hangs have
      been observed. For some other parameters, nothing may happen. Consider using
      regular non multicast operations such as *noc_async_write* in this case.
    }];

    let arguments = (ins TTKernel_L1Addr:$src_local_l1_addr,
                         TTKernel_NocAddr:$dst_noc_addr_multicast,
                         I32:$num_dests,
                         BoolAttr:$linked,
                         BoolAttr:$multicast_path_reserve);
}

def TTKernel_NocTransactionsTableOp : TTKernel_Op<"noc_transactions_table"> {
    let summary = [{define an array of parameter entries for a sequence of NOC
     read/write operations
    }];
    let description = [{
      Encapsulates parameters for a group of NOC read/write operations as
      an array of (dst, src, size, dst_x, dst_y, numElements) entries, with
      each entry slot of type int32_t.
      An entry defines an src->dst data movement operation and as either
      a read (remote src, local dst) or a write (local src, remote dst) by
      the calling core.
    }];

    let arguments = (ins DenseI32ArrayAttr:$entries);
    let results = (outs AnyStaticShapeMemRef:$table);
}

//===----------------------------------------------------------------------===//
// TTKernel Compile and runtime arguments operations
//===----------------------------------------------------------------------===//

def TTKernel_GetArgValOp : TTKernel_Op<"get_arg_val"> {
    let summary = "Get runtime arg value.";
    let description = [{
      Get runtime argument value at specified index.
    }];

    let arguments = (ins I32:$arg_index);

    let results = (outs AnyTypeOf<[TTKernel_Semaphore, I32]>:$arg_val);
}

def TTKernel_GetCompileArgValOp : TTKernel_Op<"get_compile_time_arg_val"> {
    let summary = "Get compile-time arg value.";
    let description = [{
      Get compile-time argument value at specified index.
    }];

    let arguments = (ins I32:$arg_index);

    let results = (outs AnyTypeOf<[TTKernel_Semaphore, I32]>:$arg_val);
}

//===----------------------------------------------------------------------===//
// TTKernel Helper functions
//===----------------------------------------------------------------------===//

def TTKernel_CastToL1PtrOp : TTKernel_Op<"reinterpret_cast<volatile tt_l1_ptr uint32_t*>"> {
    let summary = "CastToL1Ptr";
    let description = [{
      Cast specified addr to L1 pointer.
    }];

    let arguments = (ins AnyTypeOf<[I32, TTKernel_L1Addr]>:$addr);

    let results = (outs TTKernel_L1AddrPtr:$l1_ptr);
}

def TTKernel_StoreToL1Op : TTKernel_Op<"store_to_l1"> {
    let summary = "StoreToL1";
    let description = [{
      Store value to L1.
    }];

    let arguments = (ins I32:$value, TTKernel_L1AddrPtr:$l1_ptr, I32:$offset);
}

//===----------------------------------------------------------------------===//
// TTKernel Multicast NoC operations
//===----------------------------------------------------------------------===//

def TTKernel_GetNocMulticastAddrOp : TTKernel_Op<"get_noc_multicast_addr"> {
  let summary = "GetNocMulticastAddr";
  let description = [{
    GetNocMulticastAddr
  }];

  let arguments = (ins I32:$noc_x_start, I32:$noc_y_start, I32:$noc_x_end, I32:$noc_y_end, I32:$addr, Optional<I8>:$noc);
  let results = (outs TTKernel_NocAddr:$mcastNocAddr);
}

def TTKernel_NocAsyncWriteMulticastOnePacketOp : TTKernel_Op<"noc_async_write_multicast_one_packet"> {
  let summary = "NocAsyncWriteMulticastOnePacket";
  let description = [{
    NocAsyncWriteMulticastOnePacket
    this issues only a single packet with size <= NOC_MAX_BURST_SIZE (ie maximum packet size)
  }];

  let arguments = (ins I32:$srcLocalL1Addr, TTKernel_NocAddr:$dstNocAddrMulticast, I32:$size, I32:$num_dests, OptionalAttr<BoolAttr>:$linked, OptionalAttr<BoolAttr>:$multicast_path_reserve, Optional<I8>:$noc);
}

def TTKernel_NocAsyncWriteMulticastOp : TTKernel_Op<"noc_async_write_multicast"> {
  let summary = "NocAsyncWriteMulticast";
  let description = [{
    Initiates an asynchronous write from a source address in L1 memory on the
    Tensix core executing this function call to a rectangular destination grid.
    The destinations are specified using a uint64_t encoding referencing an
    on-chip grid of nodes located at NOC coordinate range
    (x_start,y_start,x_end,y_end) and a local address created using
    *get_noc_multicast_addr* function. Also, *see noc_async_write_barrier*.

    The destination nodes can only be a set of Tensix cores + L1 memory address.
    The destination nodes must form a rectangular grid. The destination L1
    memory address must be the same on all destination nodes.

    With this API, the multicast sender cannot be part of the multicast
    destinations. If the multicast sender has to be in the multicast
    destinations (i.e. must perform a local L1 write), the other API variant
    *noc_async_write_multicast_loopback_src* can be used.

    Note: The number of destinations needs to be non-zero. Besides that,
    there is no restriction on the number of destinations, i.e. the
    multicast destinations can span the full chip. However, as mentioned
    previously, the multicast source cannot be part of the destinations. So, the
    maximum number of destinations is 119.
  }];

  let arguments = (ins I32:$srcLocalL1Addr, TTKernel_NocAddr:$dstNocAddrMulticast, I32:$size, I32:$num_dests, OptionalAttr<BoolAttr>:$linked, OptionalAttr<BoolAttr>:$multicast_path_reserve, Optional<I8>:$noc);
}

def TTKernel_NocAsyncWriteMulticastLoopbackSrcOp : TTKernel_Op<"noc_async_write_multicast_loopback_src"> {
  let summary = "NocAsyncWriteMulticastLoopbackSrc";
  let description = [{
    NocAsyncWriteMulticastLoopbackSrc
  }];

  let arguments = (ins I32:$srcLocalL1Addr, TTKernel_NocAddr:$dstNocAddrMulticast, I32:$size, I32:$num_dests, OptionalAttr<BoolAttr>:$linked, OptionalAttr<BoolAttr>:$multicast_path_reserve, Optional<I8>:$noc);
}

//===----------------------------------------------------------------------===//
// TTKernel Misc operations
//===----------------------------------------------------------------------===//

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

def TTKernel_MemZerosBaseOp : TTKernel_Op<"mem_zeros_base"> {
    let summary = "Op corresponding to MEM_ZEROS_BASE macro in kernels.";
    let description = [{
      Op corresponding to MEM_ZEROS_BASE macro in kernels.
    }];

    let results = (outs I32:$result);
}

def TTKernel_MemZerosSizeOp : TTKernel_Op<"mem_zeros_size"> {
    let summary = "Op corresponding to MEM_ZEROS_SIZE macro in kernels.";
    let description = [{
      Op corresponding to MEM_ZEROS_SIZE macro in kernels.
    }];

    let results = (outs I32:$result);
}

def TTKernel_GetWritePtrOp : TTKernel_Op<"get_write_ptr"> {
    let summary = "GetWritePtr";
    let description = [{
      GetWritePtr operation
    }];

    let arguments = (ins TTKernel_CB:$cb);
    let results = (outs I32:$writePtr);
}

def TTKernel_GetReadPtrOp : TTKernel_Op<"get_read_ptr"> {
    let summary = "GetReadPtr";
    let description = [{
      GetReadPtr operation
    }];

    let arguments = (ins TTKernel_CB:$cb);
    let results = (outs I32:$readPtr);
}

def TTKernel_GetTileSizeOp : TTKernel_Op<"get_tile_size"> {
    let summary = "Get the tile size in bytes of a given CB";
    let description = [{
      get_tile_size operation
    }];
    let arguments = (ins TTKernel_CB:$cb);
    let results = (outs I32:$tileSizeBytes);
}


#endif
