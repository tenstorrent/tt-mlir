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

def TTKernel_MacroOp : TTKernel_Op<"macro"> {
    let summary = "Macro op.";
    let description = [{
      Macro operation to mimic C++ macros
    }];

    let arguments = (ins FlatSymbolRefAttr:$name);

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

#endif
