// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELOPS_TD
#define TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELOPS_TD

include "ttmlir/Dialect/TTKernel/IR/TTKernelBase.td"
include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.td"
include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

def AnyMemRefOrCB: AnyTypeOf<[AnyNon0RankedMemRef, TTKernel_CB]>;

def IndexLike: AnyTypeOf<[Index, I32]>;

//===----------------------------------------------------------------------===//
// TTKernel Compute Kernel Hardware Startup operation
//===----------------------------------------------------------------------===//

def TTKernel_ComputeKernelHWStartupOp : TTKernel_InitOp<"compute_kernel_hw_startup"> {
    let summary = "compute_kernel_hw_startup";
    let description = [{
      Must be run at the start of compute kernel.
    }];

    let arguments = (ins TTKernel_CB:$icb0,
                         Optional<TTKernel_CB>:$icb1,
                         TTKernel_CB:$ocb);

    let assemblyFormat = [{
      `(` $icb0 (`,` $icb1^)? `,` $ocb `)` attr-dict `:` functional-type(operands, results)
    }];
}

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

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_TileRegsCommitOp : TTKernel_Op<"tile_regs_commit"> {
    let summary = "tile_regs_commit";
    let description = [{
      Release lock on DST register by MATH thread. The lock had to be previously acquired with tile_regs_acquire.
    }];

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_TileRegsWaitOp : TTKernel_Op<"tile_regs_wait", [TTKernel_DeviceZoneOpTrait]> {
    let summary = "tile_regs_wait";
    let description = [{
      Acquire an exclusive lock on the DST register for the PACK thread.
      It waits for the MATH thread to commit the DST register.
      This is a blocking function, i.e. this function will wait until the lock is acquired.
    }];

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_TileRegsReleaseOp : TTKernel_Op<"tile_regs_release"> {
    let summary = "tile_regs_release";
    let description = [{
      Release lock on DST register by PACK thread. The lock had to be previously acquired with tile_regs_wait.
    }];

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
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

    let arguments = (ins IndexLike:$dst_index, TTKernel_CB:$out_cb, IndexLike:$out_index, DefaultValuedAttr<BoolAttr, "false">:$out_of_order);

    let assemblyFormat = [{
      `(` $dst_index `,` $out_cb `,` $out_index `,` $out_of_order`)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_PackReconfigL1AccOp : TTKernel_Op<"pack_reconfig_l1_acc"> {
    let summary = "Reconfigure packer to L1 accumulation mode.";
    let description = [{
      Reconfigures the packer to accumulate when packing from DST to L1.
    }];

    let arguments = (ins I32:$l1_acc_en);

    let assemblyFormat = [{
      `(` $l1_acc_en `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_CopyTileInitOp : TTKernel_InitOp<"copy_tile_init"> {
    let summary = "Perform the init for copy tile. This does not reconfigure the unpacker data types.";
    let description = [{
      Must be called before copy_tile.
    }];

    let arguments = (ins TTKernel_CB:$cb0);

    let assemblyFormat = [{
      `(` $cb0 `)` attr-dict `:` functional-type(operands, results)
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

    let arguments = (ins TTKernel_CB:$cb0, IndexLike:$tile_index_cb, IndexLike:$tile_index_dst);

    let assemblyFormat = [{
      `(` $cb0 `,` $tile_index_cb `,` $tile_index_dst `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_TypecastTileInitOp : TTKernel_InitOp<"typecast_tile_init"> {
    let summary = "Init function for typecast_tile operation. Refer to documentation for any init function.";
    let description = [{
      Must be run before typecast_tile.
    }];

    let arguments = (ins TTCore_DataTypeAttr:$in_dtype,
                         TTCore_DataTypeAttr:$out_dtype);

    let assemblyFormat = [{
      `(` $in_dtype `,` $out_dtype `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_TypecastTileOp : TTKernel_SFPUOp<"typecast_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Cast the dataformat of the tile in the DST at specified index.";
    let description = [{
      Performs element-wise typecast operation
      DST[dst0_index] <- typecast<in_dataformat, out_dataformat>(DST[dst0_index])
      on DST register operands. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call.
    }];

    let arguments = (ins IndexLike:$dst0_index,
                         TTCore_DataTypeAttr:$in_dtype,
                         TTCore_DataTypeAttr:$out_dtype);

    let assemblyFormat = [{
      `(` $dst0_index `,` $in_dtype `,` $out_dtype `)` attr-dict `:` functional-type(operands, results)
    }];
}

//===----------------------------------------------------------------------===//
// TTKernel FPU operations
//===----------------------------------------------------------------------===//

def TTKernel_UnaryOpInitCommonOp : TTKernel_InitOp<"unary_op_init_common"> {
    let summary = "Initialization function for unary operations.";
    let description = [{
      This operation initializes all necessary components for unary operations,
      including unpacking, packing, and math configurations.
    }];

    let arguments = (ins TTKernel_CB:$icb, TTKernel_CB:$ocb);

    let assemblyFormat = [{
      `(` $icb `,` $ocb `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_BinaryOpInitCommonOp : TTKernel_InitOp<"binary_op_init_common"> {
    let summary = "Init function for all binary ops";
    let description = [{
      Followed by the specific init required with an opcode (binrary_op_specific_init).
    }];

    let arguments = (ins TTKernel_CB:$in0_cb, TTKernel_CB:$in1_cb, TTKernel_CB:$out_cb);

    let assemblyFormat = [{
      `(` $in0_cb `,` $in1_cb `,` $out_cb `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_MatmulInitOp : TTKernel_InitOp<"mm_init"> {
    let summary = "Matmul init function";
    let description = [{
      Can only be run ONCE per kernel. Should be run before matmul.
    }];

    let arguments = (ins TTKernel_CB:$in0_cb, TTKernel_CB:$in1_cb, TTKernel_CB:$out_cb, I32:$transpose);

    let assemblyFormat = [{
      `(` $in0_cb `,` $in1_cb `,` $out_cb `,` $transpose `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_MatmulInitShortOp : TTKernel_InitOp<"mm_init_short"> {
    let summary = "Matmul short init function";
    let description = [{
      Can be run MULTIPLE times per kernel. Should be run before matmul. Use this if some other init was called between mm_init and matmul_tiles. (i.e. in a loop)
    }];

    let arguments = (ins TTKernel_CB:$in0_cb, TTKernel_CB:$in1_cb, I32:$transpose);

    let assemblyFormat = [{
      `(` $in0_cb `,` $in1_cb `,` $transpose `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_MatmulTilesOp : TTKernel_FPUOp<"matmul_tiles", [TTKernel_TernaryOpTrait, TTKernel_DstAccumulatingOpTrait]> {
    let summary = "Matmul tiles operation";
    let description = [{
      Performs tile-sized matrix multiplication *C=A\*B* between the tiles in two
      specified input CBs and writes the result to DST. The DST register buffer
      must be in acquired state via ttkernel.tile_regs_acquire call. This call is blocking and
      is only available on the compute engine.
    }];

    let arguments = (ins TTKernel_CB:$in0_cb_id,
                         TTKernel_CB:$in1_cb_id,
                         IndexLike:$in0_tile_idx,
                         IndexLike:$in1_tile_idx,
                         IndexLike:$dst_tile_idx);

    let assemblyFormat = [{
      `(` $in0_cb_id `,` $in1_cb_id `,` $in0_tile_idx `,` $in1_tile_idx `,` $dst_tile_idx `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_MatmulBlockInitOp : TTKernel_InitOp<"mm_block_init"> {
    let summary = "Matmul init function";
    let description = [{
      Initialization for matmul_block operation. Must be called before matmul_block.
    }];

    let arguments = (ins TTKernel_CB:$in0_cb, TTKernel_CB:$in1_cb, TTKernel_CB:$out_cb, I32:$transpose, I32:$ct_dim, I32:$rt_dim, I32:$kt_dim);
}

def TTKernel_MatmulBlockInitShortOp : TTKernel_InitOp<"mm_block_init_short"> {
    let summary = "Matmul short init function";
    let description = [{
      A short version of matmul_block initialization.
    }];

    let arguments = (ins TTKernel_CB:$in0_cb, TTKernel_CB:$in1_cb, I32:$transpose,  I32:$ct_dim, I32:$rt_dim, I32:$kt_dim);
}

def TTKernel_ExperimentalMatmulBlockOp : TTKernel_FPUOp<"experimental::matmul_block", [TTKernel_TernaryOpTrait, TTKernel_DstAccumulatingOpTrait]> {
    let summary = "Matmul tiles operation";
    let description = [{
      Performs block-sized matrix multiplication *C=A\*B* between the blocks in two
      different input CBs and writes the result to DST. The DST register buffer
      must be in acquired state via *acquire_dst* call. This call is blocking and
      is only available on the compute engine.
    }];

    let arguments = (ins TTKernel_CB:$in0_cb_id,
                         TTKernel_CB:$in1_cb_id,
                         IndexLike:$in0_tile_idx,
                         IndexLike:$in1_tile_idx,
                         IndexLike:$dst_tile_idx,
                         I32:$transpose,
                         I32:$ct_dim,
                         I32:$rt_dim,
                         I32:$kt_dim,
                         I32:$nt_dim);
}


def TTKernel_AddTilesInitOp : TTKernel_InitOp<"add_tiles_init"> {
    let summary = "Short init function";
    let description = [{
      Must be run before add_tiles.
    }];

    let arguments = (ins TTKernel_CB:$in0_cb, TTKernel_CB:$in1_cb); // FIXME: , BOOL:$acc_to_dst);

     let assemblyFormat = [{
      `(` $in0_cb `,` $in1_cb `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_AddTilesOp : TTKernel_FPUOp<"add_tiles", [TTKernel_BinaryOpTrait]> {
    let summary = "Add operation";
    let description = [{
      Performs element-wise addition C=A+B of tiles in two CBs at given indices
      and writes the result to the DST register at index dst_tile_index. The DST
      register buffer must be in acquired state via *tile_regs_acquire* call. This call
      is blocking and is only available on the compute engine.
    }];

    let arguments = (ins TTKernel_CB:$in0_cb, TTKernel_CB:$in1_cb, IndexLike:$in0_tile_index, IndexLike:$in1_tile_index, IndexLike:$dst_index);

    let assemblyFormat = [{
      `(` $in0_cb `,` $in1_cb `,` $in0_tile_index `,` $in1_tile_index `,` $dst_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_SubTilesInitOp : TTKernel_InitOp<"sub_tiles_init"> {
    let summary = "Short init function";
    let description = [{
      Must be run before sub_tiles.
    }];

    let arguments = (ins TTKernel_CB:$in0_cb, TTKernel_CB:$in1_cb); // FIXME: , BOOL:$acc_to_dst);

    let assemblyFormat = [{
      `(` $in0_cb `,` $in1_cb `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_SubTilesOp : TTKernel_FPUOp<"sub_tiles", [TTKernel_BinaryOpTrait]> {
    let summary = "Sub operation";
    let description = [{
      Performs element-wise subtraction C=A-B of tiles in two CBs at given
      indices and writes the result to the DST register at index dst_tile_index.
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins TTKernel_CB:$in0_cb, TTKernel_CB:$in1_cb, IndexLike:$in0_tile_index, IndexLike:$in1_tile_index, IndexLike:$dst_index);

    let assemblyFormat = [{
      `(` $in0_cb `,` $in1_cb `,` $in0_tile_index `,` $in1_tile_index `,` $dst_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_MulTilesInitOp : TTKernel_InitOp<"mul_tiles_init"> {
    let summary = "Short init function";
    let description = [{
      Must be run before mul_tiles.
    }];

    let arguments = (ins TTKernel_CB:$in0_cb, TTKernel_CB:$in1_cb);

    let assemblyFormat = [{
      `(` $in0_cb `,` $in1_cb `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_MulTilesOp : TTKernel_FPUOp<"mul_tiles", [TTKernel_BinaryOpTrait]> {
    let summary = "Mul operation";
    let description = [{
      Performs element-wise multiplication C=A*B of tiles in two CBs at given
      indices and writes the result to the DST register at index dst_tile_index.
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins TTKernel_CB:$in0_cb, TTKernel_CB:$in1_cb, IndexLike:$in0_tile_index, IndexLike:$in1_tile_index, IndexLike:$dst_index);

    let assemblyFormat = [{
      `(` $in0_cb `,` $in1_cb `,` $in0_tile_index `,` $in1_tile_index `,` $dst_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_BinaryDestReuseTilesInitOp : TTKernel_InitOp<"binary_dest_reuse_tiles_init"> {
    let summary = "Init for binary op with dest reuse";
    let description = [{
      Init function for binary_dest_reuse_tiles operation.
      Must be run before binary_dest_reuse_tiles.
      `eltwise_binary_type` specifies the operation (add/sub/mul).
      `reuse_type` specifies which source register gets the DST operand.
    }];

    let arguments = (ins TTKernel_CB:$in_cb,
                         TTKernel_EltwiseBinaryTypeAttr:$eltwise_binary_type,
                         TTKernel_BinaryDestReuseTypeAttr:$reuse_type);

    let assemblyFormat = [{
      `(` $in_cb `,` $eltwise_binary_type `,` $reuse_type `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_BinaryDestReuseTilesOp : TTKernel_FPUOp<"binary_dest_reuse_tiles"> {
    let summary = "Binary op with one operand from DST";
    let description = [{
      Performs element-wise binary op where one operand comes from DST and one
      from a CB. If reuse_type is dest_to_srca, DST[dst_tile_index] is loaded
      to SRCA and CB tile is loaded to SRCB. If dest_to_srcb, the opposite.
      Result is written back to DST[dst_tile_index].
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      `eltwise_binary_type` specifies the operation (add/sub/mul).
    }];

    let arguments = (ins TTKernel_CB:$in_cb,
                         IndexLike:$in_tile_index,
                         IndexLike:$dst_tile_index,
                         TTKernel_EltwiseBinaryTypeAttr:$eltwise_binary_type,
                         TTKernel_BinaryDestReuseTypeAttr:$reuse_type);

    let assemblyFormat = [{
      `(` $in_cb `,` $in_tile_index `,` $dst_tile_index `,` $eltwise_binary_type `,` $reuse_type `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_UnaryBcastInitOp : TTKernel_InitOp<"unary_bcast_init"> {
    let summary = "Init function";
    let description = [{
      Must be run before bcast_tile.
    }];

    let arguments = (ins TTKernel_CB:$in_cb,
                         TTKernel_CB:$out_cb,
                         TTKernel_BcastTypeAttr:$bcast_type
                        );

    let assemblyFormat = [{
      `(` $in_cb `,` $out_cb `,` $bcast_type `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_UnaryBcastTileOp : TTKernel_FPUOp<"unary_bcast", [TTKernel_UnaryOpTrait]> {
    let summary = "Broadcast operation";
    let description = [{
      Performs a broadcast operation *B = broadcast(A)* using bcast_dim for
      dimension expansion on a tile in the CB at a given index and writes the
      result to the DST register at index *dst_tile_index*. The supported
      broadcast dimensions are `row`, `col`, `scalar` (both row and column). The
      DST register buffer must be in acquired state via *tile_regs_acquire*
      call. This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins TTKernel_CB:$in_cb,
                         IndexLike:$in_tile_index,
                         IndexLike:$dst_tile_index,
                         TTKernel_BcastTypeAttr:$bcast_type
    );

    let assemblyFormat = [{
      `(` $in_cb `,` $in_tile_index `,` $dst_tile_index `,` $bcast_type `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ReduceInitOp : TTKernel_InitOp<"reduce_init"> {
    let summary = "Init function";
    let description = [{
      Must be run before reduce_tile.
    }];

    let arguments = (ins TTKernel_CB:$in_cb,
                         TTKernel_CB:$scaling_cb,
                         TTKernel_CB:$out_cb,
                         TTKernel_ReduceTypeAttr:$reduce_type,
                         TTKernel_ReduceDimAttr:$reduce_dim,
                         UnitAttr:$full_fp32);

    let assemblyFormat = [{
      `(` $in_cb `,` $scaling_cb `,` $out_cb `,` $reduce_type `,` $reduce_dim `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ReduceUninitOp : TTKernel_InitOp<"reduce_uninit"> {
    let summary = "Init function for reduce_uninit operation.";
    let description = [{
       Resets the packer edge mask configuration to its default state by clearing any previously set masks. Needs to be called after
       reduce_tile if the next operation requires default packer state. In case that the next operation is reduce operation across the
       same dimension, this call can be omitted. If this function is not called, the packer will continue to use the edge masks set
       by the latest reduce_init call, which may lead to incorrect packing behavior in subsequent operations.

       This function is not in line with our programming mode. To be removed by end of 2025. tt-metal#22904.
    }];

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ReduceTileOp : TTKernel_FPUOp<"reduce_tile", [TTKernel_TernaryOpTrait, TTKernel_DstAccumulatingOpTrait]> {
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
                         IndexLike:$in_tile_index,
                         IndexLike:$scaling_tile_index,
                         IndexLike:$dst_index,
                         TTKernel_ReduceTypeAttr:$reduce_type,
                         TTKernel_ReduceDimAttr:$reduce_dim,
                         UnitAttr:$full_fp32);

    let assemblyFormat = [{
      `(` $in_cb `,` $scaling_cb `,` $in_tile_index `,` $scaling_tile_index `,` $dst_index `,` $reduce_type `,` $reduce_dim `)` attr-dict `:` functional-type(operands, results)
    }];
}

//===----------------------------------------------------------------------===//
// TTKernel SFPU operations
//===----------------------------------------------------------------------===//

def TTKernel_InitSFPUOp : TTKernel_InitOp<"init_sfpu"> {
    let summary = "Initialization function for SFPU operations.";
    let description = [{
      This operation initializes all necessary components for SFPU operations,
      including unpacking, packing, and math configurations.
    }];

    let arguments = (ins TTKernel_CB:$icb, TTKernel_CB:$ocb);

    let assemblyFormat = [{
      `(` $icb `,` $ocb `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_CopyDestValuesInitOp : TTKernel_InitOp<"copy_dest_values_init"> {
    let summary = "Short init function";
    let description = [{
      Must be run before copy_dest_values.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_CopyDestValuesOp : TTKernel_SFPUOp<"copy_dest_values", [TTKernel_BinaryOpTrait]> {
    let summary = "Copies all values from the tile in idst1 to the tile in idst0 in the DST register buffer.";
    let description = [{
      Copies all values from the tile in idst1 to the tile in idst0 in the DST
      register buffer. Performs element-wise computation of copy operation
      DST[dst0_index] <- DST[dst1_index]
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins IndexLike:$dst0_index, IndexLike:$dst1_index);

    let assemblyFormat = [{
      `(` $dst0_index `,` $dst1_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_BinaryMaxTileInitOp : TTKernel_InitOp<"binary_max_tile_init"> {
    let summary = "Short init function";
    let description = [{
      Must be run before binary_max_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_BinaryMaxTileOp : TTKernel_SFPUOp<"binary_max_tile", [TTKernel_BinaryOpTrait]> {
    let summary = "Elementwise maximum operation";
    let description = [{
      Performs element-wise computation of maximum operation
      DST[odst_index] <- max(DST[dst0_index], DST[dst1_index])
      on DST register operands. The DST register buffer must be in
      acquired state via *acquire_dst* call.
    }];

    let arguments = (ins IndexLike:$dst0_index, IndexLike:$dst1_index, IndexLike:$odst_index);

    let assemblyFormat = [{
      `(` $dst0_index `,` $dst1_index `,` $odst_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_BinaryMinTileInitOp : TTKernel_InitOp<"binary_min_tile_init"> {
    let summary = "Short init function";
    let description = [{
      Must be run before binary_min_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_BinaryMinTileOp : TTKernel_SFPUOp<"binary_min_tile", [TTKernel_BinaryOpTrait]> {
    let summary = "Elementwise minimum operation";
    let description = [{
      Performs element-wise computation of minimum operation
      DST[odst_index] <- min(DST[dst0_index], DST[dst1_index])
      on DST register operands. The DST register buffer must be in
      acquired state via *acquire_dst* call.
    }];

    let arguments = (ins IndexLike:$dst0_index, IndexLike:$dst1_index, IndexLike:$odst_index);

    let assemblyFormat = [{
      `(` $dst0_index `,` $dst1_index `,` $odst_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ClampScalarTileInitOp : TTKernel_InitOp<"clamp_tile_init"> {
    let summary = "Init function for clamp_scalar_tile operation.";
    let description = [{
      Must be run before clamp_scalar_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ClampScalarTileOp : TTKernel_SFPUOp<"clamp_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Clamp tile elements to scalar range";
    let description = [{
      Performs element-wise clamping of tile values to the range [min, max]
      DST[dst_index] <- clamp(DST[dst_index], min, max)
      on DST register operand. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call.
      min and max are uint 32-bit representations of the float values.
    }];

    let arguments = (ins IndexLike:$dst_index, I32:$min_param, I32:$max_param);

    let assemblyFormat = [{
      `(` $dst_index `,` $min_param `,` $max_param `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_WhereTileInitOp : TTKernel_InitOp<"where_tile_init"> {
    let summary = "Short init function";
    let description = [{
      Must be run before where_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_WhereTileOp : TTKernel_SFPUOp<"where_tile", [TTKernel_TernaryOpTrait]> {
    let summary = "Conditional selection operation";
    let description = [{
      Performs element-wise conditional selection
      DST[odst_index] <- condition ? DST[dst_true_index] : DST[dst_false_index]
      For each element, if condition is non-zero, selects from dst_true_index,
      otherwise selects from dst_false_index.
      The DST register buffer must be in acquired state via *acquire_dst* call.
    }];

    let arguments = (ins IndexLike:$dst_cond_index,
                         IndexLike:$dst_true_index,
                         IndexLike:$dst_false_index,
                         IndexLike:$odst_index,
                         TTCore_DataTypeAttr:$dtype);

    let assemblyFormat = [{
      `(` $dst_cond_index `,` $dst_true_index `,` $dst_false_index `,` $odst_index `,` $dtype `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_AddBinaryTilesInitOp : TTKernel_InitOp<"add_binary_tile_init"> {
    let summary = "Short init function";
    let description = [{
      Must be run before add_binary_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_AddBinaryTilesOp : TTKernel_SFPUOp<"add_binary_tile", [TTKernel_BinaryOpTrait]> {
    let summary = "Addition operation between two tiles";
    let description = [{
      Performs element-wise computation of addition operation
      DST[odst_index] <- DST[dst0_index] + DST[dst1_index]
      on DST register operands. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call.
    }];

    let arguments = (ins IndexLike:$dst0_index, IndexLike:$dst1_index, IndexLike:$odst_index);

    let assemblyFormat = [{
      `(` $dst0_index `,` $dst1_index `,` $odst_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_MulBinaryTilesInitOp : TTKernel_InitOp<"mul_binary_tile_init"> {
    let summary = "Short init function";
    let description = [{
      Must be run before mul_binary_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_MulBinaryTilesOp : TTKernel_SFPUOp<"mul_binary_tile", [TTKernel_BinaryOpTrait]> {
    let summary = "Multiplication operation between two tiles";
    let description = [{
      Performs element-wise computation of multiplication operation
      DST[odst_index] <- DST[dst0_index] * DST[dst1_index]
      on DST register operands. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call.
    }];

    let arguments = (ins IndexLike:$dst0_index, IndexLike:$dst1_index, IndexLike:$odst_index);

    let assemblyFormat = [{
      `(` $dst0_index `,` $dst1_index `,` $odst_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_SubBinaryTilesInitOp : TTKernel_InitOp<"sub_binary_tile_init"> {
    let summary = "Short init function";
    let description = [{
      Must be run before sub_binary_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_SubBinaryTilesOp : TTKernel_SFPUOp<"sub_binary_tile", [TTKernel_BinaryOpTrait]> {
    let summary = "Subtraction operation between two tiles";
    let description = [{
      Performs element-wise computation of subtraction operation
      DST[odst_index] <- DST[dst0_index] - DST[dst1_index]
      on DST register operands. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call.
    }];

    let arguments = (ins IndexLike:$dst0_index, IndexLike:$dst1_index, IndexLike:$odst_index);

    let assemblyFormat = [{
      `(` $dst0_index `,` $dst1_index `,` $odst_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_DivBinaryTilesInitOp : TTKernel_InitOp<"div_binary_tile_init"> {
    let summary = "Short init function";
    let description = [{
      Must be run before div_binary_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:`functional-type(operands, results)
    }];
}

def TTKernel_DivBinaryTilesOp : TTKernel_SFPUOp<"div_binary_tile", [TTKernel_BinaryOpTrait]> {
    let summary = "Divide operation between two tiles";
    let description = [{
      Performs element-wise computation of division operation
      DST[odst_index] <- DST[dst0_index] / DST[dst1_index]
      on DST register operands. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call.
    }];

    let arguments = (ins IndexLike:$dst0_index, IndexLike:$dst1_index, IndexLike:$odst_index);

    let assemblyFormat = [{
      `(` $dst0_index `,` $dst1_index `,` $odst_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_BinaryBitwiseTileInitOp : TTKernel_InitOp<"binary_bitwise_tile_init"> {
    let summary = "Init function for binary bitwise operations (AND, OR, XOR). Refer to documentation for any init function.";
    let description = [{
      Must be run before bitwise_and_binary_tile, bitwise_or_binary_tile, or bitwise_xor_binary_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_BitwiseAndBinaryTilesOp : TTKernel_SFPUOp<"bitwise_and_binary_tile", [TTKernel_BinaryOpTrait]> {
    let summary = "Bitwise AND operation between two tiles";
    let description = [{
      Performs element-wise computation of bitwise AND operation
      DST[odst_index] <- DST[dst0_index] & DST[dst1_index]
      on DST register operands. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call.
    }];

    let arguments = (ins IndexLike:$dst0_index,
                         IndexLike:$dst1_index,
                         IndexLike:$odst_index,
                         TTCore_DataTypeAttr:$dtype);

    let assemblyFormat = [{
      `(` $dst0_index `,` $dst1_index `,` $odst_index `,` $dtype `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_BitwiseOrBinaryTilesOp : TTKernel_SFPUOp<"bitwise_or_binary_tile", [TTKernel_BinaryOpTrait]> {
    let summary = "Bitwise OR operation between two tiles";
    let description = [{
      Performs element-wise computation of bitwise OR operation
      DST[odst_index] <- DST[dst0_index] | DST[dst1_index]
      on DST register operands. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call.
    }];

    let arguments = (ins IndexLike:$dst0_index,
                         IndexLike:$dst1_index,
                         IndexLike:$odst_index,
                         TTCore_DataTypeAttr:$dtype);

    let assemblyFormat = [{
      `(` $dst0_index `,` $dst1_index `,` $odst_index `,` $dtype `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_BitwiseXorBinaryTilesOp : TTKernel_SFPUOp<"bitwise_xor_binary_tile", [TTKernel_BinaryOpTrait]> {
    let summary = "Bitwise XOR operation between two tiles";
    let description = [{
      Performs element-wise computation of bitwise XOR operation
      DST[odst_index] <- DST[dst0_index] ^ DST[dst1_index]
      on DST register operands. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call.
    }];

    let arguments = (ins IndexLike:$dst0_index,
                         IndexLike:$dst1_index,
                         IndexLike:$odst_index,
                         TTCore_DataTypeAttr:$dtype);

    let assemblyFormat = [{
      `(` $dst0_index `,` $dst1_index `,` $odst_index `,` $dtype `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_BinopWithScalarTileInitOp : TTKernel_InitOp<"binop_with_scalar_tile_init"> {
    let summary = "Init function for binary operations with scalar tile operations.";
    let description = [{
      Must be run before binary operations with scalar like mul_unary_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_AddUnaryTileOp : TTKernel_SFPUOp<"add_unary_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Add by scalar operation";
    let description = [{
      Performs element-wise addition of a tile by a scalar value.
      DST[dst0_index] <- DST[dst0_index] + scalar
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins IndexLike:$dst0_index, I32:$param);

    let assemblyFormat = [{
      `(` $dst0_index `,` $param `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_MulUnaryTileOp : TTKernel_SFPUOp<"mul_unary_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Multiply by scalar operation";
    let description = [{
      Performs element-wise multiplication of a tile by a scalar value.
      DST[dst0_index] <- DST[dst0_index] * scalar
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins IndexLike:$dst0_index, I32:$param);

    let assemblyFormat = [{
      `(` $dst0_index `,` $param `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_SubUnaryTileOp : TTKernel_SFPUOp<"sub_unary_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Subtract by scalar operation";
    let description = [{
      Performs element-wise subtraction of a tile by a scalar value.
      DST[dst0_index] <- DST[dst0_index] - scalar
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins IndexLike:$dst0_index, I32:$param);

    let assemblyFormat = [{
      `(` $dst0_index `,` $param `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_DivUnaryTileOp : TTKernel_SFPUOp<"div_unary_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Divide by scalar operation";
    let description = [{
      Performs element-wise division of a tile by a scalar value.
      DST[dst0_index] <- DST[dst0_index] / scalar
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins IndexLike:$dst0_index, I32:$param);

    let assemblyFormat = [{
      `(` $dst0_index `,` $param `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_PowerTileInitOp : TTKernel_InitOp<"power_tile_init"> {
    let summary = "Init function for power_tile operation.";
    let description = [{
      Must be run before power_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_PowUnaryTileOp : TTKernel_SFPUOp<"power_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Power by scalar operation";
    let description = [{
      Performs element-wise exponentiation of a tile by a scalar value.
      DST[dst0_index] <- DST[dst0_index] ^ scalar
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins IndexLike:$dst0_index, I32:$param);

    let assemblyFormat = [{
      `(` $dst0_index `,` $param `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_RecipTileInitOp : TTKernel_InitOp<"recip_tile_init"> {
    let summary = "Init function for recip_tile operation. Refer to documentation for any init function.";
    let description = [{
      Must be called before recip_tile function.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_RecipTileOp : TTKernel_SFPUOp<"recip_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Recip tile in the DST at specified index.";
    let description = [{
      Performs element-wise computation of the reciprocal on each element of a tile
      in DST register at index tile_index. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call. This call is blocking and is only
      available on the compute engine.
      Only works for Float32, Float16_b, Bfp8_b data formats for full accuracy.
    }];

    let arguments = (ins IndexLike:$tile_index);

    let assemblyFormat = [{
      `(` $tile_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_PowBinaryTilesInitOp : TTKernel_InitOp<"power_binary_tile_init"> {
    let summary = "Short init function which configures compute unit for execution of power_binary_tile.";
    let description = [{
      Must be run before power_binary_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_PowBinaryTilesOp : TTKernel_SFPUOp<"power_binary_tile", [TTKernel_BinaryOpTrait]> {
    let summary = "Power operation between two tiles";
    let description = [{
      Performs element-wise computation of power operation
      DST[odst_index] <- DST[dst0_index] ^ DST[dst1_index]
      on DST register operands. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call.
    }];

    let arguments = (ins IndexLike:$dst0_index, IndexLike:$dst1_index, IndexLike:$odst_index);

    let assemblyFormat = [{
      `(` $dst0_index `,` $dst1_index `,` $odst_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ExpTileInitOp : TTKernel_InitOp<"exp_tile_init"> {
    let summary = "Short init function which configures compute unit for execution of exp_tile.";
    let description = [{
      Must be run before exp_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ExpTileOp : TTKernel_SFPUOp<"exp_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Exp operation";
    let description = [{
      Performs element-wise computation of exponential on each element of a tile
      in DST register at index tile_index. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call. This call is blocking and is only
      available on the compute engine.
    }];

    let arguments = (ins IndexLike:$tile_index);

    let assemblyFormat = [{
      `(` $tile_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_LogTileInitOp : TTKernel_InitOp<"log_tile_init"> {
    let summary = "Short init function which configures compute unit for execution of log_tile.";
    let description = [{
      Must be run before log_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_LogTileOp : TTKernel_SFPUOp<"log_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Log operation";
    let description = [{
      Performs element-wise computation of log on each element of a tile
      in DST register at index tile_index. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call. This call is blocking and is only
      available on the compute engine.
    }];

    let arguments = (ins IndexLike:$tile_index);

    let assemblyFormat = [{
      `(` $tile_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NegativeTileInitOp : TTKernel_InitOp<"negative_tile_init"> {
    let summary = "Short init function which configures compute unit for execution of negative_tile.";
    let description = [{
      Must be run before negative_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NegativeTileOp : TTKernel_SFPUOp<"negative_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Negative operation";
    let description = [{
      Performs element-wise computation of the negative on each element of a tile
      in DST register at index tile_index. The DST register buffer must be in
      acquired state via *acquire_dst* call. This call is blocking and is only
      available on the compute engine.
    }];

    let arguments = (ins IndexLike:$tile_index);

    let assemblyFormat = [{
      `(` $tile_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_CosTileInitOp : TTKernel_InitOp<"cos_tile_init"> {
    let summary = "Short init function which configures compute unit for execution of cos_tile.";
    let description = [{
      Must be run before cos_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_CosTileOp : TTKernel_SFPUOp<"cos_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Cos operation";
    let description = [{
      Performs element-wise computation of the trigonometric cosine operation on each element of a tile
      in DST register at index tile_index. The DST register buffer must be in
      acquired state via *acquire_dst* call. This call is blocking and is only
      available on the compute engine.
    }];

    let arguments = (ins IndexLike:$tile_index);

    let assemblyFormat = [{
      `(` $tile_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_TanTileInitOp : TTKernel_InitOp<"tan_tile_init"> {
    let summary = "Short init function which configures compute unit for execution of tan_tile.";
    let description = [{
      Must be run before tan_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_TanTileOp : TTKernel_SFPUOp<"tan_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Tan operation";
    let description = [{
      Performs element-wise computation of the trigonometric tangent operation on each element of a tile
      in DST register at index tile_index. The DST register buffer must be in
      acquired state via *acquire_dst* call. This call is blocking and is only
      available on the compute engine.
    }];

    let arguments = (ins IndexLike:$tile_index);

    let assemblyFormat = [{
      `(` $tile_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_TanhTileInitOp : TTKernel_InitOp<"tanh_tile_init"> {
    let summary = "Short init function which configures compute unit for execution of tanh_tile.";
    let description = [{
      Must be run before tanh_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_TanhTileOp : TTKernel_SFPUOp<"tanh_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Tanh operation";
    let description = [{
      Performs element-wise computation of the hyperbolic tangent operation on each element of a tile
      in DST register at index tile_index. The DST register buffer must be in
      acquired state via *acquire_dst* call. This call is blocking and is only
      available on the compute engine.
    }];

    let arguments = (ins IndexLike:$dst0_index);

    let assemblyFormat = [{
      `(` $dst0_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_SqrtTileInitOp : TTKernel_InitOp<"sqrt_tile_init"> {
    let summary = "Short init function which configures compute unit for execution of sqrt_tile.";
    let description = [{
      Must be run before sqrt_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_SqrtTileOp : TTKernel_SFPUOp<"sqrt_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Sqrt operation";
    let description = [{
      Performs element-wise computation of sqrt on each element of a tile
      in DST register at index tile_index. The DST register buffer must be in
      acquired state via *acquire_dst* call. This call is blocking and is only
      available on the compute engine.
    }];

    let arguments = (ins IndexLike:$tile_index);

    let assemblyFormat = [{
      `(` $tile_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_RsqrtTileInitOp : TTKernel_InitOp<"rsqrt_tile_init"> {
    let summary = "Short init function which configures compute unit for execution of rsqrt_tile.";
    let description = [{
      Must be run before rsqrt_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_RsqrtTileOp : TTKernel_SFPUOp<"rsqrt_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Rsqrt operation";
    let description = [{
      Performs element-wise computation of reciprocal sqrt on each element of a tile
      in DST register at index tile_index. The DST register buffer must be in
      acquired state via *acquire_dst* call. This call is blocking and is only
      available on the compute engine.
    }];

    let arguments = (ins IndexLike:$tile_index);

    let assemblyFormat = [{
      `(` $tile_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_SinTileInitOp : TTKernel_InitOp<"sin_tile_init"> {
    let summary = "Init function for sin_tile operation. Refer to documentation for any init function.";
    let description = [{
      Must be run before sin_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_SinTileOp : TTKernel_SFPUOp<"sin_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Sine tile in the DST at specified index.";
    let description = [{
      Performs element-wise computation of sine operation
      DST[dst0_index] <- sin(DST[dst0_index])
      on DST register operands. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call.
    }];

    let arguments = (ins IndexLike:$dst0_index);

    let assemblyFormat = [{
      `(` $dst0_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_RoundingTileInitOp : TTKernel_InitOp<"rounding_op_tile_init"> {
    let summary = "Init function for ceil/floor/round_tile operation. Refer to documentation for any init function.";
    let description = [{
      Must be run before ceil/floor/round_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_CeilTileOp : TTKernel_SFPUOp<"ceil_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Ceil tile in the DST at specified index.";
    let description = [{
      Performs element-wise computation of ceil operation
      DST[dst0_index] <- ceil(DST[dst0_index])
      on DST register operands. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call.
    }];

    let arguments = (ins IndexLike:$dst0_index);

    let assemblyFormat = [{
      `(` $dst0_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_FloorTileOp : TTKernel_SFPUOp<"floor_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Floor tile in the DST at specified index.";
    let description = [{
      Performs element-wise computation of floor operation
      DST[dst0_index] <- floor(DST[dst0_index])
      on DST register operands. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call.
    }];

    let arguments = (ins IndexLike:$dst0_index);

    let assemblyFormat = [{
      `(` $dst0_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_AbsTileInitOp : TTKernel_InitOp<"abs_tile_init"> {
    let summary = "Init function for abs_tile operation. Refer to documentation for any init function.";
    let description = [{
      Must be run before abs_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_AbsTileOp : TTKernel_SFPUOp<"abs_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Absolute value tile in the DST at specified index.";
    let description = [{
      Performs element-wise computation of absolute value operation
      DST[dst0_index] <- abs(DST[dst0_index])
      on DST register operands. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call.
    }];

    let arguments = (ins IndexLike:$dst0_index);

    let assemblyFormat = [{
      `(` $dst0_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_AbsTileI32Op : TTKernel_SFPUOp<"abs_tile_int32", [TTKernel_UnaryOpTrait]> {
    let summary = "Absolute value i32 tile in the DST at specified index.";
    let description = [{
      Performs element-wise computation of absolute value operation
      DST[dst0_index] <- abs(DST[dst0_index])
      on DST register operands. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call.
    }];

    let arguments = (ins IndexLike:$dst0_index);

    let assemblyFormat = [{
      `(` $dst0_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_BitwiseNotTileInitOp : TTKernel_InitOp<"bitwise_not_tile_init"> {
    let summary = "Init function for bitwise_not_tile operation. Refer to documentation for any init function.";
    let description = [{
      Must be run before bitwise_not_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_BitwiseNotTileOp : TTKernel_SFPUOp<"bitwise_not_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Bitwise Not operation on tile in the DST at specified index.";
    let description = [{
      Performs element-wise computation of sign operation
      DST[tile_index] <- bitwise_not(DST[tile_index])
      on DST register operands.
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins IndexLike:$tile_index);

    let assemblyFormat = [{
      `(` $tile_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_SignTileInitOp : TTKernel_InitOp<"sign_tile_init"> {
    let summary = "Init function for sign_tile operation. Refer to documentation for any init function.";
    let description = [{
      Must be run before sign_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_SignTileOp : TTKernel_SFPUOp<"sign_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Sign operation";
    let description = [{
      Performs element-wise computation of sign on each element of a tile
      in DST register at index tile_index. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call. This call is blocking and is only
      available on the compute engine.
    }];

    let arguments = (ins IndexLike:$tile_index);
    let assemblyFormat = [{ `(` $tile_index `)` attr-dict `:` functional-type(operands, results) }];
}

def TTKernel_LogicalNotUnaryTileInitOp : TTKernel_InitOp<"logical_not_unary_tile_init"> {
    let summary = "Init function for logical_not_unary_tile(_int32) operation. Refer to documentation for any init function.";
    let description = [{
      Must be run before logical_not_unary_tile(_int32).
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_LogicalNotUnaryTileOp : TTKernel_SFPUOp<"logical_not_unary_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Logical negation tile in the DST at specified index.";
    let description = [{
      Performs element-wise computation of logical negation operation
      DST[dst0_index] <- !DST[dst0_index]
      on DST register operands. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call.
    }];

    let arguments = (ins IndexLike:$dst0_index);

    let assemblyFormat = [{
      `(` $dst0_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_LogicalNotUnaryTileI32Op : TTKernel_SFPUOp<"logical_not_unary_tile_int32", [TTKernel_UnaryOpTrait]> {
    let summary = "Logical negation tile in the DST at specified index.";
    let description = [{
      Performs element-wise computation of logical negation operation for the int32 data type
      DST[dst0_index] <- !DST[dst0_index]
      on DST register operands. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call.
    }];

    let arguments = (ins IndexLike:$dst0_index);

    let assemblyFormat = [{
      `(` $dst0_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_EqzTileInitOp : TTKernel_InitOp<"eqz_tile_init"> {
    let summary = "Init function for eqz() operation. Refer to documentation for any init function.";
    let description = [{
      Must be run before eqz_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_EqzTileOp : TTKernel_SFPUOp<"eqz_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Equal to zero tile in the DST at specified index.";
    let description = [{
      Performs element-wise equality on DST register tiles.
      DST[dst0_index] <- (DST[dst0_index] == 0)
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins IndexLike:$dst0_index);

    let assemblyFormat = [{
      `(` $dst0_index `)` attr-dict `:` functional-type(operands, results)
    }];

}

def TTKernel_EqzTileI32Op : TTKernel_SFPUOp<"eqz_tile_int32", [TTKernel_UnaryOpTrait]> {
    let summary = "Equal to zero tile in the DST at specified index.";
    let description = [{
      Performs element-wise equality on DST register tiles for int32 data type.
      DST[dst0_index] <- (DST[dst0_index] == 0)
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins IndexLike:$dst0_index);

    let assemblyFormat = [{
      `(` $dst0_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NezTileInitOp : TTKernel_InitOp<"nez_tile_init"> {
    let summary = "Init function for nez() operation. Refer to documentation for any init function.";
    let description = [{
      Must be run before nez_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NezTileOp : TTKernel_SFPUOp<"nez_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Not equal to zero tile in the DST at specified index.";
    let description = [{
      Performs element-wise inequality on DST register tiles.
      DST[dst0_index] <- (DST[dst0_index] != 0)
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins IndexLike:$dst0_index);

    let assemblyFormat = [{
      `(` $dst0_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NezTileI32Op : TTKernel_SFPUOp<"nez_tile_int32", [TTKernel_UnaryOpTrait]> {
    let summary = "Not equal to zero tile in the DST at specified index.";
    let description = [{
      Performs element-wise inequality on DST register tiles for int32 data type.
      DST[dst0_index] <- (DST[dst0_index] != 0)
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins IndexLike:$dst0_index);

    let assemblyFormat = [{
      `(` $dst0_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_GtzTileInitOp : TTKernel_InitOp<"gtz_tile_init"> {
    let summary = "Init function for gtz() operation. Refer to documentation for any init function.";
    let description = [{
      Must be run before gtz_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_GtzTileOp : TTKernel_SFPUOp<"gtz_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Greater than zero tile in the DST at specified index.";
    let description = [{
      Performs element-wise greater than zero comparison on DST register tiles.
      DST[dst0_index] <- (DST[dst0_index] > 0)
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins IndexLike:$dst0_index);

    let assemblyFormat = [{
      `(` $dst0_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_GtzTileI32Op : TTKernel_SFPUOp<"gtz_tile_int32", [TTKernel_UnaryOpTrait]> {
    let summary = "Greater than zero tile in the DST at specified index.";
    let description = [{
      Performs element-wise greater than zero comparison on DST register tiles for int32 data type.
      DST[dst0_index] <- (DST[dst0_index] > 0)
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins IndexLike:$dst0_index);

    let assemblyFormat = [{
      `(` $dst0_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_GezTileInitOp : TTKernel_InitOp<"gez_tile_init"> {
    let summary = "Init function for gez() operation. Refer to documentation for any init function.";
    let description = [{
      Must be run before gez_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_GezTileOp : TTKernel_SFPUOp<"gez_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Greater than or equal to zero tile in the DST at specified index.";
    let description = [{
      Performs element-wise greater than or equal to zero comparison on DST register tiles.
      DST[dst0_index] <- (DST[dst0_index] >= 0)
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins IndexLike:$dst0_index);

    let assemblyFormat = [{
      `(` $dst0_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_GezTileI32Op : TTKernel_SFPUOp<"gez_tile_int32", [TTKernel_UnaryOpTrait]> {
    let summary = "Greater than or equal to zero tile in the DST at specified index.";
    let description = [{
      Performs element-wise greater than or equal to zero comparison on DST register tiles for int32 data type.
      DST[dst0_index] <- (DST[dst0_index] >= 0)
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins IndexLike:$dst0_index);

    let assemblyFormat = [{
      `(` $dst0_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_LtzTileInitOp : TTKernel_InitOp<"ltz_tile_init"> {
    let summary = "Init function for ltz() operation. Refer to documentation for any init function.";
    let description = [{
      Must be run before ltz_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_LtzTileOp : TTKernel_SFPUOp<"ltz_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Less than zero tile in the DST at specified index.";
    let description = [{
      Performs element-wise less than zero comparison on DST register tiles.
      DST[dst0_index] <- (DST[dst0_index] < 0)
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins IndexLike:$dst0_index);

    let assemblyFormat = [{
      `(` $dst0_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_LtzTileI32Op : TTKernel_SFPUOp<"ltz_tile_int32", [TTKernel_UnaryOpTrait]> {
    let summary = "Less than zero tile in the DST at specified index.";
    let description = [{
      Performs element-wise less than zero comparison on DST register tiles for int32 data type.
      DST[dst0_index] <- (DST[dst0_index] < 0)
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins IndexLike:$dst0_index);

    let assemblyFormat = [{
      `(` $dst0_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_LezTileInitOp : TTKernel_InitOp<"lez_tile_init"> {
    let summary = "Init function for lez() operation. Refer to documentation for any init function.";
    let description = [{
      Must be run before lez_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_LezTileOp : TTKernel_SFPUOp<"lez_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Less than or equal to zero tile in the DST at specified index.";
    let description = [{
      Performs element-wise less than or equal to zero comparison on DST register tiles.
      DST[dst0_index] <- (DST[dst0_index] <= 0)
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins IndexLike:$dst0_index);

    let assemblyFormat = [{
      `(` $dst0_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_LezTileI32Op : TTKernel_SFPUOp<"lez_tile_int32", [TTKernel_UnaryOpTrait]> {
    let summary = "Less than or equal to zero tile in the DST at specified index.";
    let description = [{
      Performs element-wise less than or equal to zero comparison on DST register tiles for int32 data type.
      DST[dst0_index] <- (DST[dst0_index] <= 0)
      The DST register buffer must be in acquired state via *tile_regs_acquire* call.
      This call is blocking and is only available on the compute engine.
    }];

    let arguments = (ins IndexLike:$dst0_index);

    let assemblyFormat = [{
      `(` $dst0_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_FillTileInitOp : TTKernel_InitOp<"fill_tile_init"> {
    let summary = "Init function for fill_tile operation. Refer to documentation for any init function.";
    let description = [{
      Must be run before fill_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_FillTileOp : TTKernel_SFPUOp<"fill_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Fill tile with specified value.";
    let description = [{
      Fills supplied DST register tile with a supplied f32 value. The DST
      register must be in acquired state via *tile_regs_acquire* call.

      Example:
      ```
      ttkernel.fill_tile(%dst_index, %value);
      ```
    }];

    let arguments = (ins IndexLike: $dst_index, F32:$value);

    let assemblyFormat = [{
      `(` $dst_index `,` $value `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_SigmoidTileInitOp : TTKernel_InitOp<"sigmoid_tile_init"> {
    let summary = "Short init function which configures compute unit for execution of sigmoid_tile.";
    let description = [{
      Must be run before sigmoid_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_SigmoidTileOp : TTKernel_SFPUOp<"sigmoid_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Sigmoid operation";
    let description = [{
      Performs element-wise computation of sigmoid on each element of a tile
      in DST register at index tile_index. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call. This call is blocking and is only
      available on the compute engine.
    }];

    let arguments = (ins IndexLike:$tile_index);

    let assemblyFormat = [{
      `(` $tile_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_HardsigmoidTileInitOp : TTKernel_InitOp<"hardsigmoid_tile_init"> {
    let summary = "Short init function which configures compute unit for execution of hardsigmoid_tile.";
    let description = [{
      Must be run before hardsigmoid_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_HardsigmoidTileOp : TTKernel_SFPUOp<"hardsigmoid_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Hardsigmoid operation";
    let description = [{
      Performs element-wise computation of hardsigmoid on each element of a tile
      in DST register at index tile_index. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call. This call is blocking and is only
      available on the compute engine.
    }];

    let arguments = (ins IndexLike:$tile_index);

    let assemblyFormat = [{
      `(` $tile_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_SiluTileInitOp : TTKernel_InitOp<"silu_tile_init"> {
    let summary = "Short init function which configures compute unit for execution of silu_tile.";
    let description = [{
      Must be run before silu_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_SiluTileOp : TTKernel_SFPUOp<"silu_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Silu operation";
    let description = [{
      Performs element-wise computation of silu on each element of a tile
      in DST register at index tile_index. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call. This call is blocking and is only
      available on the compute engine.
    }];

    let arguments = (ins IndexLike:$tile_index);

    let assemblyFormat = [{
      `(` $tile_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ReluTileInitOp : TTKernel_InitOp<"relu_tile_init"> {
    let summary = "Short init function which configures compute unit for execution of relu_tile(_int32).";
    let description = [{
      Must be run before relu_tile(_int32).
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ReluTileOp : TTKernel_SFPUOp<"relu_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Relu operation";
    let description = [{
      Performs element-wise computation of relu on each element of a tile
      in DST register at index tile_index. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call. This call is blocking and is only
      available on the compute engine.
    }];

    let arguments = (ins IndexLike:$tile_index);

    let assemblyFormat = [{
      `(` $tile_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ReluTileI32Op : TTKernel_SFPUOp<"relu_tile_int32", [TTKernel_UnaryOpTrait]> {
    let summary = "Relu operation (for int32 type)";
    let description = [{
      Performs element-wise computation of relu on each element of a tile
      in DST register at index tile_index. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call. This call is blocking and is only
      available on the compute engine.
    }];

    let arguments = (ins IndexLike:$tile_index);

    let assemblyFormat = [{
      `(` $tile_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_GeluTileInitOp : TTKernel_InitOp<"gelu_tile_init"> {
    let summary = "Short init function which configures compute unit for execution of gelu_tile.";
    let description = [{
      Must be run before gelu_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_GeluTileOp : TTKernel_SFPUOp<"gelu_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "GELU operation";
    let description = [{
      Performs element-wise computation of GELU on each element of a tile
      in DST register at index tile_index. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call. This call is blocking and is only
      available on the compute engine.
    }];

    let arguments = (ins IndexLike:$tile_index);

    let assemblyFormat = [{
      `(` $tile_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ErfTileInitOp : TTKernel_InitOp<"erf_tile_init"> {
    let summary = "Short init function which configures compute unit for execution of erf_tile.";
    let description = [{
      Must be run before erf_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ErfTileOp : TTKernel_SFPUOp<"erf_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Erf operation";
    let description = [{
      Performs element-wise computation of error function (erf) on each element of a tile
      in DST register at index tile_index. The DST register buffer must be in
      acquired state via *acquire_dst* call. This call is blocking and is only
      available on the compute engine.
    }];

    let arguments = (ins IndexLike:$tile_index);

    let assemblyFormat = [{
      `(` $tile_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ErfcTileInitOp : TTKernel_InitOp<"erfc_tile_init"> {
    let summary = "Short init function which configures compute unit for execution of erfc_tile.";
    let description = [{
      Must be run before erfc_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ErfcTileOp : TTKernel_SFPUOp<"erfc_tile", [TTKernel_UnaryOpTrait]> {
    let summary = "Erfc operation";
    let description = [{
      Performs element-wise computation of complementary error function (erfc) on each element of a tile
      in DST register at index tile_index. The DST register buffer must be in
      acquired state via *acquire_dst* call. This call is blocking and is only
      available on the compute engine.
    }];

    let arguments = (ins IndexLike:$tile_index);

    let assemblyFormat = [{
      `(` $tile_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_InvokeSFPIOp : TTKernel_SFPUOp<"invoke_sfpi", [NoTerminator]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins);

    let regions = (region AnyRegion:$region);

    let assemblyFormat = [{
      attr-dict-with-keyword $region
    }];
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

    let assemblyFormat = [{
      `(` $cb `,` $numPages `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_CBPopFrontOp : TTKernel_Op<"cb_pop_front"> {
    let summary = "CBPopFront call.";
    let description = [{
      CBPopFront operation
    }];

    let arguments = (ins TTKernel_CB:$cb, I32:$numPages);

    let hasVerifier = 1;

    let assemblyFormat = [{
      `(` $cb `,` $numPages `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_CBReserveBackOp : TTKernel_Op<"cb_reserve_back"> {
    let summary = "CBReserveBack call.";
    let description = [{
      CBReserveBack operation
    }];

    let arguments = (ins TTKernel_CB:$cb, I32:$numPages);

    let hasVerifier = 1;

    let assemblyFormat = [{
      `(` $cb `,` $numPages `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_CBWaitFrontOp : TTKernel_Op<"cb_wait_front", [TTKernel_DeviceZoneOpTrait]> {
    let summary = "CBWaitFront call.";
    let description = [{
      CBWaitFront operation
    }];

    let arguments = (ins TTKernel_CB:$cb, I32:$numPages);

    let hasVerifier = 1;

    let assemblyFormat = [{
      `(` $cb `,` $numPages `)` attr-dict `:` functional-type(operands, results)
    }];
}

//===----------------------------------------------------------------------===//
// TTKernel Tile operations
//===----------------------------------------------------------------------===//

def TTKernel_TilizeBlockOp : TTKernel_Op<"tilize_block"> {
    let summary = "TilizeBlockOp call.";
    let description = [{
      TilizeBlockOp operation
    }];

    let arguments = (ins TTKernel_CB:$cbIn, I32:$numTiles, TTKernel_CB:$cbOut);

    let hasVerifier = 1;

    let assemblyFormat = [{
      `(` $cbIn `,` $numTiles `,` $cbOut `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ExperimentalTilizeBlockOp : TTKernel_Op<"experimental::tilize_block"> {
    let summary = "Experimental TilizeBlockOp call.";
    let description = [{
      This is a custom tilize block LLK that takes the dimensions of the block, and properly tilizes each row.
    }];

    let arguments = (ins TTKernel_CB:$cbIn, TTKernel_CB:$cbOut, I32:$blockR, I32:$blockC);

    let hasVerifier = 1;

    let assemblyFormat = [{
      `(` $cbIn `,` $cbOut `,` $blockR `,` $blockC `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_TilizeInitOp : TTKernel_Op<"tilize_init"> {
    let summary = "TilizeInitOp call.";
    let description = [{
      Initialize the tilize operation. To be called once at beginning of a
      kernel.
    }];

    let arguments = (ins TTKernel_CB:$cbIn, I32:$numTiles, TTKernel_CB:$cbOut);

    let hasVerifier = 1;

    let assemblyFormat = [{
      `(` $cbIn `,` $numTiles `,` $cbOut `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_TilizeUninitOp : TTKernel_Op<"tilize_uninit"> {
    let summary = "TilizeUninitOp call.";
    let description = [{
      Uninitialize tilize operation before re-initializing for another operation.
    }];

    let arguments = (ins TTKernel_CB:$cbI, TTKernel_CB:$cbOut);

    let assemblyFormat = [{
      `(` $cbI `,` $cbOut `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_UntilizeBlockOp : TTKernel_Op<"untilize_block"> {
    let summary = "UntilizeBlockOp call.";
    let description = [{
      UntilizeBlockOp operation
    }];

    let arguments = (ins TTKernel_CB:$cbIn, I32:$numTiles, TTKernel_CB:$cbOut);

    let hasVerifier = 1;

    let assemblyFormat = [{
      `(` $cbIn `,` $numTiles `,` $cbOut `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ExperimentalUntilizeBlockOp : TTKernel_Op<"experimental::untilize_block"> {
    let summary = "Experimental UntilizeBlockOp call.";
    let description = [{
      This is a custom untilize block LLK that takes the dimensions of the block.
    }];

    let arguments = (ins TTKernel_CB:$cbIn, TTKernel_CB:$cbOut, I32:$blockR, I32:$blockC);

    let hasVerifier = 1;

    let assemblyFormat = [{
      `(` $cbIn `,` $cbOut `,` $blockR `,` $blockC `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ExperimentalTileFillOp : TTKernel_SFPUOp<"experimental::tile_fill", [TTKernel_UnaryOpTrait]> {
    let summary = "Experimental Tile Fill Op";
    let description = [{
      Creates a tile in DST filled with a constant scalar value. All elements in the
      resulting tile will have the same value as the input scalar. The DST register
      buffer must be in acquired state via *tile_regs_acquire* call. This call is
      blocking and is only available on the compute engine.
    }];

    let arguments = (ins IndexLike:$dst_index, AnyFloat:$value);

    let assemblyFormat = [{
      `(` $dst_index `,` $value `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ExperimentalWriteRowMaskTileOp : TTKernel_Op<"experimental::write_row_mask_tile"> {
    let summary = "Experimental Write Row Mask Tile Op";
    let description = [{
      Writes a row mask tile pattern to a CB, where element[i,j] = 1.0 if i < validRows, else 0.0.
      This is used in dataflow kernels to generate OOB mask tiles directly in L1 memory,
      avoiding DST register pressure in the compute kernel.
    }];

    let arguments = (ins I32:$validRows, TTKernel_CB:$cb);

    let assemblyFormat = [{
      `(` $validRows `,` $cb `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ExperimentalWriteColMaskTileOp : TTKernel_Op<"experimental::write_col_mask_tile"> {
    let summary = "Experimental Write Col Mask Tile Op";
    let description = [{
      Writes a column mask tile pattern to a CB, where element[i,j] = 1.0 if j < validCols, else 0.0.
      This is used in dataflow kernels to generate OOB mask tiles directly in L1 memory,
      avoiding DST register pressure in the compute kernel.
    }];

    let arguments = (ins I32:$validCols, TTKernel_CB:$cb);

    let assemblyFormat = [{
      `(` $validCols `,` $cb `)` attr-dict `:` functional-type(operands, results)
    }];
}


def TTKernel_ExperimentalFillArangeTileOp : TTKernel_Op<"experimental::fill_arange_tile"> {
    let summary = "Experimental Write Full Linear Index Tile Op";
    let description = [{
      Writes a full linear index tile pattern to a CB, where element[i,j] = i * 32 + j
      (linear index as float, 0-1023). This is used for arange operations to generate
      index tiles directly in L1 memory.

      The resulting tile looks like:
      [[  0,   1,   2, ...,  31],
       [ 32,  33,  34, ...,  63],
       ...
       [992, 993, 994, ..., 1023]]
    }];

    let arguments = (ins TTKernel_CB:$cb);

    let assemblyFormat = [{
      `(` $cb `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_UntilizeInitOp : TTKernel_Op<"untilize_init"> {
    let summary = "UntilizeInitOp call.";
    let description = [{
      Init function for untilize operations, to be used at the beginning of the
      kernel.
    }];

    let arguments = (ins TTKernel_CB:$cbIn);

    let hasVerifier = 1;

    let assemblyFormat = [{
      `(` $cbIn `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_UntilizeUninitOp : TTKernel_Op<"untilize_uninit"> {
    let summary = "UntilizeUninitOp call.";
    let description = [{
      Uninitialize untilize operation, to allow initializing another operation.
    }];

    let arguments = (ins TTKernel_CB:$cbIn);

    let assemblyFormat = [{
      `(` $cbIn `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_TransposeInitOp : TTKernel_Op<"transpose_wh_init"> {
    let summary = "TransposeInitOp call.";
    let description = [{
      Initialize the transpose operation. To be called once at beginning of a
      kernel before transpose operations.
    }];

    let arguments = (ins TTKernel_CB:$cbIn, TTKernel_CB:$cbOut);

    let hasVerifier = 1;

    let assemblyFormat = [{
      `(` $cbIn `,` $cbOut `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_TransposeTileOp : TTKernel_Op<"transpose_wh_tile"> {
    let summary = "Transpose WH tile operation";
    let description = [{
      Performs a 32x32 transpose operation on a tile in the CB
      at a given index and writes the result to the DST register at index
      dst_tile_index.
    }];

    let arguments = (ins TTKernel_CB:$icb, IndexLike:$itile, IndexLike:$idst);

    let hasVerifier = 1;

    let assemblyFormat = [{
      `(` $icb `,` $itile `,` $idst `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_TransposeUninitOp : TTKernel_Op<"transpose_wh_uninit"> {
    let summary = "TransposeUninitOp call.";
    let description = [{
      Uninitialize transpose operation before re-initializing for another operation.
    }];

    let arguments = (ins TTKernel_CB:$cbIn, TTKernel_CB:$cbOut);

    let assemblyFormat = [{
      `(` $cbIn `,` $cbOut `)` attr-dict `:` functional-type(operands, results)
    }];
}

//===----------------------------------------------------------------------===//
// TTKernel NOC operations
//===----------------------------------------------------------------------===//

def TTKernel_GetNocAddrOp : TTKernel_Op<"get_noc_addr"> {
    let summary = "GetNocAddr";
    let description = [{
      GetNocAddr api including core coordinates
    }];

    let arguments = (ins IndexLike:$x, IndexLike:$y, AnyTypeOf<[I32, TTKernel_L1Addr, TTKernel_Semaphore]>:$l1Address);
    let results = (outs TTKernel_NocAddr:$nocAddr);

    let assemblyFormat = [{
      `(` $x `,` $y `,` $l1Address `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_GetNocAddrFromBankIDOp: TTKernel_Op<"get_noc_addr_from_bank_id"> {
    let summary = "GetNocAddrFromBankID";
    let description = [{
      GetNocAddrFromBankID api
    }];

    let arguments = (ins I32:$bank_id, I32:$bankAddressOffset);
    let results = (outs TTKernel_NocAddr:$nocAddr);

    let assemblyFormat = [{
      `(` $bank_id `,` $bankAddressOffset `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NocAsyncReadOp : TTKernel_Op<"noc_async_read"> {
    let summary = "NocAsyncRead";
    let description = [{
      NocAsyncRead
    }];

    let arguments = (ins TTKernel_NocAddr:$srcNocAddr, I32:$dstLocalL1Addr, I32:$size);

    let assemblyFormat = [{
      `(` $srcNocAddr `,` $dstLocalL1Addr `,` $size `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NocAsyncReadTileOp : TTKernel_Op<"noc_async_read_tile"> {
    let summary = "NocAsyncReadTile";
    let description = [{
      NocAsyncReadTile
    }];

    let arguments = (ins I32:$id,
                    AnyTypeOf<[TTKernel_InterleavedAddrGenFast, TTKernel_TensorAccessor]>:$addrGenStruct,
                    I32:$dstLocalL1Addr);

    let assemblyFormat = [{
      `(` $id `,` $addrGenStruct `,` $dstLocalL1Addr `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NocAsyncReadOnePacketSetStateOp : TTKernel_Op<"noc_async_read_one_packet_set_state"> {
    let summary = "NocAsyncReadOnePacketSetState";
    let description = [{
      NocAsyncReadOnePacketSetState
    }];

    let arguments = (ins TTKernel_NocAddr:$srcNocAddr, I32:$size);

    let assemblyFormat = [{
      `(` $srcNocAddr `,` $size `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NocAsyncReadOnePacketWithStateOp : TTKernel_Op<"noc_async_read_one_packet_with_state"> {
    let summary = "NocAsyncReadOnePacketWithState";
    let description = [{
      NocAsyncReadOnePacketWithState
    }];

    let arguments = (ins TTKernel_NocAddr:$srcNocAddr, AnyTypeOf<[I32, TTKernel_L1Addr]>:$dstLocalL1Addr);

    let assemblyFormat = [{
      `(` $srcNocAddr `,` $dstLocalL1Addr `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NocAsyncReadSetTridOp : TTKernel_Op<"noc_async_read_set_trid", [TTKernel_TridNocOpTrait]> {
    let summary = "NocAsyncReadSetTrid";
    let description = [{
      Sets the transaction ID for subsequent NOC reads. TRID is 0-15, NOC is 0 or 1.

      Example:
      ```
      ttkernel.noc_async_read_set_trid(%trid, %noc_idx) : (i32, i8) -> ()
      ```
    }];

    let arguments = (ins I32:$trid, Optional<I8>:$noc);

    let assemblyFormat = [{
      `(` $trid (`,` $noc^)? `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NocAsyncReadOnePacketWithStateWithTridOp : TTKernel_Op<"noc_async_read_one_packet_with_state_with_trid", [TTKernel_TridNocOpTrait]> {
    let summary = "NocAsyncReadOnePacketWithStateWithTrid";
    let description = [{
      Issues a one-packet NOC read with a specific transaction ID. TRID is 0-15, NOC is 0 or 1.

      Example:
      ```
      // Must set TRID before calling.
      // ttkernel.noc_async_read_set_trid(%trid, %noc_idx) : (i32, i8) -> ()
      ttkernel.noc_async_read_one_packet_with_state_with_trid(%src_base, %src_addr, %dst_l1, %trid, %noc_idx) : (i32, i32, i32, i32, i8) -> ()
      // TRID-specific barrier should follow.
      // ttkernel.noc_async_read_barrier_with_trid(%trid, %noc_idx) : (i32, i8) -> ()
      ```
    }];

    let arguments = (ins I32:$srcBaseAddr, I32:$srcAddr, AnyTypeOf<[I32, TTKernel_L1Addr]>:$dstLocalL1Addr, I32:$trid, Optional<I8>:$noc);

    let assemblyFormat = [{
      `(` $srcBaseAddr `,` $srcAddr `,` $dstLocalL1Addr `,` $trid (`,` $noc^)? `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NocAsyncReadBarrierOp : TTKernel_Op<"noc_async_read_barrier", [TTKernel_DeviceZoneOpTrait]> {
    let summary = "NocAsyncReadBarrier";
    let description = [{
      NocAsyncReadBarrier
    }];

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NocAsyncReadBarrierWithTridOp : TTKernel_Op<"noc_async_read_barrier_with_trid", [TTKernel_DeviceZoneOpTrait, TTKernel_TridNocOpTrait]> {
    let summary = "NocAsyncReadBarrierWithTrid";
    let description = [{
      Waits for outstanding read transactions matching a transaction ID. TRID is 0-15, NOC is 0 or 1.

      Example:
      ```
      ttkernel.noc_async_read_barrier_with_trid(%trid, %noc_idx) : (i32, i8) -> ()
      ```
    }];

    let arguments = (ins I32:$trid, Optional<I8>:$noc);

    let assemblyFormat = [{
      `(` $trid (`,` $noc^)? `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NocAsyncWriteOp : TTKernel_Op<"noc_async_write"> {
    let summary = "NocAsyncWrite";
    let description = [{
      NocAsyncWrite
    }];

    let arguments = (ins I32:$srcLocalL1Addr, TTKernel_NocAddr:$dstNocAddr, I32:$size);

    let assemblyFormat = [{
      `(` $srcLocalL1Addr `,` $dstNocAddr `,` $size `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NocAsyncWriteTileOp : TTKernel_Op<"noc_async_write_tile"> {
    let summary = "NocAsyncWriteTile";
    let description = [{
      NocAsyncWriteTilie
    }];

    let arguments = (ins IndexLike:$id,
                    AnyTypeOf<[TTKernel_InterleavedAddrGenFast, TTKernel_TensorAccessor]>:$addrGenStruct,
                    I32:$srcLocalL1Addr);

    let assemblyFormat = [{
      `(` $id `,` $addrGenStruct `,` $srcLocalL1Addr `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NocAsyncWriteSetTridOp : TTKernel_Op<"noc_async_write_set_trid", [TTKernel_TridNocOpTrait]> {
    let summary = "NocAsyncWriteSetTrid";
    let description = [{
      Sets the transaction ID for subsequent NOC writes. TRID is 0-15, NOC is 0 or 1.

      Example:
      ```
      ttkernel.noc_async_write_set_trid(%trid, %noc_idx) : (i32, i8) -> ()
      ```
    }];

    let arguments = (ins I32:$trid, Optional<I8>:$noc);

    let assemblyFormat = [{
      `(` $trid (`,` $noc^)? `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NocAsyncWriteOnePacketWithTridOp : TTKernel_Op<"noc_async_write_one_packet_with_trid", [TTKernel_TridNocOpTrait]> {
    let summary = "NocAsyncWriteOnePacketWithTrid";
    let description = [{
      Issues a one-packet NOC write with a specific transaction ID. TRID is 0-15, NOC is 0 or 1.

      Example:
      ```
      // Must set TRID before calling.
      // ttkernel.noc_async_write_set_trid(%trid, %noc_idx) : (i32, i8) -> ()
      ttkernel.noc_async_write_one_packet_with_trid(%l1_src, %dst_noc, %size, %trid, %noc_idx) : (i32, !ttkernel.noc_addr, i32, i32, i8) -> ()
      // TRID-specific barrier should follow.
      // ttkernel.noc_async_write_barrier_with_trid(%trid, %noc_idx) : (i32, i8) -> ()
      ```
    }];

    let arguments = (ins I32:$srcLocalL1Addr, TTKernel_NocAddr:$dstNocAddr, I32:$size, I32:$trid, Optional<I8>:$noc);

    let assemblyFormat = [{
      `(` $srcLocalL1Addr `,` $dstNocAddr `,` $size `,` $trid (`,` $noc^)? `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NocAsyncWriteBarrierOp : TTKernel_Op<"noc_async_write_barrier", [TTKernel_DeviceZoneOpTrait]> {
    let summary = "NocAsyncWriteBarrier";
    let description = [{
      NocAsyncWriteBarrier
    }];

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NocAsyncWriteBarrierWithTridOp : TTKernel_Op<"noc_async_write_barrier_with_trid", [TTKernel_DeviceZoneOpTrait, TTKernel_TridNocOpTrait]> {
    let summary = "NocAsyncWriteBarrierWithTrid";
    let description = [{
      Waits for outstanding write transactions matching a transaction ID. TRID is 0-15, NOC is 0 or 1.

      Example:
      ```
      ttkernel.noc_async_write_barrier_with_trid(%trid, %noc_idx) : (i32, i8) -> ()
      ```
    }];

    let arguments = (ins I32:$trid, Optional<I8>:$noc);

    let assemblyFormat = [{
      `(` $trid (`,` $noc^)? `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ResetNocTridBarrierCounterOp : TTKernel_Op<"reset_noc_trid_barrier_counter"> {
    let summary = "ResetNocTridBarrierCounter";
    let description = [{
      Resets the barrier counters for a set of transaction IDs on a given NOC. Mask bits correspond to TRIDs 0-15; NOC is 0 or 1.

      Example:
      ```
      ttkernel.reset_noc_trid_barrier_counter(%mask, %noc_idx) : (i32, i8) -> ()
      ```
    }];

    let arguments = (ins I32:$idMask, Optional<I8>:$noc);

    let hasVerifier = 1;

    let assemblyFormat = [{
      `(` $idMask (`,` $noc^)? `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_GetSemaphoreOp : TTKernel_Op<"get_semaphore"> {
    let summary = "GetSemaphoreOp";
    let description = [{
      Get L1 addr of the semaphore with specified semaphore id
    }];

    let arguments = (ins IndexLike:$semaphore);
    let results = (outs TTKernel_Semaphore:$sem_addr);

    let assemblyFormat = [{
      `(` $semaphore `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NocSemaphoreIncOp : TTKernel_Op<"noc_semaphore_inc"> {
    let summary = "NocSemaphoreInc";
    let description = [{
      The Tensix core executing this function call initiates an atomic increment
      (with 32-bit wrap) of a remote Tensix core L1 memory address. This L1 memory
      address is used as a semaphore of size 4 Bytes, as a synchronization
      mechanism.
    }];

    let arguments = (ins TTKernel_NocAddr:$addr, IndexLike:$incr, Optional<I8>:$noc_id);

    let assemblyFormat = [{
      `(` $addr `,` $incr (`,` $noc_id^)? `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NocSemaphoreSetOp : TTKernel_Op<"noc_semaphore_set"> {
    let summary = "NocSemaphoreSet";
    let description = [{
      Sets the value of a local L1 memory address on the Tensix core executing
      this function to a specific value. This L1 memory address is used as a
      semaphore of size 4 Bytes, as a synchronization mechanism. Also, see
      *noc_semaphore_wait*.
    }];

    let arguments = (ins TTKernel_L1AddrPtr:$sem_addr, IndexLike:$val);

    let assemblyFormat = [{
      `(` $sem_addr `,` $val `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NocSemaphoreWaitOp : TTKernel_Op<"noc_semaphore_wait"> {
    let summary = "NocSemaphoreWait";
    let description = [{
      A blocking call that waits until the value of a local L1 memory address on
      the Tensix core executing this function becomes equal to a target value.
      This L1 memory address is used as a semaphore of size 4 Bytes, as a
      synchronization mechanism. Also, see *noc_semaphore_set*.
    }];

    let arguments = (ins TTKernel_L1AddrPtr:$sem_addr, IndexLike:$val);

    let assemblyFormat = [{
      `(` $sem_addr `,` $val `)` attr-dict `:` functional-type(operands, results)
    }];
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

    let assemblyFormat = [{
      `(` $sem_addr `,` $val `)` attr-dict `:` functional-type(operands, results)
    }];
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

    let arguments = (ins AnyTypeOf<[TTKernel_L1Addr, TTKernel_Semaphore]>:$src_local_l1_addr,
                         TTKernel_NocAddr:$dst_noc_addr_multicast,
                         I32:$num_dests,
                         OptionalAttr<BoolAttr>:$linked,
                         OptionalAttr<BoolAttr>:$multicast_path_reserve);

    let assemblyFormat = [{
      `(` $src_local_l1_addr `,` $dst_noc_addr_multicast `,` $num_dests (`,` $linked^)? (`,` $multicast_path_reserve^)? `)` attr-dict `:` functional-type(operands, results)
    }];
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

    let arguments = (ins AnyTypeOf<[TTKernel_L1Addr, TTKernel_Semaphore]>:$src_local_l1_addr,
                         TTKernel_NocAddr:$dst_noc_addr_multicast,
                         I32:$num_dests,
                         OptionalAttr<BoolAttr>:$linked);

    let assemblyFormat = [{
      `(` $src_local_l1_addr `,` $dst_noc_addr_multicast `,` $num_dests (`,` $linked^)? `)` attr-dict `:` functional-type(operands, results)
    }];
}

//===----------------------------------------------------------------------===//
// TTKernel Compile and runtime arguments operations
//===----------------------------------------------------------------------===//

def TTKernel_ArgResultType : AnyTypeOf<[I32, TTKernel_CB, TTKernel_L1Addr]>;

def TTKernel_GetArgValOp : TTKernel_Op<"get_arg_val", [Pure]> {
    let summary = "Get runtime arg value.";
    let description = [{
      Get runtime argument value at specified index.
    }];

    let arguments = (ins IndexLike:$arg_index);

    let results = (outs TTKernel_ArgResultType:$arg_val);

    let assemblyFormat = [{
      `(` $arg_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_GetCommonArgValOp : TTKernel_Op<"get_common_arg_val", [Pure]> {
    let summary = "Get common runtime arg value.";
    let description = [{
      Get runtime argument value at specified index. (Indexes from different location compared to get_arg_val)
    }];

    let arguments = (ins IndexLike:$arg_index);

    let results = (outs TTKernel_ArgResultType:$arg_val);

    let assemblyFormat = [{
      `(` $arg_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_GetCompileArgValOp : TTKernel_Op<"get_compile_time_arg_val",
                                              [Pure]> {
    let summary = "Get compile-time arg value.";
    let description = [{
      Get compile-time argument value at specified index.
    }];

    let arguments = (ins I32Attr:$arg_index);

    let results = (outs TTKernel_ArgResultType:$arg_val);

    let assemblyFormat = [{
      `(` $arg_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

//===----------------------------------------------------------------------===//
// TTKernel Helper functions
//===----------------------------------------------------------------------===//

def TTKernel_CastToL1PtrOp : TTKernel_Op<"reinterpret_cast<volatile tt_l1_ptr uint32_t*>"> {
    let summary = "CastToL1Ptr";
    let description = [{
      Cast specified addr to L1 pointer.
    }];

    let arguments = (ins AnyTypeOf<[I32, TTKernel_L1Addr, TTKernel_Semaphore]>:$addr);

    let results = (outs TTKernel_L1AddrPtr:$l1_ptr);

    let assemblyFormat = [{
      `(` $addr `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_StoreToL1Op : TTKernel_Op<"store_to_l1"> {
    let summary = "StoreToL1";
    let description = [{
      Store value to L1.
    }];

    let arguments = (ins I32:$value, TTKernel_L1AddrPtr:$l1_ptr, I32:$offset);

    let assemblyFormat = [{
      `(` $value `,` $l1_ptr `,` $offset `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_GetInterleavedAddrGenFastOp : TTKernel_Op<"get_interleaved_addr_gen_fast"> {
    let summary = "GetInterleavedAddrGenFastOp";
    let description = [{
      Returns an InterleavedAddrGenFast type.
    }];

    let arguments = (ins I1:$DRAM, I32:$bank_base_address, I32:$page_size, TTKernel_DataFormat:$data_format);
    let results = (outs TTKernel_InterleavedAddrGenFast:$result);

    let assemblyFormat = [{
      `(` $DRAM `,` $bank_base_address `,` $page_size `,` $data_format `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_InterleavedAddrGenFastGetNocAddrOp : TTKernel_Op<"interleaved_addr_gen_fast.get_noc_addr"> {
    let summary = "InterleavedAddrGenFastGetNocAddr";
    let description = [{
      Returns an raw noc address from an interleaved addr gen struct.
    }];

    let arguments = (ins TTKernel_InterleavedAddrGenFast:$self, I32:$id, I32:$offset, Optional<I8>:$noc);
    let results = (outs TTKernel_NocAddr:$result);

    let assemblyFormat = [{
      `(` $self `,` $id `,` $offset `,` $noc `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_MyXOp : TTKernel_Op<"my_x", [Pure]> {
    let summary = "MyX";
    let description = [{
      Lowers to the tt-metal supported MY_X macro. This represents the virtual X coordinate of the current core.
    }];

    let arguments = (ins Optional<I8>:$noc);
    let results = (outs Index:$x);

    let assemblyFormat = [{
      `(` ($noc^)? `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_MyYOp : TTKernel_Op<"my_y", [Pure]> {
    let summary = "MyY";
    let description = [{
      Lowers to the tt-metal supported MY_Y macro. This represents the virtual Y coordinate of the current core.
    }];

    let arguments = (ins Optional<I8>:$noc);
    let results = (outs Index:$y);

    let assemblyFormat = [{
      `(` ($noc^)? `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_MyLogicalXOp : TTKernel_Op<"my_logical_x_",
  [Pure, DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]> {
    let summary = "MyLogicalX";
    let description = [{
      Lowers to the tt-metal supported my_logical_x_ global. This represents the logical X coordinate of the current core.
    }];

    let arguments = (ins);
    let results = (outs Index:$x);

    let assemblyFormat = [{
      attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_MyLogicalYOp : TTKernel_Op<"my_logical_y_",
  [Pure, DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]> {
    let summary = "MyLogicalY";
    let description = [{
      Lowers to the tt-metal supported my_logical_y_ global. This represents the logical Y coordinate of the current core.
    }];

    let arguments = (ins);
    let results = (outs Index:$y);

    let assemblyFormat = [{
      attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ConvertLogicalXToTranslatedOp : TTKernel_Op<"experimental::convert_logical_x_to_translated"> {
  let summary = "ConvertLogicalToTranslatedX";
  let description = [{
    this converts the x coordinate from the LOGICAL coordinate system to TRANSLATED
  }];

  let arguments = (ins IndexLike:$logical_x);
  let results = (outs IndexLike:$virtual_x);

  let assemblyFormat = [{
    `(` $logical_x `)` attr-dict `:` functional-type(operands, results)
  }];
}

def TTKernel_ConvertLogicalYToTranslatedOp : TTKernel_Op<"experimental::convert_logical_y_to_translated"> {
  let summary = "ConvertLogicalToTranslatedY";
  let description = [{
    this converts the y coordinate from the LOGICAL coordinate system to TRANSLATED
  }];

  let arguments = (ins IndexLike:$logical_y);
  let results = (outs IndexLike:$virtual_y);

  let assemblyFormat = [{
    `(` $logical_y `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// TTKernel Fabric API operations
//===----------------------------------------------------------------------===//

def TTKernel_GetMyDeviceIdOp : TTKernel_Op<"experimental::get_my_device_id", [Pure]> {
  let summary = "GetMyDeviceId";
  let description = [{
    Get my device id. This is a 16 bit value.
  }];

  let arguments = (ins);
  let results = (outs I16:$my_device_id);

  let assemblyFormat = [{
    `(` `)` attr-dict `:` functional-type(operands, results)
  }];
}

def TTKernel_FabricWriteOp : TTKernel_Op<"experimental::fabric_fast_write_any_len"> {
  let summary = "FabricWriteOp";
  let description = [{
    FabricWriteOp
  }];

  let arguments = (ins TTKernel_FabricConnectionManager:$fabric_connection_manager, I16:$dst_mesh_id, I16:$dst_dev_id, TTKernel_NocAddr:$dst_addr, I32:$src_addr, I32:$len_bytes);
  let results = (outs);

  let assemblyFormat = [{
    `(` $fabric_connection_manager `,` $dst_mesh_id `,` $dst_dev_id `,` $dst_addr `,` $src_addr `,` $len_bytes `)` attr-dict `:` functional-type(operands, results)
  }];
}

def TTKernel_FabricMulticastWriteOp : TTKernel_Op<"experimental::fabric_mcast_fast_write_any_len"> {
  let summary = "FabricMulticastWriteOp";
  let description = [{
    FabricMulticastWriteOp
  }];

  let arguments = (ins TTKernel_FabricConnectionManager:$fabric_connection_manager, I16:$dst_mesh_id, I16:$dst_dev_id_start, I16:$dst_dev_id_end, TTKernel_NocAddr:$dst_addr, I32:$src_addr, I32:$len_bytes);
  let results = (outs);

  let assemblyFormat = [{
    `(` $fabric_connection_manager `,` $dst_mesh_id `,` $dst_dev_id_start `,` $dst_dev_id_end `,` $dst_addr `,` $src_addr `,` $len_bytes `)` attr-dict `:` functional-type(operands, results)
  }];
}

def TTKernel_FabricSemIncOp : TTKernel_Op<"experimental::fabric_sem_inc"> {

  let summary = "FabricSemIncOp";
  let description = [{
    FabricSemIncOp. This operation increments a semaphore on a remote device.
  }];

  let arguments = (ins TTKernel_FabricConnectionManager:$fabric_connection_manager, I16:$dst_mesh_id, I16:$dst_dev_id, TTKernel_NocAddr:$semaphore_addr, IndexLike:$incr);
  let results = (outs);

  let assemblyFormat = [{
    `(` $fabric_connection_manager `,` $dst_mesh_id `,` $dst_dev_id `,` $semaphore_addr `,` $incr `)` attr-dict `:` functional-type(operands, results)
  }];
}

def TTKernel_CreateFabricConnectionManagerOp : TTKernel_Op<"experimental::create_fabric_connection_manager"> {
  let summary = "CreateFabricConnectionManager";
  let description = [{
    Create fabric connection manager. The fabric connection manager is required for all fabric operations.
  }];

  let arguments = (ins);
  let results = (outs TTKernel_FabricConnectionManager:$fabric_connection_manager);

  let assemblyFormat = [{
    `(` `)` attr-dict `:` functional-type(operands, results)
  }];
}

def TTKernel_SetupFabricConnectionsOp : TTKernel_Op<"experimental::setup_fabric_connections"> {
  let summary = "SetupFabricConnections";
  let description = [{
    Setup fabric connections for inter-device communication. The connection scheme
    is derived from the attribute `TTMetal_FabricConnectionConfigAttr` in the
    `EnqueueProgramOp`.

   `TTMetal_FabricConnectionConfigAttr` specifies:
      - `noc_index`: Which NOC the fabric uses (must match kernel's NocConfig)
      - `topology`: The routing scheme to use for the mesh device (e.g. Line, Ring)
      - `cluster_axis`: The axis along which the to route for 1D topologies
      - `num_links`: Number of routing planes (connections to fabric routers)

    After setup, the `FabricConnectionManager` can be used with ops like
    `fabric_fast_write_any_len` to send data to other devices in the mesh.
  }];

  let arguments = (ins TTKernel_FabricConnectionManager:$fabric_connection_manager);
  let results = (outs);

  let assemblyFormat = [{
    `(` $fabric_connection_manager `)` attr-dict `:` functional-type(operands, results)
  }];
}

def TTKernel_CloseFabricConnectionsOp : TTKernel_Op<"experimental::close_fabric_connections"> {
  let summary = "CloseFabricConnections";
  let description = [{
    Close fabric connections.
  }];

  let arguments = (ins TTKernel_FabricConnectionManager:$fabric_connection_manager);
  let results = (outs);

  let assemblyFormat = [{
    `(` $fabric_connection_manager `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// TTKernel Logical Mesh Position operations
//===----------------------------------------------------------------------===//

def TTKernel_GetLogicalMeshPositionOp : TTKernel_Op<"experimental::get_logical_mesh_position"> {
  let summary = "GetLogicalMeshPosition";
  let description = [{
    Get the logical mesh position for a given device ID.
  }];

  let arguments = (ins TTKernel_FabricConnectionManager:$fcm, I32:$device_id);
  let results = (outs TTKernel_MeshPosition:$position);

  let assemblyFormat = [{
    `(` $fcm `,` $device_id `)` attr-dict `:` functional-type(operands, results)
  }];
}

def TTKernel_GetDeviceIdFromLogicalMeshPositionOp : TTKernel_Op<"experimental::get_device_id_from_logical_mesh_position"> {
  let summary = "GetDeviceIdFromLogicalMeshPosition";
  let description = [{
    Get the device ID for a given logical mesh position.
    Takes a MeshPosition and returns the corresponding device ID.
    Maps to get_device_id_from_logical_mesh_position(fcm, position) in device code.
  }];

  let arguments = (ins TTKernel_FabricConnectionManager:$fcm, TTKernel_MeshPosition:$position);
  let results = (outs I16:$device_id);

  let assemblyFormat = [{
    `(` $fcm `,` $position `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// TTKernel Multicast NoC operations
//===----------------------------------------------------------------------===//

def TTKernel_GetNocMulticastAddrOp : TTKernel_Op<"get_noc_multicast_addr"> {
  let summary = "GetNocMulticastAddr";
  let description = [{
    Default tt-metal get_noc_multicast_addr
  }];

  let arguments = (ins IndexLike:$noc_x_start, IndexLike:$noc_y_start, IndexLike:$noc_x_end, IndexLike:$noc_y_end, AnyTypeOf<[I32, TTKernel_L1Addr, TTKernel_Semaphore]>:$addr, Optional<I8>:$noc);
  let results = (outs TTKernel_NocAddr:$mcastNocAddr);

    let assemblyFormat = [{
      `(` $noc_x_start `,` $noc_y_start `,` $noc_x_end `,` $noc_y_end `,` $addr (`,` $noc^)? `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ExperimentalGetNocMulticastAddrOp : TTKernel_Op<"experimental::get_noc_multicast_addr"> {
  let summary = "Experimental GetNocMulticastAddr";
  let description = [{
    Default tt-metal get_noc_multicast_addr, but flips mcast start and end coordinates on NOC1.
  }];

  let arguments = (ins IndexLike:$noc_x_start, IndexLike:$noc_y_start, IndexLike:$noc_x_end, IndexLike:$noc_y_end, AnyTypeOf<[I32, TTKernel_L1Addr, TTKernel_Semaphore]>:$addr, Optional<I8>:$noc);
  let results = (outs TTKernel_NocAddr:$mcastNocAddr);

    let assemblyFormat = [{
      `(` $noc_x_start `,` $noc_y_start `,` $noc_x_end `,` $noc_y_end `,` $addr (`,` $noc^)? `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_NocAsyncWriteMulticastOnePacketOp : TTKernel_Op<"noc_async_write_multicast_one_packet"> {
  let summary = "NocAsyncWriteMulticastOnePacket";
  let description = [{
    NocAsyncWriteMulticastOnePacket
    this issues only a single packet with size <= NOC_MAX_BURST_SIZE (ie maximum packet size)
  }];

  let arguments = (ins I32:$srcLocalL1Addr, TTKernel_NocAddr:$dstNocAddrMulticast, I32:$size, I32:$num_dests, OptionalAttr<BoolAttr>:$linked, OptionalAttr<BoolAttr>:$multicast_path_reserve, Optional<I8>:$noc);

    let assemblyFormat = [{
      `(` $srcLocalL1Addr `,` $dstNocAddrMulticast `,` $size `,` $num_dests (`,` $linked^)? (`,` $multicast_path_reserve^)? (`,` $noc^)? `)` attr-dict `:` functional-type(operands, results)
    }];
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

  let assemblyFormat = [{
    `(` $srcLocalL1Addr `,` $dstNocAddrMulticast `,` $size `,` $num_dests (`,` $linked^)? (`,` $multicast_path_reserve^)? (`,` $noc^)? `)` attr-dict `:`functional-type(operands, results)
  }];
}

def TTKernel_NocAsyncWriteMulticastLoopbackSrcOp : TTKernel_Op<"noc_async_write_multicast_loopback_src"> {
  let summary = "NocAsyncWriteMulticastLoopbackSrc";
  let description = [{
    NocAsyncWriteMulticastLoopbackSrc
  }];

  let arguments = (ins I32:$srcLocalL1Addr, TTKernel_NocAddr:$dstNocAddrMulticast, I32:$size, I32:$num_dests, OptionalAttr<BoolAttr>:$linked, OptionalAttr<BoolAttr>:$multicast_path_reserve, Optional<I8>:$noc);

  let assemblyFormat = [{
    `(` $srcLocalL1Addr `,` $dstNocAddrMulticast `,` $size `,` $num_dests (`,` $linked^)? (`,` $multicast_path_reserve^)? (`,` $noc^)? `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// TTKernel TensorAccessor operations
//===----------------------------------------------------------------------===//

def TTKernel_TensorAccessorArgsOp : TTKernel_Op<"TensorAccessorArgs",
    [AttrSizedOperandSegments]> {
  let summary = "TensorAccessorArgs";
  let description = [{
    TensorAccessorArgs struct constructor.

    CTA (compile-time args) and CRTA (compile/runtime args) offsets are determined as follows.
    1. If cta_expr is provided: use the constexpr string expression.
    2. Else if `prev_args` is provided, use chaining:
      `prev_args.next_compile_time_args_offset()` (for CTA) or
      `prev_args.next_common_runtime_args_offset()` (for CRTA).
    3. Otherwise, use cta_base/crta_base integer constants.

    Examples:
    ```mlir
    // Literal offsets
    %c0 = arith.constant 0 : i32
    %args = ttkernel.TensorAccessorArgs(%c0, %c0) : (i32, i32) -> !ttkernel.TensorAccessorArgs
    // Generates: TensorAccessorArgs<0, 0>()

    // CTA+CRTA chaining (common pattern)
    %args_src = ttkernel.TensorAccessorArgs(%c0, %c0) : (i32, i32) -> !ttkernel.TensorAccessorArgs
    %args_dst = ttkernel.TensorAccessorArgs(prev = %args_src) : (!ttkernel.TensorAccessorArgs) -> !ttkernel.TensorAccessorArgs
    // Generates: TensorAccessorArgs<args_src.next_compile_time_args_offset(), args_src.next_common_runtime_args_offset()>()

    // Selective override: chain CTA from prev, use literal CRTA
    %args_custom = ttkernel.TensorAccessorArgs(prev = %args_src) {crta_expr = "0"} : (!ttkernel.TensorAccessorArgs) -> !ttkernel.TensorAccessorArgs
    // Generates: TensorAccessorArgs<args_src.next_compile_time_args_offset(), 0>()

    // Constexpr expression (no chaining)
    %args_expr = ttkernel.TensorAccessorArgs(%c0, %c0) {cta_expr = "get_offset()"} : (i32, i32) -> !ttkernel.TensorAccessorArgs
    // Generates: TensorAccessorArgs<get_offset(), 0>()
    ```
  }];

  let arguments = (ins
    Optional<I32>:$cta_base,
    Optional<I32>:$crta_base,
    Optional<TTKernel_TensorAccessorArgs>:$prev_args,
    OptionalAttr<StrAttr>:$cta_expr,
    OptionalAttr<StrAttr>:$crta_expr
  );
  let results = (outs TTKernel_TensorAccessorArgs:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def TTKernel_TensorAccessorOp : TTKernel_Op<"TensorAccessor"> {
  let summary = "MakeTensorAccessorFromArgs";
  let description = [{
    TensorAccessor constructor.
  }];

  let arguments = (ins TTKernel_TensorAccessorArgs:$args, I32:$bank_base_address_in, I32:$page_size_in);
  let results = (outs TTKernel_TensorAccessor:$result);

  let assemblyFormat = [{
    `(` $args `,` $bank_base_address_in `,` $page_size_in `)` attr-dict `:` functional-type(operands, results)
  }];
}

def TTKernel_TensorAccessorGetNocAddrOp : TTKernel_Op<"tensor_accessor.get_noc_addr"> {
  let summary = "TensorAccessor's get_noc_addr";
  let description = [{
    get_noc_addr using information stored in the TensorAccessor.
  }];
  let arguments = (ins TTKernel_TensorAccessor:$tensor_accessor, I32:$id, I32:$offset, Optional<I8>:$noc);
  let results = (outs TTKernel_NocAddr:$nocAddr);

  let assemblyFormat = [{
    `(` $tensor_accessor `,` $id `,` $offset (`,` $noc^)? `)` attr-dict `:` functional-type(operands, results)
  }];
}

def TTKernel_TensorAccessorGetShardNocAddrOp : TTKernel_Op<"tensor_accessor.get_shard_noc_addr"> {
  let summary = "TensorAccessor's get_shard_noc_addr";
  let description = [{
    Returns noc addr of a shard.
  }];
  let arguments = (ins TTKernel_TensorAccessor:$tensor_accessor, I32:$shard_id, I32:$offset, Optional<I8>:$noc);
  let results = (outs I32:$shardNocAddr);

  let assemblyFormat = [{
    `(` $tensor_accessor `,` $shard_id `,` $offset (`,` $noc^)? `)` attr-dict `:` functional-type(operands, results)
  }];
}

def TTKernel_TensorAccessorGetBankAndOffsetOp : TTKernel_Op<"tensor_accessor.get_bank_and_offset"> {
  let summary = "TensorAccessor's get_bank_and_offset";
  let description = [{
    Returns bank id and page offset.
  }];
  let arguments = (ins TTKernel_TensorAccessor:$tensor_accessor, I32:$page_id);
  let results = (outs TTKernel_TensorAccessorPageMapping:$bank_id_and_offset);

  let assemblyFormat = [{
    `(` $tensor_accessor `,` $page_id `)` attr-dict `:` functional-type(operands, results)
  }];
}

def TTKernel_TensorAccessorIsLocalBankOp : TTKernel_Op<"tensor_accessor.is_local_bank"> {
  let summary = "TensorAccessor's is_local_bank";
  let description = [{
    Returns bool indicating bank locality.
  }];
  let arguments = (ins TTKernel_TensorAccessor:$tensor_accessor, I32:$id, I32:$offset, Optional<I8>:$noc);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    `(` $tensor_accessor `,` $id `,` $offset (`,` $noc^)? `)` attr-dict `:` functional-type(operands, results)
  }];
}

def TTKernel_TensorAccessorIsLocalAddrOp : TTKernel_Op<"tensor_accessor.is_local_addr"> {
  let summary = "TensorAccessor's is_local_addr";
  let description = [{
    Returns bool indicating addr locality.
  }];
  let arguments = (ins TTKernel_TensorAccessor:$tensor_accessor, I32:$virtual_x, I32:$virtual_y, Optional<I8>:$noc);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    `(` $tensor_accessor `,` $virtual_x `,` $virtual_y (`,` $noc^)? `)` attr-dict `:` functional-type(operands, results)
  }];
}

def TTKernel_TensorAccessorIsLocalPageOp : TTKernel_Op<"tensor_accessor.is_local_page"> {
  let summary = "TensorAccessor's is_local_page";
  let description = [{
    Returns bool indicating page locality.
  }];
  let arguments = (ins TTKernel_TensorAccessor:$tensor_accessor, I32:$page_id, Optional<I8>:$noc);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    `(` $tensor_accessor `,` $page_id (`,` $noc^)? `)` attr-dict `:` functional-type(operands, results)
  }];
}

def TTKernel_TensorAccessorIsLocalShardOp : TTKernel_Op<"tensor_accessor.is_local_shard"> {
  let summary = "TensorAccessor's is_local_shard";
  let description = [{
    Returns bool indicating shard locality.
  }];
  let arguments = (ins TTKernel_TensorAccessor:$tensor_accessor, I32:$shard_id, Optional<I8>:$noc);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    `(` $tensor_accessor `,` $shard_id (`,` $noc^)? `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// TTKernel Misc operations
//===----------------------------------------------------------------------===//

def TTKernel_UnreachableOp : TTKernel_Op<"unreachable", [Pure, ReturnLike, Terminator]> {
    let summary = "Unreachable op.";
    let description = [{
      Unreachable operation
    }];

  let assemblyFormat = [{
    `(` `)` attr-dict `:` functional-type(operands, results)
  }];
}

def TTKernel_MemZerosBaseOp : TTKernel_Op<"mem_zeros_base"> {
    let summary = "Op corresponding to MEM_ZEROS_BASE macro in kernels.";
    let description = [{
      Op corresponding to MEM_ZEROS_BASE macro in kernels.
    }];

    let results = (outs I32:$result);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_MemZerosSizeOp : TTKernel_Op<"mem_zeros_size"> {
    let summary = "Op corresponding to MEM_ZEROS_SIZE macro in kernels.";
    let description = [{
      Op corresponding to MEM_ZEROS_SIZE macro in kernels.
    }];

    let results = (outs I32:$result);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_GetWritePtrOp : TTKernel_Op<"get_write_ptr"> {
    let summary = "GetWritePtr";
    let description = [{
      GetWritePtr operation
    }];

    let arguments = (ins TTKernel_CB:$cb);
    let results = (outs I32:$writePtr);

    let assemblyFormat = [{
      `(` $cb `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_GetReadPtrOp : TTKernel_Op<"get_read_ptr"> {
    let summary = "GetReadPtr";
    let description = [{
      GetReadPtr operation
    }];

    let arguments = (ins TTKernel_CB:$cb);
    let results = (outs I32:$readPtr);

    let assemblyFormat = [{
      `(` $cb `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_GetTileSizeOp : TTKernel_Op<"get_tile_size"> {
    let summary = "Get the tile size in bytes of a given CB";
    let description = [{
      get_tile_size operation
    }];
    let arguments = (ins TTKernel_CB:$cb);
    let results = (outs I32:$tileSizeBytes);

    let assemblyFormat = [{
      `(` $cb `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_GetDataFormatOp: TTKernel_Op<"get_dataformat"> {
    let summary = "Get the data format of a given CB";
    let description = [{
      get_dataformat operation
    }];
    let arguments = (ins TTKernel_CB:$cb);
    let results = (outs TTKernel_DataFormat:$dataFormat);

    let assemblyFormat = [{
      `(` $cb `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_DPrintOp : TTKernel_Op<"dprint", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "Print to output stream from kernel.";
    let description = [{
      std::format style format string:
      ```c++
      rewriter.create<ttkernel::DPrintOp>(loc, "nocY={} nocX={} addr={}\\n",
                                        nocY, nocX, addr);
      ```
      ```
      ttkernel.dprint("virtY {} virtX {} addr {}\\n", %14, %15, %13) : (index, index, i32)
      ```
      Notes:
        - Only trivial format specifier currently supported, i.e. `{}`.
        - Must double escape newline character or other special characters.
        - When a CB operand is provided, calls print_cb_details, printing underlying CB pointers and details.
    }];
    let arguments = (ins StrAttr:$fmt, Variadic<AnyType>:$argv);

    let hasVerifier = 1;

    let assemblyFormat = "`(` $fmt (`,` $argv^)? `)` attr-dict `:` functional-type($argv, results)";

    let extraClassDeclaration = [{
      template <typename... Args>
      static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, const char* fmt, Args &&...argv) {
        return build(odsBuilder, odsState, StringRef(fmt), ValueRange{std::forward<Args>(argv)...});
      }
    }];
}

#endif
