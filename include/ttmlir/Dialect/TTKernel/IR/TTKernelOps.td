// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELOPS_TD
#define TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELOPS_TD

include "ttmlir/Dialect/TTKernel/IR/TTKernelBase.td"
include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.td"
include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td"
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

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
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

def TTKernel_MatmulTilesOp : TTKernel_FPUOp<"matmul_tiles", [TTKernel_TernaryOpTrait]> {
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
                         IndexLike:$dst_tile_idx,
                         I32:$transpose);

    let assemblyFormat = [{
      `(` $in0_cb_id `,` $in1_cb_id `,` $in0_tile_idx `,` $in1_tile_idx `,` $dst_tile_idx `,` $transpose `)` attr-dict `:` functional-type(operands, results)
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

def TTKernel_ExperimentalMatmulBlockOp : TTKernel_FPUOp<"experimental::matmul_block", [TTKernel_TernaryOpTrait]> {
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

def TTKernel_ReduceTileOp : TTKernel_FPUOp<"reduce_tile", [TTKernel_TernaryOpTrait]> {
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

def TTKernel_MaxTilesInitOp : TTKernel_InitOp<"max_tile_init"> {
    let summary = "Short init function";
    let description = [{
      Must be run before max_tile.
    }];

    let arguments = (ins);

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_MaxTilesOp : TTKernel_SFPUOp<"max_tile", [TTKernel_BinaryOpTrait]> {
    let summary = "Max operation";
    let description = [{
      Performs element-wise computation of maximum operation
      DST[dst0_index] <- max(DST[dst0_index], DST[dst1_index])
      on DST register operands. The DST register buffer must be in
      acquired state via *tile_regs_acquire* call.
    }];

    let arguments = (ins IndexLike:$dst0_index, IndexLike:$dst1_index);

    let assemblyFormat = [{
      `(` $dst0_index `,` $dst1_index `)` attr-dict `:` functional-type(operands, results)
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

def TTKernel_CeilTileF32Op : TTKernel_SFPUOp<"ceil_tile_float32", [TTKernel_UnaryOpTrait]> {
    let summary = "Ceil f32 tile in the DST at specified index.";
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

def TTKernel_FloorTileF32Op : TTKernel_SFPUOp<"floor_tile_float32", [TTKernel_UnaryOpTrait]> {
    let summary = "Floor f32 tile in the DST at specified index.";
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

def TTKernel_NocAsyncReadBarrierOp : TTKernel_Op<"noc_async_read_barrier", [TTKernel_DeviceZoneOpTrait]> {
    let summary = "NocAsyncReadBarrier";
    let description = [{
      NocAsyncReadBarrier
    }];

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
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

def TTKernel_NocAsyncWriteBarrierOp : TTKernel_Op<"noc_async_write_barrier", [TTKernel_DeviceZoneOpTrait]> {
    let summary = "NocAsyncWriteBarrier";
    let description = [{
      NocAsyncWriteBarrier
    }];

    let assemblyFormat = [{
      `(` `)` attr-dict `:` functional-type(operands, results)
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

    let arguments = (ins TTKernel_Semaphore:$src_local_l1_addr,
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

    let arguments = (ins TTKernel_Semaphore:$src_local_l1_addr,
                         TTKernel_NocAddr:$dst_noc_addr_multicast,
                         I32:$num_dests,
                         BoolAttr:$linked,
                         BoolAttr:$multicast_path_reserve);

    let assemblyFormat = [{
      `(` $src_local_l1_addr `,` $dst_noc_addr_multicast `,` $num_dests `,` $linked `,` $multicast_path_reserve `)` attr-dict `:` functional-type(operands, results)
    }];
}

//===----------------------------------------------------------------------===//
// TTKernel Compile and runtime arguments operations
//===----------------------------------------------------------------------===//

def TTKernel_ArgResultType : AnyTypeOf<[I32, TTKernel_CB, TTKernel_L1Addr]>;

def TTKernel_GetArgValOp : TTKernel_Op<"get_arg_val"> {
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

def TTKernel_GetCommonArgValOp : TTKernel_Op<"get_common_arg_val"> {
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

def TTKernel_TensorAccessorArgsOp : TTKernel_Op<"TensorAccessorArgs"> {
  let summary = "TensorAccessorArgs";
  let description = [{
    TensorAccessorArgs struct constructor.
  }];

  let arguments = (ins I32:$cta_base, I32:$crta_base);
  let results = (outs TTKernel_TensorAccessorArgs:$result);

  let assemblyFormat = [{
    `(` $cta_base `,` $crta_base `)` attr-dict `:` functional-type(operands, results)
  }];
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

def TTKernel_TensorAccessorGetNocAddrOp : TTKernel_Op<"tensor_accessor_get_noc_addr"> {
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

def TTKernel_TensorAccessorGetShardNocAddrOp : TTKernel_Op<"tensor_accessor_get_shard_noc_addr"> {
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

def TTKernel_TensorAccessorGetBankAndOffsetOp : TTKernel_Op<"tensor_accessor_get_bank_and_offset"> {
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

def TTKernel_TensorAccessorIsLocalBankOp : TTKernel_Op<"tensor_accessor_is_local_bank"> {
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

def TTKernel_TensorAccessorIsLocalAddrOp : TTKernel_Op<"tensor_accessor_is_local_addr"> {
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

def TTKernel_TensorAccessorIsLocalPageOp : TTKernel_Op<"tensor_accessor_is_local_page"> {
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

def TTKernel_TensorAccessorIsLocalShardOp : TTKernel_Op<"tensor_accessor_is_local_shard"> {
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

def TTKernel_CBReinterpretShapeOp : TTKernel_Op<"cb_reinterpret_shape"> {
    let summary = "Get the data format of a given CB";
    let description = [{
      get_dataformat operation
    }];
    let arguments = (ins TTKernel_CB:$input);
    let results = (outs TTKernel_CB:$output);

    let hasVerifier = 1;

    let assemblyFormat = [{
      `(` $input `)` attr-dict `:` functional-type(operands, results)
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
      ```mlir
      ttkernel.dprint("virtY {} virtX {} addr {}\\n", %14, %15, %13) : (index, index, i32)
      ```
      Notes:
        - Only trivial format specifier currently supported, i.e. `{}`.
        - Must double escape newline character or other special characters.
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
