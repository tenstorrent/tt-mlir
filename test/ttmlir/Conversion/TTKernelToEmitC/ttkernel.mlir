// RUN: ttmlir-opt --convert-ttkernel-to-emitc -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>

!cb0_scalar = !ttkernel.cb<memref<64x128xf32, #l1_>>
!cb1_scalar = !ttkernel.cb<memref<64x128xf32, #l1_>>
!cb2_scalar = !ttkernel.cb<memref<64x128xf32, #l1_>>

!cb0_tiles = !ttkernel.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
!cb1_tiles = !ttkernel.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
!cb2_tiles = !ttkernel.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>

module {
  //===----------------------------------------------------------------------===//
  // TTKernel Compute Kernel Hardware Startup operation
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: func @compute_kernel_hw_startup_unary
  func.func @compute_kernel_hw_startup_unary() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    // CHECK: %[[INCB:.*]] = emitc.literal "get_compile_time_arg_val(0)"
    %icb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
    // CHECK: %[[OCB:.*]] = emitc.literal "get_compile_time_arg_val(1)"
    %ocb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb2_tiles
    // CHECK: emitc.call_opaque "compute_kernel_hw_startup"(%[[INCB]], %[[OCB]])
    "ttkernel.compute_kernel_hw_startup"(%icb, %ocb) : (!cb0_tiles, !cb2_tiles) -> ()
    return
  }

  // CHECK-LABEL: func @compute_kernel_hw_startup_binary
  func.func @compute_kernel_hw_startup_binary() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    // CHECK: %[[INCB0:.*]] = emitc.literal "get_compile_time_arg_val(0)"
    %icb0 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
    // CHECK: %[[INCB1:.*]] = emitc.literal "get_compile_time_arg_val(1)"
    %icb1 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
    // CHECK: %[[OCB:.*]] = emitc.literal "get_compile_time_arg_val(2)"
    %ocb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 2 : i32}> : () -> !cb2_tiles
    // CHECK: emitc.call_opaque "compute_kernel_hw_startup"(%[[INCB0]], %[[INCB1]], %[[OCB]])
    "ttkernel.compute_kernel_hw_startup"(%icb0, %icb1, %ocb) : (!cb0_tiles, !cb1_tiles, !cb2_tiles) -> ()
    return
  }

  //===----------------------------------------------------------------------===//
  // TTKernel Register operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: ttkernel_register_operations
  module @ttkernel_register_operations {

    // CHECK-LABEL: func @tile_regs_acquire
    func.func @tile_regs_acquire() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "tile_regs_acquire"()
      "ttkernel.tile_regs_acquire"() : () -> ()
      return
    }

    // CHECK-LABEL: func @tile_regs_commit
    func.func @tile_regs_commit() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "tile_regs_commit"()
      "ttkernel.tile_regs_commit"() : () -> ()
      return
    }

    // CHECK-LABEL: func @tile_regs_wait
    func.func @tile_regs_wait() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "tile_regs_wait"()
      "ttkernel.tile_regs_wait"() : () -> ()
      return
    }

    // CHECK-LABEL: func @tile_regs_release
    func.func @tile_regs_release() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "tile_regs_release"()
      "ttkernel.tile_regs_release"() : () -> ()
      return
    }

    // CHECK-LABEL: func @pack_tile
    func.func @pack_tile() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[OUT_CB:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %out_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : index
      // CHECK: %[[OUT_CB_INDEX:.*]] = "emitc.constant"
      %out_cb_index = arith.constant 1 : index
      // CHECK: emitc.call_opaque "pack_tile"(%[[DST_INDEX]], %[[OUT_CB]], %[[OUT_CB_INDEX]])
      "ttkernel.pack_tile"(%dst_index, %out_cb, %out_cb_index) : (index, !cb0_tiles, index) -> ()
      return
    }

    // CHECK-LABEL: func @copy_tile_init
    func.func @copy_tile_init() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[CB:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: emitc.call_opaque "copy_tile_init"(%[[CB]])
      "ttkernel.copy_tile_init"(%cb) : (!cb0_tiles) -> ()
      return
    }

    // CHECK-LABEL: func @copy_tile
    func.func @copy_tile() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[CB:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[CB_INDEX:.*]] = "emitc.constant"
      %cb_index = arith.constant 2 : index
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 1 : index
      // CHECK: emitc.call_opaque "copy_tile"(%[[CB]], %[[CB_INDEX]], %[[DST_INDEX]])
      "ttkernel.copy_tile"(%cb, %cb_index, %dst_index) : (!cb0_tiles, index, index) -> ()
      return
    }

    // CHECK-LABEL: func @typecast_tile_init
    func.func @typecast_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "typecast_tile_init"() : () -> ()
      "ttkernel.typecast_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @typecast_tile
    func.func @typecast_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "typecast_tile"(%[[DST0_INDEX]]) {template_args =
      // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float32)">
      // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float16_b)">
      "ttkernel.typecast_tile"(%dst0_index) <{in_dtype = #ttcore.supportedDataTypes<f32>, out_dtype = #ttcore.supportedDataTypes<bf16>}> : (i32) -> ()
      // CHECK: emitc.call_opaque "typecast_tile"(%[[DST0_INDEX]]) {template_args =
      // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float16)">
      // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Int32)">
      "ttkernel.typecast_tile"(%dst0_index) <{in_dtype = #ttcore.supportedDataTypes<f16>, out_dtype = #ttcore.supportedDataTypes<si32>}> : (i32) -> ()
      // CHECK: emitc.call_opaque "typecast_tile"(%[[DST0_INDEX]]) {template_args =
      // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Bfp8_b)">
      // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Bfp8)">
      "ttkernel.typecast_tile"(%dst0_index) <{in_dtype = #ttcore.supportedDataTypes<bfp_bf8>, out_dtype = #ttcore.supportedDataTypes<bfp_f8>}> : (i32) -> ()
      // CHECK: emitc.call_opaque "typecast_tile"(%[[DST0_INDEX]]) {template_args =
      // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Bfp4_b)">
      // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Bfp4)">
      "ttkernel.typecast_tile"(%dst0_index) <{in_dtype = #ttcore.supportedDataTypes<bfp_bf4>, out_dtype = #ttcore.supportedDataTypes<bfp_f4>}> : (i32) -> ()
      // CHECK: emitc.call_opaque "typecast_tile"(%[[DST0_INDEX]]) {template_args =
      // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Bfp2_b)">
      // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Bfp2)">
      "ttkernel.typecast_tile"(%dst0_index) <{in_dtype = #ttcore.supportedDataTypes<bfp_bf2>, out_dtype = #ttcore.supportedDataTypes<bfp_f2>}> : (i32) -> ()
      // CHECK: emitc.call_opaque "typecast_tile"(%[[DST0_INDEX]]) {template_args =
      // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::UInt32)">
      // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::UInt16)">
      "ttkernel.typecast_tile"(%dst0_index) <{in_dtype = #ttcore.supportedDataTypes<u32>, out_dtype = #ttcore.supportedDataTypes<u16>}> : (i32) -> ()
      // CHECK: emitc.call_opaque "typecast_tile"(%[[DST0_INDEX]]) {template_args =
      // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::UInt16)">
      // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::UInt8)">
      "ttkernel.typecast_tile"(%dst0_index) <{in_dtype = #ttcore.supportedDataTypes<u16>, out_dtype = #ttcore.supportedDataTypes<u8>}> : (i32) -> ()
      return
    }

  } // module

  //===----------------------------------------------------------------------===//
  // TTKernel FPU operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: ttkernel_fpu_operations
  module @ttkernel_fpu_operations {

    // CHECK-LABEL: func @unary_op_init_common
    func.func @unary_op_init_common() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[IN_CB:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %in_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[OUT_CB:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      %out_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
      // CHECK: emitc.call_opaque "unary_op_init_common"(%[[IN_CB]], %[[OUT_CB]])
      "ttkernel.unary_op_init_common"(%in_cb, %out_cb) : (!cb0_tiles, !cb1_tiles) -> ()
      return
    }

    // CHECK-LABEL: func @binary_op_init_common
    func.func @binary_op_init_common() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[CB0:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %cb0 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[CB1:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      %cb1 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
      // CHECK: %[[OUT_CB:.*]] = emitc.literal "get_compile_time_arg_val(2)"
      %out_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 2 : i32}> : () -> !cb2_tiles
      // CHECK: emitc.call_opaque "binary_op_init_common"(%[[CB0]], %[[CB1]], %[[OUT_CB]])
      "ttkernel.binary_op_init_common"(%cb0, %cb1, %out_cb) : (!cb0_tiles, !cb1_tiles, !cb2_tiles) -> ()
      return
    }

    // CHECK-LABEL: func @add_tiles_init
    func.func @add_tiles_init() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[CB0:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %cb0 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[CB1:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      %cb1 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
      // CHECK: emitc.call_opaque "add_tiles_init"(%[[CB0]], %[[CB1]])
      "ttkernel.add_tiles_init"(%cb0, %cb1) : (!cb0_tiles, !cb1_tiles) -> ()
      return
    }

    // CHECK-LABEL: func @add_tiles
    func.func @add_tiles() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[CB0:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %cb0 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[CB1:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      %cb1 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
      // CHECK: %[[CB0_INDEX:.*]] = "emitc.constant"
      %cb0_index = arith.constant 1 : index
      // CHECK: %[[CB1_INDEX:.*]] = "emitc.constant"
      %cb1_index = arith.constant 2 : index
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : index
      // CHECK: emitc.call_opaque "add_tiles"(%[[CB0]], %[[CB1]], %[[CB0_INDEX]], %[[CB1_INDEX]], %[[DST_INDEX]])
      "ttkernel.add_tiles"(%cb0, %cb1, %cb0_index, %cb1_index, %dst_index) : (!cb0_tiles, !cb1_tiles, index, index, index) -> ()
      return
    }

    // CHECK-LABEL: func @unary_bcast_init
    func.func @unary_bcast_init() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      %in_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      %out_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
      // CHECK: %[[IN_CB:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      // CHECK: %[[OUT_CB:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      "ttkernel.unary_bcast_init"(%in_cb, %out_cb) <{bcast_type = #ttkernel.bcast_type<bcast_type_row>}> : (!cb0_tiles, !cb1_tiles) -> ()
      "ttkernel.unary_bcast_init"(%in_cb, %out_cb) <{bcast_type = #ttkernel.bcast_type<bcast_type_col>}> : (!cb0_tiles, !cb1_tiles) -> ()
      "ttkernel.unary_bcast_init"(%in_cb, %out_cb) <{bcast_type = #ttkernel.bcast_type<bcast_type_scalar>}> : (!cb0_tiles, !cb1_tiles) -> ()
      "ttkernel.unary_bcast_init"(%in_cb, %out_cb) <{bcast_type = #ttkernel.bcast_type<bcast_type_none>}> : (!cb0_tiles, !cb1_tiles) -> ()
      // CHECK: call_opaque "unary_bcast_init"(%[[IN_CB]], %[[OUT_CB]]) {template_args = [#emitc.opaque<"BroadcastType::ROW">]} : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
      // CHECK: call_opaque "unary_bcast_init"(%[[IN_CB]], %[[OUT_CB]]) {template_args = [#emitc.opaque<"BroadcastType::COL">]} : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
      // CHECK: call_opaque "unary_bcast_init"(%[[IN_CB]], %[[OUT_CB]]) {template_args = [#emitc.opaque<"BroadcastType::SCALAR">]} : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
      // CHECK: call_opaque "unary_bcast_init"(%[[IN_CB]], %[[OUT_CB]]) {template_args = [#emitc.opaque<"BroadcastType::NONE">]} : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
      return
    }

    // CHECK-LABEL: func @unary_bcast
    func.func @unary_bcast() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      %in_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      %in_tile_index = arith.constant 1 : index
      %dst_index = arith.constant 3 : index
      // CHECK: %[[IN_CB:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      // CHECK: %[[IN_TILE_INDEX:.*]] = "emitc.constant"
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      "ttkernel.unary_bcast"(%in_cb, %in_tile_index, %dst_index) <{bcast_type = #ttkernel.bcast_type<bcast_type_row>}> : (!cb0_tiles, index, index) -> ()
      "ttkernel.unary_bcast"(%in_cb, %in_tile_index, %dst_index) <{bcast_type = #ttkernel.bcast_type<bcast_type_col>}> : (!cb0_tiles, index, index) -> ()
      "ttkernel.unary_bcast"(%in_cb, %in_tile_index, %dst_index) <{bcast_type = #ttkernel.bcast_type<bcast_type_scalar>}> : (!cb0_tiles, index, index) -> ()
      "ttkernel.unary_bcast"(%in_cb, %in_tile_index, %dst_index) <{bcast_type = #ttkernel.bcast_type<bcast_type_none>}> : (!cb0_tiles, index, index) -> ()
      // CHECK: emitc.call_opaque "unary_bcast"(%[[IN_CB]], %[[IN_TILE_INDEX]], %[[DST_INDEX]]) {template_args = [#emitc.opaque<"BroadcastType::ROW">]}
      // CHECK: emitc.call_opaque "unary_bcast"(%[[IN_CB]], %[[IN_TILE_INDEX]], %[[DST_INDEX]]) {template_args = [#emitc.opaque<"BroadcastType::COL">]}
      // CHECK: emitc.call_opaque "unary_bcast"(%[[IN_CB]], %[[IN_TILE_INDEX]], %[[DST_INDEX]]) {template_args = [#emitc.opaque<"BroadcastType::SCALAR">]}
      // CHECK: emitc.call_opaque "unary_bcast"(%[[IN_CB]], %[[IN_TILE_INDEX]], %[[DST_INDEX]]) {template_args = [#emitc.opaque<"BroadcastType::NONE">]}
      return
    }

    // CHECK-LABEL: func @sub_tiles_init
    func.func @sub_tiles_init() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[CB0:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %cb0 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[CB1:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      %cb1 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
      // CHECK: emitc.call_opaque "sub_tiles_init"(%[[CB0]], %[[CB1]])
      "ttkernel.sub_tiles_init"(%cb0, %cb1) : (!cb0_tiles, !cb1_tiles) -> ()
      return
    }

    // CHECK-LABEL: func @sub_tiles
    func.func @sub_tiles() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[CB0:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %cb0 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[CB1:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      %cb1 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
      // CHECK: %[[CB0_INDEX:.*]] = "emitc.constant"
      %cb0_index = arith.constant 1 : index
      // CHECK: %[[CB1_INDEX:.*]] = "emitc.constant"
      %cb1_index = arith.constant 2 : index
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : index
      // CHECK: emitc.call_opaque "sub_tiles"(%[[CB0]], %[[CB1]], %[[CB0_INDEX]], %[[CB1_INDEX]], %[[DST_INDEX]])
      "ttkernel.sub_tiles"(%cb0, %cb1, %cb0_index, %cb1_index, %dst_index) : (!cb0_tiles, !cb1_tiles, index, index, index) -> ()
      return
    }

    // CHECK-LABEL: func @mul_tiles_init
    func.func @mul_tiles_init() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[CB0:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %cb0 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[CB1:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      %cb1 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
      // CHECK: emitc.call_opaque "mul_tiles_init"(%[[CB0]], %[[CB1]])
      "ttkernel.mul_tiles_init"(%cb0, %cb1) : (!cb0_tiles, !cb1_tiles) -> ()
      return
    }

    // CHECK-LABEL: func @mul_tiles
    func.func @mul_tiles() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[CB0:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %cb0 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[CB1:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      %cb1 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
      // CHECK: %[[CB0_INDEX:.*]] = "emitc.constant"
      %cb0_index = arith.constant 1 : i32
      // CHECK: %[[CB1_INDEX:.*]] = "emitc.constant"
      %cb1_index = arith.constant 2 : i32
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "mul_tiles"(%[[CB0]], %[[CB1]], %[[CB0_INDEX]], %[[CB1_INDEX]], %[[DST_INDEX]])
      "ttkernel.mul_tiles"(%cb0, %cb1, %cb0_index, %cb1_index, %dst_index) : (!cb0_tiles, !cb1_tiles, i32, i32, i32) -> ()
      return
    }

    // CHECK-LABEL: func @mm_init
    func.func @mm_init() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[CB_A:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %cb_A = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[CB_B:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      %cb_B = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
      // CHECK: %[[CB_C:.*]] = emitc.literal "get_compile_time_arg_val(2)"
      %cb_C = "ttkernel.get_compile_time_arg_val"() <{arg_index = 2 : i32}> : () -> !cb2_tiles
      // CHECK: %[[TRANSPOSE:.*]] = "emitc.constant"
      %transpose = arith.constant 0 : i32
      // CHECK: emitc.call_opaque "mm_init"(%[[CB_A]], %[[CB_B]], %[[CB_C]], %[[TRANSPOSE]])
      "ttkernel.mm_init"(%cb_A, %cb_B, %cb_C, %transpose) : (!cb0_tiles, !cb1_tiles, !cb2_tiles, i32) -> ()
      return
    }

    // CHECK-LABEL: func @mm_init_short
    func.func @mm_init_short() -> () attributes {ttkernerl.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[CB_A:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %cb_A = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[CB_B:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      %cb_B = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
      // CHECK: %[[TRANSPOSE:.*]] = "emitc.constant"
      %transpose = arith.constant 0 : i32
      // CHECK: emitc.call_opaque "mm_init_short"(%[[CB_A]], %[[CB_B]], %[[TRANSPOSE]])
      "ttkernel.mm_init_short"(%cb_A, %cb_B, %transpose) : (!cb0_tiles, !cb1_tiles, i32) -> ()
      return
    }

    // CHECK-LABEL: func @matmul_tiles
    func.func @matmul_tiles() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[CB_A:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %cb_A = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[CB_B:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      %cb_B = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
      // CHECK: %[[TRANSPOSE:.*]] = "emitc.constant"
      // CHECK: %[[CB_IDX_A:.*]] = "emitc.constant"
      // CHECK: %[[CB_IDX_B:.*]] = "emitc.constant"
      // CHECK: %[[CB_IDX_C:.*]] = "emitc.constant"
      %transpose = arith.constant 0 : i32
      %cb_idx_A = arith.constant 1 : i32
      %cb_idx_B = arith.constant 2 : i32
      %cb_idx_C = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "matmul_tiles"(%[[CB_A]], %[[CB_B]], %[[CB_IDX_A]], %[[CB_IDX_B]], %[[CB_IDX_C]], %[[TRANSPOSE]])
      "ttkernel.matmul_tiles"(%cb_A, %cb_B, %cb_idx_A, %cb_idx_B, %cb_idx_C, %transpose) : (!cb0_tiles, !cb1_tiles, i32, i32, i32, i32) -> ()
      return
    }

    // CHECK-LABEL: func @matmul_block_init
    func.func @matmul_block_init() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[CB_A:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %cb_A = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[CB_B:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      %cb_B = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
      // CHECK: %[[CB_C:.*]] = emitc.literal "get_compile_time_arg_val(2)"
      %cb_C = "ttkernel.get_compile_time_arg_val"() <{arg_index = 2 : i32}> : () -> !cb2_tiles
      // CHECK: %[[TRANSPOSE:.*]] = "emitc.constant"
      // CHECK: %[[CT_DIM:.*]] = "emitc.constant"
      // CHECK: %[[RT_DIM:.*]] = "emitc.constant"
      // CHECK: %[[KT_DIM:.*]] = "emitc.constant"
      %transpose = arith.constant 0 : i32
      %ct_dim = arith.constant 1 : i32
      %rt_dim = arith.constant 2 : i32
      %kt_dim = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "mm_block_init"(%[[CB_A]], %[[CB_B]], %[[CB_C]], %[[TRANSPOSE]], %[[CT_DIM]], %[[RT_DIM]], %[[KT_DIM]])
      "ttkernel.mm_block_init"(%cb_A, %cb_B, %cb_C, %transpose, %ct_dim, %rt_dim, %kt_dim) : (!cb0_tiles, !cb1_tiles, !cb2_tiles, i32, i32, i32, i32) -> ()
      return
    }

    // CHECK-LABEL: func @matmul_block_init_short
    func.func @matmul_block_init_short() -> () attributes {ttkernerl.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[CB_A:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %cb_A = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[CB_B:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      %cb_B = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
      // CHECK: %[[TRANSPOSE:.*]] = "emitc.constant"
      // CHECK: %[[CT_DIM:.*]] = "emitc.constant"
      // CHECK: %[[RT_DIM:.*]] = "emitc.constant"
      // CHECK: %[[KT_DIM:.*]] = "emitc.constant"
      %transpose = arith.constant 0 : i32
      %ct_dim = arith.constant 1 : i32
      %rt_dim = arith.constant 2 : i32
      %kt_dim = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "mm_block_init_short"(%[[CB_A]], %[[CB_B]], %[[TRANSPOSE]], %[[CT_DIM]], %[[RT_DIM]], %[[KT_DIM]])
      "ttkernel.mm_block_init_short"(%cb_A, %cb_B, %transpose, %ct_dim, %rt_dim, %kt_dim) : (!cb0_tiles, !cb1_tiles, i32, i32, i32, i32) -> ()
      return
    }

    // CHECK-LABEL: func @matmul_block
    func.func @matmul_block() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[CB_A:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %cb_A = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[CB_B:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      %cb_B = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
      // CHECK: %[[TRANSPOSE:.*]] = "emitc.constant"
      // CHECK: %[[IN0_TILE_INDEX:.*]] = "emitc.constant"
      // CHECK: %[[IN1_TILE_INDEX:.*]] = "emitc.constant"
      // CHECK: %[[DST_TILE_INDEX:.*]] = "emitc.constant"
      // CHECK: %[[CT_DIM:.*]] = "emitc.constant"
      // CHECK: %[[RT_DIM:.*]] = "emitc.constant"
      // CHECK: %[[KT_DIM:.*]] = "emitc.constant"
      // CHECK: %[[NT_DIM:.*]] = "emitc.constant"
      %transpose = arith.constant 0 : i32
      %in0_tile_index = arith.constant 1 : i32
      %in1_tile_index = arith.constant 2 : i32
      %dst_tile_index = arith.constant 0 : i32
      %ct_dim = arith.constant 2 : i32
      %rt_dim = arith.constant 2 : i32
      %kt_dim = arith.constant 2 : i32
      %nt_dim = arith.constant 2 : i32
      // CHECK: emitc.call_opaque "experimental::matmul_block"(%[[CB_A]], %[[CB_B]], %[[IN0_TILE_INDEX]], %[[IN1_TILE_INDEX]], %[[DST_TILE_INDEX]], %[[TRANSPOSE]], %[[CT_DIM]], %[[RT_DIM]], %[[KT_DIM]], %[[NT_DIM]])
      "ttkernel.experimental::matmul_block"(%cb_A, %cb_B, %in0_tile_index, %in1_tile_index, %dst_tile_index, %transpose, %ct_dim, %rt_dim, %kt_dim, %nt_dim) : (!cb0_tiles, !cb1_tiles, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      return
    }

    // CHECK-LABEL: func @reduce_init
    func.func @reduce_init() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[IN_CB:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %in_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[SCALING_CB:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      %scaling_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
      // CHECK: %[[OUT_CB:.*]] = emitc.literal "get_compile_time_arg_val(2)"
      %out_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 2 : i32}> : () -> !cb2_tiles
      // CHECK: emitc.call_opaque "reduce_init"(%[[IN_CB]], %[[SCALING_CB]], %[[OUT_CB]]) {template_args = [#emitc.opaque<"PoolType::SUM">, #emitc.opaque<"ReduceDim::REDUCE_SCALAR">, #emitc.opaque<"false">]}
      "ttkernel.reduce_init"(%in_cb, %scaling_cb, %out_cb) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_scalar>, reduce_type = #ttkernel.reduce_type<reduce_sum>}> : (!cb0_tiles, !cb1_tiles, !cb2_tiles) -> ()
      return
    }

    // CHECK-LABEL: func @reduce_tile
    func.func @reduce_tile() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[IN_CB:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %in_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[SCALING_CB:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      %scaling_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
      // CHECK: %[[IN_TILE_INDEX:.*]] = "emitc.constant"
      %in_tile_index = arith.constant 1 : i32
      // CHECK: %[[SCALING_TILE_INDEX:.*]] = "emitc.constant"
      %scaling_tile_index = arith.constant 2 : i32
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "reduce_tile"(%[[IN_CB]], %[[SCALING_CB]],  %[[IN_TILE_INDEX]], %[[SCALING_TILE_INDEX]], %[[DST_INDEX]]) {template_args = [#emitc.opaque<"PoolType::MAX">, #emitc.opaque<"ReduceDim::REDUCE_ROW">, #emitc.opaque<"false">]}
      "ttkernel.reduce_tile"(%in_cb, %scaling_cb, %in_tile_index, %scaling_tile_index, %dst_index) <{
        reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_max>
        }> : (!cb0_tiles, !cb1_tiles, i32, i32, i32) -> ()
      return
    }

  } // module

  //===----------------------------------------------------------------------===//
  // TTKernel SFPU operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: ttkernel_sfpu_operations
  module @ttkernel_sfpu_operations {

    // CHECK-LABEL: func @init_sfpu
    func.func @init_sfpu() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[IN_CB:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %in_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[OUT_CB:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      %out_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
      // CHECK: emitc.call_opaque "init_sfpu"(%[[IN_CB]], %[[OUT_CB]])
      "ttkernel.init_sfpu"(%in_cb, %out_cb) : (!cb0_tiles, !cb1_tiles) -> ()
      return
    }

    // CHECK-LABEL: func @max_tile_init
    func.func @max_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "max_tile_init"()
      "ttkernel.max_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @max_tile
    func.func @max_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: %[[DST1_INDEX:.*]] = "emitc.constant"
      %dst1_index = arith.constant 2 : i32
      // CHECK: emitc.call_opaque "max_tile"(%[[DST0_INDEX]], %[[DST1_INDEX]])
      "ttkernel.max_tile"(%dst0_index, %dst1_index) : (i32, i32) -> ()
      return
    }

    // CHECK-LABEL: func.func @add_binary_tile_init
    func.func @add_binary_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "add_binary_tile_init"()
      "ttkernel.add_binary_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func.func @test_add_binary_tile
    func.func @test_add_binary_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      // CHECK: %[[DST1_INDEX:.*]] = "emitc.constant"
      // CHECK: %[[ODST_INDEX:.*]] = "emitc.constant"
      // CHECK: emitc.call_opaque "add_binary_tile"(%[[DST0_INDEX]], %[[DST1_INDEX]], %[[ODST_INDEX]])
      "ttkernel.add_binary_tile"(%c0, %c1, %c2) : (index, index, index) -> ()
      return
    }

    // CHECK-LABEL: func.func @mul_binary_tile_init
    func.func @mul_binary_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "mul_binary_tile_init"()
      "ttkernel.mul_binary_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func.func @test_mul_binary_tile
    func.func @test_mul_binary_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      // CHECK: %[[DST1_INDEX:.*]] = "emitc.constant"
      // CHECK: %[[ODST_INDEX:.*]] = "emitc.constant"
      // CHECK: emitc.call_opaque "mul_binary_tile"(%[[DST0_INDEX]], %[[DST1_INDEX]], %[[ODST_INDEX]])
      "ttkernel.mul_binary_tile"(%c0, %c1, %c2) : (index, index, index) -> ()
      return
    }

    // CHECK-LABEL: func.func @sub_binary_tile_init
    func.func @sub_binary_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "sub_binary_tile_init"()
      "ttkernel.sub_binary_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func.func @test_sub_binary_tile
    func.func @test_sub_binary_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      // CHECK: %[[DST1_INDEX:.*]] = "emitc.constant"
      // CHECK: %[[ODST_INDEX:.*]] = "emitc.constant"
      // CHECK: emitc.call_opaque "sub_binary_tile"(%[[DST0_INDEX]], %[[DST1_INDEX]], %[[ODST_INDEX]])
      "ttkernel.sub_binary_tile"(%c0, %c1, %c2) : (index, index, index) -> ()
      return
    }

    // CHECK-LABEL: func.func @copy_dest_values_init
    func.func @copy_dest_values_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "copy_dest_values_init"()
      "ttkernel.copy_dest_values_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func.func @test_copy_dest_values
    func.func @test_copy_dest_values() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      // CHECK: %[[DST1_INDEX:.*]] = "emitc.constant"
      // CHECK: emitc.call_opaque "copy_dest_values"(%[[DST0_INDEX]], %[[DST1_INDEX]])
      "ttkernel.copy_dest_values"(%c0, %c1) : (index, index) -> ()
      return
    }

    // CHECK-LABEL: func @div_binary_tile_init
    func.func @div_binary_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "div_binary_tile_init"()
      "ttkernel.div_binary_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @div_binary_tile
    func.func @div_binary_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: %[[DST1_INDEX:.*]] = "emitc.constant"
      %dst1_index = arith.constant 2 : i32
      // CHECK: %[[ODST_INDEX:.*]] = "emitc.constant"
      %odst_index = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "div_binary_tile"(%[[DST0_INDEX]], %[[DST1_INDEX]], %[[ODST_INDEX]])
      "ttkernel.div_binary_tile"(%dst0_index, %dst1_index, %odst_index) : (i32, i32, i32) -> ()
      return
    }

    // CHECK-LABEL: func @recip_tile_init
    func.func @recip_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "recip_tile_init"()
      "ttkernel.recip_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @recip_tile
    func.func @recip_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "recip_tile"(%[[DST_INDEX]])
      "ttkernel.recip_tile"(%dst_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @power_binary_tile_init
    func.func @power_binary_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "power_binary_tile_init"()
      "ttkernel.power_binary_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @power_binary_tile
    func.func @power_binary_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: %[[DST1_INDEX:.*]] = "emitc.constant"
      %dst1_index = arith.constant 2 : i32
      // CHECK: %[[ODST_INDEX:.*]] = "emitc.constant"
      %odst_index = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "power_binary_tile"(%[[DST0_INDEX]], %[[DST1_INDEX]], %[[ODST_INDEX]])
      "ttkernel.power_binary_tile"(%dst0_index, %dst1_index, %odst_index) : (i32, i32, i32) -> ()
      return
    }

    // CHECK-LABEL: func @exp_tile_init
    func.func @exp_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "exp_tile_init"()
      "ttkernel.exp_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @exp_tile
    func.func @exp_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "exp_tile"(%[[DST_INDEX]])
      "ttkernel.exp_tile"(%dst_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @log_tile_init
    func.func @log_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "log_tile_init"()
      "ttkernel.log_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @log_tile
    func.func @log_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "log_tile"(%[[DST_INDEX]])
      "ttkernel.log_tile"(%dst_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @negative_tile_init
    func.func @negative_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "negative_tile_init"()
      "ttkernel.negative_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @negative_tile
    func.func @negative_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "negative_tile"(%[[DST_INDEX]])
      "ttkernel.negative_tile"(%dst_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @cos_tile_init
    func.func @cos_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "cos_tile_init"()
      "ttkernel.cos_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @cos_tile
    func.func @cos_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "cos_tile"(%[[DST_INDEX]])
      "ttkernel.cos_tile"(%dst_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @tan_tile_init
    func.func @tan_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "tan_tile_init"()
      "ttkernel.tan_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @tan_tile
    func.func @tan_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "tan_tile"(%[[DST_INDEX]])
      "ttkernel.tan_tile"(%dst_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @sqrt_tile_init
    func.func @sqrt_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "sqrt_tile_init"()
      "ttkernel.sqrt_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @sqrt_tile
    func.func @sqrt_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "sqrt_tile"(%[[DST_INDEX]])
      "ttkernel.sqrt_tile"(%dst_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @rsqrt_tile_init
    func.func @rsqrt_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "rsqrt_tile_init"()
      "ttkernel.rsqrt_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @rsqrt_tile
    func.func @rsqrt_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "rsqrt_tile"(%[[DST_INDEX]])
      "ttkernel.rsqrt_tile"(%dst_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @sin_tile_init
    func.func @sin_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "sin_tile_init"()
      "ttkernel.sin_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @sin_tile
    func.func @sin_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "sin_tile"(%[[DST0_INDEX]])
      "ttkernel.sin_tile"(%dst0_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @sigmoid_tile_init
    func.func @sigmoid_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "sigmoid_tile_init"()
      "ttkernel.sigmoid_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @sigmoid_tile
    func.func @sigmoid_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "sigmoid_tile"(%[[DST_INDEX]])
      "ttkernel.sigmoid_tile"(%dst_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @rounding_op_tile_init
    func.func @rounding_op_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "rounding_op_tile_init"()
      "ttkernel.rounding_op_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @ceil_tile
    func.func @ceil_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "ceil_tile"(%[[DST0_INDEX]])
      "ttkernel.ceil_tile"(%dst0_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @ceil_tile_float32
    func.func @ceil_tile_float32() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "ceil_tile_float32"(%[[DST0_INDEX]])
      "ttkernel.ceil_tile_float32"(%dst0_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @floor_tile
    func.func @floor_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "floor_tile"(%[[DST0_INDEX]])
      "ttkernel.floor_tile"(%dst0_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @floor_tile_float32
    func.func @floor_tile_float32() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "floor_tile_float32"(%[[DST0_INDEX]])
      "ttkernel.floor_tile_float32"(%dst0_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @abs_tile_init
    func.func @abs_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "abs_tile_init"()
      "ttkernel.abs_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @abs_tile
    func.func @abs_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "abs_tile"(%[[DST0_INDEX]])
      "ttkernel.abs_tile"(%dst0_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @abs_tile_int32
    func.func @abs_tile_int32() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "abs_tile_int32"(%[[DST0_INDEX]])
      "ttkernel.abs_tile_int32"(%dst0_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @logical_not_unary_tile_init
    func.func @logical_not_unary_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "logical_not_unary_tile_init"()
      "ttkernel.logical_not_unary_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @logical_not_unary_tile
    func.func @logical_not_unary_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "logical_not_unary_tile"(%[[DST0_INDEX]])
      "ttkernel.logical_not_unary_tile"(%dst0_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @logical_not_unary_tile_int32
    func.func @logical_not_unary_tile_int32() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "logical_not_unary_tile_int32"(%[[DST0_INDEX]])
      "ttkernel.logical_not_unary_tile_int32"(%dst0_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @fill_tile_init
    func.func @fill_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "fill_tile_init"()
      "ttkernel.fill_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @fill_tile
    func.func @fill_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : i32
      // CHECK: %[[VAL:.*]] = "emitc.constant"
      %val = arith.constant 1.0 : f32
      // CHECK: emitc.call_opaque "fill_tile"(%[[DST_INDEX]], %[[VAL]])
      "ttkernel.fill_tile"(%dst_index, %val) : (i32, f32) -> ()
      return
    }

    // CHECK-LABEL: func @eqz_tile_init
    func.func @eqz_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "eqz_tile_init"()
      "ttkernel.eqz_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @eqz_tile
    func.func @eqz_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "eqz_tile"(%[[DST0_INDEX]])
      "ttkernel.eqz_tile"(%dst0_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @nez_tile_init
    func.func @nez_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "nez_tile_init"()
      "ttkernel.nez_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @nez_tile
    func.func @nez_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "nez_tile"(%[[DST0_INDEX]])
      "ttkernel.nez_tile"(%dst0_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @gtz_tile_init
    func.func @gtz_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "gtz_tile_init"()
      "ttkernel.gtz_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @gtz_tile
    func.func @gtz_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "gtz_tile"(%[[DST0_INDEX]])
      "ttkernel.gtz_tile"(%dst0_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @gez_tile_init
    func.func @gez_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "gez_tile_init"()
      "ttkernel.gez_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @gez_tile
    func.func @gez_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "gez_tile"(%[[DST0_INDEX]])
      "ttkernel.gez_tile"(%dst0_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @ltz_tile_init
    func.func @ltz_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "ltz_tile_init"()
      "ttkernel.ltz_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @ltz_tile
    func.func @ltz_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "ltz_tile"(%[[DST0_INDEX]])
      "ttkernel.ltz_tile"(%dst0_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @lez_tile_init
    func.func @lez_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: emitc.call_opaque "lez_tile_init"()
      "ttkernel.lez_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @lez_tile
    func.func @lez_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "lez_tile"(%[[DST0_INDEX]])
      "ttkernel.lez_tile"(%dst0_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @eqz_tile_int32
    func.func @eqz_tile_int32() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "eqz_tile_int32"(%[[DST0_INDEX]])
      "ttkernel.eqz_tile_int32"(%dst0_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @nez_tile_int32
    func.func @nez_tile_int32() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "nez_tile_int32"(%[[DST0_INDEX]])
      "ttkernel.nez_tile_int32"(%dst0_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @gtz_tile_int32
    func.func @gtz_tile_int32() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "gtz_tile_int32"(%[[DST0_INDEX]])
      "ttkernel.gtz_tile_int32"(%dst0_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @gez_tile_int32
    func.func @gez_tile_int32() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "gez_tile_int32"(%[[DST0_INDEX]])
      "ttkernel.gez_tile_int32"(%dst0_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @ltz_tile_int32
    func.func @ltz_tile_int32() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "ltz_tile_int32"(%[[DST0_INDEX]])
      "ttkernel.ltz_tile_int32"(%dst0_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @lez_tile_int32
    func.func @lez_tile_int32() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "lez_tile_int32"(%[[DST0_INDEX]])
      "ttkernel.lez_tile_int32"(%dst0_index) : (i32) -> ()
      return
    }
  } // module

  //===----------------------------------------------------------------------===//
  // TTKernel CB operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: ttkernel_cb_operations
  module @ttkernel_cb_operations {

    // CHECK-LABEL: func @cb_push_back
    func.func @cb_push_back() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[CB:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[NUM_PAGES:.*]] = "emitc.constant"
      %num_pages = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "cb_push_back"(%[[CB]], %[[NUM_PAGES]])
      "ttkernel.cb_push_back"(%cb, %num_pages) : (!cb0_tiles, i32) -> ()
      return
    }

    // CHECK-LABEL: func @cb_pop_front
    func.func @cb_pop_front() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[CB:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[NUM_PAGES:.*]] = "emitc.constant"
      %num_pages = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "cb_pop_front"(%[[CB]], %[[NUM_PAGES]])
      "ttkernel.cb_pop_front"(%cb, %num_pages) : (!cb0_tiles, i32) -> ()
      return
    }

    // CHECK-LABEL: func @cb_reserve_back
    func.func @cb_reserve_back() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[CB:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[NUM_PAGES:.*]] = "emitc.constant"
      %num_pages = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "cb_reserve_back"(%[[CB]], %[[NUM_PAGES]])
      "ttkernel.cb_reserve_back"(%cb, %num_pages) : (!cb0_tiles, i32) -> ()
      return
    }

    // CHECK-LABEL: func @cb_wait_front
    func.func @cb_wait_front() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[CB:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[NUM_PAGES:.*]] = "emitc.constant"
      %num_pages = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "cb_wait_front"(%[[CB]], %[[NUM_PAGES]])
      "ttkernel.cb_wait_front"(%cb, %num_pages) : (!cb0_tiles, i32) -> ()
      return
    }

  } // module

  //===----------------------------------------------------------------------===//
  // TTKernel Tile operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: ttkernel_tile_operations
  module @ttkernel_tile_operations {

    // CHECK-LABEL: func @tilize_init
    func.func @tilize_init() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[IN_CB:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %in_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_scalar
      // CHECK: %[[OUT_CB:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      %out_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
      // CHECK: %[[NUM_TILES:.*]] = "emitc.constant"
      %num_tiles = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "tilize_init"(%[[IN_CB]], %[[NUM_TILES]], %[[OUT_CB]])
      "ttkernel.tilize_init"(%in_cb, %num_tiles, %out_cb) : (!cb0_scalar, i32, !cb1_tiles) -> ()
      return
    }


    // CHECK-LABEL: func @tilize_uninit
    func.func @tilize_uninit() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[IN_CB:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %in_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_scalar
      // CHECK: %[[OUT_CB:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      %out_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
      // CHECK: emitc.call_opaque "tilize_uninit"(%[[IN_CB]], %[[OUT_CB]])
      "ttkernel.tilize_uninit"(%in_cb, %out_cb) : (!cb0_scalar, !cb1_tiles) -> ()
      return
    }

    // CHECK-LABEL: func @untilize_init
    func.func @untilize_init() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[IN_CB:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %in_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: emitc.call_opaque "untilize_init"(%[[IN_CB]])
      "ttkernel.untilize_init"(%in_cb) : (!cb0_tiles) -> ()
      return
    }

    // CHECK-LABEL: func @untilize_uninit
    func.func @untilize_uninit() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[IN_CB:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %in_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: emitc.call_opaque "untilize_uninit"(%[[IN_CB]])
      "ttkernel.untilize_uninit"(%in_cb) : (!cb0_tiles) -> ()
      return
    }

    // CHECK-LABEL: func @tilize_block
    func.func @tilize_block() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[IN_CB:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %in_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_scalar
      // CHECK: %[[OUT_CB:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      %out_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_tiles
      // CHECK: %[[NUM_TILES:.*]] = "emitc.constant"
      %num_tiles = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "tilize_block"(%[[IN_CB]], %[[NUM_TILES]], %[[OUT_CB]])
      "ttkernel.tilize_block"(%in_cb, %num_tiles, %out_cb) : (!cb0_scalar, i32, !cb1_tiles) -> ()
      return
    }

    // CHECK-LABEL: func @untilize_block
    func.func @untilize_block() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
      // CHECK: %[[IN_CB:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %in_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[OUT_CB:.*]] = emitc.literal "get_compile_time_arg_val(1)"
      %out_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1_scalar
      // CHECK: %[[NUM_TILES:.*]] = "emitc.constant"
      %num_tiles = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "untilize_block"(%[[IN_CB]], %[[NUM_TILES]], %[[OUT_CB]])
      "ttkernel.untilize_block"(%in_cb, %num_tiles, %out_cb) : (!cb0_tiles, i32, !cb1_scalar) -> ()
      return
    }

  } // module

  //===----------------------------------------------------------------------===//
  // TTKernel NOC operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: ttkernel_noc_operations
  module @ttkernel_noc_operations {

    // CHECK-LABEL: func @get_noc_addr
    func.func @get_noc_addr() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
      // CHECK: %[[X:.*]] = "emitc.constant"
      %x = arith.constant 1 : index
      // CHECK: %[[Y:.*]] = "emitc.constant"
      %y = arith.constant 2 : index
      // CHECK: %[[ADDR:.*]] = "emitc.constant"
      %addr = arith.constant 262400 : i32
      // note: ttkernel.get_noc_addr() converts to one of metallium get_noc_addr() overloads
      // CHECK: emitc.call_opaque "get_noc_addr"(%[[X]], %[[Y]], %[[ADDR]])
      "ttkernel.get_noc_addr"(%x, %y, %addr) : (index, index, i32) -> (!ttkernel.noc_addr)
      return
    }

    // CHECK-LABEL: func @get_noc_addr_from_bank_id
    func.func @get_noc_addr_from_bank_id() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
      // CHECK: %[[BANK_ID:.*]] = "emitc.constant"
      %bank_id = arith.constant 1 : i32
      // CHECK: %[[ADDR_OFFSET:.*]] = "emitc.constant"
      %addr_offset = arith.constant 262400 : i32
      // CHECK: emitc.call_opaque "get_noc_addr_from_bank_id"(%[[BANK_ID]], %[[ADDR_OFFSET]])
      "ttkernel.get_noc_addr_from_bank_id"(%bank_id, %addr_offset) : (i32, i32) -> (!ttkernel.noc_addr)
      return
    }

    // CHECK-LABEL: func @noc_async_read
    func.func @noc_async_read() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
      // CHECK: %[[SRC_ADDR:.*]] = emitc.call_opaque "get_noc_addr"
      %x = arith.constant 1 : index
      %y = arith.constant 1 : index
      %temp = arith.constant 262400 : i32
      %src_addr = "ttkernel.get_noc_addr"(%x, %y, %temp) : (index, index, i32) -> (!ttkernel.noc_addr)
      // CHECK: %[[DST_ADDR:.*]] = "emitc.constant"
      %dst_addr = arith.constant 303104 : i32
      // CHECK: %[[SIZE:.*]] = "emitc.constant"
      %size = arith.constant 2048 : i32
      // CHECK: emitc.call_opaque "noc_async_read"(%[[SRC_ADDR]], %[[DST_ADDR]], %[[SIZE]])
      "ttkernel.noc_async_read"(%src_addr, %dst_addr, %size) : (!ttkernel.noc_addr, i32, i32) -> ()
      return
    }

    // CHECK-LABEL: func @noc_async_read_one_packet_set_state
    func.func @noc_async_read_one_packet_set_state() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
      // CHECK: %[[SRC_ADDR:.*]] = emitc.call_opaque "get_noc_addr"
      %x = arith.constant 1 : index
      %y = arith.constant 1 : index
      %temp = arith.constant 262400 : i32
      %src_addr = "ttkernel.get_noc_addr"(%x, %y, %temp) : (index, index, i32) -> (!ttkernel.noc_addr)
      // CHECK: %[[SIZE:.*]] = "emitc.constant"
      %size = arith.constant 2048 : i32
      // CHECK: emitc.call_opaque "noc_async_read_one_packet_set_state"(%[[SRC_ADDR]], %[[SIZE]])
      "ttkernel.noc_async_read_one_packet_set_state"(%src_addr, %size) : (!ttkernel.noc_addr, i32) -> ()
      return
    }

    // CHECK-LABEL: func @noc_async_read_one_packet_with_state
    func.func @noc_async_read_one_packet_with_state() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
      // CHECK: %[[SRC_ADDR:.*]] = emitc.call_opaque "get_noc_addr"
      %x = arith.constant 1 : index
      %y = arith.constant 1 : index
      %temp = arith.constant 262400 : i32
      %src_addr = "ttkernel.get_noc_addr"(%x, %y, %temp) : (index, index, i32) -> (!ttkernel.noc_addr)
      // CHECK: %[[DST_ADDR:.*]] = "emitc.constant"
      %dst_addr = arith.constant 327680 : i32
      // CHECK: emitc.call_opaque "noc_async_read_one_packet_with_state"(%[[SRC_ADDR]], %[[DST_ADDR]])
      "ttkernel.noc_async_read_one_packet_with_state"(%src_addr, %dst_addr) : (!ttkernel.noc_addr, i32) -> ()
      // TODO: test %dst_addr of type TTKernel_L1Addr?
      return
    }

    // CHECK-LABEL: func @noc_async_read_barrier
    func.func @noc_async_read_barrier() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
      // CHECK: emitc.call_opaque "noc_async_read_barrier"()
      "ttkernel.noc_async_read_barrier"() : () -> ()
      return
    }

    // CHECK-LABEL: func @noc_async_write
    func.func @noc_async_write() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
      // CHECK: %[[SRC_ADDR:.*]] = "emitc.constant"
      %src_addr = arith.constant 303104 : i32
      // CHECK: %[[DST_ADDR:.*]] = emitc.call_opaque "get_noc_addr"
      %x = arith.constant 1 : index
      %y = arith.constant 1 : index
      %temp = arith.constant 262400 : i32
      %dst_addr = "ttkernel.get_noc_addr"(%x, %y, %temp) : (index, index, i32) -> (!ttkernel.noc_addr)
      // CHECK: %[[SIZE:.*]] = "emitc.constant"
      %size = arith.constant 2048 : i32
      // CHECK: emitc.call_opaque "noc_async_write"(%[[SRC_ADDR]], %[[DST_ADDR]], %[[SIZE]])
      "ttkernel.noc_async_write"(%src_addr, %dst_addr, %size) : (i32, !ttkernel.noc_addr, i32) -> ()
      return
    }

    // CHECK-LABEL: func @noc_async_write_barrier
    func.func @noc_async_write_barrier() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
      // CHECK: emitc.call_opaque "noc_async_write_barrier"()
      "ttkernel.noc_async_write_barrier"() : () -> ()
      return
    }

    // CHECK-LABEL: func @get_semaphore
    func.func @get_semaphore() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
      // CHECK: %[[SEMAPHORE_ID:.*]] = "emitc.constant"
      %semaphore_id = arith.constant 2 : i32
      // CHECK: emitc.call_opaque "get_semaphore"(%[[SEMAPHORE_ID]])
      "ttkernel.get_semaphore"(%semaphore_id) : (i32) -> (!ttkernel.semaphore)
      return
    }

    // CHECK-LABEL: func @noc_semaphore_inc
    func.func @noc_semaphore_inc() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
      // CHECK: %[[ADDR:.*]] = emitc.call_opaque "get_noc_addr"
      %x = arith.constant 1 : index
      %y = arith.constant 1 : index
      %temp = arith.constant 262400 : i32
      %addr = "ttkernel.get_noc_addr"(%x, %y, %temp) : (index, index, i32) -> (!ttkernel.noc_addr)
      // CHECK: %[[INCR:.*]] = "emitc.constant"
      %incr = arith.constant 1 : i32
      // CHECK: %[[NOC_ID:.*]] = "emitc.constant"
      %noc_id = arith.constant 3 : i8
      // CHECK: emitc.call_opaque "noc_semaphore_inc"(%[[ADDR]], %[[INCR]], %[[NOC_ID]])
      "ttkernel.noc_semaphore_inc"(%addr, %incr, %noc_id) : (!ttkernel.noc_addr, i32, i8) -> ()
      return
    }

    // CHECK-LABEL: func @noc_semaphore_set
    func.func @noc_semaphore_set() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
      // CHECK: %[[ADDR:.*]] = emitc.call_opaque "reinterpret_cast
      %temp = arith.constant 262400 : i32
      %addr = "ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>"(%temp) : (i32) -> (!ttkernel.l1_addr_ptr) // a dummy l1 addr ptr
      // CHECK: %[[VAL:.*]] = "emitc.constant"
      %val = arith.constant 123 : i32
      // CHECK: emitc.call_opaque "noc_semaphore_set"(%[[ADDR]], %[[VAL]])
      "ttkernel.noc_semaphore_set"(%addr, %val) : (!ttkernel.l1_addr_ptr, i32) -> ()
      return
    }

    // CHECK-LABEL: func @noc_semaphore_wait
    func.func @noc_semaphore_wait() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
      // CHECK: %[[ADDR:.*]] = emitc.call_opaque "reinterpret_cast
      %temp = arith.constant 262400 : i32
      %addr = "ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>"(%temp) : (i32) -> (!ttkernel.l1_addr_ptr) // a dummy l1 addr ptr
      // CHECK: %[[VAL:.*]] = "emitc.constant"
      %val = arith.constant 123 : i32
      // CHECK: emitc.call_opaque "noc_semaphore_wait"(%[[ADDR]], %[[VAL]])
      "ttkernel.noc_semaphore_wait"(%addr, %val) : (!ttkernel.l1_addr_ptr, i32) -> ()
      return
    }

    // CHECK-LABEL: func @noc_semaphore_wait_min
    func.func @noc_semaphore_wait_min() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
      // CHECK: %[[ADDR:.*]] = emitc.call_opaque "reinterpret_cast
      %temp = arith.constant 262400 : i32
      %addr = "ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>"(%temp) : (i32) -> (!ttkernel.l1_addr_ptr) // a dummy l1 addr ptr
      // CHECK: %[[VAL:.*]] = "emitc.constant"
      %val = arith.constant 123 : i32
      // CHECK: emitc.call_opaque "noc_semaphore_wait_min"(%[[ADDR]], %[[VAL]])
      "ttkernel.noc_semaphore_wait_min"(%addr, %val) : (!ttkernel.l1_addr_ptr, i32) -> ()
      return
    }

    // CHECK-LABEL: func @noc_semaphore_set_multicast
    func.func @noc_semaphore_set_multicast() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
      // CHECK: %[[SRC_ADDR:.*]] = emitc.call_opaque "get_semaphore"
      %temp1 = arith.constant 2 : i32
      %src_addr = "ttkernel.get_semaphore"(%temp1) : (i32) -> (!ttkernel.semaphore) // a dummy l1 addr
      // CHECK: %[[DST_MCAST_ADDR:.*]] = emitc.call_opaque "get_noc_addr"
      %x = arith.constant 1 : index
      %y = arith.constant 1 : index
      %temp2 = arith.constant 262400 : i32
      %dst_mcast_addr = "ttkernel.get_noc_addr"(%x, %y, %temp2) : (index, index, i32) -> (!ttkernel.noc_addr) // dummy l1 addr (use mcast getter)
      // CHECK: %[[NUM_DSTS:.*]] = "emitc.constant"
      %num_dsts = arith.constant 8 : i32
      // TODO(#2229): emitc lowering ignores 'linked' and 'multicast_path_reserve'
      // CHECK: emitc.call_opaque "noc_semaphore_set_multicast"(%[[SRC_ADDR]], %[[DST_MCAST_ADDR]], %[[NUM_DSTS]])
      "ttkernel.noc_semaphore_set_multicast"(%src_addr, %dst_mcast_addr, %num_dsts) <{
          linked = false, multicast_path_reserve = true
        }> : (!ttkernel.semaphore, !ttkernel.noc_addr, i32) -> ()
      return
    }

    // CHECK-LABEL: func @noc_semaphore_set_multicast_loopback_src
    func.func @noc_semaphore_set_multicast_loopback_src() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
      // CHECK: %[[SRC_ADDR:.*]] = emitc.call_opaque "get_semaphore"
      %temp1 = arith.constant 2 : i32
      %src_addr = "ttkernel.get_semaphore"(%temp1) : (i32) -> (!ttkernel.semaphore) // a dummy l1 addr
      // CHECK: %[[DST_MCAST_ADDR:.*]] = emitc.call_opaque "get_noc_addr"
      %x = arith.constant 1 : index
      %y = arith.constant 1 : index
      %temp2 = arith.constant 303104 : i32
      %dst_mcast_addr = "ttkernel.get_noc_addr"(%x, %y, %temp2) : (index, index, i32) -> (!ttkernel.noc_addr) // dummy l1 addr (use mcast getter)
      // CHECK: %[[NUM_DSTS:.*]] = "emitc.constant"
      %num_dsts = arith.constant 8 : i32
      // TODO(#2229): emitc lowering ignores 'linked' and 'multicast_path_reserve'
      // CHECK: emitc.call_opaque "noc_semaphore_set_multicast_loopback_src"(%[[SRC_ADDR]], %[[DST_MCAST_ADDR]], %[[NUM_DSTS]])
      "ttkernel.noc_semaphore_set_multicast_loopback_src"(%src_addr, %dst_mcast_addr, %num_dsts) <{
          linked = false, multicast_path_reserve = true
        }> : (!ttkernel.semaphore, !ttkernel.noc_addr, i32) -> ()
      return
    }

    // CHECK-LABEL: func @interleaved_addr_gen_fast_funcs
    func.func @interleaved_addr_gen_fast_funcs() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<noc>} {
      // CHECK: %[[CB:.*]] = emitc.literal "get_compile_time_arg_val(0)"
      %cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0_tiles
      // CHECK: %[[DATA_FORMAT:.*]]= emitc.call_opaque "get_dataformat"
      %data_format = "ttkernel.get_dataformat"(%cb) : (!cb0_tiles) -> !ttkernel.DataFormat
      // CHECK: = "emitc.constant"() <{value = true}>
      // CHECK: %[[TEMP_ADDR:.*]] = "emitc.constant"()
      // CHECK: %[[TILE_SIZE:.*]] = "emitc.constant"()
      // CHECK: %[[TILE:.*]] = "emitc.constant"()
      %is_dram = arith.constant 1 : i1
      %temp_addr = arith.constant 262400 : i32
      %tile_size = arith.constant 8 : i32
      %tile = arith.constant 1 : i32
      // CHECK: %[[VAR:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"InterleavedAddrGenFast<true>">>
      // CHECK: "emitc.member"(%[[VAR]]) <{member = "bank_base_address"}>
      // CHECK: "emitc.member"(%[[VAR]]) <{member = "page_size"}>
      // CHECK: "emitc.member"(%[[VAR]]) <{member = "data_format"}>
      // CHECK: emitc.assign %[[TEMP_ADDR]]
      // CHECK: emitc.assign %[[TILE_SIZE]]
      // CHECK: emitc.assign %[[DATA_FORMAT]]
      // CHECK: %[[ADDR_GEN:.*]] = emitc.load %[[VAR]] : <!emitc.opaque<"InterleavedAddrGenFast<true>">>
      %s = "ttkernel.get_interleaved_addr_gen_fast"(%is_dram, %temp_addr, %tile_size, %data_format) : (i1, i32, i32, !ttkernel.DataFormat) -> !ttkernel.interleaved_addr_gen_fast
      // CHECK: emitc.call_opaque "noc_async_write_tile"(%[[TILE]], %[[ADDR_GEN]], %[[TEMP_ADDR]])
      "ttkernel.noc_async_write_tile"(%tile, %s, %temp_addr) : (i32, !ttkernel.interleaved_addr_gen_fast, i32) -> ()
      // CHECK: emitc.call_opaque "noc_async_read_tile"(%[[TILE]], %[[ADDR_GEN]], %[[TEMP_ADDR]])
      "ttkernel.noc_async_read_tile"(%tile, %s, %temp_addr) : (i32, !ttkernel.interleaved_addr_gen_fast, i32) -> ()
      return
    }

    // CHECK-LABEL: func @tensor_accessor
    func.func @tensor_accessor() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
      // CHECK: "emitc.constant"() <{value = [[CTA_OFFSET:.*]] : i32}>
      // CHECK: "emitc.constant"() <{value = [[CRTA_OFFSET:.*]] : i32}>
      %cta_offset = arith.constant 2 : i32
      %crta_offset = arith.constant 0 : i32
      // CHECK: %[[ADDR:.*]] = "emitc.constant"
      // CHECK: %[[SIZE:.*]] = "emitc.constant"
      %bank_address = arith.constant 303104 : i32
      %page_size = arith.constant 32 : i32
      // CHECK: %[[ARGS:.*]] = emitc.call_opaque "TensorAccessorArgs"() {template_args = [[[CTA_OFFSET]] : i32, [[CRTA_OFFSET]] : i32]} : () -> !emitc.opaque<"TensorAccessorArgs">
      %tensor_accessor_args = "ttkernel.TensorAccessorArgs"(%cta_offset, %crta_offset) : (i32, i32) -> !ttkernel.TensorAccessorArgs
      // CHECK: %[[TENSOR_ACCESSOR:.*]] = emitc.call_opaque "TensorAccessor"(%[[ARGS]], %[[ADDR]], %[[SIZE]]) : (!emitc.opaque<"TensorAccessorArgs">, i32, i32) -> !emitc.opaque<"TensorAccessor">
      %tensor_accessor = "ttkernel.TensorAccessor"(%tensor_accessor_args, %bank_address, %page_size) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
      %temp1 = arith.constant 0 : i32
      %temp2 = arith.constant 32: i32
      // CHECK: emitc.verbatim "uint32_t [[NOC_ADDR:.*]] = {}.get_noc_addr({}, {});" args %[[TENSOR_ACCESSOR]], {{.*}} : !emitc.opaque<"TensorAccessor">, i32, i32
      // CHECK: emitc.literal "[[NOC_ADDR]]" : i64
      %noc_addr = "ttkernel.tensor_accessor_get_noc_addr"(%tensor_accessor, %temp1, %temp2) : (!ttkernel.TensorAccessor, i32, i32) -> !ttkernel.noc_addr
      // CHECK: emitc.verbatim "uint32_t [[SHARD_ADDR:.*]] = {}.get_shard_noc_addr({}, {});" args %[[TENSOR_ACCESSOR]], {{.*}} : !emitc.opaque<"TensorAccessor">, i32, i32
      // CHECK: emitc.literal "[[SHARD_ADDR]]"
      %shard_noc_addr = "ttkernel.tensor_accessor_get_shard_noc_addr"(%tensor_accessor, %temp1, %temp2) : (!ttkernel.TensorAccessor, i32, i32) -> i32
      // CHECK: emitc.verbatim "uint32_t [[BANK_AND_OFFSET:.*]] = {}.get_bank_and_offset({});" args %[[TENSOR_ACCESSOR]], {{.*}} : !emitc.opaque<"TensorAccessor">, i32
      // CHECK: emitc.literal "[[BANK_AND_OFFSET]]" : !emitc.opaque<"PageMapping">
      %bank_and_offset = "ttkernel.tensor_accessor_get_bank_and_offset"(%tensor_accessor, %temp1) : (!ttkernel.TensorAccessor, i32) -> !ttkernel.PageMapping
      // CHECK: emitc.verbatim "uint32_t [[IS_LOCAL_BANK:.*]] = {}.is_local_bank({}, {});" args %[[TENSOR_ACCESSOR]], {{.*}} : !emitc.opaque<"TensorAccessor">, i32, i32
      // CHECK: emitc.literal "[[IS_LOCAL_BANK]]" : i1
      %is_local_bank = "ttkernel.tensor_accessor_is_local_bank"(%tensor_accessor, %temp1, %temp2) : (!ttkernel.TensorAccessor, i32, i32) -> i1
      // CHECK: emitc.verbatim "uint32_t [[IS_LOCAL_ADDR:.*]] = {}.is_local_addr({}, {});" args %[[TENSOR_ACCESSOR]], {{.*}} : !emitc.opaque<"TensorAccessor">, i32, i32
      // CHECK: emitc.literal "[[IS_LOCAL_ADDR]]" : i1
      %is_local_addr = "ttkernel.tensor_accessor_is_local_addr"(%tensor_accessor, %temp1, %temp2) : (!ttkernel.TensorAccessor, i32, i32) -> i1
      // CHECK: emitc.verbatim "uint32_t [[IS_LOCAL_PAGE:.*]] = {}.is_local_page({});" args %[[TENSOR_ACCESSOR]], {{.*}} : !emitc.opaque<"TensorAccessor">, i32
      // CHECK: emitc.literal "[[IS_LOCAL_PAGE]]" : i1
      %is_local_page = "ttkernel.tensor_accessor_is_local_page"(%tensor_accessor, %temp1) : (!ttkernel.TensorAccessor, i32) -> i1
      // CHECK: emitc.verbatim "uint32_t [[IS_LOCAL_SHARD:.*]] = {}.is_local_shard({});" args %[[TENSOR_ACCESSOR]], {{.*}} : !emitc.opaque<"TensorAccessor">, i32
      // CHECK: emitc.literal "[[IS_LOCAL_SHARD]]" : i1
      %is_local_shard = "ttkernel.tensor_accessor_is_local_shard"(%tensor_accessor, %temp1) : (!ttkernel.TensorAccessor, i32) -> i1
      return
    }


  } // module

} // module
