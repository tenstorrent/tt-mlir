// Verify that ttkernel.dfb_* ops emit experimental::DataflowBuffer
// method-call syntax (e.g. `dfb.reserve_back(n);`) via emitc.verbatim.

// RUN: ttmlir-opt --convert-ttkernel-to-emitc -o %t %s
// RUN: FileCheck %s --input-file=%t

!dfb_tiles = !ttkernel.dfb<8, !ttcore.tile<32x32, f32>, 1, 1>

module {
  // CHECK-LABEL: func.func @dfb_reserve_back
  func.func @dfb_reserve_back() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = dfb_id, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    // CHECK: %[[DFB:.*]] = emitc.literal "get_compile_time_arg_val(0)" {ttkernel.cb_ctarg_idx = 0 : i32} : !emitc.opaque<"experimental::DataflowBuffer">
    %dfb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !dfb_tiles
    %n = arith.constant 1 : i32
    // CHECK: emitc.verbatim "{}.reserve_back({});" args %[[DFB]],
    "ttkernel.dfb_reserve_back"(%dfb, %n) : (!dfb_tiles, i32) -> ()
    return
  }

  // CHECK-LABEL: func.func @dfb_push_back
  func.func @dfb_push_back() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = dfb_id, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    %dfb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !dfb_tiles
    %n = arith.constant 1 : i32
    // CHECK: emitc.verbatim "{}.push_back({});"
    "ttkernel.dfb_push_back"(%dfb, %n) : (!dfb_tiles, i32) -> ()
    return
  }

  // CHECK-LABEL: func.func @dfb_wait_front
  func.func @dfb_wait_front() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = dfb_id, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    %dfb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !dfb_tiles
    %n = arith.constant 1 : i32
    // CHECK: emitc.verbatim "{}.wait_front({});"
    "ttkernel.dfb_wait_front"(%dfb, %n) : (!dfb_tiles, i32) -> ()
    return
  }

  // CHECK-LABEL: func.func @dfb_pop_front
  func.func @dfb_pop_front() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = dfb_id, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    %dfb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !dfb_tiles
    %n = arith.constant 1 : i32
    // CHECK: emitc.verbatim "{}.pop_front({});"
    "ttkernel.dfb_pop_front"(%dfb, %n) : (!dfb_tiles, i32) -> ()
    return
  }

  // CHECK-LABEL: func.func @dfb_finish
  func.func @dfb_finish() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = dfb_id, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    %dfb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !dfb_tiles
    // CHECK: emitc.verbatim "{}.finish();"
    "ttkernel.dfb_finish"(%dfb) : (!dfb_tiles) -> ()
    return
  }
}
