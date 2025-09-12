// RUN: ttmlir-opt --convert-ttkernel-to-emitc -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK-LABEL: func @gelu_tile_init
  func.func @gelu_tile_init() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    // CHECK: emitc.call_opaque "gelu_tile_init"()
    "ttkernel.gelu_tile_init"() : () -> ()
    return
  }

  // CHECK-LABEL: func @gelu_tile
  func.func @gelu_tile() -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
    %dst_index = arith.constant 3 : i32
    // CHECK: emitc.call_opaque "gelu_tile"(%[[DST_INDEX]])
    "ttkernel.gelu_tile"(%dst_index) : (i32) -> ()
    return
  }
}
