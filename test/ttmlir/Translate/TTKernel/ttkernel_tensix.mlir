// RUN: ttmlir-translate --tensixkernel-to-cpp %s
#l1_ = #tt.memory_space<l1>

func.func @ttkernel_tensix(%arg1: !ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>,
                            %arg2: !ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>) -> () {
    %c4_i32 = arith.constant 4 : i32
    // CHECK: emitc.call_opaque "untilize_init"[[C:.*]]
    "ttkernel.untilize_init"(%arg1, %arg2) : (!ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, !ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>) -> ()
    // CHECK: emitc.call_opaque "untilize_block"[[C:.*]]
    "ttkernel.untilize_block"(%arg1, %c4_i32, %arg2) : (!ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>) -> ()
    // CHECK: emitc.call_opaque "cb_pop_front"[[C:.*]]
    "ttkernel.cb_pop_front"(%arg1, %c4_i32) : (!ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
    // CHECK: emitc.call_opaque "cb_push_back"[[C:.*]]
    "ttkernel.cb_push_back"(%arg2, %c4_i32) : (!ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>, i32) -> ()
    // CHECK: emitc.call_opaque "untilize_block"[[C:.*]]
    "ttkernel.untilize_block"(%arg1, %c4_i32, %arg2) : (!ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>) -> ()
    // CHECK: emitc.call_opaque "cb_pop_front"[[C:.*]]
    "ttkernel.cb_pop_front"(%arg1, %c4_i32) : (!ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
    // CHECK: emitc.call_opaque "cb_push_back"[[C:.*]]
    "ttkernel.cb_push_back"(%arg2, %c4_i32) : (!ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>, i32) -> ()
    // CHECK: return
    "ttkernel.return"() : () -> ()
}
