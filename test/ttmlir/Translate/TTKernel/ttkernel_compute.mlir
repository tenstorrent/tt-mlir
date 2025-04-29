// RUN: ttmlir-opt --convert-ttkernel-to-emitc %s | ttmlir-translate --ttkernel-to-cpp | FileCheck %s

#l1_ = #tt.memory_space<l1>

// CHECK: void kernel_main
func.func @ttkernel_compute(%arg1: !ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>,
                            %arg2: !ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>) -> () attributes {ttkernel.thread = #ttkernel.thread<compute>} {

    // CHECK: ::tt::CB [[CBIN0:.*]] = ::tt::CB::c_in0
    // CHECK: ::tt::CB [[CBOUT0:.*]] = ::tt::CB::c_out0
    // CHECK: int32_t [[C:.*]] = 4
    %c4_i32 = arith.constant 4 : i32
    // CHECK: untilize_init([[CBIN0]], [[CBOUT0]])
    "ttkernel.untilize_init"(%arg1, %arg2) : (!ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, !ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>) -> ()
    // CHECK: untilize_block([[CBIN0]], [[C]], [[CBOUT0]])
    "ttkernel.untilize_block"(%arg1, %c4_i32, %arg2) : (!ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>) -> ()
    // CHECK: cb_pop_front([[CBIN0]], [[C]])
    "ttkernel.cb_pop_front"(%arg1, %c4_i32) : (!ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
    // CHECK: cb_push_back([[CBOUT0]], [[C]])
    "ttkernel.cb_push_back"(%arg2, %c4_i32) : (!ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>, i32) -> ()
    // CHECK: untilize_block([[CBIN0]], [[C]], [[CBOUT0]])
    "ttkernel.untilize_block"(%arg1, %c4_i32, %arg2) : (!ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>) -> ()
    // CHECK: cb_pop_front([[CBIN0]], [[C]])
    "ttkernel.cb_pop_front"(%arg1, %c4_i32) : (!ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
    // CHECK: cb_push_back([[CBOUT0]], [[C]])
    "ttkernel.cb_push_back"(%arg2, %c4_i32) : (!ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>, i32) -> ()
    // CHECK: return
    func.return
}
