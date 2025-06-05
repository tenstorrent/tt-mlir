// RUN: ttmlir-opt %s | FileCheck %s

#l1_ = #tt.memory_space<l1>
// CHECK-LABEL: func @test_untilize_init_short
func.func @test_untilize_init_short(%tilized_cb : !ttkernel.cb<cb_in0, 0, memref<4x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> () {
  "ttkernel.untilize_init_short"(%tilized_cb) : (!ttkernel.cb<cb_in0, 0, memref<4x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> ()
  return
}

// CHECK-LABEL: func @test_tilize_init_short
func.func @test_tilize_init_short(%tilized_cb : !ttkernel.cb<cb_in0, 0, memref<4x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, %untilized_cb : !ttkernel.cb<cb_in0, 0, memref<128x128xf32, #l1_>, 512, 1>, %num_tiles : i32) -> () {
  "ttkernel.tilize_init_short"(%untilized_cb, %num_tiles, %tilized_cb) : (!ttkernel.cb<cb_in0, 0, memref<128x128xf32, #l1_>, 512, 1>, i32, !ttkernel.cb<cb_in0, 0, memref<4x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> ()
  return
}

// CHECK-LABEL: func.func @test_tilize_uninit
func.func @test_tilize_uninit(%tilized_cb : !ttkernel.cb<cb_in0, 0, memref<4x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, %untilized_cb : !ttkernel.cb<cb_in0, 0, memref<128x128xf32, #l1_>, 512, 1>) -> () {
  "ttkernel.tilize_uninit"(%untilized_cb, %tilized_cb) : (!ttkernel.cb<cb_in0, 0, memref<128x128xf32, #l1_>, 512, 1>, !ttkernel.cb<cb_in0, 0, memref<4x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> ()
  return
}

// CHECK-LABEL: func.func @test_untilize_uninit
func.func @test_untilize_uninit(%tilized_cb : !ttkernel.cb<cb_in0, 0, memref<4x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> () {
  "ttkernel.untilize_uninit"(%tilized_cb) : (!ttkernel.cb<cb_in0, 0, memref<4x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> ()
  return
}

// CHECK-LABEL: func.func @test_sub_tiles
func.func @test_sub_tiles(%lhs : !ttkernel.cb<cb_in0, 0, memref<4x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, %rhs : !ttkernel.cb<cb_in0, 0, memref<4x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, %lhs_idx : index, %rhs_idx : index, %dst_idx : index) -> () {
  "ttkernel.sub_tiles"(%lhs, %rhs, %lhs_idx, %rhs_idx, %dst_idx) : (!ttkernel.cb<cb_in0, 0, memref<4x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, !ttkernel.cb<cb_in0, 0, memref<4x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, index, index, index) -> ()
  return
}

// CHECK-LABEL: func.func @test_sub_tiles_init
func.func @test_sub_tiles_init(%lhs : !ttkernel.cb<cb_in0, 0, memref<4x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, %rhs : !ttkernel.cb<cb_in0, 0, memref<4x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> () {
  "ttkernel.sub_tiles_init"(%lhs, %rhs) : (!ttkernel.cb<cb_in0, 0, memref<4x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, !ttkernel.cb<cb_in0, 0, memref<4x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>) -> ()
  return
}
