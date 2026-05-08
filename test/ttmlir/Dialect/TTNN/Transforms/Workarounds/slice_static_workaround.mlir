// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#layout_258112 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<6x8066x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_64 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<6x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module {
  // Test: output last dim (258112) causes circular buffer to exceed L1.
  func.func @slice_cb_exceeds_l1(
      %arg0: tensor<6x3x258112xf32, #layout_258112>)
      -> tensor<6x1x258112xf32, #layout_258112> {
    // CHECK-LABEL: func.func @slice_cb_exceeds_l1
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 2, 1>
    // CHECK: "ttnn.slice_static"
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 2 : i32]
    // CHECK-SAME: ends = [6 : i32, 258112 : i32, 3 : i32]
    // CHECK-SAME: step = [1 : i32, 1 : i32, 1 : i32]
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 2, 1>
    %0 = "ttnn.slice_static"(%arg0)
        <{begins = [0 : i32, 2 : i32, 0 : i32],
          ends   = [6 : i32, 3 : i32, 258112 : i32],
          step   = [1 : i32, 1 : i32, 1 : i32]}>
        : (tensor<6x3x258112xf32, #layout_258112>)
       -> tensor<6x1x258112xf32, #layout_258112>
    return %0 : tensor<6x1x258112xf32, #layout_258112>
  }

  // Test: output last dim (64) keeps circular buffer within L1 — no decomposition.
  func.func @slice_cb_fits_l1(
      %arg0: tensor<6x3x64xf32, #layout_64>)
      -> tensor<6x1x64xf32, #layout_64> {
    // CHECK-LABEL: func.func @slice_cb_fits_l1
    // CHECK-NOT: "ttnn.permute"
    // CHECK: "ttnn.slice_static"
    // CHECK-SAME: begins = [0 : i32, 2 : i32, 0 : i32]
    // CHECK-SAME: ends = [6 : i32, 3 : i32, 64 : i32]
    // CHECK-SAME: step = [1 : i32, 1 : i32, 1 : i32]
    %0 = "ttnn.slice_static"(%arg0)
        <{begins = [0 : i32, 2 : i32, 0 : i32],
          ends   = [6 : i32, 3 : i32, 64 : i32],
          step   = [1 : i32, 1 : i32, 1 : i32]}>
        : (tensor<6x3x64xf32, #layout_64>)
       -> tensor<6x1x64xf32, #layout_64>
    return %0 : tensor<6x1x64xf32, #layout_64>
  }
}
