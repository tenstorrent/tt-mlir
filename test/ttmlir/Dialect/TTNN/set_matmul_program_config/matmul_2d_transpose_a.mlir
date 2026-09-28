// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=0 mock-system-desc-arch=blackhole" -mlir-print-local-scope %s | FileCheck %s
// Test that a 2D mcast matmul with transpose_a reads M and K from the
// transposed A and reserves L1 for the transposed in0 CB: without it the
// 4x11 block with in0_block_w = 16 would fit.

// CHECK-LABEL: func.func @matmul_2d_transpose_a
func.func @matmul_2d_transpose_a(%arg0: tensor<8192x1024xbf16>, %arg1: tensor<8192x3584xbf16>) -> tensor<1024x3584xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 16, out_subblock_h = 2, out_subblock_w = 1, out_block_h = 2, out_block_w = 11, per_core_m = 4, per_core_n = 11, transpose_mcast = false, fuse_batch = true>
  // CHECK-SAME: transpose_a = true
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = true, transpose_b = false}> : (tensor<8192x1024xbf16>, tensor<8192x3584xbf16>) -> tensor<1024x3584xbf16>
  return %0 : tensor<1024x3584xbf16>
}
