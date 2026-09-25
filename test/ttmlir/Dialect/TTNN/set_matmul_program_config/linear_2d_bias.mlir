// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=0 mock-system-desc-arch=blackhole" -mlir-print-local-scope %s | FileCheck %s
// Test that ttnn.linear with a bias gets a 2D mcast program config, with L1
// reserved for the bias circular buffer.

// CHECK-LABEL: func.func @linear_2d_bias
func.func @linear_2d_bias(%arg0: tensor<1024x8192xbf16>, %arg1: tensor<8192x1024xbf16>, %arg2: tensor<1024xbf16>) -> tensor<1024x1024xbf16> {
  // CHECK: "ttnn.linear"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 32, out_subblock_h = 4, out_subblock_w = 1, out_block_h = 4, out_block_w = 3, per_core_m = 4, per_core_n = 3, transpose_mcast = false, fuse_batch = true>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x8192xbf16>, tensor<8192x1024xbf16>, tensor<1024xbf16>) -> tensor<1024x1024xbf16>
  return %0 : tensor<1024x1024xbf16>
}
