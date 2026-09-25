// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=0 mock-system-desc-arch=blackhole" -mlir-print-local-scope %s | FileCheck %s
// Test that a DRAM-interleaved matmul routed to 2D mcast gets the block that
// maximises in0_block_w * out_block_h * out_block_w, with the batch fused into
// M because B is unbatched.

// CHECK-LABEL: func.func @matmul_2d_transpose_b
func.func @matmul_2d_transpose_b(%arg0: tensor<1024x8192xbf16>, %arg1: tensor<3584x8192xbf16>) -> tensor<1024x3584xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 16, out_subblock_h = 4, out_subblock_w = 1, out_block_h = 4, out_block_w = 11, per_core_m = 4, per_core_n = 11, transpose_mcast = false, fuse_batch = true>
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = true}> : (tensor<1024x8192xbf16>, tensor<3584x8192xbf16>) -> tensor<1024x3584xbf16>
  return %0 : tensor<1024x3584xbf16>
}
