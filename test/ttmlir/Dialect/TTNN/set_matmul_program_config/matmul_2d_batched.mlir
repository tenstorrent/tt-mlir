// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=0 mock-system-desc-arch=blackhole" -mlir-print-local-scope %s | FileCheck %s
// Test that a 2D mcast matmul with matching A and B batches keeps per-batch
// rows (fuse_batch = false).

// CHECK-LABEL: func.func @matmul_2d_batched
func.func @matmul_2d_batched(%arg0: tensor<8x1024x1024xbf16>, %arg1: tensor<8x1024x1024xbf16>) -> tensor<8x1024x1024xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 16, out_subblock_h = 4, out_subblock_w = 1, out_block_h = 4, out_block_w = 3, per_core_m = 4, per_core_n = 3, transpose_mcast = false, fuse_batch = false>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<8x1024x1024xbf16>, tensor<8x1024x1024xbf16>) -> tensor<8x1024x1024xbf16>
  return %0 : tensor<8x1024x1024xbf16>
}
