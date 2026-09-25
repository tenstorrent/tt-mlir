// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=0 mock-system-desc-arch=blackhole" -mlir-print-local-scope %s | FileCheck %s
// Test that a tall matmul routed to 1D mcast_in1 keeps out_block_h =
// per_core_M and, since N < 1024, out_block_w = per_core_N.

// CHECK-LABEL: func.func @matmul_1d_mcast_in1
func.func @matmul_1d_mcast_in1(%arg0: tensor<16384x1024xbf16>, %arg1: tensor<1024x128xbf16>) -> tensor<16384x128xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 16, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 5, out_block_w = 4, per_core_m = 5, per_core_n = 4, fuse_batch = false, mcast_in0 = false, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<16384x1024xbf16>, tensor<1024x128xbf16>) -> tensor<16384x128xbf16>
  return %0 : tensor<16384x128xbf16>
}
