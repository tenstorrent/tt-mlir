// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=0 mock-system-desc-arch=blackhole" -mlir-print-local-scope %s | FileCheck %s
// Test that a wide matmul routed to 1D mcast_in0 keeps out_block_w =
// per_core_N, halves out_block_h since M >= 1024, and takes the largest
// in0_block_w that fits L1.

// CHECK-LABEL: func.func @matmul_1d_mcast_in0
func.func @matmul_1d_mcast_in0(%arg0: tensor<1024x8192xbf16>, %arg1: tensor<8192x16032xbf16>) -> tensor<1024x16032xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 8, out_subblock_h = 4, out_subblock_w = 1, out_block_h = 16, out_block_w = 5, per_core_m = 32, per_core_n = 5, fuse_batch = false, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x8192xbf16>, tensor<8192x16032xbf16>) -> tensor<1024x16032xbf16>
  return %0 : tensor<1024x16032xbf16>
}
