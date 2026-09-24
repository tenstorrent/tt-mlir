// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mock-system-desc-arch=blackhole" -mlir-print-local-scope %s | FileCheck %s
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mock-system-desc-arch=blackhole enable-matmul-program-config=false" -mlir-print-local-scope %s | FileCheck %s --check-prefix=DISABLED

// DISABLED-NOT: matmul_program_config

module {
  // 2D mcast: maximise in0_block_w * out_block_h * out_block_w.
  // CHECK-LABEL: func.func @matmul_2d_transpose_b
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 16, out_subblock_h = 4, out_subblock_w = 1, out_block_h = 4, out_block_w = 11, per_core_m = 4, per_core_n = 11, transpose_mcast = false, fuse_batch = true>
  func.func @matmul_2d_transpose_b(%arg0: tensor<1024x8192xbf16>, %arg1: tensor<3584x8192xbf16>) -> tensor<1024x3584xbf16> {
    %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = true}> : (tensor<1024x8192xbf16>, tensor<3584x8192xbf16>) -> tensor<1024x3584xbf16>
    return %0 : tensor<1024x3584xbf16>
  }

  // Wide shape, 1D mcast_in0: out_block_w = per_core_N, out_block_h halved
  // since M >= 1024, then the largest in0_block_w that fits L1.
  // CHECK-LABEL: func.func @matmul_1d_mcast_in0
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 8, out_subblock_h = 4, out_subblock_w = 1, out_block_h = 16, out_block_w = 5, per_core_m = 32, per_core_n = 5, fuse_batch = false, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>
  func.func @matmul_1d_mcast_in0(%arg0: tensor<1024x8192xbf16>, %arg1: tensor<8192x16032xbf16>) -> tensor<1024x16032xbf16> {
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x8192xbf16>, tensor<8192x16032xbf16>) -> tensor<1024x16032xbf16>
    return %0 : tensor<1024x16032xbf16>
  }

  // Tall shape, 1D mcast_in1: out_block_h = per_core_M, N < 1024 keeps
  // out_block_w = per_core_N.
  // CHECK-LABEL: func.func @matmul_1d_mcast_in1
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 16, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 5, out_block_w = 4, per_core_m = 5, per_core_n = 4, fuse_batch = false, mcast_in0 = false
  func.func @matmul_1d_mcast_in1(%arg0: tensor<16384x1024xbf16>, %arg1: tensor<1024x128xbf16>) -> tensor<16384x128xbf16> {
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<16384x1024xbf16>, tensor<1024x128xbf16>) -> tensor<16384x128xbf16>
    return %0 : tensor<16384x128xbf16>
  }

  // CHECK-LABEL: func.func @linear_2d_bias
  // CHECK: "ttnn.linear"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 32, out_subblock_h = 4, out_subblock_w = 1, out_block_h = 4, out_block_w = 3, per_core_m = 4, per_core_n = 3, transpose_mcast = false, fuse_batch = true>
  func.func @linear_2d_bias(%arg0: tensor<1024x8192xbf16>, %arg1: tensor<8192x1024xbf16>, %arg2: tensor<1024xbf16>) -> tensor<1024x1024xbf16> {
    %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x8192xbf16>, tensor<8192x1024xbf16>, tensor<1024xbf16>) -> tensor<1024x1024xbf16>
    return %0 : tensor<1024x1024xbf16>
  }

  // Matching batches keep per-batch rows (fuse_batch = false).
  // CHECK-LABEL: func.func @matmul_2d_batched
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: per_core_m = 4, per_core_n = 3, transpose_mcast = false, fuse_batch = false>
  func.func @matmul_2d_batched(%arg0: tensor<8x1024x1024xbf16>, %arg1: tensor<8x1024x1024xbf16>) -> tensor<8x1024x1024xbf16> {
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<8x1024x1024xbf16>, tensor<8x1024x1024xbf16>) -> tensor<8x1024x1024xbf16>
    return %0 : tensor<8x1024x1024xbf16>
  }

  // A batched B broadcast over an unbatched A is left to tt-metal.
  // CHECK-LABEL: func.func @matmul_batch_broadcast_skipped
  // CHECK: "ttnn.matmul"
  // CHECK-NOT: matmul_program_config
  // CHECK: return
  func.func @matmul_batch_broadcast_skipped(%arg0: tensor<1024x1024xbf16>, %arg1: tensor<8x1024x1024xbf16>) -> tensor<8x1024x1024xbf16> {
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x1024xbf16>, tensor<8x1024x1024xbf16>) -> tensor<8x1024x1024xbf16>
    return %0 : tensor<8x1024x1024xbf16>
  }
}
