// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=0 mock-system-desc-arch=blackhole" -mlir-print-local-scope %s | FileCheck %s

// A DRAM-interleaved matmul routed to 2D mcast gets the block that maximises
// in0_block_w * out_block_h * out_block_w, with the batch fused into M because
// B is unbatched.
// CHECK-LABEL: func.func @matmul_2d_transpose_b
func.func @matmul_2d_transpose_b(%arg0: tensor<1024x8192xbf16>, %arg1: tensor<3584x8192xbf16>) -> tensor<1024x3584xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 16, out_subblock_h = 4, out_subblock_w = 1, out_block_h = 4, out_block_w = 11, per_core_m = 4, per_core_n = 11, transpose_mcast = false, fuse_batch = true>
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = true}> : (tensor<1024x8192xbf16>, tensor<3584x8192xbf16>) -> tensor<1024x3584xbf16>
  return %0 : tensor<1024x3584xbf16>
}

// With transpose_a, M and K come from the transposed A and L1 is reserved for
// the transposed in0 CB: without it the 4x11 block with in0_block_w = 16 would
// fit.
// CHECK-LABEL: func.func @matmul_2d_transpose_a
func.func @matmul_2d_transpose_a(%arg0: tensor<8192x1024xbf16>, %arg1: tensor<8192x3584xbf16>) -> tensor<1024x3584xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 16, out_subblock_h = 2, out_subblock_w = 1, out_block_h = 2, out_block_w = 11, per_core_m = 4, per_core_n = 11, transpose_mcast = false, fuse_batch = true>
  // CHECK-SAME: transpose_a = true
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = true, transpose_b = false}> : (tensor<8192x1024xbf16>, tensor<8192x3584xbf16>) -> tensor<1024x3584xbf16>
  return %0 : tensor<1024x3584xbf16>
}

// Matching A and B batches keep per-batch rows (fuse_batch = false).
// CHECK-LABEL: func.func @matmul_2d_batched
func.func @matmul_2d_batched(%arg0: tensor<8x1024x1024xbf16>, %arg1: tensor<8x1024x1024xbf16>) -> tensor<8x1024x1024xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 16, out_subblock_h = 4, out_subblock_w = 1, out_block_h = 4, out_block_w = 3, per_core_m = 4, per_core_n = 3, transpose_mcast = false, fuse_batch = false>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<8x1024x1024xbf16>, tensor<8x1024x1024xbf16>) -> tensor<8x1024x1024xbf16>
  return %0 : tensor<8x1024x1024xbf16>
}

// A batched A with transpose_a over an unbatched B keeps per-batch rows
// (fuse_batch = false): tt-metal rejects fuse_batch with transpose_a when A
// has batches of more than one M tile.
// CHECK-LABEL: func.func @matmul_2d_batched_transpose_a
func.func @matmul_2d_batched_transpose_a(%arg0: tensor<8x1024x1024xbf16>, %arg1: tensor<1024x1024xbf16>) -> tensor<8x1024x1024xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 16, out_subblock_h = 4, out_subblock_w = 1, out_block_h = 4, out_block_w = 3, per_core_m = 4, per_core_n = 3, transpose_mcast = false, fuse_batch = false>
  // CHECK-SAME: transpose_a = true
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = true, transpose_b = false}> : (tensor<8x1024x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<8x1024x1024xbf16>
  return %0 : tensor<8x1024x1024xbf16>
}

// ttnn.linear with a bias gets a 2D mcast config, with L1 reserved for the
// bias CB.
// CHECK-LABEL: func.func @linear_2d_bias
func.func @linear_2d_bias(%arg0: tensor<1024x8192xbf16>, %arg1: tensor<8192x1024xbf16>, %arg2: tensor<1024xbf16>) -> tensor<1024x1024xbf16> {
  // CHECK: "ttnn.linear"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 32, out_subblock_h = 4, out_subblock_w = 1, out_block_h = 4, out_block_w = 3, per_core_m = 4, per_core_n = 3, transpose_mcast = false, fuse_batch = true>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x8192xbf16>, tensor<8192x1024xbf16>, tensor<1024xbf16>) -> tensor<1024x1024xbf16>
  return %0 : tensor<1024x1024xbf16>
}

// A wide matmul routed to 1D mcast_in0 keeps out_block_w = per_core_N, halves
// out_block_h since M >= 1024, and takes the largest in0_block_w that fits L1.
// CHECK-LABEL: func.func @matmul_1d_mcast_in0
func.func @matmul_1d_mcast_in0(%arg0: tensor<1024x8192xbf16>, %arg1: tensor<8192x16032xbf16>) -> tensor<1024x16032xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 8, out_subblock_h = 4, out_subblock_w = 1, out_block_h = 16, out_block_w = 5, per_core_m = 32, per_core_n = 5, fuse_batch = false, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x8192xbf16>, tensor<8192x16032xbf16>) -> tensor<1024x16032xbf16>
  return %0 : tensor<1024x16032xbf16>
}

// A tall matmul routed to 1D mcast_in1 keeps out_block_h = per_core_M and,
// since N < 1024, out_block_w = per_core_N.
// CHECK-LABEL: func.func @matmul_1d_mcast_in1
func.func @matmul_1d_mcast_in1(%arg0: tensor<16384x1024xbf16>, %arg1: tensor<1024x128xbf16>) -> tensor<16384x128xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 16, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 5, out_block_w = 4, per_core_m = 5, per_core_n = 4, fuse_batch = false, mcast_in0 = false, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<16384x1024xbf16>, tensor<1024x128xbf16>) -> tensor<16384x128xbf16>
  return %0 : tensor<16384x128xbf16>
}

// 1D mcast_in0 with transpose_a reserves L1 for the transposed in0 CB: the
// same problem without transpose_a (@matmul_1d_mcast_in0) takes
// in0_block_w = 8.
// CHECK-LABEL: func.func @matmul_1d_transpose_a
func.func @matmul_1d_transpose_a(%arg0: tensor<8192x1024xbf16>, %arg1: tensor<8192x16032xbf16>) -> tensor<1024x16032xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 4, out_subblock_h = 4, out_subblock_w = 1, out_block_h = 16, out_block_w = 5, per_core_m = 32, per_core_n = 5, fuse_batch = false, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>
  // CHECK-SAME: transpose_a = true
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = true, transpose_b = false}> : (tensor<8192x1024xbf16>, tensor<8192x16032xbf16>) -> tensor<1024x16032xbf16>
  return %0 : tensor<1024x16032xbf16>
}

// A batched B broadcast over an unbatched A is left to tt-metal.
// CHECK-LABEL: func.func @skip_batch_broadcast
func.func @skip_batch_broadcast(%arg0: tensor<1024x1024xbf16>, %arg1: tensor<8x1024x1024xbf16>) -> tensor<8x1024x1024xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-NOT: matmul_program_config
  // CHECK: return
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x1024xbf16>, tensor<8x1024x1024xbf16>) -> tensor<8x1024x1024xbf16>
  return %0 : tensor<8x1024x1024xbf16>
}

// Batch dimensions broadcast in both directions are left to tt-metal, even
// though A's and B's batch volumes match.
// CHECK-LABEL: func.func @skip_partial_batch_broadcast
func.func @skip_partial_batch_broadcast(%arg0: tensor<1x8x1024x1024xbf16>, %arg1: tensor<8x1x1024x1024xbf16>) -> tensor<8x8x1024x1024xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-NOT: matmul_program_config
  // CHECK: return
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1x8x1024x1024xbf16>, tensor<8x1x1024x1024xbf16>) -> tensor<8x8x1024x1024xbf16>
  return %0 : tensor<8x8x1024x1024xbf16>
}

// A zero-sized dimension is left to tt-metal.
// CHECK-LABEL: func.func @skip_zero_dim
func.func @skip_zero_dim(%arg0: tensor<0x1024xbf16>, %arg1: tensor<1024x1024xbf16>) -> tensor<0x1024xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-NOT: matmul_program_config
  // CHECK: return
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<0x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<0x1024xbf16>
  return %0 : tensor<0x1024xbf16>
}
