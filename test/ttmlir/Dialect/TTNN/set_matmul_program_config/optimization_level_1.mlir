// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=1 mock-system-desc-arch=blackhole compute-cfg-math-fidelity=hifi4 compute-cfg-fp32-dest-acc-en=true" -mlir-print-local-scope %s | FileCheck %s
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=1 mock-system-desc-arch=blackhole" -mlir-print-local-scope %s | FileCheck %s --check-prefix=DEFAULT
// Test the pass after the optimizer at optimization-level=1. With a compute
// config the configs match optimization-level=0. By default this level leaves
// the compute config unset, so the pass leaves every matmul to tt-metal.

// DEFAULT-NOT: matmul_program_config

// CHECK-LABEL: func.func @matmul_2d
func.func @matmul_2d(%arg0: tensor<1024x8192xbf16>, %arg1: tensor<3584x8192xbf16>) -> tensor<1024x3584xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 16, out_subblock_h = 4, out_subblock_w = 1, out_block_h = 4, out_block_w = 11, per_core_m = 4, per_core_n = 11, transpose_mcast = false, fuse_batch = true>
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = true}> : (tensor<1024x8192xbf16>, tensor<3584x8192xbf16>) -> tensor<1024x3584xbf16>
  return %0 : tensor<1024x3584xbf16>
}

// CHECK-LABEL: func.func @matmul_1d_mcast_in0
func.func @matmul_1d_mcast_in0(%arg0: tensor<1024x8192xbf16>, %arg1: tensor<8192x16032xbf16>) -> tensor<1024x16032xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 8, out_subblock_h = 4, out_subblock_w = 1, out_block_h = 16, out_block_w = 5, per_core_m = 32, per_core_n = 5, fuse_batch = false, mcast_in0 = true
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x8192xbf16>, tensor<8192x16032xbf16>) -> tensor<1024x16032xbf16>
  return %0 : tensor<1024x16032xbf16>
}

// CHECK-LABEL: func.func @matmul_1d_mcast_in1
func.func @matmul_1d_mcast_in1(%arg0: tensor<16384x1024xbf16>, %arg1: tensor<1024x128xbf16>) -> tensor<16384x128xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 16, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 5, out_block_w = 4, per_core_m = 5, per_core_n = 4, fuse_batch = false, mcast_in0 = false
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<16384x1024xbf16>, tensor<1024x128xbf16>) -> tensor<16384x128xbf16>
  return %0 : tensor<16384x128xbf16>
}

// CHECK-LABEL: func.func @linear_2d_bias
func.func @linear_2d_bias(%arg0: tensor<1024x8192xbf16>, %arg1: tensor<8192x1024xbf16>, %arg2: tensor<1024xbf16>) -> tensor<1024x1024xbf16> {
  // CHECK: "ttnn.linear"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 32, out_subblock_h = 4, out_subblock_w = 1, out_block_h = 4, out_block_w = 3, per_core_m = 4, per_core_n = 3, transpose_mcast = false, fuse_batch = true>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x8192xbf16>, tensor<8192x1024xbf16>, tensor<1024xbf16>) -> tensor<1024x1024xbf16>
  return %0 : tensor<1024x1024xbf16>
}
