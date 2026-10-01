// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=1 mock-system-desc-arch=blackhole compute-cfg-math-fidelity=hifi4 compute-cfg-fp32-dest-acc-en=true tensor-l1-usage-cap=0.8" -mlir-print-local-scope %s | FileCheck %s
// Test that the L1 budget follows the tensor L1 usage cap. With 0.8 of usable
// L1 (about 1.18 MB) the 4x11 block of matmul_2d.mlir no longer fits, so the
// pass halves out_block_h; the smaller config passes OpModel validation.

// CHECK-LABEL: func.func @matmul_2d
func.func @matmul_2d(%arg0: tensor<1024x8192xbf16>, %arg1: tensor<3584x8192xbf16>) -> tensor<1024x3584xbf16> {
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<11, 10>, in0_block_w = 16, out_subblock_h = 2, out_subblock_w = 1, out_block_h = 2, out_block_w = 11, per_core_m = 4, per_core_n = 11, transpose_mcast = false, fuse_batch = true>
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = true}> : (tensor<1024x8192xbf16>, tensor<3584x8192xbf16>) -> tensor<1024x3584xbf16>
  return %0 : tensor<1024x3584xbf16>
}
