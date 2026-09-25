// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=0 mock-system-desc-arch=blackhole" -mlir-print-local-scope %s | FileCheck %s --check-prefix=FP32
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=0 mock-system-desc-arch=blackhole compute-cfg-fp32-dest-acc-en=false" -mlir-print-local-scope %s | FileCheck %s --check-prefix=NOFP32
// Test that fp32 dest accumulation caps the output subblock at 4 tiles, while
// without it the 2x3 subblock covering the 4x3 block's width is allowed.

// FP32-LABEL: func.func @subblock_area
// NOFP32-LABEL: func.func @subblock_area
func.func @subblock_area(%arg0: tensor<1024x8192xbf16>, %arg1: tensor<8192x1024xbf16>) -> tensor<1024x1024xbf16> {
  // FP32: "ttnn.matmul"
  // FP32-SAME: out_subblock_h = 4, out_subblock_w = 1, out_block_h = 4, out_block_w = 3
  // NOFP32: "ttnn.matmul"
  // NOFP32-SAME: out_subblock_h = 2, out_subblock_w = 3, out_block_h = 4, out_block_w = 3
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x8192xbf16>, tensor<8192x1024xbf16>) -> tensor<1024x1024xbf16>
  return %0 : tensor<1024x1024xbf16>
}
