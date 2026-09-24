// RUN: ttmlir-opt --ttcore-register-device="mock-system-desc-arch=blackhole" --ttnn-layout --convert-ttir-to-ttnn --ttnn-set-compute-kernel-config --ttnn-set-matmul-program-config -mlir-print-local-scope %s | FileCheck %s
// RUN: ttmlir-opt --ttcore-register-device="mock-system-desc-arch=blackhole" --ttnn-layout --convert-ttir-to-ttnn --ttnn-set-compute-kernel-config="fp32-dest-acc-en=false" --ttnn-set-matmul-program-config -mlir-print-local-scope %s | FileCheck %s --check-prefix=NOFP32
// RUN: ttmlir-opt --ttcore-register-device="mock-system-desc-arch=blackhole" --ttnn-layout --convert-ttir-to-ttnn --ttnn-set-matmul-program-config -mlir-print-local-scope %s | FileCheck %s --check-prefix=NOCFG

// Without a compute_config tt-metal would lower its default math fidelity
// once a program config is present, so the op is left alone.
// NOCFG-NOT: matmul_program_config

module {
  // fp32 dest accumulation caps the subblock area at 4; without it the 2x3
  // subblock covering the whole 4x3 block's width is allowed.
  // CHECK-LABEL: func.func @subblock_area
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: out_subblock_h = 4, out_subblock_w = 1, out_block_h = 4, out_block_w = 3
  // NOFP32-LABEL: func.func @subblock_area
  // NOFP32: "ttnn.matmul"
  // NOFP32-SAME: out_subblock_h = 2, out_subblock_w = 3, out_block_h = 4, out_block_w = 3
  func.func @subblock_area(%arg0: tensor<1024x8192xbf16>, %arg1: tensor<8192x1024xbf16>) -> tensor<1024x1024xbf16> {
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x8192xbf16>, tensor<8192x1024xbf16>) -> tensor<1024x1024xbf16>
    return %0 : tensor<1024x1024xbf16>
  }

  // CHECK-LABEL: func.func @transpose_a_skipped
  // CHECK: "ttnn.matmul"
  // CHECK-NOT: matmul_program_config
  // CHECK: return
  func.func @transpose_a_skipped(%arg0: tensor<8192x1024xbf16>, %arg1: tensor<8192x1024xbf16>) -> tensor<1024x1024xbf16> {
    %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = true, transpose_b = false}> : (tensor<8192x1024xbf16>, tensor<8192x1024xbf16>) -> tensor<1024x1024xbf16>
    return %0 : tensor<1024x1024xbf16>
  }
}
