// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn -o %t %s
// RUN: FileCheck %s --input-file=%t

// Quetzal pilot 1.4 / 3.3: per-op lowering must emit a HiFi4 +
// fp32_dest_acc_en=true compute_config on every ttnn.matmul. This is the
// numerics floor that lands when the optimizer pipeline (and its
// TTNNSetComputeKernelConfig pass) is not running. Use the bare
// `--convert-ttir-to-ttnn` pass so this test pinpoints MatmulOpConversionPattern
// itself rather than the optimizer-side defaults.

module {
  // CHECK-LABEL: func.func @matmul_default_compute_config
  func.func @matmul_default_compute_config(%arg0: tensor<128x256xbf16>, %arg1: tensor<256x512xbf16>) -> tensor<128x512xbf16> {
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<
    // CHECK-SAME: math_fidelity = hifi4
    // CHECK-SAME: fp32_dest_acc_en = true
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<128x256xbf16>, tensor<256x512xbf16>) -> tensor<128x512xbf16>
    return %0 : tensor<128x512xbf16>
  }

  // CHECK-LABEL: func.func @matmul_decode_residual
  // Single-row decode shape — this is the residual-feeding case where HiFi4 +
  // fp32_dest_acc_en stability is critical (Quetzal ttnn_codegen.py:2075-2109
  // force-enables it specifically for residual-feeding matmuls).
  func.func @matmul_decode_residual(%arg0: tensor<1x4096xbf16>, %arg1: tensor<4096x4096xbf16>) -> tensor<1x4096xbf16> {
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<
    // CHECK-SAME: math_fidelity = hifi4
    // CHECK-SAME: fp32_dest_acc_en = true
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<1x4096xbf16>
    return %0 : tensor<1x4096xbf16>
  }
}
