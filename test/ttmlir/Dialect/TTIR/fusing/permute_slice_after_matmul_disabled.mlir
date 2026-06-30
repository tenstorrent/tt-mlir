// RUN: ttmlir-opt --ttir-fusing="enable-permute-slice-after-matmul-fusion=false" %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

// With the fusion disabled, the matmul keeps its full output and the slice
// stays below it (mirrors @left_matrix in permute_slice_after_matmul.mlir,
// which fuses under the default-on flag).
// CHECK-LABEL: func.func @left_matrix_fusion_disabled
func.func @left_matrix_fusion_disabled(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x128256xbf16> {
  // CHECK: %[[M:.*]] = "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  // CHECK: "ttir.slice_static"(%[[M]]) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1 : tensor<1x128256xbf16>
}
