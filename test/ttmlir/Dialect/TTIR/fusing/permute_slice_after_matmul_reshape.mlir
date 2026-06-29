// RUN: ttmlir-opt --ttir-erase-inverse-ops %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --check-prefix=EIO
// RUN: ttmlir-opt --ttir-erase-inverse-ops --ttir-fusing %s -o %t2.mlir
// RUN: FileCheck %s --input-file=%t2.mlir

// EIO-LABEL: func.func @left_matrix_reshape
// EIO: %[[M:.*]] = "ttir.matmul"(%arg0, %arg1)
// EIO: %[[S:.*]] = "ttir.slice_static"(%[[M]])
// EIO: "ttir.reshape"(%[[S]])
// CHECK-LABEL: func.func @left_matrix_reshape
func.func @left_matrix_reshape(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<1x4096xbf16>
  // CHECK: %[[M:.*]] = "ttir.matmul"(%[[A]], %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1x128256xbf16>
  // CHECK: "ttir.reshape"(%[[M]]) <{shape = [1 : i32, 1 : i32, 128256 : i32]}>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 1024 : i32, 128256 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x1024x128256xbf16>
  %2 = "ttir.slice_static"(%1) <{begins = [0 : i32, 1023 : i32, 0 : i32], ends = [1 : i32, 1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1024x128256xbf16>) -> tensor<1x1x128256xbf16>
  return %2 : tensor<1x1x128256xbf16>
}

// EIO-LABEL: func.func @right_matrix_reshape
// EIO: %[[M:.*]] = "ttir.matmul"(%arg0, %arg1)
// EIO: %[[S:.*]] = "ttir.slice_static"(%[[M]])
// EIO: "ttir.reshape"(%[[S]])
// CHECK-LABEL: func.func @right_matrix_reshape
func.func @right_matrix_reshape(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x1024x64xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32], ends = [4096 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4096x128256xbf16>) -> tensor<4096x64xbf16>
  // CHECK: %[[M:.*]] = "ttir.matmul"(%arg0, %[[B]]) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x64xbf16>) -> tensor<1024x64xbf16>
  // CHECK: "ttir.reshape"(%[[M]]) <{shape = [1 : i32, 1024 : i32, 64 : i32]}>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 1024 : i32, 128256 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x1024x128256xbf16>
  %2 = "ttir.slice_static"(%1) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1024 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1024x128256xbf16>) -> tensor<1x1024x64xbf16>
  return %2 : tensor<1x1024x64xbf16>
}

// EIO-LABEL: func.func @left_matrix_transpose_reshape
// EIO: %[[M:.*]] = "ttir.matmul"(%arg0, %arg1)
// EIO: %[[S:.*]] = "ttir.slice_static"(%[[M]])
// EIO: "ttir.reshape"(%[[S]])
// CHECK-LABEL: func.func @left_matrix_transpose_reshape
func.func @left_matrix_transpose_reshape(%arg0: tensor<4096x1024xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 1023 : i32], ends = [4096 : i32, 1024 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4096x1024xbf16>) -> tensor<4096x1xbf16>
  // CHECK: %[[M:.*]] = "ttir.matmul"(%[[A]], %arg1) <{transpose_a = true, transpose_b = false}> : (tensor<4096x1xbf16>, tensor<4096x128256xbf16>) -> tensor<1x128256xbf16>
  // CHECK: "ttir.reshape"(%[[M]]) <{shape = [1 : i32, 1 : i32, 128256 : i32]}>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = true, transpose_b = false}> : (tensor<4096x1024xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 1024 : i32, 128256 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x1024x128256xbf16>
  %2 = "ttir.slice_static"(%1) <{begins = [0 : i32, 1023 : i32, 0 : i32], ends = [1 : i32, 1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1024x128256xbf16>) -> tensor<1x1x128256xbf16>
  return %2 : tensor<1x1x128256xbf16>
}
