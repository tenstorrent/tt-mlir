// RUN: ttmlir-opt --ttir-fusing %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

// CHECK-LABEL: func.func @left_matrix
func.func @left_matrix(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<1x4096xbf16>
  // CHECK: "ttir.matmul"(%[[A]], %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1x128256xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1 : tensor<1x128256xbf16>
}

// CHECK-LABEL: func.func @left_matrix_multiple_uses_negative
func.func @left_matrix_multiple_uses_negative(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> (tensor<1x128256xbf16>, tensor<1024x128256xbf16>) {
  // CHECK: "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1, %0 : tensor<1x128256xbf16>, tensor<1024x128256xbf16>
}

// CHECK-LABEL: func.func @right_matrix
func.func @right_matrix(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1024x64xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32], ends = [4096 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4096x128256xbf16>) -> tensor<4096x64xbf16>
  // CHECK: "ttir.matmul"(%arg0, %[[B]]) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x64xbf16>) -> tensor<1024x64xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x64xbf16>
  return %1 : tensor<1024x64xbf16>
}

// CHECK-LABEL: func.func @right_matrix_middle_cols
func.func @right_matrix_middle_cols(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1024x12xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 5 : i32], ends = [4096 : i32, 17 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4096x128256xbf16>) -> tensor<4096x12xbf16>
  // CHECK: "ttir.matmul"(%arg0, %[[B]]) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x12xbf16>) -> tensor<1024x12xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 5 : i32], ends = [1024 : i32, 17 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x12xbf16>
  return %1 : tensor<1024x12xbf16>
}

// CHECK-LABEL: func.func @right_matrix_transpose
func.func @right_matrix_transpose(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<128256x4096xbf16>) -> tensor<1024x64xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32], ends = [64 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<128256x4096xbf16>) -> tensor<64x4096xbf16>
  // CHECK: "ttir.matmul"(%arg0, %[[B]]) <{transpose_a = false, transpose_b = true}> : (tensor<1024x4096xbf16>, tensor<64x4096xbf16>) -> tensor<1024x64xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = true}> : (tensor<1024x4096xbf16>, tensor<128256x4096xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x64xbf16>
  return %1 : tensor<1024x64xbf16>
}

// CHECK-LABEL: func.func @both_matrix_negative
func.func @both_matrix_negative(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x64xbf16> {
  // CHECK: "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x64xbf16>
  return %1 : tensor<1x64xbf16>
}

// CHECK-LABEL: func.func @left_matrix_strided_rows
func.func @left_matrix_strided_rows(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<512x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 4096 : i32], step = [2 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<512x4096xbf16>
  // CHECK: "ttir.matmul"(%[[A]], %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<512x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<512x128256xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [2 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<512x128256xbf16>
  return %1 : tensor<512x128256xbf16>
}

// CHECK-LABEL: func.func @right_matrix_strided_cols
func.func @right_matrix_strided_cols(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1024x42752xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32], ends = [4096 : i32, 128256 : i32], step = [1 : i32, 3 : i32]}> : (tensor<4096x128256xbf16>) -> tensor<4096x42752xbf16>
  // CHECK: "ttir.matmul"(%arg0, %[[B]]) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x42752xbf16>) -> tensor<1024x42752xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 3 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x42752xbf16>
  return %1 : tensor<1024x42752xbf16>
}

// CHECK-LABEL: func.func @left_matrix_strided_rows_offset
func.func @left_matrix_strided_rows_offset(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [1022 : i32, 0 : i32], ends = [1024 : i32, 4096 : i32], step = [2 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<1x4096xbf16>
  // CHECK: "ttir.matmul"(%[[A]], %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1x128256xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1022 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [2 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1 : tensor<1x128256xbf16>
}

// CHECK-LABEL: func.func @left_matrix_strided_rows_offset_transpose
func.func @left_matrix_strided_rows_offset_transpose(%arg0: tensor<4096x1024xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 1022 : i32], ends = [4096 : i32, 1024 : i32], step = [1 : i32, 2 : i32]}> : (tensor<4096x1024xbf16>) -> tensor<4096x1xbf16>
  // CHECK: "ttir.matmul"(%[[A]], %arg1) <{transpose_a = true, transpose_b = false}> : (tensor<4096x1xbf16>, tensor<4096x128256xbf16>) -> tensor<1x128256xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = true, transpose_b = false}> : (tensor<4096x1024xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1022 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [2 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1 : tensor<1x128256xbf16>
}

// CHECK-LABEL: func.func @left_vector_matmul
func.func @left_vector_matmul(%arg0: tensor<4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<64xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32], ends = [4096 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4096x128256xbf16>) -> tensor<4096x64xbf16>
  // CHECK: "ttir.matmul"(%arg0, %[[B]]) <{transpose_a = false, transpose_b = false}> : (tensor<4096xbf16>, tensor<4096x64xbf16>) -> tensor<64xbf16>
  // CHECK-NOT: tensor<128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<4096xbf16>, tensor<4096x128256xbf16>) -> tensor<128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32], ends = [64 : i32], step = [1 : i32]}> : (tensor<128256xbf16>) -> tensor<64xbf16>
  return %1 : tensor<64xbf16>
}

// CHECK-LABEL: func.func @right_vector_matmul
func.func @right_vector_matmul(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096xbf16>) -> tensor<64xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32], ends = [64 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<64x4096xbf16>
  // CHECK: "ttir.matmul"(%[[A]], %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<64x4096xbf16>, tensor<4096xbf16>) -> tensor<64xbf16>
  // CHECK-NOT: tensor<1024xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096xbf16>) -> tensor<1024xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32], ends = [64 : i32], step = [1 : i32]}> : (tensor<1024xbf16>) -> tensor<64xbf16>
  return %1 : tensor<64xbf16>
}

// CHECK-LABEL: func.func @batched_right_vector_matmul
func.func @batched_right_vector_matmul(%arg0: tensor<2x1024x4096xbf16>, %arg1: tensor<4096xbf16>) -> tensor<2x64xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 64 : i32, 4096 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x4096xbf16>) -> tensor<2x64x4096xbf16>
  // CHECK: "ttir.matmul"(%[[A]], %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<2x64x4096xbf16>, tensor<4096xbf16>) -> tensor<2x64xbf16>
  // CHECK-NOT: tensor<2x1024xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<2x1024x4096xbf16>, tensor<4096xbf16>) -> tensor<2x1024xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x1024xbf16>) -> tensor<2x64xbf16>
  return %1 : tensor<2x64xbf16>
}

// CHECK-LABEL: func.func @both_vector_matmul_negative
func.func @both_vector_matmul_negative(%arg0: tensor<4096xbf16>, %arg1: tensor<4096xbf16>) -> tensor<1xbf16> {
  // CHECK: "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<4096xbf16>, tensor<4096xbf16>) -> tensor<1xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<4096xbf16>, tensor<4096xbf16>) -> tensor<1xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<1xbf16>) -> tensor<1xbf16>
  return %1 : tensor<1xbf16>
}

// CHECK-LABEL: func.func @left_matrix_middle_rows
func.func @left_matrix_middle_rows(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<12x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [5 : i32, 0 : i32], ends = [17 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<12x4096xbf16>
  // CHECK: "ttir.matmul"(%[[A]], %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<12x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<12x128256xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [5 : i32, 0 : i32], ends = [17 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<12x128256xbf16>
  return %1 : tensor<12x128256xbf16>
}

// CHECK-LABEL: func.func @left_matrix_negative_indices
func.func @left_matrix_negative_indices(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<3x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [-4 : i32, 0 : i32], ends = [-1 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<3x4096xbf16>
  // CHECK: "ttir.matmul"(%[[A]], %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<3x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<3x128256xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [-4 : i32, 0 : i32], ends = [-1 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<3x128256xbf16>
  return %1 : tensor<3x128256xbf16>
}

// CHECK-LABEL: func.func @left_matrix_batched
func.func @left_matrix_batched(%arg0: tensor<2x1024x4096xbf16>, %arg1: tensor<2x4096x128256xbf16>) -> tensor<2x1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 1023 : i32, 0 : i32], ends = [2 : i32, 1024 : i32, 4096 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x4096xbf16>) -> tensor<2x1x4096xbf16>
  // CHECK: "ttir.matmul"(%[[A]], %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<2x1x4096xbf16>, tensor<2x4096x128256xbf16>) -> tensor<2x1x128256xbf16>
  // CHECK-NOT: tensor<2x1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<2x1024x4096xbf16>, tensor<2x4096x128256xbf16>) -> tensor<2x1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 1023 : i32, 0 : i32], ends = [2 : i32, 1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x128256xbf16>) -> tensor<2x1x128256xbf16>
  return %1 : tensor<2x1x128256xbf16>
}

// CHECK-LABEL: func.func @left_matrix_transpose
func.func @left_matrix_transpose(%arg0: tensor<4096x1024xbf16>, %arg1: tensor<4096x128256xbf16>) -> tensor<1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 1023 : i32], ends = [4096 : i32, 1024 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4096x1024xbf16>) -> tensor<4096x1xbf16>
  // CHECK: "ttir.matmul"(%[[A]], %arg1) <{transpose_a = true, transpose_b = false}> : (tensor<4096x1xbf16>, tensor<4096x128256xbf16>) -> tensor<1x128256xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = true, transpose_b = false}> : (tensor<4096x1024xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1 : tensor<1x128256xbf16>
}

// CHECK-LABEL: func.func @left_matrix_transpose_both
func.func @left_matrix_transpose_both(%arg0: tensor<4096x1024xbf16>, %arg1: tensor<128256x4096xbf16>) -> tensor<1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 1023 : i32], ends = [4096 : i32, 1024 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4096x1024xbf16>) -> tensor<4096x1xbf16>
  // CHECK: "ttir.matmul"(%[[A]], %arg1) <{transpose_a = true, transpose_b = true}> : (tensor<4096x1xbf16>, tensor<128256x4096xbf16>) -> tensor<1x128256xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = true, transpose_b = true}> : (tensor<4096x1024xbf16>, tensor<128256x4096xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1 : tensor<1x128256xbf16>
}

// CHECK-LABEL: func.func @linear_left
func.func @linear_left(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>, %arg2: tensor<128256xbf16>) -> tensor<1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<1x4096xbf16>
  // CHECK: "ttir.linear"(%[[A]], %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1x4096xbf16>, tensor<4096x128256xbf16>, tensor<128256xbf16>) -> tensor<1x128256xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1 : tensor<1x128256xbf16>
}

// CHECK-LABEL: func.func @linear_left_bias_2d
func.func @linear_left_bias_2d(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>, %arg2: tensor<1024x128256xbf16>) -> tensor<1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<1x4096xbf16>
  // CHECK: %[[BIAS:.*]] = "ttir.slice_static"(%arg2) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  // CHECK: "ttir.linear"(%[[A]], %arg1, %[[BIAS]]) <{transpose_a = false, transpose_b = false}> : (tensor<1x4096xbf16>, tensor<4096x128256xbf16>, tensor<1x128256xbf16>) -> tensor<1x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<1024x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1 : tensor<1x128256xbf16>
}

// CHECK-LABEL: func.func @linear_left_bias_broadcast_m
func.func @linear_left_bias_broadcast_m(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>, %arg2: tensor<1x128256xbf16>) -> tensor<1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<1x4096xbf16>
  // CHECK: "ttir.linear"(%[[A]], %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1x4096xbf16>, tensor<4096x128256xbf16>, tensor<1x128256xbf16>) -> tensor<1x128256xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<1x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1 : tensor<1x128256xbf16>
}

// CHECK-LABEL: func.func @linear_left_transpose
func.func @linear_left_transpose(%arg0: tensor<4096x1024xbf16>, %arg1: tensor<4096x128256xbf16>, %arg2: tensor<128256xbf16>) -> tensor<1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 1023 : i32], ends = [4096 : i32, 1024 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4096x1024xbf16>) -> tensor<4096x1xbf16>
  // CHECK: "ttir.linear"(%[[A]], %arg1, %arg2) <{transpose_a = true, transpose_b = false}> : (tensor<4096x1xbf16>, tensor<4096x128256xbf16>, tensor<128256xbf16>) -> tensor<1x128256xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) <{transpose_a = true, transpose_b = false}> : (tensor<4096x1024xbf16>, tensor<4096x128256xbf16>, tensor<128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1 : tensor<1x128256xbf16>
}

// CHECK-LABEL: func.func @linear_right
func.func @linear_right(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>, %arg2: tensor<128256xbf16>) -> tensor<1024x64xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32], ends = [4096 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4096x128256xbf16>) -> tensor<4096x64xbf16>
  // CHECK: %[[BIAS:.*]] = "ttir.slice_static"(%arg2) <{begins = [0 : i32], ends = [64 : i32], step = [1 : i32]}> : (tensor<128256xbf16>) -> tensor<64xbf16>
  // CHECK: "ttir.linear"(%arg0, %[[B]], %[[BIAS]]) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x64xbf16>, tensor<64xbf16>) -> tensor<1024x64xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x64xbf16>
  return %1 : tensor<1024x64xbf16>
}

// CHECK-LABEL: func.func @linear_right_bias_2d
func.func @linear_right_bias_2d(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>, %arg2: tensor<1024x128256xbf16>) -> tensor<1024x64xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32], ends = [4096 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4096x128256xbf16>) -> tensor<4096x64xbf16>
  // CHECK: %[[BIAS:.*]] = "ttir.slice_static"(%arg2) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x64xbf16>
  // CHECK: "ttir.linear"(%arg0, %[[B]], %[[BIAS]]) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x64xbf16>, tensor<1024x64xbf16>) -> tensor<1024x64xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<1024x128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x64xbf16>
  return %1 : tensor<1024x64xbf16>
}

// CHECK-LABEL: func.func @linear_right_bias_broadcast_n
func.func @linear_right_bias_broadcast_n(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>, %arg2: tensor<1024x1xbf16>) -> tensor<1024x64xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32], ends = [4096 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4096x128256xbf16>) -> tensor<4096x64xbf16>
  // CHECK: "ttir.linear"(%arg0, %[[B]], %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x64xbf16>, tensor<1024x1xbf16>) -> tensor<1024x64xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<1024x1xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x64xbf16>
  return %1 : tensor<1024x64xbf16>
}

// CHECK-LABEL: func.func @linear_right_bias_scalar
func.func @linear_right_bias_scalar(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>, %arg2: tensor<1xbf16>) -> tensor<1024x64xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32], ends = [4096 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4096x128256xbf16>) -> tensor<4096x64xbf16>
  // CHECK: "ttir.linear"(%arg0, %[[B]], %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x64xbf16>, tensor<1xbf16>) -> tensor<1024x64xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<1xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x64xbf16>
  return %1 : tensor<1024x64xbf16>
}

// CHECK-LABEL: func.func @linear_right_transpose
func.func @linear_right_transpose(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<128256x4096xbf16>, %arg2: tensor<128256xbf16>) -> tensor<1024x64xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32], ends = [64 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<128256x4096xbf16>) -> tensor<64x4096xbf16>
  // CHECK: %[[BIAS:.*]] = "ttir.slice_static"(%arg2) <{begins = [0 : i32], ends = [64 : i32], step = [1 : i32]}> : (tensor<128256xbf16>) -> tensor<64xbf16>
  // CHECK: "ttir.linear"(%arg0, %[[B]], %[[BIAS]]) <{transpose_a = false, transpose_b = true}> : (tensor<1024x4096xbf16>, tensor<64x4096xbf16>, tensor<64xbf16>) -> tensor<1024x64xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = true}> : (tensor<1024x4096xbf16>, tensor<128256x4096xbf16>, tensor<128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x64xbf16>
  return %1 : tensor<1024x64xbf16>
}

// CHECK-LABEL: func.func @linear_right_strided
func.func @linear_right_strided(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>, %arg2: tensor<128256xbf16>) -> tensor<1024x42752xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32], ends = [4096 : i32, 128256 : i32], step = [1 : i32, 3 : i32]}> : (tensor<4096x128256xbf16>) -> tensor<4096x42752xbf16>
  // CHECK: %[[BIAS:.*]] = "ttir.slice_static"(%arg2) <{begins = [0 : i32], ends = [128256 : i32], step = [3 : i32]}> : (tensor<128256xbf16>) -> tensor<42752xbf16>
  // CHECK: "ttir.linear"(%arg0, %[[B]], %[[BIAS]]) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x42752xbf16>, tensor<42752xbf16>) -> tensor<1024x42752xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 3 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x42752xbf16>
  return %1 : tensor<1024x42752xbf16>
}

// CHECK-LABEL: func.func @linear_right_strided_transpose_bias
func.func @linear_right_strided_transpose_bias(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<128256x4096xbf16>, %arg2: tensor<128256xbf16>) -> tensor<1024x42752xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32], ends = [128256 : i32, 4096 : i32], step = [3 : i32, 1 : i32]}> : (tensor<128256x4096xbf16>) -> tensor<42752x4096xbf16>
  // CHECK: %[[BIAS:.*]] = "ttir.slice_static"(%arg2) <{begins = [0 : i32], ends = [128256 : i32], step = [3 : i32]}> : (tensor<128256xbf16>) -> tensor<42752xbf16>
  // CHECK: "ttir.linear"(%arg0, %[[B]], %[[BIAS]]) <{transpose_a = false, transpose_b = true}> : (tensor<1024x4096xbf16>, tensor<42752x4096xbf16>, tensor<42752xbf16>) -> tensor<1024x42752xbf16>
  // CHECK-NOT: tensor<1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = true}> : (tensor<1024x4096xbf16>, tensor<128256x4096xbf16>, tensor<128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 3 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1024x42752xbf16>
  return %1 : tensor<1024x42752xbf16>
}

// CHECK-LABEL: func.func @linear_both_negative
func.func @linear_both_negative(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>, %arg2: tensor<128256xbf16>) -> tensor<1x64xbf16> {
  // CHECK: "ttir.linear"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<128256xbf16>) -> tensor<1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x64xbf16>
  return %1 : tensor<1x64xbf16>
}

// CHECK-LABEL: func.func @linear_batched_bias_3d
func.func @linear_batched_bias_3d(%arg0: tensor<2x1024x4096xbf16>, %arg1: tensor<2x4096x128256xbf16>, %arg2: tensor<2x1024x128256xbf16>) -> tensor<2x1024x64xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 4096 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x4096x128256xbf16>) -> tensor<2x4096x64xbf16>
  // CHECK: %[[BIAS:.*]] = "ttir.slice_static"(%arg2) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 1024 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x128256xbf16>) -> tensor<2x1024x64xbf16>
  // CHECK: "ttir.linear"(%arg0, %[[B]], %[[BIAS]]) <{transpose_a = false, transpose_b = false}> : (tensor<2x1024x4096xbf16>, tensor<2x4096x64xbf16>, tensor<2x1024x64xbf16>) -> tensor<2x1024x64xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<2x1024x4096xbf16>, tensor<2x4096x128256xbf16>, tensor<2x1024x128256xbf16>) -> tensor<2x1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 1024 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x128256xbf16>) -> tensor<2x1024x64xbf16>
  return %1 : tensor<2x1024x64xbf16>
}

// CHECK-LABEL: func.func @linear_batched_bias_3d_row
func.func @linear_batched_bias_3d_row(%arg0: tensor<2x1024x4096xbf16>, %arg1: tensor<2x4096x128256xbf16>, %arg2: tensor<2x1024x128256xbf16>) -> tensor<2x1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 1023 : i32, 0 : i32], ends = [2 : i32, 1024 : i32, 4096 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x4096xbf16>) -> tensor<2x1x4096xbf16>
  // CHECK: %[[BIAS:.*]] = "ttir.slice_static"(%arg2) <{begins = [0 : i32, 1023 : i32, 0 : i32], ends = [2 : i32, 1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x128256xbf16>) -> tensor<2x1x128256xbf16>
  // CHECK: "ttir.linear"(%[[A]], %arg1, %[[BIAS]]) <{transpose_a = false, transpose_b = false}> : (tensor<2x1x4096xbf16>, tensor<2x4096x128256xbf16>, tensor<2x1x128256xbf16>) -> tensor<2x1x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<2x1024x4096xbf16>, tensor<2x4096x128256xbf16>, tensor<2x1024x128256xbf16>) -> tensor<2x1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 1023 : i32, 0 : i32], ends = [2 : i32, 1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x128256xbf16>) -> tensor<2x1x128256xbf16>
  return %1 : tensor<2x1x128256xbf16>
}

// CHECK-LABEL: func.func @linear_batched_bias_broadcast_batch
func.func @linear_batched_bias_broadcast_batch(%arg0: tensor<2x1024x4096xbf16>, %arg1: tensor<2x4096x128256xbf16>, %arg2: tensor<1x1024x128256xbf16>) -> tensor<2x1024x64xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 4096 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x4096x128256xbf16>) -> tensor<2x4096x64xbf16>
  // CHECK: %[[BIAS:.*]] = "ttir.slice_static"(%arg2) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1024 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1024x128256xbf16>) -> tensor<1x1024x64xbf16>
  // CHECK: "ttir.linear"(%arg0, %[[B]], %[[BIAS]]) <{transpose_a = false, transpose_b = false}> : (tensor<2x1024x4096xbf16>, tensor<2x4096x64xbf16>, tensor<1x1024x64xbf16>) -> tensor<2x1024x64xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<2x1024x4096xbf16>, tensor<2x4096x128256xbf16>, tensor<1x1024x128256xbf16>) -> tensor<2x1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 1024 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x128256xbf16>) -> tensor<2x1024x64xbf16>
  return %1 : tensor<2x1024x64xbf16>
}

// CHECK-LABEL: func.func @linear_batched_bias_broadcast_m_row
func.func @linear_batched_bias_broadcast_m_row(%arg0: tensor<2x1024x4096xbf16>, %arg1: tensor<2x4096x128256xbf16>, %arg2: tensor<2x1x128256xbf16>) -> tensor<2x1x128256xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 1023 : i32, 0 : i32], ends = [2 : i32, 1024 : i32, 4096 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x4096xbf16>) -> tensor<2x1x4096xbf16>
  // CHECK: "ttir.linear"(%[[A]], %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<2x1x4096xbf16>, tensor<2x4096x128256xbf16>, tensor<2x1x128256xbf16>) -> tensor<2x1x128256xbf16>
  // CHECK-NOT: tensor<2x1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<2x1024x4096xbf16>, tensor<2x4096x128256xbf16>, tensor<2x1x128256xbf16>) -> tensor<2x1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 1023 : i32, 0 : i32], ends = [2 : i32, 1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x128256xbf16>) -> tensor<2x1x128256xbf16>
  return %1 : tensor<2x1x128256xbf16>
}

// CHECK-LABEL: func.func @linear_batched_bias_broadcast_n_col
func.func @linear_batched_bias_broadcast_n_col(%arg0: tensor<2x1024x4096xbf16>, %arg1: tensor<2x4096x128256xbf16>, %arg2: tensor<2x1024x1xbf16>) -> tensor<2x1024x64xbf16> {
  // CHECK: %[[B:.*]] = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 4096 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x4096x128256xbf16>) -> tensor<2x4096x64xbf16>
  // CHECK: "ttir.linear"(%arg0, %[[B]], %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<2x1024x4096xbf16>, tensor<2x4096x64xbf16>, tensor<2x1024x1xbf16>) -> tensor<2x1024x64xbf16>
  // CHECK-NOT: tensor<2x1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<2x1024x4096xbf16>, tensor<2x4096x128256xbf16>, tensor<2x1024x1xbf16>) -> tensor<2x1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 1024 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x128256xbf16>) -> tensor<2x1024x64xbf16>
  return %1 : tensor<2x1024x64xbf16>
}

// CHECK-LABEL: func.func @linear_multiple_uses_negative
func.func @linear_multiple_uses_negative(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x128256xbf16>, %arg2: tensor<128256xbf16>) -> (tensor<1x128256xbf16>, tensor<1024x128256xbf16>) {
  // CHECK: "ttir.linear"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<128256xbf16>) -> tensor<1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>, tensor<128256xbf16>) -> tensor<1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x128256xbf16>) -> tensor<1x128256xbf16>
  return %1, %0 : tensor<1x128256xbf16>, tensor<1024x128256xbf16>
}

// CHECK-LABEL: func.func @linear_batch_slice_negative
func.func @linear_batch_slice_negative(%arg0: tensor<2x1024x4096xbf16>, %arg1: tensor<2x4096x128256xbf16>, %arg2: tensor<128256xbf16>) -> tensor<1x1024x128256xbf16> {
  // CHECK: "ttir.linear"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<2x1024x4096xbf16>, tensor<2x4096x128256xbf16>, tensor<128256xbf16>) -> tensor<2x1024x128256xbf16>
  %0 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<2x1024x4096xbf16>, tensor<2x4096x128256xbf16>, tensor<128256xbf16>) -> tensor<2x1024x128256xbf16>
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1024 : i32, 128256 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x1024x128256xbf16>) -> tensor<1x1024x128256xbf16>
  return %1 : tensor<1x1024x128256xbf16>
}

// CHECK-LABEL: func.func @shared_lhs_fusion_not_undone
func.func @shared_lhs_fusion_not_undone(%arg0: tensor<32x512xbf16>, %arg1: tensor<512x384xbf16>, %arg2: tensor<512x384xbf16>, %arg3: tensor<512x384xbf16>) -> (tensor<32x384xbf16>, tensor<32x384xbf16>, tensor<32x384xbf16>) {
  // CHECK: %[[W:.*]] = "ttir.concat"({{.*}}) <{dim = 1 : si32}> : (tensor<512x384xbf16>, tensor<512x384xbf16>, tensor<512x384xbf16>) -> tensor<512x1152xbf16>
  // CHECK: %[[M:.*]] = "ttir.matmul"(%arg0, %[[W]]) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x1152xbf16>) -> tensor<32x1152xbf16>
  // CHECK: "ttir.slice_static"(%[[M]])
  // CHECK: "ttir.slice_static"(%[[M]])
  // CHECK: "ttir.slice_static"(%[[M]])
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
  %1 = "ttir.matmul"(%arg0, %arg2) : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
  %2 = "ttir.matmul"(%arg0, %arg3) : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
  return %0, %1, %2 : tensor<32x384xbf16>, tensor<32x384xbf16>, tensor<32x384xbf16>
}

// CHECK-LABEL: func.func @shared_lhs_fusion_with_final_slice
func.func @shared_lhs_fusion_with_final_slice(%arg0: tensor<32x512xbf16>, %arg1: tensor<512x384xbf16>, %arg2: tensor<512x384xbf16>, %arg3: tensor<512x384xbf16>, %arg4: tensor<384x256xbf16>) -> (tensor<32x384xbf16>, tensor<32x384xbf16>, tensor<1x256xbf16>) {
  // CHECK: %[[W:.*]] = "ttir.concat"({{.*}}) <{dim = 1 : si32}> : (tensor<512x384xbf16>, tensor<512x384xbf16>, tensor<512x384xbf16>) -> tensor<512x1152xbf16>
  // CHECK: %[[FUSED:.*]] = "ttir.matmul"(%arg0, %[[W]]) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x1152xbf16>) -> tensor<32x1152xbf16>
  // CHECK: "ttir.slice_static"(%[[FUSED]]) <{begins = [0 : i32, 384 : i32], ends = [32 : i32, 768 : i32], step = [1 : i32, 1 : i32]}>
  // CHECK: "ttir.slice_static"(%[[FUSED]]) <{begins = [0 : i32, 768 : i32], ends = [32 : i32, 1152 : i32], step = [1 : i32, 1 : i32]}>
  // CHECK: %[[L:.*]] = "ttir.slice_static"(%[[FUSED]]) <{begins = [31 : i32, 0 : i32], ends = [32 : i32, 384 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x1152xbf16>) -> tensor<1x384xbf16>
  // CHECK: "ttir.matmul"(%[[L]], %arg4) <{transpose_a = false, transpose_b = false}> : (tensor<1x384xbf16>, tensor<384x256xbf16>) -> tensor<1x256xbf16>
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
  %1 = "ttir.matmul"(%arg0, %arg2) : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
  %2 = "ttir.matmul"(%arg0, %arg3) : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
  %3 = "ttir.matmul"(%2, %arg4) : (tensor<32x384xbf16>, tensor<384x256xbf16>) -> tensor<32x256xbf16>
  %s = "ttir.slice_static"(%3) <{begins = [31 : i32, 0 : i32], ends = [32 : i32, 256 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x256xbf16>) -> tensor<1x256xbf16>
  return %0, %1, %s : tensor<32x384xbf16>, tensor<32x384xbf16>, tensor<1x256xbf16>
}

// CHECK-LABEL: func.func @shared_lhs_wins_collision
func.func @shared_lhs_wins_collision(%arg0: tensor<32x512xbf16>, %arg1: tensor<512x384xbf16>, %arg2: tensor<512x384xbf16>, %arg3: tensor<512x384xbf16>) -> (tensor<32x384xbf16>, tensor<32x384xbf16>, tensor<1x384xbf16>) {
  // CHECK: %[[W:.*]] = "ttir.concat"({{.*}}) <{dim = 1 : si32}> : (tensor<512x384xbf16>, tensor<512x384xbf16>, tensor<512x384xbf16>) -> tensor<512x1152xbf16>
  // CHECK: %[[FUSED:.*]] = "ttir.matmul"(%arg0, %[[W]]) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x1152xbf16>) -> tensor<32x1152xbf16>
  // CHECK: "ttir.slice_static"(%[[FUSED]]) <{begins = [0 : i32, 384 : i32], ends = [32 : i32, 768 : i32], step = [1 : i32, 1 : i32]}>
  // CHECK: "ttir.slice_static"(%[[FUSED]]) <{begins = [0 : i32, 768 : i32], ends = [32 : i32, 1152 : i32], step = [1 : i32, 1 : i32]}>
  // CHECK: "ttir.slice_static"(%[[FUSED]]) <{begins = [31 : i32, 0 : i32], ends = [32 : i32, 384 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x1152xbf16>) -> tensor<1x384xbf16>
  // CHECK-NOT: "ttir.matmul"
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
  %1 = "ttir.matmul"(%arg0, %arg2) : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
  %2 = "ttir.matmul"(%arg0, %arg3) : (tensor<32x512xbf16>, tensor<512x384xbf16>) -> tensor<32x384xbf16>
  %s = "ttir.slice_static"(%2) <{begins = [31 : i32, 0 : i32], ends = [32 : i32, 384 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x384xbf16>) -> tensor<1x384xbf16>
  return %0, %1, %s : tensor<32x384xbf16>, tensor<32x384xbf16>, tensor<1x384xbf16>
}

// CHECK-LABEL: func.func @cascade_three_matmuls
func.func @cascade_three_matmuls(%arg0: tensor<1024x4096xbf16>, %arg1: tensor<4096x2048xbf16>, %arg2: tensor<2048x1024xbf16>, %arg3: tensor<1024x512xbf16>) -> tensor<1x512xbf16> {
  // CHECK: %[[A:.*]] = "ttir.slice_static"(%arg0) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 4096 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<1x4096xbf16>
  // CHECK: %[[M1:.*]] = "ttir.matmul"(%[[A]], %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1x4096xbf16>, tensor<4096x2048xbf16>) -> tensor<1x2048xbf16>
  // CHECK: %[[M2:.*]] = "ttir.matmul"(%[[M1]], %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1x1024xbf16>
  // CHECK: "ttir.matmul"(%[[M2]], %arg3) <{transpose_a = false, transpose_b = false}> : (tensor<1x1024xbf16>, tensor<1024x512xbf16>) -> tensor<1x512xbf16>
  // CHECK-NOT: "ttir.slice_static"
  %m1 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x2048xbf16>) -> tensor<1024x2048xbf16>
  %m2 = "ttir.matmul"(%m1, %arg2) : (tensor<1024x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<1024x1024xbf16>
  %m3 = "ttir.matmul"(%m2, %arg3) : (tensor<1024x1024xbf16>, tensor<1024x512xbf16>) -> tensor<1024x512xbf16>
  %s = "ttir.slice_static"(%m3) <{begins = [1023 : i32, 0 : i32], ends = [1024 : i32, 512 : i32], step = [1 : i32, 1 : i32]}> : (tensor<1024x512xbf16>) -> tensor<1x512xbf16>
  return %s : tensor<1x512xbf16>
}
