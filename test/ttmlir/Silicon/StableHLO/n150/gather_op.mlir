// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  // Examples 0 to 5 are from models.
  // CHECK-LABEL: func.func @gather_0
  func.func @gather_0(%operand: tensor<32000x1024xbf16>, %start_indices: tensor<1x32xi32>) -> tensor<1x32x1024xbf16> {
    // CHECK: "ttnn.embedding"
    // CHECK-SAME: (tensor<1x32xui32, {{.+}}>, tensor<32000x1024xbf16, {{.+}}>) -> tensor<1x32x1024xbf16, {{.+}}>
    %1 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1024>}> : (tensor<32000x1024xbf16>, tensor<1x32xi32>) -> tensor<1x32x1024xbf16>
    return %1 : tensor<1x32x1024xbf16>
  }

  // CHECK-LABEL: func.func @gather_1
  func.func @gather_1(%operand: tensor<448x384xbf16>, %start_indices: tensor<1x2x1xi32>) -> tensor<1x2x384xbf16> {
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.embedding"
    // CHECK-SAME: (tensor<1x2xui32, {{.+}}>, tensor<448x384xbf16, {{.+}}>) -> tensor<1x2x384xbf16, {{.+}}>
    %1 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 384>}> : (tensor<448x384xbf16>, tensor<1x2x1xi32>) -> tensor<1x2x384xbf16>
    return %1 : tensor<1x2x384xbf16>
  }

  // CHECK-LABEL: func.func @gather_2
  func.func @gather_2(%operand: tensor<51864x384xbf16>, %start_indices: tensor<1x2xi32>) -> tensor<1x2x384xbf16> {
    // CHECK: "ttnn.embedding"
    // CHECK-SAME: (tensor<1x2xui32, {{.+}}>, tensor<51864x384xbf16, {{.+}}>) -> tensor<1x2x384xbf16, {{.+}}>
    %1 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 384>}> : (tensor<51864x384xbf16>, tensor<1x2xi32>) -> tensor<1x2x384xbf16>
    return %1 : tensor<1x2x384xbf16>
  }

  // CHECK-LABEL: func.func @gather_3
  func.func @gather_3(%operand: tensor<732x12xf32>, %start_indices: tensor<38809x1xi32>) -> tensor<38809x12xf32> {
    // CHECK: "ttnn.embedding"
    // CHECK-SAME: (tensor<38809x1xui32, {{.+}}>, tensor<732x12xbf16, {{.+}}>) -> tensor<38809x1x12xbf16, {{.+}}>
    // CHECK: "ttnn.reshape"
    %1 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 12>}> : (tensor<732x12xf32>, tensor<38809x1xi32>) -> tensor<38809x12xf32>
    return %1 : tensor<38809x12xf32>
  }

  // CHECK-LABEL: func.func @gather_4
  func.func @gather_4(%operand: tensor<2048x1x200xf32>, %start_indices: tensor<1x2x1xi32>) -> tensor<1x2x1x200xf32> {
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.embedding"
    // CHECK-SAME: (tensor<1x2xui32, {{.+}}>, tensor<2048x200xbf16, {{.+}}>) -> tensor<1x2x200xbf16, {{.+}}>
    // CHECK: "ttnn.reshape"
    %1 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 200>}> : (tensor<2048x1x200xf32>, tensor<1x2x1xi32>) -> tensor<1x2x1x200xf32>
    return %1 : tensor<1x2x1x200xf32>
  }

  // CHECK-LABEL: func.func @gather_5
  func.func @gather_5(%operand: tensor<2x7x512xf32>, %start_indices: tensor<2x2xi32>) -> (tensor<2x512xf32> {jax.result_info = "result"}) {
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.typecast"
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.embedding"
    // CHECK-SAME: (tensor<2x1xui32, {{.+}}>, tensor<14x512xbf16, {{.+}}>) -> tensor<2x1x512xbf16, {{.+}}>
    // CHECK: "ttnn.reshape"
    %1 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 512>}> : (tensor<2x7x512xf32>, tensor<2x2xi32>) -> tensor<2x512xf32>
    return %1 : tensor<2x512xf32>
  }

  // Examples 6 to 8 test different rank combinations for input, start indices and output.
  // CHECK-LABEL: func.func @gather_6
  func.func @gather_6(%operand: tensor<6xbf16>, %start_indices: tensor<1xi32>) -> (tensor<1xbf16> {jax.result_info = "result"}) {
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.embedding"
    // CHECK-SAME: (tensor<1x1xui32, {{.+}}>, tensor<6x1xbf16, {{.+}}>) -> tensor<1x1x1xbf16, {{.+}}>
    // CHECK: "ttnn.reshape"
    %1 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<6xbf16>, tensor<1xi32>) -> tensor<1xbf16>
    return %1 : tensor<1xbf16>
  }

  // CHECK-LABEL: func.func @gather_7
  func.func @gather_7(%operand: tensor<6x4xbf16>, %start_indices: tensor<3x1xi32>) -> tensor<3x4xbf16> {
    // CHECK: "ttnn.embedding"
    // CHECK-SAME: (tensor<3x1xui32, {{.+}}>, tensor<6x4xbf16, {{.+}}>) -> tensor<3x1x4xbf16, {{.+}}>
    // CHECK: "ttnn.reshape"
    %1 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4>}> : (tensor<6x4xbf16>, tensor<3x1xi32>) -> tensor<3x4xbf16>
    return %1 : tensor<3x4xbf16>
  }

  // CHECK-LABEL: func.func @gather_8
  func.func @gather_8(%operand: tensor<6x4xbf16>, %start_indices: tensor<3x1xi32>) -> tensor<3x1x4xbf16> {
    // CHECK: "ttnn.embedding"
    // CHECK-SAME: (tensor<3x1xui32, {{.+}}>, tensor<6x4xbf16, {{.+}}>) -> tensor<3x1x4xbf16, {{.+}}>
    %1 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4>}> : (tensor<6x4xbf16>, tensor<3x1xi32>) -> tensor<3x1x4xbf16>
    return %1 : tensor<3x1x4xbf16>
  }

  // In examples 9 to 14 more than one dim is being indexed
  // CHECK-LABEL: func.func @gather_9
  func.func @gather_9(%operand: tensor<3x2x3xf32>, %start_indices: tensor<1x2xi32>) -> (tensor<1x3xf32> {jax.result_info = "result"}) {
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.typecast"
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.embedding"
    // CHECK-SAME: (tensor<1x1xui32, {{.+}}>, tensor<6x3xbf16, {{.+}}>) -> tensor<1x1x3xbf16, {{.+}}>
    // CHECK: "ttnn.reshape"
    %1 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 3>}> : (tensor<3x2x3xf32>, tensor<1x2xi32>) -> tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }

  // CHECK-LABEL: func.func @gather_10
  func.func @gather_10(%operand: tensor<7x8x2xf32>, %start_indices: tensor<2x2x2xi32>) -> (tensor<2x2x2xf32> {jax.result_info = "result"}) {
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.typecast"
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.embedding"
    // CHECK-SAME: (tensor<1x4xui32, {{.+}}>, tensor<56x2xbf16, {{.+}}>) -> tensor<1x4x2xbf16, {{.+}}>
    // CHECK: "ttnn.reshape"
    %1 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 2>}> : (tensor<7x8x2xf32>, tensor<2x2x2xi32>) -> tensor<2x2x2xf32>
    return %1 : tensor<2x2x2xf32>
  }

  // CHECK-LABEL: func.func @gather_11
  func.func @gather_11(%operand: tensor<18x17x2xf32>, %start_indices: tensor<3x1x3x2xi32>) -> (tensor<3x1x3x2xf32> {jax.result_info = "result"}) {
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.typecast"
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.embedding"
    // CHECK-SAME: (tensor<1x9xui32, {{.+}}>, tensor<306x2xbf16, {{.+}}>) -> tensor<1x9x2xbf16, {{.+}}>
    // CHECK: "ttnn.reshape"
    %1 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [3], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 2>}> : (tensor<18x17x2xf32>, tensor<3x1x3x2xi32>) -> tensor<3x1x3x2xf32>
    return %1 : tensor<3x1x3x2xf32>
  }

  // CHECK-LABEL: func.func @gather_12
  func.func @gather_12(%operand: tensor<4x5x2x2xf32>, %start_indices: tensor<2x1x1x2xi32>) -> (tensor<2x1x1x2x2xf32> {jax.result_info = "result"}) {
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.typecast"
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.embedding"
    // CHECK-SAME: (tensor<1x2xui32, {{.+}}>, tensor<20x4xbf16, {{.+}}>) -> tensor<1x2x4xbf16, {{.+}}>
    // CHECK: "ttnn.reshape"
    %1 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [3, 4], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 2, 2>}> : (tensor<4x5x2x2xf32>, tensor<2x1x1x2xi32>) -> tensor<2x1x1x2x2xf32>
    return %1 : tensor<2x1x1x2x2xf32>
  }

  // CHECK-LABEL: func.func @gather_13
  func.func @gather_13(%operand: tensor<1x7x2xbf16>, %start_indices: tensor<1x2xi32>) -> (tensor<1x2xbf16> {jax.result_info = "result"}) {
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.embedding"
    // CHECK-SAME: (tensor<1x1xui32, {{.+}}>, tensor<7x2xbf16, {{.+}}>) -> tensor<1x1x2xbf16, {{.+}}>
    // CHECK: "ttnn.reshape"
    %1 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 2>}> : (tensor<1x7x2xbf16>, tensor<1x2xi32>) -> tensor<1x2xbf16>
    return %1 : tensor<1x2xbf16>
  }

  // CHECK-LABEL: func.func @gather_14
  func.func @gather_14(%operand: tensor<3x2x3x4xf32>, %start_indices: tensor<1x3xi32>) -> (tensor<1x4xf32> {jax.result_info = "result"}) {
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.typecast"
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.embedding"
    // CHECK-SAME: (tensor<1x1xui32, {{.+}}>, tensor<18x4xbf16, {{.+}}>) -> tensor<1x1x4xbf16, {{.+}}>
    // CHECK: "ttnn.reshape"
    %1 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1, 4>}> : (tensor<3x2x3x4xf32>, tensor<1x3xi32>) -> tensor<1x4xf32>
    return %1 : tensor<1x4xf32>
  }

  // In examples 15 and 16 permute ops are needed
  // CHECK-LABEL: func.func @gather_15
  func.func @gather_15(%operand: tensor<1x2x3x5xf32>, %start_indices: tensor<4x1xi32>) -> (tensor<1x2x3x4xf32> {jax.result_info = "result"}) {
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.embedding"
    // CHECK-SAME: (tensor<4x1xui32, {{.+}}>, tensor<5x6xbf16, {{.+}}>) -> tensor<4x1x6xbf16, {{.+}}>
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.permute"
    %1 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 2, 3, 1>}> : (tensor<1x2x3x5xf32>, tensor<4x1xi32>) -> tensor<1x2x3x4xf32>
    return %1 : tensor<1x2x3x4xf32>
  }

  // CHECK-LABEL: func.func @gather_16
  func.func @gather_16(%operand: tensor<1x2x5x7xf32>, %start_indices: tensor<3x2xi32>) -> (tensor<1x2x3xf32> {jax.result_info = "result"}) {
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.typecast"
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.embedding"
    // CHECK-SAME: (tensor<3x1xui32, {{.+}}>, tensor<35x2xbf16, {{.+}}>) -> tensor<3x1x2xbf16, {{.+}}>
    // CHECK: "ttnn.permute"
    %1 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2, 3], start_index_map = [2, 3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 2, 1, 1>}> : (tensor<1x2x5x7xf32>, tensor<3x2xi32>) -> tensor<1x2x3xf32>
    return %1 : tensor<1x2x3xf32>
  }

  // Example 17: i64 start indices
  // CHECK-LABEL: func.func @gather_17
  func.func @gather_17(%operand: tensor<32128x512xbf16>, %start_indices: tensor<1x15xi64>) -> tensor<1x15x512xbf16> {
    // CHECK: "ttnn.embedding"
    // CHECK-SAME: (tensor<1x15xui32, {{.+}}>, tensor<32128x512xbf16, {{.+}}>) -> tensor<1x15x512xbf16, {{.+}}>
    %1 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 512>}> : (tensor<32128x512xbf16>, tensor<1x15xi64>) -> tensor<1x15x512xbf16>
    return %1 : tensor<1x15x512xbf16>
  }
}
