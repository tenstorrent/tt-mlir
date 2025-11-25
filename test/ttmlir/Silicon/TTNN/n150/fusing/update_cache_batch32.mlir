
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// %135 = "ttir.scatter"(%arg6, %29, %133, %134) <{index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 2>, scatter_dims_to_operand_dims = array<i32: 2>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 0, 1, 3>}> : (tensor<32x8x128x128xbf16>, tensor<1x1xi64>, tensor<32x8x1x128xbf16>, tensor<32x8x128x128xbf16>) -> tensor<32x8x128x128xbf16> loc(#loc372)

module {
  func.func @forward(%cache: tensor<32x8x128x128xbf16>, %indices: tensor<1xi64>, %fill_value: tensor<32x8x1x128xbf16>) -> tensor<32x8x128x128xbf16> {
    // CHECK: "ttnn.permute"
    // CHECK: "ttnn.paged_update_cache"
    %dps = ttir.empty() : tensor<32x8x128x128xbf16>
    %updated_cache = "ttir.scatter"(%cache, %indices, %fill_value, %dps) <{index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 2>, scatter_dims_to_operand_dims = array<i32: 2>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 0, 1, 3>}> : (tensor<32x8x128x128xbf16>, tensor<1xi64>, tensor<32x8x1x128xbf16>, tensor<32x8x128x128xbf16>) -> tensor<32x8x128x128xbf16>
    return %updated_cache : tensor<32x8x128x128xbf16>
  }
}
