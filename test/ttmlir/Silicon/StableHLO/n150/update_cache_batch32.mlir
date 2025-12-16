
// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s -o %t.mlir --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%"
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s


module {
  func.func @forward(%cache: tensor<32x8x128x128xbf16>, %indices: tensor<1xi64>, %fill_value: tensor<32x8x1x128xbf16>) -> tensor<32x8x128x128xbf16> {
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.paged_update_cache"
    %updated_cache = "stablehlo.scatter"(%cache, %indices, %fill_value) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 3], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<bf16>, %arg4: tensor<bf16>):
      stablehlo.return %arg4 : tensor<bf16>
    }) : (tensor<32x8x128x128xbf16>, tensor<1xi64>, tensor<32x8x1x128xbf16>) -> tensor<32x8x128x128xbf16>
    return %updated_cache : tensor<32x8x128x128xbf16>
  }
}
