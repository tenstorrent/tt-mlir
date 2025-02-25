// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// XFAIL: *
module attributes {} {
  func.func @forward(%operand: tensor<32000x1024xbf16>, %start_indices: tensor<1x32xbf16>) -> tensor<1x32x1024xbf16> {
    // CHECK: = "ttnn.empty"
    %0 = tensor.empty() : tensor<1x32x1024xbf16>
    // CHECK: %[[C:.*]] = "ttnn.embedding"(%start_indices, %operand, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32xbf16>, tensor<32000x1024xbf16>, tensor<1x32x1024xbf16>) -> tensor<1x32x1024xbf16>
    %1 = "ttir.gather"(%operand, %start_indices, %0) {
        offset_dims = array<i64: 2>,
        collapsed_slice_dims = array<i64: 0>,
        operand_batching_dims = array<i64: 0>,
        start_indices_batching_dims = array<i64: 0>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 1 : si64,
        slice_sizes = array<i64: 1, 1024>,
        indices_are_sorted = false
    } : (tensor<32000x1024xbf16>, tensor<1x32xbf16>, tensor<1x32x1024xbf16>) -> tensor<1x32x1024xbf16>
    return %1 : tensor<1x32x1024xbf16>
  }
}
