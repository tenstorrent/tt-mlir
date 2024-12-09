// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @gather_0(%operand: tensor<32000x1024xbf16>, %start_indices: tensor<1x32xi32>) -> tensor<1x32x1024xbf16> {
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %0 = tensor.empty() : tensor<1x32x1024xbf16>
    // CHECK: %[[C:.*]] = "ttnn.embedding"[[C:.*]]
    %1 = "ttir.gather"(%operand, %start_indices, %0) {
        offset_dims = array<i64: 2>,
        collapsed_slice_dims = array<i64: 0>,
        operand_batching_dims = array<i64: 0>,
        start_indices_batching_dims = array<i64: 0>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 1 : si64,
        slice_sizes = array<i64: 1, 1024>,
        indices_are_sorted = false,
        operand_constraints = [#any_device, #any_device, #any_device]
    } : (tensor<32000x1024xbf16>, tensor<1x32xi32>, tensor<1x32x1024xbf16>) -> tensor<1x32x1024xbf16>
    return %1 : tensor<1x32x1024xbf16>
  }

  func.func @gather_1(%operand: tensor<448x384xbf16>, %start_indices: tensor<1x2x1xi32>) -> tensor<1x2x384xbf16> {
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %0 = tensor.empty() : tensor<1x2x384xbf16>
    // CHECK: %[[C:.*]] = "ttnn.embedding"[[C:.*]]
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
        offset_dims = array<i64: 2>,
        collapsed_slice_dims = array<i64: 0>,
        operand_batching_dims = array<i64: 0>,
        start_indices_batching_dims = array<i64: 0>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 2 : si64,
        slice_sizes = array<i64: 1, 384>,
        indices_are_sorted = false,
        operand_constraints = [#any_device, #any_device, #any_device]
      }> : (tensor<448x384xbf16>, tensor<1x2x1xi32>, tensor<1x2x384xbf16>) -> tensor<1x2x384xbf16>
    return %1 : tensor<1x2x384xbf16>
  }

  func.func @gather_2(%operand: tensor<51864x384xbf16>, %start_indices: tensor<1x2xi32>) -> tensor<1x2x384xbf16> {
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %0 = tensor.empty() : tensor<1x2x384xbf16>
    // CHECK: %[[C:.*]] = "ttnn.embedding"[[C:.*]]
    %1 = "ttir.gather"(%operand, %start_indices, %0) <{
        offset_dims = array<i64: 2>,
        collapsed_slice_dims = array<i64: 0>,
        operand_batching_dims = array<i64: 0>,
        start_indices_batching_dims = array<i64: 0>,
        start_index_map = array<i64: 0>,
        index_vector_dim = 2 : si64,
        slice_sizes = array<i64: 1, 384>,
        indices_are_sorted = false,
        operand_constraints = [#any_device, #any_device, #any_device]
      }> : (tensor<51864x384xbf16>, tensor<1x2xi32>, tensor<1x2x384xbf16>) -> tensor<1x2x384xbf16>
    return %1 : tensor<1x2x384xbf16>
  }
}
