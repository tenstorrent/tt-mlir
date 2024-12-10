// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=BFInterleaved" %s | FileCheck %s
//
//         A
//         |
//         B
//       /   \
//      C     D
//       \   /
//         E
//         |
//         F
//
// There is enough L1 memory to schedule this fork-join without any DRAM spill even though
// output tensors of C and D cannot be in L1 memory at the same time.
//
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<4096x5120xbf16>, %arg1: tensor<5120x1024xbf16>, %arg2: tensor<5120x1024xbf16>) -> tensor<4096x1024xbf16> {
    // CHECK: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    // CHECK: #[[LAYOUT_5:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<512x640xbf16, #l1_>, <interleaved>>
    // CHECK: #[[LAYOUT_6:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<512x128xbf16, #l1_>, <interleaved>>
    %0 = tensor.empty() : tensor<4096x5120xbf16>
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<4096x5120xbf16, #[[LAYOUT_5]]>
    %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<4096x5120xbf16>, tensor<4096x5120xbf16>) -> tensor<4096x5120xbf16>
    %2 = tensor.empty() : tensor<4096x1024xbf16>
    // CHECK: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<4096x1024xbf16, #[[LAYOUT_6]]>
    %3 = "ttir.matmul"(%1, %arg1, %2) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<4096x5120xbf16>, tensor<5120x1024xbf16>, tensor<4096x1024xbf16>) -> tensor<4096x1024xbf16>
    %4 = tensor.empty() : tensor<4096x1024xbf16>
    // CHECK: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<4096x1024xbf16, #[[LAYOUT_6]]>
    %5 = "ttir.matmul"(%1, %arg2, %4) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<4096x5120xbf16>, tensor<5120x1024xbf16>, tensor<4096x1024xbf16>) -> tensor<4096x1024xbf16>
    %6 = tensor.empty() : tensor<4096x1024xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<4096x1024xbf16, #[[LAYOUT_6]]>
    %7 = "ttir.add"(%3, %5, %6) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<4096x1024xbf16>, tensor<4096x1024xbf16>, tensor<4096x1024xbf16>) -> tensor<4096x1024xbf16>
    return %7 : tensor<4096x1024xbf16>
  }
}
