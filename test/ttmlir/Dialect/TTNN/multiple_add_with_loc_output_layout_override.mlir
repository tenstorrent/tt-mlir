// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true override-output-layout=add_1_0=4x4:dram:interleaved,add_2_0=4x4:l1:interleaved" %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
#loc = loc("test_ops.py:17_0_0":0:0)
module attributes {} {
  func.func @main(%arg0: tensor<1x32x32xf32> loc("test_ops.py:17_0_0":0:0), %arg1: tensor<1x32x32xf32> loc("test_ops.py:17_0_0":0:0), %arg2: tensor<1x32x32xf32> loc("test_ops.py:17_0_0":0:0)) -> (tensor<1x32x32xf32>, tensor<1x32x32xf32>) {
    // CHECK: #[[L1_:.*]] = #tt.memory_space<l1>
    // CHECK: #[[LAYOUT_0:.*]] = #tt.layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), undef, <1x1>, memref<32x32xf32, #system>>
    // CHECK: #[[LAYOUT_1:.*]] = #tt.layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), undef, <4x4>, memref<8x8xf32, #dram>, interleaved>
    // CHECK: #[[LAYOUT_2:.*]] = #tt.layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), undef, <4x4>, memref<8x8xf32, #l1_>, interleaved>
    // CHECK: #[[LAYOUT_3:.*]] = #tt.layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), undef, <8x8>, memref<4x4xf32, #dram>, interleaved>
    %0 = tensor.empty() : tensor<1x32x32xf32> loc(#loc5)
    // CHECK: %[[C:.*]] = "ttnn.add"[[C:.*]] -> tensor<1x32x32xf32, #[[LAYOUT_1]]>
    %1 = "ttir.add"(%arg1, %arg2, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc5)
    %2 = tensor.empty() : tensor<1x32x32xf32> loc(#loc6)
    // CHECK: %[[C:.*]] = "ttnn.add"[[C:.*]] -> tensor<1x32x32xf32, #[[LAYOUT_2]]>
    %3 = "ttir.add"(%1, %arg0, %2) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc6)
    %4 = tensor.empty() : tensor<1x32x32xf32> loc(#loc7)
    // CHECK: %[[C:.*]] = "ttnn.add"[[C:.*]] -> tensor<1x32x32xf32, #[[LAYOUT_3]]>
    %5 = "ttir.add"(%arg2, %arg1, %4) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc7)
    // CHECK: return %[[R0:.*]], %[[R1:.*]] : tensor<1x32x32xf32, #[[LAYOUT_0]]>, tensor<1x32x32xf32, #[[LAYOUT_0]]>
    return %3, %5 : tensor<1x32x32xf32>, tensor<1x32x32xf32> loc(#loc4)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("test_ops.py:17_0_0":0:4)
#loc2 = loc("test_ops.py:17_0_0":0:6)
#loc3 = loc("test_ops.py:17_0_0":0:3)
#loc4 = loc(unknown)
#loc5 = loc("add_1_0"(#loc1))
#loc6 = loc("add_2_0"(#loc2))
#loc7 = loc("add_0"(#loc3))
