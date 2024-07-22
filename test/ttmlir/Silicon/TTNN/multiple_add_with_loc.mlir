// RUN: ttmlir-opt --ttir-load-system-desc --ttir-to-ttnn-backend-pipeline %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
#loc = loc("test_ops.py:17_0_0":0:0)
module @pybuda_graph attributes {} {
  func.func @main(%arg0: tensor<1x32x32xf32> loc("test_ops.py:17_0_0":0:0), %arg1: tensor<1x32x32xf32> loc("test_ops.py:17_0_0":0:0), %arg2: tensor<1x32x32xf32> loc("test_ops.py:17_0_0":0:0)) -> (tensor<1x32x32xf32>, tensor<1x32x32xf32>) {
    // CHECK: #layout1 = #tt.layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), undef, <8x8>, memref<4x4xf32, #system>>
    // CHECK: #layout2 = #tt.layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), undef, <8x8>, memref<4x4xf32, #l1_>>
    %0 = tensor.empty() : tensor<1x32x32xf32> loc(#loc5)
    // CHECK: %[[C:.*]] = "ttnn.add"[[C:.*]] -> tensor<1x32x32xf32, #layout2>
    %1 = "ttir.add"(%arg1, %arg2, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc5)
    %2 = tensor.empty() : tensor<1x32x32xf32> loc(#loc6)
    // CHECK: %[[C:.*]] = "ttnn.add"[[C:.*]] -> tensor<1x32x32xf32, #layout2>
    %3 = "ttir.add"(%1, %arg0, %2) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc6)
    %4 = tensor.empty() : tensor<1x32x32xf32> loc(#loc7)
    // CHECK: %[[C:.*]] = "ttnn.add"[[C:.*]] -> tensor<1x32x32xf32, #layout2>
    %5 = "ttir.add"(%arg2, %arg1, %4) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<1x32x32xf32>, tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> tensor<1x32x32xf32> loc(#loc7)
    // CHECK: return %20, %22 : tensor<1x32x32xf32, #layout1>, tensor<1x32x32xf32, #layout1>
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
