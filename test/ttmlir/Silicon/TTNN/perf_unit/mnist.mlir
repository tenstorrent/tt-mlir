// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
#loc = loc("MNISTLinear":4294967295:0)
module @"tt-forge-graph" attributes {} {
  func.func @main(%arg0: tensor<1x784xf32> loc("MNISTLinear":4294967295:0), %arg1: tensor<1x10xf32> loc("MNISTLinear":4294967295:0), %arg2: tensor<256x10xf32> loc("MNISTLinear":4294967295:0), %arg3: tensor<1x256xf32> loc("MNISTLinear":4294967295:0), %arg4: tensor<784x256xf32> loc("MNISTLinear":4294967295:0)) -> tensor<1x10xf32> {
    // CHECK: #[[LAYOUT_10:.*]] = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x8>, memref<1x32xf32, #l1_>, block_sharded>
    // CHECK: #[[LAYOUT_11:.*]] = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<1x10xf32, #l1_>, block_sharded>
    %0 = tensor.empty() : tensor<1x256xf32> loc(#loc8)
    // CHECK: %[[C:.*]] = "ttnn.matmul"[[C:.*]] -> tensor<1x256xf32, #[[LAYOUT_10]]>
    %1 = "ttir.matmul"(%arg0, %arg4, %0) <{operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<1x784xf32>, tensor<784x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32> loc(#loc8)
    %2 = tensor.empty() : tensor<1x256xf32> loc(#loc9)
    // CHECK: %[[C:.*]] = "ttnn.add"[[C:.*]] -> tensor<1x256xf32, #[[LAYOUT_10]]>
    %3 = "ttir.add"(%1, %arg3, %2) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32> loc(#loc9)
    %4 = tensor.empty() : tensor<1x256xf32> loc(#loc10)
    // CHECK: %[[C:.*]] = "ttnn.relu"[[C:.*]] -> tensor<1x256xf32, #[[LAYOUT_10]]>
    %5 = "ttir.relu"(%3, %4) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32> loc(#loc10)
    %6 = tensor.empty() : tensor<1x10xf32> loc(#loc11)
    // CHECK: %[[C:.*]] = "ttnn.matmul"[[C:.*]] -> tensor<1x10xf32, #[[LAYOUT_11]]>
    %7 = "ttir.matmul"(%5, %arg2, %6) <{operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<1x256xf32>, tensor<256x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32> loc(#loc11)
    %8 = tensor.empty() : tensor<1x10xf32> loc(#loc12)
    // CHECK: %[[C:.*]] = "ttnn.add"[[C:.*]] -> tensor<1x10xf32, #[[LAYOUT_11]]>
    %9 = "ttir.add"(%7, %arg1, %8) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<1x10xf32>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32> loc(#loc12)
    %10 = tensor.empty() : tensor<1x10xf32> loc(#loc13)
    %11 = "ttir.softmax"(%9, %10) <{dimension = 1 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32> loc(#loc13)
    return %11 : tensor<1x10xf32> loc(#loc7)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("MNISTLinear":4294967295:10)
#loc2 = loc("MNISTLinear":4294967295:8)
#loc3 = loc("MNISTLinear":4294967295:6)
#loc4 = loc("MNISTLinear":4294967295:4)
#loc5 = loc("MNISTLinear":4294967295:3)
#loc6 = loc("MNISTLinear":4294967295:2)
#loc7 = loc(unknown)
#loc8 = loc("matmul_1"(#loc1))
#loc9 = loc("add_2"(#loc2))
#loc10 = loc("relu_3"(#loc3))
#loc11 = loc("matmul_5"(#loc4))
#loc12 = loc("add_6"(#loc5))
#loc13 = loc("softmax_7"(#loc6))
