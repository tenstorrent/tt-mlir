// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true memreconfig-enabled=true insert-memreconfig=relu=0 override-output-layout=relu=tile row-major-enabled=true" %s | FileCheck %s

module attributes {} {
  func.func @main(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x3x224xf32> {
    %0 = ttir.empty() : tensor<1x224x3x224xf32> loc(#loc1)
    // CHECK: #[[LAYOUT_HS:.*]] = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 672 + d1 * 224 + d2, d3), <62x1, (d0, d1) -> (0, d0 floordiv 8, d0 mod 8)>, memref<11x224xf32, #l1_>, <height_sharded>>
    // CHECK: %4 = "ttnn.transpose"(%3){{.*}}#[[LAYOUT_HS]]>
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x3x224x224xf32>, tensor<1x224x3x224xf32>) -> tensor<1x224x3x224xf32> loc(#loc2)

    %2 = ttir.empty() : tensor<1x224x3x224xf32> loc(#loc3)
    %3 = "ttir.relu"(%1, %2) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x224x3x224xf32>, tensor<1x224x3x224xf32>) -> tensor<1x224x3x224xf32> loc(#loc4)
    return %3 : tensor<1x224x3x224xf32> loc(#loc5)
  }
}

#loc1 = loc("loc":0:0)
#loc2 = loc("loc":5:0)
#loc3 = loc("loc:":10:0)
#loc4 = loc("relu"(#loc3))
#loc5 = loc("loc":20:0)
