// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true insert-memreconfig=relu=0 override-output-layout=relu=tile row-major-enabled=true" -o shard_transpose.mlir %s
// RUN: FileCheck %s --input-file=shard_transpose.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer shard_transpose.mlir > %t.ttnn
// UNSUPPORTED: true
// Test is failing with an ND hang after metal commit daf8fbadc8
// See issue https://github.com/tenstorrent/tt-mlir/issues/3743

// TODO(rpavlovicTT): transpose-op sharding still not supported by default
// It will be fixed with #3205 and #2637.

module attributes {} {
  func.func @main(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x3x224xf32> {
    %0 = ttir.empty() : tensor<1x224x3x224xf32> loc(#loc1)
    // CHECK: #[[LAYOUT_HS:.*]] = #ttnn.ttnn_layout{{.*}}<62x1{{.*}}memref<11x224xf32, #l1>, <height_sharded>>
    // CHECK: {{.*}}"ttnn.transpose"{{.*}}#[[LAYOUT_HS]]>
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x3x224x224xf32>, tensor<1x224x3x224xf32>) -> tensor<1x224x3x224xf32> loc(#loc2)

    %2 = ttir.empty() : tensor<1x224x3x224xf32> loc(#loc3)
    %3 = "ttir.relu"(%1, %2) : (tensor<1x224x3x224xf32>, tensor<1x224x3x224xf32>) -> tensor<1x224x3x224xf32> loc(#loc4)
    return %3 : tensor<1x224x3x224xf32> loc(#loc5)
  }
}

#loc1 = loc("loc":0:0)
#loc2 = loc("loc":5:0)
#loc3 = loc("loc:":10:0)
#loc4 = loc("relu"(#loc3))
#loc5 = loc("loc":20:0)
