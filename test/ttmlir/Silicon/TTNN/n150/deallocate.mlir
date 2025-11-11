// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
#loc = loc("Dealloc":4294967295:0)
module @"dealloc_test" attributes {} {
  func.func @main(%arg0: tensor<1x784xf32> loc("Dealloc":4294967295:0), %arg1: tensor<1x10xf32> loc("Dealloc":4294967295:0), %arg2: tensor<256x10xf32> loc("Dealloc":4294967295:0), %arg3: tensor<1x256xf32> loc("Dealloc":4294967295:0), %arg4: tensor<784x256xf32> loc("Dealloc":4294967295:0)) -> tensor<1x10xf32> {
    %0 = ttir.empty() : tensor<1x256xf32> loc(#loc8)
    %1 = "ttir.matmul"(%arg0, %arg4, %0) : (tensor<1x784xf32>, tensor<784x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32> loc(#loc8)
    %2 = ttir.empty() : tensor<1x256xf32> loc(#loc9)
    %3 = "ttir.add"(%1, %arg3, %2) : (tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32> loc(#loc9)
    %4 = ttir.empty() : tensor<1x256xf32> loc(#loc10)
    %5 = "ttir.relu"(%3, %4) : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32> loc(#loc10)
    // CHECK: %{{.+}} = "ttnn.relu"([[I1:%.+]]) {{.+}} -> tensor<1x256xf32, {{.+}}>
    %6 = ttir.empty() : tensor<1x10xf32> loc(#loc11)
    %7 = "ttir.matmul"(%5, %arg2, %6) : (tensor<1x256xf32>, tensor<256x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32> loc(#loc11)
    %8 = ttir.empty() : tensor<1x10xf32> loc(#loc12)
    %9 = "ttir.add"(%7, %arg1, %8) : (tensor<1x10xf32>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32> loc(#loc12)
    %10 = ttir.empty() : tensor<1x10xf32> loc(#loc13)
    %11 = "ttir.softmax"(%9, %10) <{dimension = 1 : si32}> : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32> loc(#loc13)
    // CHECK: %{{.+}} = "ttnn.softmax"([[I1:%.+]]) {{.+}} -> tensor<1x10xf32, {{.+}}>
    return %11 : tensor<1x10xf32> loc(#loc7)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("Dealloc":4294967295:10)
#loc2 = loc("Dealloc":4294967295:8)
#loc3 = loc("Dealloc":4294967295:6)
#loc4 = loc("Dealloc":4294967295:4)
#loc5 = loc("Dealloc":4294967295:3)
#loc6 = loc("Dealloc":4294967295:2)
#loc7 = loc(unknown)
#loc8 = loc("matmul_1"(#loc1))
#loc9 = loc("add_2"(#loc2))
#loc10 = loc("relu_3"(#loc3))
#loc11 = loc("matmul_5"(#loc4))
#loc12 = loc("add_6"(#loc5))
#loc13 = loc("softmax_7"(#loc6))
