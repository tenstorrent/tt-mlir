// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s -o %t.mlir --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%"
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

func.func @test_sort_ascending_none_stable(%arg0: tensor<1x128x256x256xbf16>) -> (tensor<1x128x256x256xbf16>, tensor<1x128x256x256xi32>) {
  // CHECK-LABEL: @test_sort_ascending_none_stable
  %0 = stablehlo.iota dim = 0 : tensor<256xi32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<256xi32>) -> tensor<1x128x256x256xi32>
  // CHECK: %{{.*}}, %{{.*}} = "ttnn.sort"(%arg0)
  // CHECK-SAME: <{descending = false, dim = 3 : si8, stable = false}>
  // CHECK-SAME: (tensor<1x128x256x256xbf16,
  // CHECK-SAME: -> (tensor<1x128x256x256xbf16,
  // CHECK-SAME: tensor<1x128x256x256xui16,
  %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 3 : i64}> ({
  ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>, %arg3: tensor<i32>, %arg4: tensor<i32>):
    %4 = stablehlo.compare  LT, %arg1, %arg2,  TOTALORDER : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  }) : (tensor<1x128x256x256xbf16>, tensor<1x128x256x256xi32>) -> (tensor<1x128x256x256xbf16>, tensor<1x128x256x256xi32>)
  return %2#0, %2#1 : tensor<1x128x256x256xbf16>, tensor<1x128x256x256xi32>
}

func.func @test_sort_descending_stable(%arg0: tensor<1x128x256x256xbf16>) -> (tensor<1x128x256x256xbf16>, tensor<1x128x256x256xi32>) {
  // CHECK-LABEL: @test_sort_descending_stable
  %0 = stablehlo.iota dim = 0 : tensor<256xi32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<256xi32>) -> tensor<1x128x256x256xi32>
  // CHECK: %{{.*}}, %{{.*}} = "ttnn.sort"(%arg0)
  // CHECK-SAME: <{descending = true, dim = 3 : si8, stable = true}>
  // CHECK-SAME: (tensor<1x128x256x256xbf16,
  // CHECK-SAME: -> (tensor<1x128x256x256xbf16,
  // CHECK-SAME: tensor<1x128x256x256xui16,
  %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 3 : i64, is_stable = true}> ({
  ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>, %arg3: tensor<i32>, %arg4: tensor<i32>):
    %4 = stablehlo.compare  GT, %arg1, %arg2,  TOTALORDER : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  }) : (tensor<1x128x256x256xbf16>, tensor<1x128x256x256xi32>) -> (tensor<1x128x256x256xbf16>, tensor<1x128x256x256xi32>)
  return %2#0, %2#1 : tensor<1x128x256x256xbf16>, tensor<1x128x256x256xi32>
}
