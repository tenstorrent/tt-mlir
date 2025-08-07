// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @test_sort_ascending_none_stable(%arg0: tensor<1x128x256x256xbf16>) -> (tensor<1x128x256x256xbf16>, tensor<1x128x256x256xi32>) {
  // CHECK-LABEL: @test_sort_ascending_none_stable
  %0 = stablehlo.iota dim = 0 : tensor<256xi32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<256xi32>) -> tensor<1x128x256x256xi32>
  // CHECK: %{{.*}}, %{{.*}} = "ttir.sort"(%arg0, %{{[0-9]+}}, %{{[0-9]+}})
  // CHECK-SAME: <{descending = false, dim = 3 : si32, stable = false}>
  // CHECK-SAME: (tensor<1x128x256x256xbf16>, tensor<1x128x256x256xbf16>, tensor<1x128x256x256xi32>)
  // CHECK-SAME: -> (tensor<1x128x256x256xbf16>, tensor<1x128x256x256xi32>)
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
  // CHECK: %{{.*}}, %{{.*}} = "ttir.sort"(%arg0, %{{[0-9]+}}, %{{[0-9]+}})
  // CHECK-SAME: <{descending = true, dim = 3 : si32, stable = true}>
  // CHECK-SAME: (tensor<1x128x256x256xbf16>, tensor<1x128x256x256xbf16>, tensor<1x128x256x256xi32>)
  // CHECK-SAME: -> (tensor<1x128x256x256xbf16>, tensor<1x128x256x256xi32>)
  %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 3 : i64, is_stable = true}> ({
  ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>, %arg3: tensor<i32>, %arg4: tensor<i32>):
    %4 = stablehlo.compare  GT, %arg1, %arg2,  TOTALORDER : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  }) : (tensor<1x128x256x256xbf16>, tensor<1x128x256x256xi32>) -> (tensor<1x128x256x256xbf16>, tensor<1x128x256x256xi32>)
  return %2#0, %2#1 : tensor<1x128x256x256xbf16>, tensor<1x128x256x256xi32>
}
