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

func.func @test_sort_values_only(%arg0: tensor<2x3xi32>) -> (tensor<2x3xi32> {jax.result_info = "result"}) {
  // CHECK-LABEL: @test_sort_values_only
  // CHECK: %{{.*}}, %{{.*}} = "ttir.sort"(%arg0, %{{[0-9]+}}, %{{[0-9]+}})
  // CHECK-SAME: <{descending = false, dim = 1 : si32, stable = true}>
  // CHECK-SAME: (tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>)
  // CHECK-SAME: -> (tensor<2x3xi32>, tensor<2x3xi32>)
  %0 = "stablehlo.sort"(%arg0) <{dimension = 1 : i64, is_stable = true}> ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %1 = stablehlo.compare  LT, %arg1, %arg2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  }) : (tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

func.func public @test_sort_key_values(%arg0: tensor<1x4x64x64xf32>, %arg1: tensor<1x4x64x64xi32>, %arg2: tensor<1x4x64x64xi32>, %arg3: tensor<1x4x64x64xi32>) -> (tensor<1x4x64x64xf32>, tensor<1x4x64x64xi32>, tensor<1x4x64x64xi32>, tensor<1x4x64x64xi32>) {
  // CHECK-LABEL: @test_sort_key_values
  // CHECK: %{{.*}}, %[[INDICES:.*]] = "ttir.sort"(%arg0, %{{[0-9]+}}, %{{[0-9]+}})
  // CHECK-SAME: <{descending = false, dim = 3 : si32, stable = true}>
  // CHECK-SAME: (tensor<1x4x64x64xf32>, tensor<1x4x64x64xf32>, tensor<1x4x64x64xi32>)
  // CHECK-SAME: -> (tensor<1x4x64x64xf32>, tensor<1x4x64x64xi32>)
  // CHECK: %[[INDICES_RESHAPE:[0-9]+]] = "ttir.reshape"(%[[INDICES]], %{{[0-9]+}})
  // CHECK-SAME: <{shape = [1 : i32, 4 : i32, 64 : i32, 64 : i32, 1 : i32]}>
  // CHECK-SAME: (tensor<1x4x64x64xi32>, tensor<1x4x64x64x1xi32>) -> tensor<1x4x64x64x1xi32>
  // CHECK: %[[ARANGE0:[0-9]+]] = "ttir.arange"()
  // CHECK-SAME: <{arange_dimension = 0 : i64, dtype = i32, end = 1 : si64, start = 0 : si64, step = 1 : si64}>
  // CHECK-SAME: -> tensor<1x4x64x64x1xi32>
  // CHECK: %[[ARANGE1:[0-9]+]] = "ttir.arange"()
  // CHECK-SAME: <{arange_dimension = 1 : i64, dtype = i32, end = 4 : si64, start = 0 : si64, step = 1 : si64}>
  // CHECK-SAME: -> tensor<1x4x64x64x1xi32>
  // CHECK: %[[ARANGE2:[0-9]+]] = "ttir.arange"()
  // CHECK-SAME: <{arange_dimension = 2 : i64, dtype = i32, end = 64 : si64, start = 0 : si64, step = 1 : si64}>
  // CHECK-SAME: -> tensor<1x4x64x64x1xi32>
  // CHECK: %[[REORDER_INDICES:[0-9]+]] = "ttir.concat"
  // CHECK-SAME: (%[[ARANGE0]], %[[ARANGE1]], %[[ARANGE2]], %[[INDICES_RESHAPE]], %{{[0-9]+}})
  // CHECK-SAME: <{dim = 4 : si32}>
  // CHECK-SAME: (tensor<1x4x64x64x1xi32>, tensor<1x4x64x64x1xi32>, tensor<1x4x64x64x1xi32>, tensor<1x4x64x64x1xi32>, tensor<1x4x64x64x4xi32>)
  // CHECK-SAME: -> tensor<1x4x64x64x4xi32>
  // CHECK: %{{.*}} = "ttir.gather"(%arg1, %[[REORDER_INDICES]], %{{[0-9]+}})
  // CHECK: %{{.*}} = "ttir.gather"(%arg2, %[[REORDER_INDICES]], %{{[0-9]+}})
  // CHECK: %{{.*}} = "ttir.gather"(%arg3, %[[REORDER_INDICES]], %{{[0-9]+}})
  %0:4 = "stablehlo.sort"(%arg0, %arg1, %arg2, %arg3) <{dimension = 3 : i64, is_stable = true}> ({
  ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<i32>, %arg7: tensor<i32>, %arg8: tensor<i32>, %arg9: tensor<i32>, %arg10: tensor<i32>, %arg11: tensor<i32>):
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.compare  EQ, %arg4, %cst,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.select %1, %cst, %arg4 : tensor<i1>, tensor<f32>
    %3 = stablehlo.compare  NE, %arg4, %arg4,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %4 = stablehlo.select %3, %cst_0, %2 : tensor<i1>, tensor<f32>
    %5 = stablehlo.compare  EQ, %arg5, %cst,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = stablehlo.select %5, %cst, %arg5 : tensor<i1>, tensor<f32>
    %7 = stablehlo.compare  NE, %arg5, %arg5,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %8 = stablehlo.select %7, %cst_0, %6 : tensor<i1>, tensor<f32>
    %9 = stablehlo.compare  LT, %4, %8,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %9 : tensor<i1>
  }) : (tensor<1x4x64x64xf32>, tensor<1x4x64x64xi32>, tensor<1x4x64x64xi32>, tensor<1x4x64x64xi32>) -> (tensor<1x4x64x64xf32>, tensor<1x4x64x64xi32>, tensor<1x4x64x64xi32>, tensor<1x4x64x64xi32>)
  return %0#0, %0#1, %0#2, %0#3 : tensor<1x4x64x64xf32>, tensor<1x4x64x64xi32>, tensor<1x4x64x64xi32>, tensor<1x4x64x64xi32>
}
