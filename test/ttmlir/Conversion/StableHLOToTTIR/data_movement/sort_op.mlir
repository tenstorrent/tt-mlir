// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @test_sort_ascending_none_stable(%arg0: tensor<1x128x256x256xbf16>) -> (tensor<1x128x256x256xbf16>, tensor<1x128x256x256xi32>) {
  // CHECK-LABEL: @test_sort_ascending_none_stable
  %0 = stablehlo.iota dim = 0 : tensor<256xi32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [3] : (tensor<256xi32>) -> tensor<1x128x256x256xi32>
  // CHECK: %{{.*}}, %{{.*}} = "ttir.sort"(%arg0)
  // CHECK-SAME: <{descending = false, dim = 3 : si32, stable = false}>
  // CHECK-SAME: (tensor<1x128x256x256xbf16>)
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
  // CHECK: %{{.*}}, %{{.*}} = "ttir.sort"(%arg0)
  // CHECK-SAME: <{descending = true, dim = 3 : si32, stable = true}>
  // CHECK-SAME: (tensor<1x128x256x256xbf16>)
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
  // CHECK: %{{.*}}, %{{.*}} = "ttir.sort"(%arg0)
  // CHECK-SAME: <{descending = false, dim = 1 : si32, stable = true}>
  // CHECK-SAME: (tensor<2x3xi32>)
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
  // CHECK: %{{.*}}, %[[INDICES:.*]] = "ttir.sort"(%arg0)
  // CHECK-SAME: <{descending = false, dim = 3 : si32, stable = true}>
  // CHECK-SAME: (tensor<1x4x64x64xf32>)
  // CHECK-SAME: -> (tensor<1x4x64x64xf32>, tensor<1x4x64x64xi32>)
  // CHECK: %[[PERM_IDX:[0-9]+]] = "ttir.permute"(%[[INDICES]])
  // CHECK-SAME: <{permutation = array<i64: 0, 1, 2, 3>}>
  // CHECK-SAME: (tensor<1x4x64x64xi32>) -> tensor<1x4x64x64xi32>
  // CHECK: %[[INDICES_2D:[0-9]+]] = "ttir.reshape"(%[[PERM_IDX]])
  // CHECK-SAME: <{shape = [256 : i32, 64 : i32]}>
  // CHECK-SAME: (tensor<1x4x64x64xi32>) -> tensor<256x64xi32>
  // CHECK: %[[OFFSETS:[0-9]+]] = "ttir.arange"()
  // CHECK-SAME: <{arange_dimension = 0 : i64, end = 16384 : si64, start = 0 : si64, step = 64 : si64}>
  // CHECK-SAME: -> tensor<256x64xi32>
  // CHECK: %[[FLAT_IDX:[0-9]+]] = "ttir.add"(%[[OFFSETS]], %[[INDICES_2D]])
  // CHECK-SAME: (tensor<256x64xi32>, tensor<256x64xi32>) -> tensor<256x64xi32>
  // CHECK: %[[PV1:[0-9]+]] = "ttir.permute"(%arg1) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<1x4x64x64xi32>) -> tensor<1x4x64x64xi32>
  // CHECK: %[[W1:[0-9]+]] = "ttir.reshape"(%[[PV1]]) <{shape = [16384 : i32, 1 : i32]}> : (tensor<1x4x64x64xi32>) -> tensor<16384x1xi32>
  // CHECK: %[[E1:[0-9]+]] = "ttir.embedding"(%[[FLAT_IDX]], %[[W1]]) : (tensor<256x64xi32>, tensor<16384x1xi32>) -> tensor<256x64x1xi32>
  // CHECK: %[[R1:[0-9]+]] = "ttir.reshape"(%[[E1]]) <{shape = [1 : i32, 4 : i32, 64 : i32, 64 : i32]}> : (tensor<256x64x1xi32>) -> tensor<1x4x64x64xi32>
  // CHECK: %{{.*}} = "ttir.permute"(%[[R1]]) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<1x4x64x64xi32>) -> tensor<1x4x64x64xi32>
  // CHECK: %[[PV2:[0-9]+]] = "ttir.permute"(%arg2) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<1x4x64x64xi32>) -> tensor<1x4x64x64xi32>
  // CHECK: %[[W2:[0-9]+]] = "ttir.reshape"(%[[PV2]]) <{shape = [16384 : i32, 1 : i32]}> : (tensor<1x4x64x64xi32>) -> tensor<16384x1xi32>
  // CHECK: %[[E2:[0-9]+]] = "ttir.embedding"(%[[FLAT_IDX]], %[[W2]]) : (tensor<256x64xi32>, tensor<16384x1xi32>) -> tensor<256x64x1xi32>
  // CHECK: %[[R2:[0-9]+]] = "ttir.reshape"(%[[E2]]) <{shape = [1 : i32, 4 : i32, 64 : i32, 64 : i32]}> : (tensor<256x64x1xi32>) -> tensor<1x4x64x64xi32>
  // CHECK: %{{.*}} = "ttir.permute"(%[[R2]]) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<1x4x64x64xi32>) -> tensor<1x4x64x64xi32>
  // CHECK: %[[PV3:[0-9]+]] = "ttir.permute"(%arg3) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<1x4x64x64xi32>) -> tensor<1x4x64x64xi32>
  // CHECK: %[[W3:[0-9]+]] = "ttir.reshape"(%[[PV3]]) <{shape = [16384 : i32, 1 : i32]}> : (tensor<1x4x64x64xi32>) -> tensor<16384x1xi32>
  // CHECK: %[[E3:[0-9]+]] = "ttir.embedding"(%[[FLAT_IDX]], %[[W3]]) : (tensor<256x64xi32>, tensor<16384x1xi32>) -> tensor<256x64x1xi32>
  // CHECK: %[[R3:[0-9]+]] = "ttir.reshape"(%[[E3]]) <{shape = [1 : i32, 4 : i32, 64 : i32, 64 : i32]}> : (tensor<256x64x1xi32>) -> tensor<1x4x64x64xi32>
  // CHECK: %{{.*}} = "ttir.permute"(%[[R3]]) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<1x4x64x64xi32>) -> tensor<1x4x64x64xi32>
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

// KeyValue sort along dim=0 (first dim). sortDim != rank-1, so permute path is taken.
// Shape [4x3]: perm=[1,0], prePost=3, dSort=4, total=12.
// Indices permuted [4x3] → [3x4], reshaped to [3x4].
// ArangeOp(step=4): row offsets [0,4,8] broadcast to [3x4].
// Per value tensor: permute [4x3]→[3x4], flatten to [12x1], embedding, reshape [3x4], permute back [4x3].
func.func public @test_sort_key_values_dim0(%arg0: tensor<4x3xf32>, %arg1: tensor<4x3xi32>) -> (tensor<4x3xf32>, tensor<4x3xi32>) {
  // CHECK-LABEL: @test_sort_key_values_dim0
  // CHECK: %{{.*}}, %[[INDICES:.*]] = "ttir.sort"(%arg0)
  // CHECK-SAME: <{descending = false, dim = 0 : si32, stable = true}>
  // CHECK-SAME: (tensor<4x3xf32>)
  // CHECK-SAME: -> (tensor<4x3xf32>, tensor<4x3xi32>)
  // CHECK: %[[PERM_IDX:.*]] = "ttir.permute"(%[[INDICES]])
  // CHECK-SAME: <{permutation = array<i64: 1, 0>}>
  // CHECK-SAME: (tensor<4x3xi32>) -> tensor<3x4xi32>
  // CHECK: %[[INDICES_2D:.*]] = "ttir.reshape"(%[[PERM_IDX]])
  // CHECK-SAME: <{shape = [3 : i32, 4 : i32]}>
  // CHECK-SAME: (tensor<3x4xi32>) -> tensor<3x4xi32>
  // CHECK: %[[OFFSETS:.*]] = "ttir.arange"()
  // CHECK-SAME: <{arange_dimension = 0 : i64, end = 12 : si64, start = 0 : si64, step = 4 : si64}>
  // CHECK-SAME: -> tensor<3x4xi32>
  // CHECK: %[[FLAT_IDX:.*]] = "ttir.add"(%[[OFFSETS]], %[[INDICES_2D]])
  // CHECK-SAME: (tensor<3x4xi32>, tensor<3x4xi32>) -> tensor<3x4xi32>
  // CHECK: %[[PERM_V1:.*]] = "ttir.permute"(%arg1)
  // CHECK-SAME: <{permutation = array<i64: 1, 0>}>
  // CHECK-SAME: (tensor<4x3xi32>) -> tensor<3x4xi32>
  // CHECK: %[[W1:.*]] = "ttir.reshape"(%[[PERM_V1]]) <{shape = [12 : i32, 1 : i32]}> : (tensor<3x4xi32>) -> tensor<12x1xi32>
  // CHECK: %[[E1:.*]] = "ttir.embedding"(%[[FLAT_IDX]], %[[W1]]) : (tensor<3x4xi32>, tensor<12x1xi32>) -> tensor<3x4x1xi32>
  // CHECK: %[[R1:.*]] = "ttir.reshape"(%[[E1]]) <{shape = [3 : i32, 4 : i32]}> : (tensor<3x4x1xi32>) -> tensor<3x4xi32>
  // CHECK: %{{.*}} = "ttir.permute"(%[[R1]])
  // CHECK-SAME: <{permutation = array<i64: 1, 0>}>
  // CHECK-SAME: (tensor<3x4xi32>) -> tensor<4x3xi32>
  %0:2 = "stablehlo.sort"(%arg0, %arg1) <{dimension = 0 : i64, is_stable = true}> ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<i32>, %arg5: tensor<i32>):
    %1 = stablehlo.compare LT, %arg2, %arg3, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  }) : (tensor<4x3xf32>, tensor<4x3xi32>) -> (tensor<4x3xf32>, tensor<4x3xi32>)
  return %0#0, %0#1 : tensor<4x3xf32>, tensor<4x3xi32>
}

// KeyValue sort along dim=1 (middle dim of 3D tensor). sortDim != rank-1, so permute path is taken.
// Shape [2x4x3]: perm=[0,2,1], prePost=6, dSort=4, total=24.
// Indices permuted [2x4x3] → [2x3x4], reshaped to [6x4].
// ArangeOp(step=4): row offsets [0,4,8,12,16,20] broadcast to [6x4].
// Per value tensor: permute [2x4x3]→[2x3x4], flatten to [24x1], embedding, reshape [2x3x4], permute back [2x4x3].
func.func public @test_sort_key_values_middle_dim(%arg0: tensor<2x4x3xf32>, %arg1: tensor<2x4x3xi32>) -> (tensor<2x4x3xf32>, tensor<2x4x3xi32>) {
  // CHECK-LABEL: @test_sort_key_values_middle_dim
  // CHECK: %{{.*}}, %[[INDICES:.*]] = "ttir.sort"(%arg0)
  // CHECK-SAME: <{descending = false, dim = 1 : si32, stable = true}>
  // CHECK-SAME: (tensor<2x4x3xf32>)
  // CHECK-SAME: -> (tensor<2x4x3xf32>, tensor<2x4x3xi32>)
  // CHECK: %[[PERM_IDX:.*]] = "ttir.permute"(%[[INDICES]])
  // CHECK-SAME: <{permutation = array<i64: 0, 2, 1>}>
  // CHECK-SAME: (tensor<2x4x3xi32>) -> tensor<2x3x4xi32>
  // CHECK: %[[INDICES_2D:.*]] = "ttir.reshape"(%[[PERM_IDX]])
  // CHECK-SAME: <{shape = [6 : i32, 4 : i32]}>
  // CHECK-SAME: (tensor<2x3x4xi32>) -> tensor<6x4xi32>
  // CHECK: %[[OFFSETS:.*]] = "ttir.arange"()
  // CHECK-SAME: <{arange_dimension = 0 : i64, end = 24 : si64, start = 0 : si64, step = 4 : si64}>
  // CHECK-SAME: -> tensor<6x4xi32>
  // CHECK: %[[FLAT_IDX:.*]] = "ttir.add"(%[[OFFSETS]], %[[INDICES_2D]])
  // CHECK-SAME: (tensor<6x4xi32>, tensor<6x4xi32>) -> tensor<6x4xi32>
  // CHECK: %[[PERM_V1:.*]] = "ttir.permute"(%arg1)
  // CHECK-SAME: <{permutation = array<i64: 0, 2, 1>}>
  // CHECK-SAME: (tensor<2x4x3xi32>) -> tensor<2x3x4xi32>
  // CHECK: %[[W1:.*]] = "ttir.reshape"(%[[PERM_V1]]) <{shape = [24 : i32, 1 : i32]}> : (tensor<2x3x4xi32>) -> tensor<24x1xi32>
  // CHECK: %[[E1:.*]] = "ttir.embedding"(%[[FLAT_IDX]], %[[W1]]) : (tensor<6x4xi32>, tensor<24x1xi32>) -> tensor<6x4x1xi32>
  // CHECK: %[[R1:.*]] = "ttir.reshape"(%[[E1]]) <{shape = [2 : i32, 3 : i32, 4 : i32]}> : (tensor<6x4x1xi32>) -> tensor<2x3x4xi32>
  // CHECK: %{{.*}} = "ttir.permute"(%[[R1]])
  // CHECK-SAME: <{permutation = array<i64: 0, 2, 1>}>
  // CHECK-SAME: (tensor<2x3x4xi32>) -> tensor<2x4x3xi32>
  %0:2 = "stablehlo.sort"(%arg0, %arg1) <{dimension = 1 : i64, is_stable = true}> ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<i32>, %arg5: tensor<i32>):
    %1 = stablehlo.compare LT, %arg2, %arg3, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  }) : (tensor<2x4x3xf32>, tensor<2x4x3xi32>) -> (tensor<2x4x3xf32>, tensor<2x4x3xi32>)
  return %0#0, %0#1 : tensor<2x4x3xf32>, tensor<2x4x3xi32>
}
