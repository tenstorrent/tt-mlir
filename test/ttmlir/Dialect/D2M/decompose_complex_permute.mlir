// RUN: ttmlir-opt --d2m-decompose-complex-permute %s | FileCheck %s

// inner -> outer
// CHECK-LABEL: @permute_4d_0312
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 1, 3, 2>}> {decomposed}
// CHECK: %[[P1:.*]] = "ttir.permute"(%[[P0]]) <{permutation = array<i64: 0, 2, 1, 3>}> {decomposed}
// CHECK: return %[[P1]]
func.func @permute_4d_0312(%arg0: tensor<3x32x32x32xf32>) -> tensor<3x32x32x32xf32> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<3x32x32x32xf32>) -> tensor<3x32x32x32xf32>
  return %0 : tensor<3x32x32x32xf32>
}

// outer -> inner -> outer
// CHECK-LABEL: @permute_4d_0321
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 1, 3>}> {decomposed}
// CHECK: %[[P1:.*]] = "ttir.permute"(%[[P0]]) <{permutation = array<i64: 0, 1, 3, 2>}> {decomposed}
// CHECK: %[[P2:.*]] = "ttir.permute"(%[[P1]]) <{permutation = array<i64: 0, 2, 1, 3>}> {decomposed}
// CHECK: return %[[P2]]
func.func @permute_4d_0321(%arg0: tensor<3x32x32x32xf32>) -> tensor<3x32x32x32xf32> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 3, 2, 1>}> : (tensor<3x32x32x32xf32>) -> tensor<3x32x32x32xf32>
  return %0 : tensor<3x32x32x32xf32>
}

// outer permute -> inner permute
// CHECK-LABEL: @permute_4d_2031
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 2, 0, 1, 3>}> {decomposed}
// CHECK: %[[P1:.*]] = "ttir.permute"(%[[P0]]) <{permutation = array<i64: 0, 1, 3, 2>}> {decomposed}
// CHECK: return %[[P1]]
func.func @permute_4d_2031(%arg0: tensor<3x32x32x32xf32>) -> tensor<32x3x32x32xf32> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 2, 0, 3, 1>}> : (tensor<3x32x32x32xf32>) -> tensor<32x3x32x32xf32>
  return %0 : tensor<32x3x32x32xf32>
}

// inner permute -> outer permute
// CHECK-LABEL: @permute_4d_1302
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 1, 3, 2>}> {decomposed}
// CHECK: %[[P1:.*]] = "ttir.permute"(%[[P0]]) <{permutation = array<i64: 1, 2, 0, 3>}> {decomposed}
// CHECK: return %[[P1]]
func.func @permute_4d_1302(%arg0: tensor<3x32x32x32xf32>) -> tensor<32x32x3x32xf32> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 3, 0, 2>}> : (tensor<3x32x32x32xf32>) -> tensor<32x32x3x32xf32>
  return %0 : tensor<32x32x3x32xf32>
}

// inner permute -> outer permute
// CHECK-LABEL: @permute_4d_3102
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 1, 3, 2>}> {decomposed}
// CHECK: %[[P1:.*]] = "ttir.permute"(%[[P0]]) <{permutation = array<i64: 2, 1, 0, 3>}> {decomposed}
// CHECK: return %[[P1]]
func.func @permute_4d_3102(%arg0: tensor<3x32x32x32xf32>) -> tensor<32x32x3x32xf32> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 3, 1, 0, 2>}> : (tensor<3x32x32x32xf32>) -> tensor<32x32x3x32xf32>
  return %0 : tensor<32x32x3x32xf32>
}

// outer permute -> inner permute -> outer permute
// CHECK-LABEL: @permute_4d_2310
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 2, 1, 0, 3>}> {decomposed}
// CHECK: %[[P1:.*]] = "ttir.permute"(%[[P0]]) <{permutation = array<i64: 0, 1, 3, 2>}> {decomposed}
// CHECK: %[[P2:.*]] = "ttir.permute"(%[[P1]]) <{permutation = array<i64: 0, 2, 1, 3>}> {decomposed}
// CHECK: return %[[P2]]
func.func @permute_4d_2310(%arg0: tensor<3x32x32x32xf32>) -> tensor<32x32x32x3xf32> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 2, 3, 1, 0>}> : (tensor<3x32x32x32xf32>) -> tensor<32x32x32x3xf32>
  return %0 : tensor<32x32x32x3xf32>
}

// outer -> inner -> outer
// CHECK-LABEL: @permute_4d_3210
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 2, 1, 0, 3>}> {decomposed}
// CHECK: %[[P1:.*]] = "ttir.permute"(%[[P0]]) <{permutation = array<i64: 0, 1, 3, 2>}> {decomposed}
// CHECK: %[[P2:.*]] = "ttir.permute"(%[[P1]]) <{permutation = array<i64: 2, 0, 1, 3>}> {decomposed}
// CHECK: return %[[P2]]
func.func @permute_4d_3210(%arg0: tensor<3x32x32x32xf32>) -> tensor<32x32x32x3xf32> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 3, 2, 1, 0>}> : (tensor<3x32x32x32xf32>) -> tensor<32x32x32x3xf32>
  return %0 : tensor<32x32x32x3xf32>
}

// CHECK-LABEL: @permute_4d_inner_only
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 1, 3, 2>}>
// CHECK-NOT: {decomposed}
// CHECK-SAME: -> tensor<3x32x32x32xf32>
// CHECK: return %[[P0]]
func.func @permute_4d_inner_only(%arg0: tensor<3x32x32x32xf32>) -> tensor<3x32x32x32xf32> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<3x32x32x32xf32>) -> tensor<3x32x32x32xf32>
  return %0 : tensor<3x32x32x32xf32>
}

// CHECK-LABEL: @permute_4d_outer_only
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 1, 3>}>
// CHECK-NOT: {decomposed}
// CHECK-SAME: -> tensor<3x32x32x32xf32>
// CHECK: return %[[P0]]
func.func @permute_4d_outer_only(%arg0: tensor<3x32x32x32xf32>) -> tensor<3x32x32x32xf32> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<3x32x32x32xf32>) -> tensor<3x32x32x32xf32>
  return %0 : tensor<3x32x32x32xf32>
}

// CHECK-LABEL: @permute_4d_identity
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 1, 2, 3>}>
// CHECK-NOT: {decomposed}
// CHECK-SAME: -> tensor<3x32x32x32xf32>
// CHECK: return %[[P0]]
func.func @permute_4d_identity(%arg0: tensor<3x32x32x32xf32>) -> tensor<3x32x32x32xf32> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<3x32x32x32xf32>) -> tensor<3x32x32x32xf32>
  return %0 : tensor<3x32x32x32xf32>
}

// CHECK-LABEL: @permute_5d_04123
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 1, 2, 4, 3>}> {decomposed}
// CHECK: %[[P1:.*]] = "ttir.permute"(%[[P0]]) <{permutation = array<i64: 0, 3, 1, 2, 4>}> {decomposed}
// CHECK: return %[[P1]]
func.func @permute_5d_04123(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<2x6x3x4x5xf32> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 4, 1, 2, 3>}> : (tensor<2x3x4x5x6xf32>) -> tensor<2x6x3x4x5xf32>
  return %0 : tensor<2x6x3x4x5xf32>
}

// CHECK-LABEL: @permute_5d_01432
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 1, 3, 2, 4>}> {decomposed}
// CHECK: %[[P1:.*]] = "ttir.permute"(%[[P0]]) <{permutation = array<i64: 0, 1, 2, 4, 3>}> {decomposed}
// CHECK: %[[P2:.*]] = "ttir.permute"(%[[P1]]) <{permutation = array<i64: 0, 1, 3, 2, 4>}> {decomposed}
// CHECK: return %[[P2]]
func.func @permute_5d_01432(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<2x3x6x5x4xf32> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 1, 4, 3, 2>}> : (tensor<2x3x4x5x6xf32>) -> tensor<2x3x6x5x4xf32>
  return %0 : tensor<2x3x6x5x4xf32>
}

// CHECK-LABEL: @permute_5d_43210
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 3, 1, 2, 0, 4>}> {decomposed}
// CHECK: %[[P1:.*]] = "ttir.permute"(%[[P0]]) <{permutation = array<i64: 0, 1, 2, 4, 3>}> {decomposed}
// CHECK: %[[P2:.*]] = "ttir.permute"(%[[P1]]) <{permutation = array<i64: 3, 0, 2, 1, 4>}> {decomposed}
// CHECK: return %[[P2]]
func.func @permute_5d_43210(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<6x5x4x3x2xf32> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 4, 3, 2, 1, 0>}> : (tensor<2x3x4x5x6xf32>) -> tensor<6x5x4x3x2xf32>
  return %0 : tensor<6x5x4x3x2xf32>
}

// CHECK-LABEL: @permute_5d_10432
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 1, 3, 2, 4>}> {decomposed}
// CHECK: %[[P1:.*]] = "ttir.permute"(%[[P0]]) <{permutation = array<i64: 0, 1, 2, 4, 3>}> {decomposed}
// CHECK: %[[P2:.*]] = "ttir.permute"(%[[P1]]) <{permutation = array<i64: 1, 0, 3, 2, 4>}> {decomposed}
// CHECK: return %[[P2]]
func.func @permute_5d_10432(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<3x2x6x5x4xf32> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0, 4, 3, 2>}> : (tensor<2x3x4x5x6xf32>) -> tensor<3x2x6x5x4xf32>
  return %0 : tensor<3x2x6x5x4xf32>
}

// CHECK-LABEL: @permute_5d_24301
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 3, 2, 1, 4>}> {decomposed}
// CHECK: %[[P1:.*]] = "ttir.permute"(%[[P0]]) <{permutation = array<i64: 0, 1, 2, 4, 3>}> {decomposed}
// CHECK: %[[P2:.*]] = "ttir.permute"(%[[P1]]) <{permutation = array<i64: 2, 3, 1, 0, 4>}> {decomposed}
// CHECK: return %[[P2]]
func.func @permute_5d_24301(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<4x6x5x2x3xf32> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 2, 4, 3, 0, 1>}> : (tensor<2x3x4x5x6xf32>) -> tensor<4x6x5x2x3xf32>
  return %0 : tensor<4x6x5x2x3xf32>
}

// CHECK-LABEL: @permute_5d_30412
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 1, 3, 2, 4>}> {decomposed}
// CHECK: %[[P1:.*]] = "ttir.permute"(%[[P0]]) <{permutation = array<i64: 0, 1, 2, 4, 3>}> {decomposed}
// CHECK: %[[P2:.*]] = "ttir.permute"(%[[P1]]) <{permutation = array<i64: 2, 0, 3, 1, 4>}> {decomposed}
// CHECK: return %[[P2]]
func.func @permute_5d_30412(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<5x2x6x3x4xf32> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 3, 0, 4, 1, 2>}> : (tensor<2x3x4x5x6xf32>) -> tensor<5x2x6x3x4xf32>
  return %0 : tensor<5x2x6x3x4xf32>
}

// CHECK-LABEL: @permute_5d_inner_only
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 1, 2, 4, 3>}>
// CHECK-NOT: {decomposed}
// CHECK-SAME: -> tensor<2x3x4x6x5xf32>
// CHECK: return %[[P0]]
func.func @permute_5d_inner_only(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x6x5xf32> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 1, 2, 4, 3>}> : (tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x6x5xf32>
  return %0 : tensor<2x3x4x6x5xf32>
}

// CHECK-LABEL: @permute_5d_outer_only
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 1, 3, 4>}>
// CHECK-NOT: {decomposed}
// CHECK-SAME: -> tensor<2x4x3x5x6xf32>
// CHECK: return %[[P0]]
func.func @permute_5d_outer_only(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<2x4x3x5x6xf32> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 1, 3, 4>}> : (tensor<2x3x4x5x6xf32>) -> tensor<2x4x3x5x6xf32>
  return %0 : tensor<2x4x3x5x6xf32>
}

// CHECK-LABEL: @permute_5d_outer_only_2
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0, 2, 3, 4>}>
// CHECK-NOT: {decomposed}
// CHECK-SAME: -> tensor<3x2x4x5x6xf32>
// CHECK: return %[[P0]]
func.func @permute_5d_outer_only_2(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<3x2x4x5x6xf32> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0, 2, 3, 4>}> : (tensor<2x3x4x5x6xf32>) -> tensor<3x2x4x5x6xf32>
  return %0 : tensor<3x2x4x5x6xf32>
}

// CHECK-LABEL: @permute_5d_identity
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 1, 2, 3, 4>}>
// CHECK-NOT: {decomposed}
// CHECK-SAME: -> tensor<2x3x4x5x6xf32>
// CHECK: return %[[P0]]
func.func @permute_5d_identity(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x5x6xf32> {
  %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 1, 2, 3, 4>}> : (tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x5x6xf32>
  return %0 : tensor<2x3x4x5x6xf32>
}
