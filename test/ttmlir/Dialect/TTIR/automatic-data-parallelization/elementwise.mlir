// RUN: ttmlir-opt --ttir-automatic-data-parallelization="mesh-shape=1,2" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

func.func @test0(%arg0: tensor<32x1x64x128xf32>, %arg1: tensor<32x1x64x128xf32>) -> tensor<32x1x64x128xf32> {
  %0 = ttir.empty() : tensor<32x1x64x128xf32>
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x1x64x128xf32>, tensor<32x1x64x128xf32>, tensor<32x1x64x128xf32>) -> tensor<32x1x64x128xf32>
  return %1 : tensor<32x1x64x128xf32>
}

// CHECK: func.func @test0
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch", ?}, {?}, {?}, {?}]>
// CHECK: ttir.empty() {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>]>}

// CHECK: "ttir.add"
// CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>]>

func.func @test1(%arg0: tensor<32x64x128xf32>, %arg1: tensor<32x64x128xf32>) -> tensor<32x64x128xf32> {
  %0 = ttir.empty() : tensor<32x64x128xf32>
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x64x128xf32>, tensor<32x64x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
  return %1 : tensor<32x64x128xf32>
}

// CHECK: func.func @test1
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>

func.func @test2(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

// CHECK: func.func @test2
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>

func.func @test3(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %0 = ttir.empty() : tensor<128xf32>
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
  return %1 : tensor<128xf32>
}

// CHECK: func.func @test3
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}]>

func.func @test4(%arg0: tensor<1x32x64x128xf32>, %arg1: tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32> {
  %0 = ttir.empty() : tensor<1x32x64x128xf32>
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x64x128xf32>, tensor<1x32x64x128xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
  return %1 : tensor<1x32x64x128xf32>
}

// CHECK: func.func @test4
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>

func.func @test5(%arg0: tensor<32x1x64x128xf32>) -> tensor<32x1x64x128xf32> {
  %0 = ttir.empty() : tensor<32x1x64x128xf32>
  %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x1x64x128xf32>, tensor<32x1x64x128xf32>) -> tensor<32x1x64x128xf32>
  return %1 : tensor<32x1x64x128xf32>
}

// CHECK: func.func @test5
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch", ?}, {?}, {?}, {?}]>
// CHECK: ttir.empty() {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>]>}

// CHECK: "ttir.relu"
// CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>]>

func.func @test6(%arg0: tensor<32x64x128xf32>) -> tensor<32x64x128xf32> {
  %0 = ttir.empty() : tensor<32x64x128xf32>
  %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x64x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
  return %1 : tensor<32x64x128xf32>
}

// CHECK: func.func @test6
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>

func.func @test7(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

// CHECK: func.func @test7
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>

func.func @test8(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %0 = ttir.empty() : tensor<128xf32>
  %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
  return %1 : tensor<128xf32>
}

// CHECK: func.func @test8
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}]>

func.func @test9(%arg0: tensor<1x64x128xf32>) -> tensor<1x64x128xf32> {
  %0 = ttir.empty() : tensor<1x64x128xf32>
  %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x64x128xf32>, tensor<1x64x128xf32>) -> tensor<1x64x128xf32>
  return %1 : tensor<1x64x128xf32>
}

// CHECK: func.func @test9
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>

func.func @test10(%arg0: tensor<32x32x12x64xbf16>, %arg1: tensor<32x32x12x64xbf16>) -> tensor<32x32x12x64xbf16> {
  %0 = ttir.empty() : tensor<32x32x12x64xbf16>
  %1 = "ttir.eq"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32x12x64xbf16>, tensor<32x32x12x64xbf16>, tensor<32x32x12x64xbf16>) -> tensor<32x32x12x64xbf16>
  %2 = ttir.empty() : tensor<32x32x12x64xbf16>
  %3 = "ttir.where"(%1, %arg0, %arg1, %2) <{operandSegmentSizes = array<i32: 3, 1>}> : (tensor<32x32x12x64xbf16>, tensor<32x32x12x64xbf16>, tensor<32x32x12x64xbf16>, tensor<32x32x12x64xbf16>) -> tensor<32x32x12x64xbf16>
  return %3 : tensor<32x32x12x64xbf16>
}

// CHECK: func.func @test10
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch", ?}, {?}, {?}, {?}]>
// CHECK: ttir.empty() {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>]>}

// CHECK: "ttir.eq"
// CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>]>

// CHECK: "ttir.where"
// CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>]>
