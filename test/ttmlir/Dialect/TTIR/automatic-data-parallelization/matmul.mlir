// RUN: ttmlir-opt --ttir-automatic-data-parallelization="mesh-shape=1,2" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

func.func @test0(%arg0: tensor<32x1x64x128xbf16>, %arg1: tensor<128x96xbf16>) -> tensor<32x1x64x96xbf16> {
  %0 = ttir.empty() : tensor<32x1x64x96xbf16>
  %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<32x1x64x128xbf16>, tensor<128x96xbf16>, tensor<32x1x64x96xbf16>) -> tensor<32x1x64x96xbf16>
  return %1 : tensor<32x1x64x96xbf16>
}

// CHECK: func.func @test0
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch", ?}, {?}, {?}, {?}]>
// CHECK: ttir.empty() {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>]>}

// CHECK: "ttir.matmul"
// CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>]>

func.func @test1(%arg0: tensor<32x1x1x128xbf16>, %arg1: tensor<128x1xbf16>) -> tensor<32x1x1x1xbf16> {
  %0 = ttir.empty() : tensor<32x1x1x1xbf16>
  %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<32x1x1x128xbf16>, tensor<128x1xbf16>, tensor<32x1x1x1xbf16>) -> tensor<32x1x1x1xbf16>
  return %1 : tensor<32x1x1x1xbf16>
}

// CHECK: func.func @test1
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch", ?}, {?}, {?}, {?}]>
// CHECK: ttir.empty() {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>]>}

// CHECK: "ttir.matmul"
// CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>]>

func.func @test2(%arg0: tensor<32x1x64x128xbf16>, %arg1: tensor<32x1x128x96xbf16>) -> tensor<32x1x64x96xbf16> {
  %0 = ttir.empty() : tensor<32x1x64x96xbf16>
  %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<32x1x64x128xbf16>, tensor<32x1x128x96xbf16>, tensor<32x1x64x96xbf16>) -> tensor<32x1x64x96xbf16>
  return %1 : tensor<32x1x64x96xbf16>
}

// CHECK: func.func @test2
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch", ?}, {?}, {?}, {?}]>
// CHECK: ttir.empty() {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>]>}

// CHECK: "ttir.matmul"
// CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>]>

func.func @test3(%arg0: tensor<7x64x128xbf16>, %arg1: tensor<7x128x64xbf16>) -> tensor<7x64x64xbf16> {
  %0 = ttir.empty() : tensor<7x64x64xbf16>
  %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<7x64x128xbf16>, tensor<7x128x64xbf16>, tensor<7x64x64xbf16>) -> tensor<7x64x64xbf16>
  return %1 : tensor<7x64x64xbf16>
}

// CHECK: func.func @test3
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>

func.func @test4(%arg0: tensor<7x64x128xbf16>, %arg1: tensor<7x128x64xbf16>) -> tensor<7x64x64xbf16> {
  %0 = ttir.empty() : tensor<7x64x64xbf16>
  %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<7x64x128xbf16>, tensor<7x128x64xbf16>, tensor<7x64x64xbf16>) -> tensor<7x64x64xbf16>
  return %1 : tensor<7x64x64xbf16>
}

// CHECK: func.func @test4
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>

func.func @test5(%arg0: tensor<32x1x64x128xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<32x1x128x128xbf16> {
  %0 = ttir.empty() : tensor<32x1x128x128xbf16>
  %1 = "ttir.matmul"(%arg0, %arg1, %0) <{transpose_a = true}> : (tensor<32x1x64x128xbf16>, tensor<64x128xbf16>, tensor<32x1x128x128xbf16>) -> tensor<32x1x128x128xbf16>
  return %1 : tensor<32x1x128x128xbf16>
}

// CHECK: func.func @test5
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch", ?}, {?}, {?}, {?}]>
// CHECK: ttir.empty() {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>]>}

// CHECK: "ttir.matmul"
// CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>]>

func.func @test6(%arg0: tensor<32x1x64x128xbf16>, %arg1: tensor<128x64xbf16>) -> tensor<32x1x128x128xbf16> {
  %0 = ttir.empty() : tensor<32x1x128x128xbf16>
  %1 = "ttir.matmul"(%arg0, %arg1, %0) <{transpose_a = true, transpose_b = true}> : (tensor<32x1x64x128xbf16>, tensor<128x64xbf16>, tensor<32x1x128x128xbf16>) -> tensor<32x1x128x128xbf16>
  return %1 : tensor<32x1x128x128xbf16>
}

// CHECK: func.func @test6
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch", ?}, {?}, {?}, {?}]>
// CHECK: ttir.empty() {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>]>}

// CHECK: "ttir.matmul"
// CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>]>
