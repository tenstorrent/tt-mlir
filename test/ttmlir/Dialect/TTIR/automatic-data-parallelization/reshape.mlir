// RUN: ttmlir-opt --ttir-automatic-data-parallelization="mesh-shape=1,2" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

func.func @test0(%arg0: tensor<32x4x64x128xbf16>) -> tensor<32x64x4x128xbf16> {
  %0 = ttir.empty() : tensor<32x64x4x128xbf16>
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [32: i32, 64: i32, 4: i32, 128: i32]}> : (tensor<32x4x64x128xbf16>, tensor<32x64x4x128xbf16>) -> tensor<32x64x4x128xbf16>
  return %1 : tensor<32x64x4x128xbf16>
}

// CHECK: func.func @test0
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>
// CHECK: sdy.sharding = #sdy.sharding<@mesh, [{"batch", ?}, {?}, {?}, {?}]>
// CHECK: ttir.empty() {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>]>}

// CHECK: "ttir.reshape"
// CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch", ?}, {?}, {?}, {?}]>]>
