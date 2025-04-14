// RUN: ttmlir-opt --shardy-automatic-parallelization="mesh-shape=1,2" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

module @jit_eltwise_add attributes {} {
  func.func public @test_add(%arg0: tensor<32x48x24x32xf32>, %arg1: tensor<32x48x24x32xf32>) -> tensor<32x48x24x32xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<32x48x24x32xf32>
    return %0 : tensor<32x48x24x32xf32>
  }
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>, <@mesh, [{"batch"}, {}, {}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>]
// CHECK: stablehlo.add %arg2, %arg3 : tensor<16x48x24x32xf32>
// CHECK: sdy.return %1 : tensor<16x48x24x32xf32>
