// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// CHECK-LABEL: module @SyncTensorsGraph.5
module @SyncTensorsGraph.5 attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  // CHECK: sdy.mesh @mesh = <["_axis_0_updated"=1, "_axis_0"=2]>
  func.func @main(%arg0: tensor<32x128xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,2]<=[2]}"}, %arg1: tensor<32x32xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}, mhlo.sharding = "{replicated}"}) -> tensor<32x128xf32> {
    // CHECK: sdy.manual_computation(%{{.*}}, %{{.*}}) in_shardings=[<@mesh, [{}, {"_axis_0"}]>, <@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}, {"_axis_0"}]>] manual_axes={"_axis_0_updated", "_axis_0"}
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
    return %0 : tensor<32x128xf32>
  }
}
