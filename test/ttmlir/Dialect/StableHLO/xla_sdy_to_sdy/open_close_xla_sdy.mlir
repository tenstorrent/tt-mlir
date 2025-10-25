// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// CHECK-LABEL: module @ClosedShardy
module @ClosedShardy attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  // CHECK: sdy.mesh @mesh = <["_axis_0_updated"=1, "_axis_0"=2]>
  func.func @main(%arg0: tensor<19456x2560xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,2]<=[2]}"}, %arg1: tensor<256x2560xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}, mhlo.sharding = "{replicated}"}) -> tensor<256x9728xbf16> {
    // CHECK: sdy.manual_computation(%{{.*}}, %{{.*}}) in_shardings=[<@mesh, [{}, {"_axis_0"}]>, <@mesh, [{}, {}]>]
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2560,19456]{0,1}"} : (tensor<19456x2560xbf16>) -> tensor<2560x19456xbf16>
    %1 = stablehlo.dot_general %arg1, %0, contracting_dims = [1] x [0] : (tensor<256x2560xbf16>, tensor<2560x19456xbf16>) -> tensor<256x19456xbf16>
    %2 = stablehlo.slice %1 [0:256, 0:9728] : (tensor<256x19456xbf16>) -> tensor<256x9728xbf16>
    return %2 : tensor<256x9728xbf16>
  }
}

// -----

// CHECK-LABEL: module @OpenShardy
module @OpenShardy attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  // CHECK: sdy.mesh @mesh = <["_axis_0_updated"=1, "_axis_0"=2]>
  func.func @main(%arg0: tensor<19456x2560xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,2]<=[2]}"}, %arg1: tensor<256x2560xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {?}]>"}, mhlo.sharding = "{replicated}"}) -> tensor<256x9728xbf16> {
    // CHECK: sdy.manual_computation(%{{.*}}, %{{.*}}) in_shardings=[<@mesh, [{}, {"_axis_0"}]>, <@mesh, [{}, {"_axis_0"}]>]
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2560,19456]{0,1}"} : (tensor<19456x2560xbf16>) -> tensor<2560x19456xbf16>
    %1 = stablehlo.dot_general %arg1, %0, contracting_dims = [1] x [0] : (tensor<256x2560xbf16>, tensor<2560x19456xbf16>) -> tensor<256x19456xbf16>
    %2 = stablehlo.slice %1 [0:256, 0:9728] : (tensor<256x19456xbf16>) -> tensor<256x9728xbf16>
    return %2 : tensor<256x9728xbf16>
  }
}
