// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline="mesh-shape=1,4" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module @reshape_split_sharded {
  sdy.mesh @mesh = <["batch"=1, "model"=4]>
  func.func public @main(%arg0: tensor<3x30720xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"model"}]>}) -> tensor<3x6x5120xbf16> {
    %0 = stablehlo.reshape %arg0 : (tensor<3x30720xbf16>) -> tensor<3x6x5120xbf16>
    return %0 : tensor<3x6x5120xbf16>
  }
}

// CHECK-LABEL: @main
// CHECK: stablehlo.all_gather
// CHECK: stablehlo.reshape
