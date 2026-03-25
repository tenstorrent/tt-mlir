// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline="mesh-shape=1,2 result-presharded=0,1" --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,2" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: ttrt run %t.ttnn

module {
  sdy.mesh @mesh = <["x"=1, "batch"=2]>
  func.func public @mixed_outputs(%arg0: tensor<32x1x64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}) -> (tensor<32x1x64x64xf32>, tensor<32x1x64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}) {
    %0 = stablehlo.add %arg0, %arg0 : tensor<32x1x64x64xf32>
    %1 = stablehlo.multiply %arg0, %arg0 : tensor<32x1x64x64xf32>
    return %0, %1 : tensor<32x1x64x64xf32>, tensor<32x1x64x64xf32>
  }
}
