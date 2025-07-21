// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline="mesh-shape=1,2" --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,2" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

sdy.mesh @mesh = <["model"=1, "batch"=2]>

func.func public @op_sequence(%arg0: tensor<32x1x8192x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}, %arg1: tensor<256x2048xf32>, %arg2: tensor<32x1x8192x2048xf32>) -> (tensor<32x1x8192x2048xf32> {jax.result_info = ""}) {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [3] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1x8192x256xf32>, tensor<256x2048xf32>) -> tensor<32x1x8192x2048xf32>
  %1 = stablehlo.add %0, %arg2 : tensor<32x1x8192x2048xf32>
  %2 = stablehlo.multiply %1, %arg2 : tensor<32x1x8192x2048xf32>
  return %2 : tensor<32x1x8192x2048xf32>
}
