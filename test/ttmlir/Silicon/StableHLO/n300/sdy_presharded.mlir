// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline="mesh-shape=1,2 result-presharded=0" --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,2" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {
  sdy.mesh @mesh = <["x"=1, "batch"=2]>
  func.func public @full_arg_full_op_annotation(%arg0: tensor<32x1x8192x784xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}, %arg1: tensor<784x2048xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<32x1x8192x2048xf32> {jax.result_info = ""}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [3] x [0], precision = [DEFAULT, DEFAULT] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch"}, {}, {}, {}]>]>} : (tensor<32x1x8192x784xf32>, tensor<784x2048xf32>) -> tensor<32x1x8192x2048xf32>
    %1 = stablehlo.add %0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch"}, {}, {}, {}]>]>} : tensor<32x1x8192x2048xf32>
    return %1 : tensor<32x1x8192x2048xf32>
  }
}
