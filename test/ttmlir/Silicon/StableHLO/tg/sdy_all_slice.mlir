// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline="mesh-shape=1,32 automatic-arg-analysis" --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,32" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {
  sdy.mesh @mesh = <["model"=1, "batch"=32]>
  func.func @all_slice_replicated_input(%arg0: tensor<1x32xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> tensor<1x32xbf16> {
    %0 = sdy.all_slice [{}, {"batch"}] %arg0 out_sharding=<@mesh, [{}, {"batch"}]> : tensor<1x32xbf16>
    return %0 : tensor<1x32xbf16>
  }
}
