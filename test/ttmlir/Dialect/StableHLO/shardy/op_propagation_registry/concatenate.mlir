// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline --split-input-file -o %t.mlir %s

module {
  sdy.mesh @mesh = <["_axis_0"=8]>
  func.func @concatenate_reshape_test(%arg0: tensor<128x2880xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> tensor<32x128x2880xbf16> {
    %0 = stablehlo.concatenate %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0,
                                %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0,
                                %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0,
                                %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0,
                                dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0"}, {}]>]>} :
      (tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>,
       tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>,
       tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>,
       tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>,
       tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>,
       tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>,
       tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>,
       tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>) -> tensor<4096x2880xbf16>

    %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_0"}, {}, {}]>]>} :
      (tensor<4096x2880xbf16>) -> tensor<32x128x2880xbf16>

    return %1 : tensor<32x128x2880xbf16>
  }
}
