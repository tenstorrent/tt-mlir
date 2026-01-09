// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline %s
module {
  sdy.mesh @mesh = <["_axis_0"=8]>
  func.func @concatenate_reshape_test(%arg0: tensor<128x2880xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg1: tensor<32x2880x5760xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}, {}]>}) -> tensor<32x128x5760xbf16> {
    %0 = stablehlo.concatenate
        %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0,
        %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0,
        %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0,
        %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0,
        dim = 0 : (
          tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>,
          tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>,
          tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>,
          tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>,
          tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>,
          tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>,
          tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>,
          tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>, tensor<128x2880xbf16>
        ) -> tensor<4096x2880xbf16>

    %1 = stablehlo.reshape %0 : (tensor<4096x2880xbf16>) -> tensor<32x128x2880xbf16>
    %2 = stablehlo.dot_general %1, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<32x128x2880xbf16>, tensor<32x2880x5760xbf16>) -> tensor<32x128x5760xbf16>

    return %2 : tensor<32x128x5760xbf16>
  }
}

// module {
  // func.func @concatenate_reshape(%arg0: tensor<2x4xbf16>) -> tensor<2x3x4xbf16> {
  //   %0 = stablehlo.concatenate %arg0, %arg0, %arg0, dim = 0 : (tensor<2x4xbf16>, tensor<2x4xbf16>, tensor<2x4xbf16>) -> tensor<6x4xbf16>
  //   %1 = stablehlo.reshape %0 : (tensor<6x4xbf16>) -> tensor<2x3x4xbf16>
  //   return %1 : tensor<2x3x4xbf16>
  //   // not a broadcast operation
  // }
  // func.func @concatenate_reshape_2(%arg0: tensor<2x4xbf16>) -> tensor<3x2x4xbf16> {
  //   %0 = stablehlo.concatenate %arg0, %arg0, %arg0, dim = 0 : (tensor<2x4xbf16>, tensor<2x4xbf16>, tensor<2x4xbf16>) -> tensor<6x4xbf16>
  //   %1 = stablehlo.reshape %0 : (tensor<6x4xbf16>) -> tensor<3x2x4xbf16>
  //   return %1 : tensor<3x2x4xbf16>
  //   // broadcast dims [1, 2]
  // }
  // func.func @concatenate_reshape_3(%arg0: tensor<2x4xbf16>) -> tensor<2x3x4xbf16> {
  //   %0 = stablehlo.concatenate %arg0, %arg0, %arg0, dim = 1 : (tensor<2x4xbf16>, tensor<2x4xbf16>, tensor<2x4xbf16>) -> tensor<2x12xbf16>
  //   %1 = stablehlo.reshape %0 : (tensor<2x12xbf16>) -> tensor<2x3x4xbf16>
  //   return %1 : tensor<2x3x4xbf16>
  //   // broadcast dims [0, 2]
  // }
  // func.func @concatenate_reshape_4(%arg0: tensor<2x4xbf16>) -> tensor<3x2x4xbf16> {
  //   %0 = stablehlo.reshape %arg0 : (tensor<2x4xbf16>) -> tensor<1x2x4xbf16>
  //   %1 = stablehlo.concatenate %0, %0, %0, dim = 0 : (tensor<1x2x4xbf16>, tensor<1x2x4xbf16>, tensor<1x2x4xbf16>) -> tensor<3x2x4xbf16>
  //   return %1 : tensor<3x2x4xbf16>
  //   // broadcast dims [1, 2]
  // }
  // func.func @concatenate_reshape_5(%arg0: tensor<2x4xbf16>) -> tensor<2x3x4xbf16> {
  //   %0 = stablehlo.reshape %arg0 : (tensor<2x4xbf16>) -> tensor<2x1x4xbf16>
  //   %1 = stablehlo.concatenate %0, %0, %0, dim = 1 : (tensor<2x1x4xbf16>, tensor<2x1x4xbf16>, tensor<2x1x4xbf16>) -> tensor<2x3x4xbf16>
  //   return %1 : tensor<2x3x4xbf16>
  //   // broadcast dims [0, 2]
  // }
// }
