// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline --split-input-file -o %t.mlir %s

module {
  sdy.mesh @mesh = <["_axis_0"=8]>
  func.func @concatenate_reshape_test(%arg0: tensor<128x2880xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg1: tensor<32x2880x5760xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}, {}]>}) -> tensor<32x128x5760xbf16> {

    %named_arg0 = sdy.named_computation <"input_source"> (%arg0) (%arg_sub: tensor<128x2880xbf16>) {
      sdy.return %arg_sub : tensor<128x2880xbf16>
    } : (tensor<128x2880xbf16>) -> tensor<128x2880xbf16>

    %0 = stablehlo.concatenate
        %named_arg0, %named_arg0, %named_arg0, %named_arg0, %named_arg0, %named_arg0, %named_arg0, %named_arg0,
        %named_arg0, %named_arg0, %named_arg0, %named_arg0, %named_arg0, %named_arg0, %named_arg0, %named_arg0,
        %named_arg0, %named_arg0, %named_arg0, %named_arg0, %named_arg0, %named_arg0, %named_arg0, %named_arg0,
        %named_arg0, %named_arg0, %named_arg0, %named_arg0, %named_arg0, %named_arg0, %named_arg0, %named_arg0,
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

module {
  sdy.mesh @mesh = <["_axis_0"=8]>

  func.func @concatenate_reshape_test(%arg0: tensor<128x2880xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg1: tensor<32x2880x5760xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}, {}]>}) -> tensor<32x128x5760xbf16> {

    %0 = sdy.named_computation <"input_concatenator"> (%arg0) (%arg_sub: tensor<128x2880xbf16>) {
      %concat = stablehlo.concatenate
        %arg_sub, %arg_sub, %arg_sub, %arg_sub, %arg_sub, %arg_sub, %arg_sub, %arg_sub,
        %arg_sub, %arg_sub, %arg_sub, %arg_sub, %arg_sub, %arg_sub, %arg_sub, %arg_sub,
        %arg_sub, %arg_sub, %arg_sub, %arg_sub, %arg_sub, %arg_sub, %arg_sub, %arg_sub,
        %arg_sub, %arg_sub, %arg_sub, %arg_sub, %arg_sub, %arg_sub, %arg_sub, %arg_sub,
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

      sdy.return %concat : tensor<4096x2880xbf16>
    } : (tensor<128x2880xbf16>) -> tensor<4096x2880xbf16>

    %1 = stablehlo.reshape %0 : (tensor<4096x2880xbf16>) -> tensor<32x128x2880xbf16>

    %2 = stablehlo.dot_general %1, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1]
      : (tensor<32x128x2880xbf16>, tensor<32x2880x5760xbf16>) -> tensor<32x128x5760xbf16>

    return %2 : tensor<32x128x5760xbf16>
  }
}
