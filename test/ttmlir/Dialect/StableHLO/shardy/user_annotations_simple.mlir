// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

sdy.mesh @mesh = <["model"=1, "batch"=2]>

func.func public @full_arg_annotation(%arg0: tensor<64x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {"model"}]>}) -> tensor<64x128xf32> {
  %0 = stablehlo.cbrt %arg0 : tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}
// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"batch"}, {"model"}]>] out_shardings=[<@mesh, [{"batch"}, {"model"}]>]
// CHECK: stablehlo.cbrt %arg1 : tensor<32x128xf32>
// CHECK: sdy.return %1 : tensor<32x128xf32>

func.func public @partial_arg_annotation(%arg0: tensor<32x48x24x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {"model"}, {}, {}]>}, %arg1: tensor<32x48x24x32xf32>) -> tensor<32x48x24x32xf32> {
  %0 = stablehlo.subtract %arg0, %arg1 : tensor<32x48x24x32xf32>
  return %0 : tensor<32x48x24x32xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {"model"}, {}, {}]>, <@mesh, [{"batch"}, {"model"}, {}, {}]>] out_shardings=[<@mesh, [{"batch"}, {"model"}, {}, {}]>]
// CHECK: stablehlo.subtract %arg2, %arg3 : tensor<16x48x24x32xf32>
// CHECK: sdy.return %1 : tensor<16x48x24x32xf32>

func.func public @partial_arg_annotation_two(%arg0: tensor<32x48x24x32xf32>, %arg1: tensor<32x48x24x32xf32>{sdy.sharding = #sdy.sharding<@mesh, [{}, {"model"}, {"batch"}, {}]>}) -> tensor<32x48x24x32xf32> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<32x48x24x32xf32>
  return %0 : tensor<32x48x24x32xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{}, {"model"}, {"batch"}, {}]>, <@mesh, [{}, {"model"}, {"batch"}, {}]>] out_shardings=[<@mesh, [{}, {"model"}, {"batch"}, {}]>]
// CHECK: stablehlo.multiply %arg2, %arg3 : tensor<32x48x12x32xf32>
// CHECK: sdy.return %1 : tensor<32x48x12x32xf32>


func.func public @full_op_annotation(%arg0: tensor<32x48x24x32xf32>, %arg1: tensor<32x48x24x32xf32>) -> tensor<32x48x24x32xf32> {
  %0 = stablehlo.multiply %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"model"}, {"batch"}, {}]>]>} : tensor<32x48x24x32xf32>
  return %0 : tensor<32x48x24x32xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{}, {"model"}, {"batch"}, {}]>, <@mesh, [{}, {"model"}, {"batch"}, {}]>] out_shardings=[<@mesh, [{}, {"model"}, {"batch"}, {}]>]
// CHECK: stablehlo.multiply %arg2, %arg3 : tensor<32x48x12x32xf32>
// CHECK: sdy.return %1 : tensor<32x48x12x32xf32>

func.func public @partial_op_annotation(%arg0: tensor<32x1x8192x784xf32>, %arg1: tensor<784x2048xf32>) -> (tensor<32x1x8192x2048xf32> {jax.result_info = ""}) {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [3] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1x8192x784xf32>, tensor<784x2048xf32>) -> tensor<32x1x8192x2048xf32>
  %1 = stablehlo.add %0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch"}, {}, {}, {}]>]>} : tensor<32x1x8192x2048xf32>
  return %1 : tensor<32x1x8192x2048xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>, <@mesh, [{}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>]
// CHECK: %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [3] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x1x8192x784xf32>, tensor<784x2048xf32>) -> tensor<16x1x8192x2048xf32>
// CHECK: %2 = stablehlo.add %1, %1 : tensor<16x1x8192x2048xf32>
// CHECK: sdy.return %2 : tensor<16x1x8192x2048xf32>

func.func public @full_arg_full_op_annotation(%arg0: tensor<32x1x8192x784xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}, %arg1: tensor<784x2048xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<32x1x8192x2048xf32> {jax.result_info = ""}) {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [3] x [0], precision = [DEFAULT, DEFAULT] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch"}, {}, {}, {}]>]>} : (tensor<32x1x8192x784xf32>, tensor<784x2048xf32>) -> tensor<32x1x8192x2048xf32>
  %1 = stablehlo.add %0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch"}, {}, {}, {}]>]>} : tensor<32x1x8192x2048xf32>
  return %1 : tensor<32x1x8192x2048xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>, <@mesh, [{}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>]
// CHECK: %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [3] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x1x8192x784xf32>, tensor<784x2048xf32>) -> tensor<16x1x8192x2048xf32>
// CHECK: %2 = stablehlo.add %1, %1 : tensor<16x1x8192x2048xf32>
// CHECK: sdy.return %2 : tensor<16x1x8192x2048xf32>

func.func @constant(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>}, %arg1: tensor<16x16xf32>) -> (tensor<8x16xf32>) {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = stablehlo.add %arg0, %0 : tensor<8x16xf32>
  %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
  return %2 : tensor<8x16xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {}]>, <@mesh, [{}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}]>]
// CHECK: %cst = stablehlo.constant dense<1.000000e+00> : tensor<4x16xf32>
// CHECK: %1 = stablehlo.add %arg2, %cst : tensor<4x16xf32>
// CHECK: %2 = stablehlo.dot_general %1, %arg3, contracting_dims = [1] x [0] : (tensor<4x16xf32>, tensor<16x16xf32>) -> tensor<4x16xf32>
// CHECK: sdy.return %2 : tensor<4x16xf32>

func.func @transpose(%arg0: tensor<256x32x64x100xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}) -> tensor<100x32x256x64xf32> {
  %0 = stablehlo.transpose %arg0, dims = [3, 1, 0, 2] : (tensor<256x32x64x100xf32>) -> tensor<100x32x256x64xf32>
  return %0 : tensor<100x32x256x64xf32>
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>] out_shardings=[<@mesh, [{}, {}, {"batch"}, {}]>]
// CHECK: %1 = stablehlo.transpose %arg1, dims = [3, 1, 0, 2] : (tensor<128x32x64x100xf32>) -> tensor<100x32x128x64xf32>
// CHECK: sdy.return %1 : tensor<100x32x128x64xf32>

func.func @concat_operands_with_same_shape(%arg0: tensor<4x16x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}]>}, %arg1: tensor<4x16x256xf32>) -> tensor<4x32x256xf32> {
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<4x16x256xf32>, tensor<4x16x256xf32>) -> tensor<4x32x256xf32>
  return %0 : tensor<4x32x256xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {}, {}]>, <@mesh, [{"batch"}, {}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}]>]
// CHECK: %1 = stablehlo.concatenate %arg2, %arg3, dim = 1 : (tensor<2x16x256xf32>, tensor<2x16x256xf32>) -> tensor<2x32x256xf32>
// CHECK: sdy.return %1 : tensor<2x32x256xf32>

func.func public @logical_and(%arg0: tensor<32x32xi1> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>}, %arg1: tensor<32x32xi1>) -> tensor<32x32xi1> {
  %0 = stablehlo.and  %arg0, %arg1 : tensor<32x32xi1>
  return %0 : tensor<32x32xi1>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {}]>, <@mesh, [{"batch"}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}]>]
// CHECK: %1 = stablehlo.and %arg2, %arg3 : tensor<16x32xi1>
// CHECK: sdy.return %1 : tensor<16x32xi1>

func.func public @multiple_output(%arg0: tensor<64x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {"model"}]>}) -> (tensor<64x128xf32>, tensor<64x128xf32>) {
  %0 = stablehlo.cbrt %arg0 : tensor<64x128xf32>
  %1 = stablehlo.cbrt %arg0 : tensor<64x128xf32>
  return %0, %1 : tensor<64x128xf32>, tensor<64x128xf32>
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"batch"}, {"model"}]>] out_shardings=[<@mesh, [{"batch"}, {"model"}]>, <@mesh, [{"batch"}, {"model"}]>]
// CHECK: %1 = stablehlo.cbrt %arg1 : tensor<32x128xf32>
// CHECK: %2 = stablehlo.cbrt %arg1 : tensor<32x128xf32>
// CHECK: sdy.return %1, %2 : tensor<32x128xf32>, tensor<32x128xf32>

func.func public @constant_operand(%arg0: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}) -> (tensor<f32>) {
  return %arg0 :tensor<f32>
}

// CHECK: %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, []>] out_shardings=[<@mesh, []>]
// CHECK: sdy.return %arg1 : tensor<f32>
