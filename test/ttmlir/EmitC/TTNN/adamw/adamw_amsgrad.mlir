// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir | FileCheck %s

module {
  func.func @adamw_amsgrad(%param: tensor<1x1x64x64xf32>, %grad: tensor<1x1x64x64xbf16>,
                           %exp_avg: tensor<1x1x64x64xf32>, %exp_avg_sq: tensor<1x1x64x64xf32>,
                           %step_size: tensor<1xf32>, %inv_sqrt_bc2: tensor<1xf32>, %decay_factor: tensor<1xf32>,
                           %max_exp_avg_sq: tensor<1x1x64x64xf32>)
      -> tensor<1x1x64x64xf32> {
    // `max_exp_avg_sq` is the last operand of the op, but the fifth argument of
    // `ttml::metal::adamw_tensor_scalars`, ahead of the scalar tensors (operands
    // four to six, arguments six to eight). Bind the operands to check that the
    // reordering keeps them apart.
    // CHECK: ::ttnn::Tensor [[STEP_SIZE:v[0-9]+]] = {{v[0-9]+}}[4];
    // CHECK: ::ttnn::Tensor [[INV_SQRT_BC2:v[0-9]+]] = {{v[0-9]+}}[5];
    // CHECK: ::ttnn::Tensor [[DECAY_FACTOR:v[0-9]+]] = {{v[0-9]+}}[6];
    // CHECK: ::ttnn::Tensor [[MAX_EXP_AVG_SQ:v[0-9]+]] = {{v[0-9]+}}[7];
    // Epsilon must survive as 1e-8 rather than being rounded down to zero.
    // CHECK: {{^ *}}ttml::metal::adamw_tensor_scalars({{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, [[MAX_EXP_AVG_SQ]], [[STEP_SIZE]], [[INV_SQRT_BC2]], [[DECAY_FACTOR]], 0.899999976f, 0.999000012f, 9.99999993E-9f, ::ttml::metal::StochasticRounding::Enabled);
    %0:4 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %step_size, %inv_sqrt_bc2, %decay_factor, %max_exp_avg_sq) <{
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32,
        stochastic_rounding = true}>
        : (tensor<1x1x64x64xf32>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1x1x64x64xf32>)
          -> (tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>)
    return %0#0 : tensor<1x1x64x64xf32>
  }
}
