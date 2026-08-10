// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // AdamW without AMSGrad (4 operands).
  func.func @adamw_optimizer(%param: tensor<1x1x64x64xbf16>, %grad: tensor<1x1x64x64xbf16>,
                             %exp_avg: tensor<1x1x64x64xbf16>, %exp_avg_sq: tensor<1x1x64x64xbf16>)
      -> tensor<1x1x64x64xbf16> {
    // CHECK-LABEL: func.func @adamw_optimizer
    // CHECK: "ttnn.adamw"
    // CHECK-SAME: -> ()
    %0:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq) <{
        lr = 1.000000e-03 : f32,
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        beta1_pow = 0.899999976 : f32,
        beta2_pow = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32,
        weight_decay = 1.000000e-02 : f32}>
        : (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>)
          -> (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>)
    return %0#0 : tensor<1x1x64x64xbf16>
  }

  // AdamW with AMSGrad (5 operands). The presence of max_exp_avg_sq is what
  // enables amsgrad in ttml, and it exercises the 5-input path through the
  // op model interface.
  func.func @adamw_amsgrad_optimizer(%param: tensor<1x1x64x64xbf16>, %grad: tensor<1x1x64x64xbf16>,
                                     %exp_avg: tensor<1x1x64x64xbf16>, %exp_avg_sq: tensor<1x1x64x64xbf16>,
                                     %max_exp_avg_sq: tensor<1x1x64x64xbf16>)
      -> tensor<1x1x64x64xbf16> {
    // CHECK-LABEL: func.func @adamw_amsgrad_optimizer
    // CHECK: "ttnn.adamw"
    // CHECK-SAME: -> ()
    %0:4 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %max_exp_avg_sq) <{
        lr = 1.000000e-03 : f32,
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        beta1_pow = 0.899999976 : f32,
        beta2_pow = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32,
        weight_decay = 1.000000e-02 : f32}>
        : (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>)
          -> (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>)
    return %0#0 : tensor<1x1x64x64xbf16>
  }

  // Grad is produced by a compute op the optimizer wants in L1. The multiply is
  // free to take an L1 sharded layout, but adamw's operands must all be DRAM
  // interleaved, so a to_memory_config has to be inserted in between rather
  // than adamw inheriting the L1 layout.
  //
  // Grad reaching adamw must be the *reshard* of the multiply rather than the
  // multiply's own result.
  func.func @adamw_grad_from_l1_producer(%param: tensor<1x1x256x256xbf16>,
                                         %g0: tensor<1x1x256x256xbf16>,
                                         %g1: tensor<1x1x256x256xbf16>,
                                         %exp_avg: tensor<1x1x256x256xbf16>,
                                         %exp_avg_sq: tensor<1x1x256x256xbf16>)
      -> tensor<1x1x256x256xbf16> {
    // CHECK-LABEL: func.func @adamw_grad_from_l1_producer
    // CHECK: %[[MUL:[0-9a-z_]+]] = "ttnn.multiply"
    // CHECK: %[[RESHARD:[0-9a-z_]+]] = "ttnn.to_memory_config"(%[[MUL]])
    // CHECK: "ttnn.adamw"(%{{[0-9a-z_]+}}, %[[RESHARD]],
    // CHECK-SAME: -> ()
    %grad = "ttir.multiply"(%g0, %g1) : (tensor<1x1x256x256xbf16>, tensor<1x1x256x256xbf16>) -> tensor<1x1x256x256xbf16>
    %0:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq) <{
        lr = 1.000000e-03 : f32,
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        beta1_pow = 0.899999976 : f32,
        beta2_pow = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32,
        weight_decay = 1.000000e-02 : f32}>
        : (tensor<1x1x256x256xbf16>, tensor<1x1x256x256xbf16>, tensor<1x1x256x256xbf16>, tensor<1x1x256x256xbf16>)
          -> (tensor<1x1x256x256xbf16>, tensor<1x1x256x256xbf16>, tensor<1x1x256x256xbf16>)
    return %0#0 : tensor<1x1x256x256xbf16>
  }
}
