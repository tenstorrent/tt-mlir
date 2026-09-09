// RUN: ttmlir-opt --ttir-to-ttnn-runtime-pipeline="enable-trace=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// ttnn.adamw reads lr / beta*_pow back to the host, a sync a captured trace
// would not replay, so it must stay outside the trace region while the compute
// feeding it (the gradient multiply) is traced.
module {
  // CHECK-LABEL: func.func private @trace_0_step
  // CHECK: "ttnn.multiply"
  // CHECK-NOT: "ttnn.adamw"

  // CHECK-LABEL: func.func @step(
  // CHECK: "ttnn.capture_or_execute_trace"
  // CHECK: "ttnn.adamw"
  func.func @step(%param: tensor<1x1x64x64xbf16>, %g0: tensor<1x1x64x64xbf16>,
                  %g1: tensor<1x1x64x64xbf16>, %exp_avg: tensor<1x1x64x64xbf16>,
                  %exp_avg_sq: tensor<1x1x64x64xbf16>, %lr: tensor<1xf32>,
                  %beta1_pow: tensor<1xf32>, %beta2_pow: tensor<1xf32>)
      -> tensor<1x1x64x64xbf16> {
    %grad = "ttir.multiply"(%g0, %g1) : (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>) -> tensor<1x1x64x64xbf16>
    %0:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %lr, %beta1_pow, %beta2_pow) <{
        beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32, weight_decay = 1.000000e-02 : f32}>
        : (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>)
          -> (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>)
    return %0#0 : tensor<1x1x64x64xbf16>
  }
}
