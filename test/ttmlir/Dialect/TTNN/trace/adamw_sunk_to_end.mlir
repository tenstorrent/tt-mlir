// RUN: ttmlir-opt --ttir-to-ttnn-runtime-pipeline="enable-trace=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// ttnn.adamw is not hoistable. When hoistable compute that does not touch its
// operands follows it, the adamw is sunk below that compute so the trace
// region stays contiguous instead of failing on a non-hoistable op in the
// middle.
module {
  // CHECK-LABEL: func.func private @trace_0_step
  // CHECK: "ttnn.multiply"
  // CHECK: "ttnn.add"
  // CHECK-NOT: "ttnn.adamw"

  // CHECK-LABEL: func.func @step(
  // CHECK: "ttnn.capture_or_execute_trace"
  // CHECK: "ttnn.adamw"
  // CHECK: "ttnn.adamw"
  // CHECK: return
  func.func @step(%p0: tensor<1x1x64x64xbf16>, %p1: tensor<1x1x64x64xbf16>,
                  %g0: tensor<1x1x64x64xbf16>, %g1: tensor<1x1x64x64xbf16>,
                  %m0: tensor<1x1x64x64xbf16>, %v0: tensor<1x1x64x64xbf16>,
                  %m1: tensor<1x1x64x64xbf16>, %v1: tensor<1x1x64x64xbf16>,
                  %a: tensor<1x1x64x64xbf16>, %b: tensor<1x1x64x64xbf16>,
                  %lr: tensor<1xf32>, %beta1_pow: tensor<1xf32>, %beta2_pow: tensor<1xf32>)
      -> (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>) {
    %grad0 = "ttir.multiply"(%g0, %g1) : (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>) -> tensor<1x1x64x64xbf16>
    %0:3 = "ttir.adamw"(%p0, %grad0, %m0, %v0, %lr, %beta1_pow, %beta2_pow) <{
        beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32, weight_decay = 1.000000e-02 : f32}>
        : (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>)
          -> (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>)
    // Unrelated hoistable compute after the first adamw.
    %out = "ttir.add"(%a, %b) : (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>) -> tensor<1x1x64x64xbf16>
    %1:3 = "ttir.adamw"(%p1, %g1, %m1, %v1, %lr, %beta1_pow, %beta2_pow) <{
        beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32, weight_decay = 1.000000e-02 : f32}>
        : (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>)
          -> (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>)
    return %0#0, %out, %1#0 : tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>
  }
}
