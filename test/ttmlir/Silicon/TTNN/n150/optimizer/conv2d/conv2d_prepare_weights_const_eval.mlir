// REQUIRES: opmodel
// RUN: ttmlir-opt --tt-populate-argument-types="argument-types=conv2d_const_eval_weights=input,constant,input" --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=false enable-const-eval=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Tests for conv2d weight preparation with constant evaluation.
// When weights are marked as constants, prepare_conv2d_weights should be hoisted
// into a separate function that can be evaluated at compile time.

module {
  // Test: Constant weights are hoisted to separate const-eval function
  // When weights are marked as constant, the prepare_conv2d_weights op should be
  // extracted into a separate function (named with _const_eval_N suffix) and
  // replaced with a load_cached call in the main function.
  //
  // CHECK: func.func @conv2d_const_eval_weights_const_eval_0
  // CHECK: = "ttnn.get_device"()
  // CHECK: = "ttnn.prepare_conv2d_weights"(%{{.*}}, %{{.*}})
  //
  // CHECK: func.func @conv2d_const_eval_weights(
  func.func @conv2d_const_eval_weights(%arg0: tensor<16x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<16x30x30x64xbf16> {
    // CHECK: ttcore.load_cached(@conv2d_const_eval_weights_const_eval_0, [%arg1])
    // CHECK: = "ttnn.get_device"()
    // CHECK: = "ttnn.conv2d"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
    %0 = "ttir.conv2d"(%arg0, %arg1, %arg2)
            <{
                stride = 1: i32,
                padding = 0: i32,
                dilation = 1: i32,
                groups = 1: i32
            }> : (tensor<16x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>) -> tensor<16x30x30x64xbf16>
    return %0 : tensor<16x30x30x64xbf16>
  }
}
