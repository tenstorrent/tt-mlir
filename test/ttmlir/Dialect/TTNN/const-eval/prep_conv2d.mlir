// REQUIRES: opmodel
// RUN: ttmlir-opt --tt-populate-argument-types="argument-types=prepare_conv2d_weights=input,constant,input" --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=false enable-const-eval=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK: func.func @prepare_conv2d_weights_const_eval_0
// CHECK: = "ttnn.get_device"()
// CHECK: = "ttnn.prepare_conv2d_weights"(%{{.*}}, %{{.*}})

// CHECK: func.func @prepare_conv2d_weights(
func.func @prepare_conv2d_weights(%arg0: tensor<16x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<16x30x30x64xbf16> {
    // CHECK: ttcore.load_cached(@prepare_conv2d_weights_const_eval_0, [%arg1])
    // CHECK: = "ttnn.get_device"()
    %0 = ttir.empty() : tensor<16x30x30x64xbf16>
    // CHECK: = "ttnn.conv2d"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
                stride = 1: i32,
                padding = 0: i32,
                dilation = 1: i32,
                groups = 1: i32
            }> : (tensor<16x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<16x30x30x64xbf16>) -> tensor<16x30x30x64xbf16>
    return %1 : tensor<16x30x30x64xbf16>
}
