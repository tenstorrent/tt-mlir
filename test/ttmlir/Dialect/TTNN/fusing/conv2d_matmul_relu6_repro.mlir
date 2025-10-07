// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true"
func.func @conv2d_to_matmul_with_relu6(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x1x1xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x32x32x64xbf16> {
    %0 = ttir.empty() : tensor<1x32x32x64xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
                stride = 1: i32,
                padding = 0: i32,
                dilation = 1: i32,
                groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x1x1xbf16>, tensor<1x1x1x64xbf16>, tensor<1x32x32x64xbf16>) -> tensor<1x32x32x64xbf16>

    %2 = ttir.empty() : tensor<1x32x32x64xbf16>
    %3 = "ttir.relu6"(%1, %2) : (tensor<1x32x32x64xbf16>, tensor<1x32x32x64xbf16>) -> tensor<1x32x32x64xbf16>

    return %3 : tensor<1x32x32x64xbf16>
}