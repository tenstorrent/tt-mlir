// RUN: ttmlir-opt --canonicalize --ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
    func.func @main(%arg0: tensor<12x256x1x1xbf16>, %arg1: tensor<256x256x1x1xbf16>) -> tensor<12x256x1x1xbf16> {
        // CHECK: "ttir.conv2d"
        // CHECK-NOT: "ttir.add"
        // CHECK-NOT: "ttir.clamp_tensor"
        // CHECK-NOT: "ttir.div"
        // CHECK: "ttir.hardsigmoid"
        %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<12x256x1x1xbf16>}> : () -> tensor<12x256x1x1xbf16>
        %1 = "ttir.constant"() <{value = dense<3.000000e+00> : tensor<12x256x1x1xbf16>}> : () -> tensor<12x256x1x1xbf16>
        %2 = "ttir.constant"() <{value = dense<6.000000e+00> : tensor<12x256x1x1xbf16>}> : () -> tensor<12x256x1x1xbf16>
        %3 = "ttir.conv2d"(%arg0, %arg1) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>, batch_dim = 0 : i64, channel_dim = 1 : i64, height_dim = 2 : i64, width_dim = 3 : i64}> : (tensor<12x256x1x1xbf16>, tensor<256x256x1x1xbf16>) -> tensor<12x256x1x1xbf16>
        %4 = "ttir.add"(%3, %1) : (tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>) -> tensor<12x256x1x1xbf16>
        %5 = "ttir.clamp_tensor"(%4, %0, %2) : (tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>) -> tensor<12x256x1x1xbf16>
        %6 = "ttir.clamp_tensor"(%5, %0, %2) : (tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>) -> tensor<12x256x1x1xbf16>
        %7 = "ttir.div"(%6, %2) : (tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>) -> tensor<12x256x1x1xbf16>
        return %7 : tensor<12x256x1x1xbf16>
    }
}
