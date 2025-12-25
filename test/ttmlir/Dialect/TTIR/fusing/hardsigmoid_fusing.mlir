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
        %3 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<12x256x1x1xbf16>) -> tensor<12x1x1x256xbf16>
        %4 = "ttir.permute"(%arg1) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<256x256x1x1xbf16>) -> tensor<256x256x1x1xbf16>
        %5 = "ttir.conv2d"(%3, %4) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<12x1x1x256xbf16>, tensor<256x256x1x1xbf16>) -> tensor<12x1x1x256xbf16>
        %6 = "ttir.permute"(%5) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<12x1x1x256xbf16>) -> tensor<12x256x1x1xbf16>
        %7 = "ttir.add"(%6, %1) : (tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>) -> tensor<12x256x1x1xbf16>
        %8 = "ttir.clamp_tensor"(%7, %0, %2) : (tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>) -> tensor<12x256x1x1xbf16>
        %9 = "ttir.clamp_tensor"(%8, %0, %2) : (tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>) -> tensor<12x256x1x1xbf16>
        %10 = "ttir.div"(%9, %2) : (tensor<12x256x1x1xbf16>, tensor<12x256x1x1xbf16>) -> tensor<12x256x1x1xbf16>
        return %10 : tensor<12x256x1x1xbf16>
    }
}
