// RUN: ttmlir-opt --ttir-fusing %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

module {
    // CHECK-LABEL: func.func @mish_fusing
    func.func @mish_fusing(%arg0:tensor<1x32x480x640xbf16>) -> tensor<1x32x480x640xbf16> {
        %1 = "ttir.constant"() <{value = dense<2.000000e+01> : tensor<1x32x480x640xbf16>}> : () -> tensor<1x32x480x640xbf16>
        %2 = "ttir.gt"(%arg0, %1) : (tensor<1x32x480x640xbf16>, tensor<1x32x480x640xbf16>) -> tensor<1x32x480x640xi1>
        %3 = "ttir.exp"(%arg0) : (tensor<1x32x480x640xbf16>) -> tensor<1x32x480x640xbf16>
        %4 = "ttir.log1p"(%3) : (tensor<1x32x480x640xbf16>) -> tensor<1x32x480x640xbf16>
        %5 = "ttir.where"(%2, %arg0, %4) : (tensor<1x32x480x640xi1>, tensor<1x32x480x640xbf16>, tensor<1x32x480x640xbf16>) -> tensor<1x32x480x640xbf16>
        %6 = "ttir.tanh"(%5) : (tensor<1x32x480x640xbf16>) -> tensor<1x32x480x640xbf16>
        // CHECK-NOT: "ttir.multiply"
        %7 = "ttir.multiply"(%arg0, %6) : (tensor<1x32x480x640xbf16>, tensor<1x32x480x640xbf16>) -> tensor<1x32x480x640xbf16>
        // CHECK: "ttir.mish"
        return %7:  tensor<1x32x480x640xbf16>
    }

    // CHECK-LABEL: func.func @mish_fusing_with_typecast
    func.func @mish_fusing_with_typecast(%arg0:tensor<1x32x480x640xbf16>) -> tensor<1x32x480x640xf32> {
        %1 = "ttir.constant"() <{value = dense<2.000000e+01> : tensor<1x32x480x640xbf16>}> : () -> tensor<1x32x480x640xbf16>
        %2 = "ttir.gt"(%arg0, %1) : (tensor<1x32x480x640xbf16>, tensor<1x32x480x640xbf16>) -> tensor<1x32x480x640xi1>
        %3 = "ttir.exp"(%arg0) : (tensor<1x32x480x640xbf16>) -> tensor<1x32x480x640xbf16>
        %4 = "ttir.log1p"(%3) : (tensor<1x32x480x640xbf16>) -> tensor<1x32x480x640xbf16>
        %5 = "ttir.where"(%2, %arg0, %4) : (tensor<1x32x480x640xi1>, tensor<1x32x480x640xbf16>, tensor<1x32x480x640xbf16>) -> tensor<1x32x480x640xbf16>
        %6 = "ttir.tanh"(%5) : (tensor<1x32x480x640xbf16>) -> tensor<1x32x480x640xbf16>
        %7 = "ttir.typecast"(%arg0) : (tensor<1x32x480x640xbf16>) -> tensor<1x32x480x640xf32>
        %8 = "ttir.typecast"(%6) : (tensor<1x32x480x640xbf16>) -> tensor<1x32x480x640xf32>
        // CHECK-NOT: "ttir.multiply"
        %9 = "ttir.multiply"(%7, %8) : (tensor<1x32x480x640xf32>, tensor<1x32x480x640xf32>) -> tensor<1x32x480x640xf32>
        // CHECK: "ttir.mish"
        // CHECK: "ttir.typecast"({{.*}}){{.*}} : (tensor<1x32x480x640xbf16>) -> tensor<1x32x480x640xf32>
        return %9:  tensor<1x32x480x640xf32>
    }
}
