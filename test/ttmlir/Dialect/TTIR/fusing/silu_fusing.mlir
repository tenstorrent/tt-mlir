// RUN: ttmlir-opt --ttir-fusing %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

module {
    // CHECK-LABEL: func.func @silu_fusing
    func.func @silu_fusing(%arg0: tensor<32x3x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg1: tensor<1x3x224x224xbf16>) -> tensor<1x32x112x112xbf16> {
        // CHECK: "ttir.conv2d"
        // CHECK-NOT: "ttir.sigmoid"
        // CHECK-NOT: "ttir.multiply"
        // CHECK: "ttir.silu"
        %0 = "ttir.permute"(%arg1) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x224x224xbf16>) -> tensor<1x224x224x3xbf16>
        %1 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<32x3x3x3xbf16>) -> tensor<32x3x3x3xbf16>
        %2 = "ttir.conv2d"(%0, %1) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> : (tensor<1x224x224x3xbf16>, tensor<32x3x3x3xbf16>) -> tensor<1x112x112x32xbf16>
        %3 = "ttir.permute"(%2) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x112x112x32xbf16>) -> tensor<1x32x112x112xbf16>
        %4 = "ttir.sigmoid"(%3) : (tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xbf16>
        %5 = "ttir.multiply"(%3, %4) : (tensor<1x32x112x112xbf16>, tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xbf16>
        return %5 : tensor<1x32x112x112xbf16>
    }

    // CHECK-LABEL: func.func @silu_fusing_with_typecast
    func.func @silu_fusing_with_typecast(%arg0: tensor<32x3x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg1: tensor<1x3x224x224xbf16>) -> tensor<1x32x112x112xbf16> {
        // CHECK: "ttir.conv2d"
        // CHECK-NOT: "ttir.sigmoid"
        // CHECK-NOT: "ttir.multiply"
        // CHECK: "ttir.silu"
        %0 = "ttir.permute"(%arg1) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x224x224xbf16>) -> tensor<1x224x224x3xbf16>
        %1 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<32x3x3x3xbf16>) -> tensor<32x3x3x3xbf16>
        %2 = "ttir.conv2d"(%0, %1) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> : (tensor<1x224x224x3xbf16>, tensor<32x3x3x3xbf16>) -> tensor<1x112x112x32xbf16>
        %3 = "ttir.permute"(%2) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x112x112x32xbf16>) -> tensor<1x32x112x112xbf16>
        %4 = "ttir.sigmoid"(%3) : (tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xbf16>
        %5 = "ttir.typecast"(%3) <{conservative_folding = false}> : (tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xf32>
        %6 = "ttir.typecast"(%4) <{conservative_folding = false}> : (tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xf32>
        %7 = "ttir.multiply"(%5, %6) : (tensor<1x32x112x112xf32>, tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
        %8 = "ttir.typecast"(%7) <{conservative_folding = false}> : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xbf16>
        return %8 : tensor<1x32x112x112xbf16>
    }
}
