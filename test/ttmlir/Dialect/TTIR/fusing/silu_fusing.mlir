// RUN: ttmlir-opt --ttir-fusing %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

module {
    // CHECK-LABEL: func.func @silu_fusing
    func.func @silu_fusing(%arg0: tensor<32x3x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg1: tensor<1x3x224x224xbf16>) -> tensor<1x32x112x112xbf16> {
        // CHECK: "ttir.conv2d"
        // CHECK-NOT: "ttir.sigmoid"
        // CHECK-NOT: "ttir.multiply"
        // CHECK: "ttir.silu"
        %0 = "ttir.conv2d"(%arg1, %arg0) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>, batch_dim = 0 : i64, channel_dim = 1 : i64, height_dim = 2 : i64, width_dim = 3 : i64}> : (tensor<1x3x224x224xbf16>, tensor<32x3x3x3xbf16>) -> tensor<1x32x112x112xbf16>
        %1 = "ttir.sigmoid"(%0) : (tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xbf16>
        %2 = "ttir.multiply"(%0, %1) : (tensor<1x32x112x112xbf16>, tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xbf16>
        return %2 : tensor<1x32x112x112xbf16>
    }

    // CHECK-LABEL: func.func @silu_fusing_with_typecast_after_sigmoid
    // Pattern: sigmoid(bf16) -> typecast(bf16->f32) -> multiply(f32) -> typecast(f32->bf16)
    func.func @silu_fusing_with_typecast_after_sigmoid(%arg0: tensor<32x3x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg1: tensor<1x3x224x224xbf16>) -> tensor<1x32x112x112xbf16> {
        // CHECK: "ttir.conv2d"
        // CHECK-NOT: "ttir.sigmoid"
        // CHECK-NOT: "ttir.multiply"
        // CHECK: "ttir.silu"
        %0 = "ttir.conv2d"(%arg1, %arg0) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>, batch_dim = 0 : i64, channel_dim = 1 : i64, height_dim = 2 : i64, width_dim = 3 : i64}> : (tensor<1x3x224x224xbf16>, tensor<32x3x3x3xbf16>) -> tensor<1x32x112x112xbf16>
        %1 = "ttir.sigmoid"(%0) : (tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xbf16>
        %2 = "ttir.typecast"(%0) <{conservative_folding = false}> : (tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xf32>
        %3 = "ttir.typecast"(%1) <{conservative_folding = false}> : (tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xf32>
        %4 = "ttir.multiply"(%2, %3) : (tensor<1x32x112x112xf32>, tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
        %5 = "ttir.typecast"(%4) <{conservative_folding = false}> : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xbf16>
        return %5 : tensor<1x32x112x112xbf16>
    }

    // CHECK-LABEL: func.func @silu_fusing_with_typecast_before_sigmoid
    // Pattern: typecast(bf16->f32) -> sigmoid(f32) -> multiply(f32) -> typecast(f32->bf16)
    // This is the EfficientNet pattern where sigmoid is done in f32 for numerical stability
    func.func @silu_fusing_with_typecast_before_sigmoid(%arg0: tensor<32x3x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg1: tensor<1x3x224x224xbf16>) -> tensor<1x32x112x112xbf16> {
        // CHECK: "ttir.conv2d"
        %0 = "ttir.conv2d"(%arg1, %arg0) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>, batch_dim = 0 : i64, channel_dim = 1 : i64, height_dim = 2 : i64, width_dim = 3 : i64}> : (tensor<1x3x224x224xbf16>, tensor<32x3x3x3xbf16>) -> tensor<1x32x112x112xbf16>
        %1 = "ttir.typecast"(%0) : (tensor<1x32x112x112xbf16>) -> tensor<1x32x112x112xf32>
        // CHECK-NOT: "ttir.sigmoid"
        %2 = "ttir.sigmoid"(%1) : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
        // CHECK-NOT: "ttir.multiply"
        %3 = "ttir.multiply"(%1, %2) : (tensor<1x32x112x112xf32>, tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
        %4 = "ttir.typecast"(%3) : (tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xbf16>
        // CHECK: "ttir.silu"
        return %4 : tensor<1x32x112x112xbf16>
    }
}
