// RUN: ttmlir-opt --ttir-fusing="ttnn-enable-conv2d-with-multiply-pattern=true" %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

// Batch norm should be decomposed only if it is following conv2d that doesn't have outher users
module {
    // CHECK-LABEL: func.func @conv_batch_norm
    func.func @conv_batch_norm(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg3: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg4: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg5: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg6: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x30x30x64xbf16> {
    // CHECK: "ttir.conv2d"
    // CHECK-NOT: "ttir.batch_norm"
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>

    %18 = ttir.empty() : tensor<1x30x30x64xbf16>
    %19 = "ttir.batch_norm"(%1, %arg3, %arg4, %arg5, %arg6, %18) <{dimension = 3 : i32, epsilon = 9.99999974E-6 : f32}> : (tensor<1x30x30x64xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %19: tensor<1x30x30x64xbf16>
    }
}
module {
    // CHECK-LABEL: func.func @batch_norm_only
    func.func @batch_norm_only(%arg0: tensor<1x30x30x64xbf16>, %arg1: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg3: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg4: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x30x30x64xbf16> {
        // CHECK: "ttir.batch_norm"
        %0 = ttir.empty() : tensor<1x30x30x64xbf16>
        %1 = "ttir.batch_norm"(%arg0, %arg1, %arg2, %arg3, %arg4, %0) <{dimension = 3 : i32, epsilon = 9.99999974E-6 : f32}> : (tensor<1x30x30x64xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
        return %1: tensor<1x30x30x64xbf16>
    }
}
module {
    // CHECK-LABEL: func.func @conv_batch_norm_multiple_uses
    func.func @conv_batch_norm_multiple_uses(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg3: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg4: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg5: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg6: tensor<64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> (tensor<1x30x30x64xbf16>, tensor<1x30x30x64xbf16>) {
    // CHECK: "ttir.conv2d"
    // CHECK: "ttir.add"
    // CHECK: "ttir.batch_norm"
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %2 = ttir.empty() : tensor<1x30x30x64xbf16>
    %3 = "ttir.add"(%1, %arg2, %2) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %4 = ttir.empty() : tensor<1x30x30x64xbf16>
    %5 = "ttir.batch_norm"(%1, %arg3, %arg4, %arg5, %arg6, %4) <{dimension = 3 : i32, epsilon = 9.99999974E-6 : f32}> : (tensor<1x30x30x64xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %3, %5 : tensor<1x30x30x64xbf16>, tensor<1x30x30x64xbf16>
    }
}
