// RUN: ttmlir-opt --canonicalize --ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    func.func @add_convs(
            %arg0: tensor<1x1x400x128xbf16>,
            %arg1: tensor<128x128x3x3xbf16>,    // weight1 - 3x3
            %arg2: tensor<1x1x1x128xbf16>,      // bias1
            %arg3: tensor<128x128x1x1xbf16>,    // weight2 - 1x1
            %arg4: tensor<1x1x1x128xbf16>       // bias2
    ) -> tensor<1x1x400x128xbf16> {
        // CHECK: %[[PAD:.*]] = "ttir.pad"(%arg3
        // CHECK: %[[ADD1:.*]] = "ttir.add"(%[[PAD]], %arg1
        // CHECK: %[[ADD2:.*]] = "ttir.add"(%arg4, %arg2
        // CHECK: %[[CONV:.*]] = "ttir.conv2d"(%arg0, %[[ADD1]], %[[ADD2]]
        // CHECK: return %[[CONV]]
        %0 = ttir.empty() : tensor<1x1x400x128xbf16>
        %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0) <{dilation = array<i32: 1, 1>, flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 20, input_width = 20>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> : (tensor<1x1x400x128xbf16>, tensor<128x128x3x3xbf16>, tensor<1x1x1x128xbf16>, tensor<1x1x400x128xbf16>) -> tensor<1x1x400x128xbf16>
        %2 = ttir.empty() : tensor<1x1x400x128xbf16>
        %3 = "ttir.conv2d"(%arg0, %arg3, %arg4, %2) <{dilation = array<i32: 1, 1>, flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 20, input_width = 20>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x1x400x128xbf16>, tensor<128x128x1x1xbf16>, tensor<1x1x1x128xbf16>, tensor<1x1x400x128xbf16>) -> tensor<1x1x400x128xbf16>
        %4 = ttir.empty() : tensor<1x1x400x128xbf16>
        %5 = "ttir.add"(%1, %3, %4) : (tensor<1x1x400x128xbf16>, tensor<1x1x400x128xbf16>, tensor<1x1x400x128xbf16>) -> tensor<1x1x400x128xbf16>
        return %5 : tensor<1x1x400x128xbf16>
    }
}
