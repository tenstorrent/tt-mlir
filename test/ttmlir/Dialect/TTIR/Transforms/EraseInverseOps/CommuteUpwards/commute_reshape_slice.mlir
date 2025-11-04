// RUN: ttmlir-opt --ttir-erase-inverse-ops="enable-commute-downwards=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
    func.func @test_reshape_slice_commute_upwards(%arg0: tensor<1x160x160x128xbf16> ) -> tensor<1x1x25600x64xbf16> {
        %0 = ttir.empty() : tensor<1x160x160x64xbf16>
        %1 = "ttir.slice_static"(%arg0, %0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 160 : i32, 160 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x160x160x128xbf16>, tensor<1x160x160x64xbf16>) -> tensor<1x160x160x64xbf16>
        %2 = ttir.empty() : tensor<1x1x25600x64xbf16>
        %3 = "ttir.reshape"(%1, %2) <{shape = [1 : i32, 1 : i32, 25600 : i32, 64 : i32]}> : (tensor<1x160x160x64xbf16>, tensor<1x1x25600x64xbf16>) -> tensor<1x1x25600x64xbf16>
        return %3 : tensor<1x1x25600x64xbf16>
    }
}
