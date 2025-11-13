// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {
    func.func @gelu_bw_default(%arg0: tensor<4x4xbf16>, %arg1: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
        %0 = ttir.empty() : tensor<4x4xbf16>
        %1 = "ttir.gelu_bw"(%arg0, %arg1, %0) : (tensor<4x4xbf16>, tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
        // CHECK: "ttnn.gelu_bw"
        // CHECK-SAME: approximate = "none"
        // CHECK-SAME: tensor<4x4xbf16
        // CHECK-SAME: tensor<4x4xbf16
        // CHECK-SAME: -> tensor<4x4xbf16
      return %1 : tensor<4x4xbf16>
    }
    func.func @gelu_bw(%arg0: tensor<4x4xbf16>, %arg1: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
        %0 = ttir.empty() : tensor<4x4xbf16>
        %1 = "ttir.gelu_bw"(%arg0, %arg1, %0) <{approximate = "tanh"}> : (tensor<4x4xbf16>, tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
        // CHECK: "ttnn.gelu_bw"
        // CHECK-SAME: approximate = "tanh"
        // CHECK-SAME: tensor<4x4xbf16
        // CHECK-SAME: tensor<4x4xbf16
        // CHECK-SAME: -> tensor<4x4xbf16
      return %1 : tensor<4x4xbf16>
    }
}
