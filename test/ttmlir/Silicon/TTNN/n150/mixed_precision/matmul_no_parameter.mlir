// REQUIRES: opmodel
// RUN: ttmlir-opt -mlir-disable-threading --ttir-to-ttnn-backend-pipeline="experimental-bfp8-weights=true enable-optimizer=true system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module {
  func.func @matmul_no_parameter(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>) -> tensor<64x96xbf16> {
    // CHECK-LABEL: func.func @matmul_no_parameter
    // CHECK-NOT: ttcore.load_cached
    // CHECK-NOT: bfp_bf8
    // CHECK: "ttnn.matmul"(%arg0, %arg1)
    %1 = "ttir.matmul"(%arg0, %arg1) : (tensor<64x128xbf16>, tensor<128x96xbf16>) -> tensor<64x96xbf16>
    return %1 : tensor<64x96xbf16>
  }
}
