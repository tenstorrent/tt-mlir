// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="experimental-bfp8-weights=true enable-optimizer=true system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module {
  func.func @matmul_transpose_rhs(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<64x64xbf16> {
    // CHECK-LABEL: func.func @matmul_transpose_rhs
    // CHECK: %[[BFP8_WEIGHT:.*]] = ttcore.load_cached({{.*}}, [%arg1]) : {{.*}} -> tensor<{{.*}}bfp_bf8{{.*}}>
    // CHECK: "ttnn.matmul"(%arg0, %[[BFP8_WEIGHT]])
    %1 = "ttir.matmul"(%arg0, %arg1) <{transpose_b = true}>: (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }
}
