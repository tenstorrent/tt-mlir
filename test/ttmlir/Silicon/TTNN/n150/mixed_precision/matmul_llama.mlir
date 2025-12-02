// REQUIRES: opmodel
// RUN: ttmlir-opt -mlir-disable-threading --ttir-to-ttnn-backend-pipeline="experimental-bfp8-weights=true enable-optimizer=true system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module {
  func.func @forward_llama_matmul(%arg0: tensor<1x11x2048xf32>, %arg1: tensor<2048x128256xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<1x11x128256xf32> {
    // CHECK-LABEL: func.func @forward_llama_matmul
    // CHECK: %[[BFP8_WEIGHT:.*]] = ttcore.load_cached({{.*}}, [%arg1]) : {{.*}} -> tensor<{{.*}}bfp_bf8{{.*}}>
    // CHECK: "ttnn.matmul"(%arg0, %[[BFP8_WEIGHT]])
    %1 = "ttir.matmul"(%arg0, %arg1) : (tensor<1x11x2048xf32>, tensor<2048x128256xf32>) -> tensor<1x11x128256xf32>
    return %1 : tensor<1x11x128256xf32>
  }
}
