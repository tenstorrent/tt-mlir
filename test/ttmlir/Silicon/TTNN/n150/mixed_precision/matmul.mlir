// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="experimental-weight-dtype=bfp_bf8 enable-optimizer=true system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<64x96xbf16> {
    // CHECK-LABEL: func.func @forward
    // CHECK: %[[BFP8_WEIGHT:.*]] = ttcore.load_cached({{.*}}, [%arg1]) : {{.*}} -> tensor<{{.*}}bfp_bf8{{.*}}>
    // CHECK: "ttnn.matmul"(%arg0, %[[BFP8_WEIGHT]])
    %1 = "ttir.matmul"(%arg0, %arg1) : (tensor<64x128xbf16>, tensor<128x96xbf16>) -> tensor<64x96xbf16>
    return %1 : tensor<64x96xbf16>
  }

  func.func @matmul_transpose_lhs(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<128x128xbf16> {
    // CHECK-LABEL: func.func @matmul_transpose_lhs
    // CHECK: %[[BFP8_WEIGHT:.*]] = ttcore.load_cached({{.*}}, [%arg1]) : {{.*}} -> tensor<{{.*}}bfp_bf8{{.*}}>
    // CHECK: "ttnn.matmul"(%arg0, %[[BFP8_WEIGHT]])
    %1 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = true}>: (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<128x128xbf16>
    return %1 : tensor<128x128xbf16>
  }

  func.func @matmul_transpose_rhs(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<64x64xbf16> {
    // CHECK-LABEL: func.func @matmul_transpose_rhs
    // CHECK: %[[BFP8_WEIGHT:.*]] = ttcore.load_cached({{.*}}, [%arg1]) : {{.*}} -> tensor<{{.*}}bfp_bf8{{.*}}>
    // CHECK: "ttnn.matmul"(%arg0, %[[BFP8_WEIGHT]])
    %1 = "ttir.matmul"(%arg0, %arg1) <{transpose_b = true}>: (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }

  func.func @forward_llama_matmul(%arg0: tensor<1x11x2048xf32>, %arg1: tensor<2048x128256xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<1x11x128256xf32> {
    // CHECK-LABEL: func.func @forward_llama_matmul
    // CHECK: %[[BFP8_WEIGHT:.*]] = ttcore.load_cached({{.*}}, [%arg1]) : {{.*}} -> tensor<{{.*}}bfp_bf8{{.*}}>
    // CHECK: "ttnn.matmul"(%arg0, %[[BFP8_WEIGHT]])
    %1 = "ttir.matmul"(%arg0, %arg1) : (tensor<1x11x2048xf32>, tensor<2048x128256xf32>) -> tensor<1x11x128256xf32>
    return %1 : tensor<1x11x128256xf32>
  }

  func.func @matmul_no_parameter(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>) -> tensor<64x96xbf16> {
    // CHECK-LABEL: func.func @matmul_no_parameter
    // CHECK-NOT: ttcore.load_cached
    // CHECK-NOT: bfp_bf8
    // CHECK: "ttnn.matmul"(%arg0, %arg1)
    %1 = "ttir.matmul"(%arg0, %arg1) : (tensor<64x128xbf16>, tensor<128x96xbf16>) -> tensor<64x96xbf16>
    return %1 : tensor<64x96xbf16>
  }
}
