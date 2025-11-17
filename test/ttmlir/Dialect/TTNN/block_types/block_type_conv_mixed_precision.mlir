// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-bfp8-conversion=true" %s | FileCheck %s

module  {
  // We take 3 args: two bf16 tensors and two f32 tensors; return bf16.
  // CHECK-LABEL: @mixed(
  // CHECK-SAME: %arg0: tensor<32x32xbf16{{.*}}>, %arg1: tensor<32x32xbf16{{.*}}>, %arg2: tensor<32x32xf32{{.*}}>, %arg3: tensor<32x32xf32{{.*}}>) -> tensor<32x32xbf16{{.*}}>
  func.func @mixed(
      %arg0 : tensor<32x32xbf16>, %arg1 : tensor<32x32xbf16>,
      %arg2 : tensor<32x32xf32>,   %arg3 : tensor<32x32xf32>
  ) -> tensor<32x32xbf16> {
    // ---- f32 path (should stay untouched) ----
    %f32_init = ttir.empty() : tensor<32x32xf32>
    %mul_f32 = "ttir.multiply"(%arg2, %arg3, %f32_init)
      : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    // f32 multiply gets converted to ttnn but stays as f32 (not bfp_bf8):
    // CHECK: "ttnn.multiply"
    // CHECK-SAME: <{dtype = #ttcore.supportedDataTypes<f32>}>
    // CHECK-SAME: (tensor<32x32xf32{{.*}}>, tensor<32x32xf32{{.*}}>) -> tensor<32x32xf32{{.*}}>

    // ---- bf16 path (should be converted inside body to tile<bfp_bf8>) ----
    %bf16_init = ttir.empty() : tensor<32x32xbf16>
    %sum_bf16 = "ttir.add"(%arg0, %arg1, %bf16_init)
      : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // After the pass:
    // - this bf16 add becomes a ttnn.add with tile<bfp_bf8> (inside the body)
    // CHECK: "ttnn.add"
    // CHECK-SAME: tensor<32x32x!ttcore.tile<32x32, bfp_bf8>
    // CHECK-SAME: tensor<32x32x!ttcore.tile<32x32, bfp_bf8>
    // CHECK-SAME: -> tensor<32x32x!ttcore.tile<32x32, bfp_bf8>

    // ---- Cast f32 to bf16 and add to bf16 result ----
    %bf16_init2 = ttir.empty() : tensor<32x32xbf16>
    %mul_f32_to_bf16 = "ttir.typecast"(%mul_f32, %bf16_init2)
      : (tensor<32x32xf32>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    %bf16_init3 = ttir.empty() : tensor<32x32xbf16>
    %final = "ttir.add"(%sum_bf16, %mul_f32_to_bf16, %bf16_init3)
      : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // This final add should also convert to bfp_bf8:
    // CHECK: "ttnn.add"
    // CHECK-SAME: tensor<32x32x!ttcore.tile<32x32, bfp_bf8>
    // CHECK-SAME: tensor<32x32x!ttcore.tile<32x32, bfp_bf8>
    // CHECK-SAME: -> tensor<32x32x!ttcore.tile<32x32, bfp_bf8>

    // And the function return type remains bf16:
    // CHECK: return
    // CHECK-SAME: tensor<32x32xbf16{{.*}}
    return %final : tensor<32x32xbf16>
  }
}
