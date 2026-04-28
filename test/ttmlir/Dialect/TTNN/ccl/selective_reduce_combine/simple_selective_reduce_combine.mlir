// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
// Unit tests for ttnn selective_reduce_combine op
// Shapes from GPT-OSS: hidden=2880, batch=128, seq=1, K=4, experts=32

// Verify lowering of ttir selective_reduce_combine to ttnn ops

module attributes {} {
  // CHECK-LABEL: selective_reduce_combine_gpt_oss
  func.func @selective_reduce_combine_gpt_oss(%arg0: tensor<16x128x1x2880xbf16>, %arg1: tensor<16x128x1x2880xbf16>, %arg2: tensor<1x128x1x4xi64>, %arg3: tensor<1x128x1x1xi64>) -> tensor<16x128x1x2880xbf16> {
    %0 = "ttir.composite"(%arg0, %arg1, %arg2, %arg3) <{name = "tt.selective_reduce_combine", op_attributes = {hidden_size = 2880 : ui32, batch_size = 128 : ui32, seq_size = 1 : ui32, select_experts_k = 4 : ui32, experts = 32 : ui32}}> : (tensor<16x128x1x2880xbf16>, tensor<16x128x1x2880xbf16>, tensor<1x128x1x4xi64>, tensor<1x128x1x1xi64>) -> tensor<16x128x1x2880xbf16>
    // CHECK: "ttnn.selective_reduce_combine"
    return %0 : tensor<16x128x1x2880xbf16>
  }
}
