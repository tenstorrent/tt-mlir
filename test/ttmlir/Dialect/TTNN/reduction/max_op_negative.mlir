// RUN: not ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="system-desc-path=ttrt-artifacts/system_desc.ttsys" %s 2>&1 | FileCheck %s
// Negative tests for Max op.
module {
  func.func @forward(%arg0: tensor<128x32x10x4xbf16>) -> tensor<128x1x1x1xbf16> {
    %0 = tensor.empty() : tensor<128x1x1x1xbf16>
    // CHECK: error: 'ttnn.max' op Reduce on more than two dimensions is not currently supported by TTNN
    %1 = "ttir.max"(%arg0, %0) <{dim_arg = [1: i32, 2: i32, 3: i32], keep_dim = true}> : (tensor<128x32x10x4xbf16>, tensor<128x1x1x1xbf16>) -> tensor<128x1x1x1xbf16>
    return %1 : tensor<128x1x1x1xbf16>
  }
}
