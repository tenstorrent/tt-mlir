// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

module attributes {} {
    func.func @test_dense_attr() -> tensor<1x2xbf16> {
        // CHECK: ttnn.constant
        // CHECK-SAME: dense_resource<dense_attr>
        %0 = stablehlo.constant dense_resource<dense_attr> : tensor<1x2xbf16>
        return %0 : tensor<1x2xbf16>
  }
}
{-#
    dialect_resources: {
        builtin: {
            // This should encode for two bfloat16 values which are both 2.0
            // 0x020000000 is a hex string blob
            // 0x0040 is 2.0 in bfloat16
            // 0x00400040 is 2.0, 2.0
            dense_attr: "0x0200000000400040"
        }
    }
#-}
