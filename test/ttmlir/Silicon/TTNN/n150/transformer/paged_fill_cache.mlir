// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
  ttcore.device_module {
    builtin.module {
      func.func @paged_fill_cache(%cache: tensor<128x4x32x256xbf16>, %input: tensor<1x12x65x256xbf16>, %page_table: tensor<8x16xi32>, %batch_idx_tensor: tensor<1xi64>) -> tensor<128x4x32x256xbf16> {
        // CHECK: "ttnn.paged_fill_cache"
        %0 = "ttir.paged_fill_cache"(%cache, %input, %page_table, %batch_idx_tensor) : (tensor<128x4x32x256xbf16>, tensor<1x12x65x256xbf16>, tensor<8x16xi32>, tensor<1xi64>) -> tensor<128x4x32x256xbf16>
        return %0 : tensor<128x4x32x256xbf16>
      }
    }
  }
}
