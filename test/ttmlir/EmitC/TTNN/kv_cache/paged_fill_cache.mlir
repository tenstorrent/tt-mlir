// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-flatbuffer-pipeline -o %t_fb.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_fb.mlir
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

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
