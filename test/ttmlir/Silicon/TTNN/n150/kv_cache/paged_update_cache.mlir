// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module {
  func.func @paged_update_cache(%cache: tensor<128x24x128x128xbf16>, %input: tensor<1x24x32x128xbf16>) -> tensor<128x24x128x128xbf16> {
    // CHECK: "ttnn.to_memory_config"(%arg1
    // CHECK-SAME: #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (7,2)>]>
    // CHECK: "ttnn.paged_update_cache"
    %update_index = "ttir.ones"() <{shape = array<i32: 24>}> : () -> tensor<24xi32>
    %page_table = "ttir.ones"() <{shape = array<i32: 24, 16>}> : () -> tensor<24x16xi32>
    %0 = "ttir.paged_update_cache"(%cache, %input, %update_index, %page_table) : (tensor<128x24x128x128xbf16>, tensor<1x24x32x128xbf16>, tensor<24xi32>, tensor<24x16xi32>) -> tensor<128x24x128x128xbf16>
    return %0 : tensor<128x24x128x128xbf16>
  }
}
