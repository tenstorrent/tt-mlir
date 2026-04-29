// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,1" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer -o %t.ttm %t.mlir

module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  func.func @permute(%arg0: tensor<1x96x32x128xf32>) -> tensor<1x96x128x32xf32> {
    // CHECK-LABEL: func.func @permute
    // CHECK: "ttmetal.create_buffer"() {{.*}}virtualGridForwardMapping
    // CHECK: "ttmetal.enqueue_read_buffer"({{.*}}, {{.*}}) : (memref<8x8x1536x4xf32, #ttcore.shard{{.*}}, #l1>, memref<1x96x128x32xf32>)
    %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x96x32x128xf32>) -> tensor<1x96x128x32xf32>
    return %0 : tensor<1x96x128x32xf32>
  }
}
