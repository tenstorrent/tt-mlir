// RUN: ttmlir-opt --ttir-load-system-desc="path=%system_desc_path%" --ttir-implicit-device --ttir-layout --ttnn-open-device --convert-ttir-to-ttnn %s  > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
module @jit_convert_element_type attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x2xf32> {mhlo.layout_mode = "default"}) -> (tensor<2x2xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
    // CHECK: "ttnn.close_device"[[C:.*]]
    return %arg0 : tensor<2x2xf32>
  }
}
