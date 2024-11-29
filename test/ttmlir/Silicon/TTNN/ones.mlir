// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @ones_2d() -> tensor<32x128xbf16> {
    // CHECK: {{.*}} = "ttnn.ones"() {{.*}}
    %0 = "ttir.ones"() <{shape = array<i32:32, 128>}> : () -> tensor<32x128xbf16>
    return %0 : tensor<32x128xbf16>
  }
}
