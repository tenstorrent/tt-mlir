// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path% default-input-memspace=dram use-tensor-accessor-dma=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer -o %t.ttm %t.mlir

func.func @abs(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: ct_args = [<tensor_accessor_args[0]>]
  // CHECK: emitc.verbatim "constexpr auto tensor_accessor_args_
  // CHECK: emitc.verbatim "const uint32_t page_id_
  // CHECK: .page_id = page_id_
  // CHECK: emitc.call_opaque "abs_tile"
  %0 = "ttir.abs"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}
