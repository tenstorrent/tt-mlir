// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer -o %t.ttm %t.mlir
// UNSUPPORTED: true
func.func @test_typecast(%arg0: tensor<64x128xf32>) -> (tensor<64x128xi32>, tensor<64x128xbf16>) {
  %0 = "ttir.empty"() : () -> tensor<64x128xi32>
  // CHECK: emitc.call_opaque "copy_tile_init"
  // CHECK-NEXT: emitc.call_opaque "copy_tile"
  // CHECK-NEXT: emitc.call_opaque "typecast_tile_init"() {template_args =
  // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float32)">
  // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Int32)">
  // CHECK-NEXT: emitc.call_opaque "typecast_tile"(%{{[0-9]+}}) {template_args =
  // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float32)">
  // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Int32)">
  %1 = "ttir.typecast"(%arg0, %0) <{conservative_folding = true}> : (tensor<64x128xf32>, tensor<64x128xi32>) -> tensor<64x128xi32>
  %2 = "ttir.empty"() : () -> tensor<64x128xbf16>
  // CHECK: emitc.call_opaque "copy_tile_init"
  // CHECK-NEXT: emitc.call_opaque "copy_tile"
  // CHECK-NEXT: emitc.call_opaque "typecast_tile_init"() {template_args =
  // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Int32)">
  // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float16_b)">
  // CHECK-NEXT: emitc.call_opaque "typecast_tile"(%{{[0-9]+}}) {template_args =
  // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Int32)">
  // CHECK-SAME: #emitc.opaque<"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float16_b)">
  %3 = "ttir.typecast"(%1, %2) <{conservative_folding = true}> : (tensor<64x128xi32>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
  return %1, %3 : tensor<64x128xi32>, tensor<64x128xbf16>
}
