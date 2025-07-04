// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer %t.mlir > %t.ttm
func.func @multiply(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  // CHECK: emitc.call_opaque "binary_op_init_common"
  // CHECK: emitc.call_opaque "mul_tiles_init"
  // CHECK-NEXT: emitc.call_opaque "mul_tiles"
  %1 = "ttir.multiply"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  // CHECK: emitc.call_opaque "binary_op_init_common"
  // CHECK: emitc.call_opaque "add_tiles_init"
  // CHECK-NEXT: emitc.call_opaque "add_tiles"
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @subtract(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  // CHECK: emitc.call_opaque "binary_op_init_common"
  // CHECK: emitc.call_opaque "sub_tiles_init"
  // CHECK-NEXT: emitc.call_opaque "sub_tiles"
  %1 = "ttir.subtract"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @maximum(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  // CHECK: emitc.call_opaque "init_sfpu"
  // CHECK: emitc.call_opaque "copy_tile_init"(%[[CB0:.+]]) :
  // CHECK-NEXT: emitc.call_opaque "copy_tile"(%[[CB0]], %{{.+}}, %[[DST_IDX0:.+]])
  // CHECK: emitc.call_opaque "copy_tile_init"(%[[CB1:.+]]) :
  // CHECK-NOT: emitc.call_opaque "copy_tile"(%{{.+}}, %{{.+}}, %[[DST_IDX0]])
  // CHECK-NEXT: emitc.call_opaque "copy_tile"(%[[CB1]], %{{.+}}, %[[DST_IDX1:.+]])
  // CHECK: emitc.call_opaque "max_tile_init"
  // CHECK-NEXT: emitc.call_opaque "max_tile"
  %1 = "ttir.maximum"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @div(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  // CHECK: emitc.call_opaque "init_sfpu"
  // CHECK: emitc.call_opaque "copy_tile_init"(%[[CB0:.+]]) :
  // CHECK-NEXT: emitc.call_opaque "copy_tile"(%[[CB0]], %{{.+}}, %[[DST_IDX0:.+]])
  // CHECK: emitc.call_opaque "copy_tile_init"(%[[CB1:.+]]) :
  // CHECK-NOT: emitc.call_opaque "copy_tile"(%{{.+}}, %{{.+}}, %[[DST_IDX0]])
  // CHECK-NEXT: emitc.call_opaque "copy_tile"(%[[CB1]], %{{.+}}, %[[DST_IDX1:.+]])
  // CHECK: emitc.call_opaque "div_binary_tile_init"
  // CHECK-NEXT: emitc.call_opaque "div_binary_tile"
  %1 = "ttir.div"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @pow(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  // CHECK: emitc.call_opaque "init_sfpu"
  // CHECK: emitc.call_opaque "copy_tile_init"(%[[CB0:.+]]) :
  // CHECK-NEXT: emitc.call_opaque "copy_tile"(%[[CB0]], %{{.+}}, %[[DST_IDX0:.+]])
  // CHECK: emitc.call_opaque "copy_tile_init"(%[[CB1:.+]]) :
  // CHECK-NOT: emitc.call_opaque "copy_tile"(%{{.+}}, %{{.+}}, %[[DST_IDX0]])
  // CHECK-NEXT: emitc.call_opaque "copy_tile"(%[[CB1]], %{{.+}}, %[[DST_IDX1:.+]])
  // CHECK: emitc.call_opaque "power_binary_tile_init"
  // CHECK-NEXT: emitc.call_opaque "power_binary_tile"
  %1 = "ttir.pow"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @exp(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  // CHECK: emitc.call_opaque "init_sfpu"
  // CHECK: emitc.call_opaque "copy_tile_init"(%[[CB0:.+]]) :
  // CHECK-NEXT: emitc.call_opaque "copy_tile"(%[[CB0]], %{{.+}}, %{{.+}})
  // CHECK: emitc.call_opaque "exp_tile_init"
  // CHECK-NEXT: emitc.call_opaque "exp_tile"
  %1 = "ttir.exp"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @sin(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  // CHECK: emitc.call_opaque "init_sfpu"
  // CHECK: emitc.call_opaque "copy_tile_init"(%[[CB0:.+]]) :
  // CHECK-NEXT: emitc.call_opaque "copy_tile"(%[[CB0]], %{{.+}}, %{{.+}})
  // CHECK: emitc.call_opaque "sin_tile_init"
  // CHECK-NEXT: emitc.call_opaque "sin_tile"
  %1 = "ttir.sin"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @cos(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  // CHECK: emitc.call_opaque "init_sfpu"
  // CHECK: emitc.call_opaque "copy_tile_init"(%[[CB0:.+]]) :
  // CHECK-NEXT: emitc.call_opaque "copy_tile"(%[[CB0]], %{{.+}}, %{{.+}})
  // CHECK: emitc.call_opaque "cos_tile_init"
  // CHECK-NEXT: emitc.call_opaque "cos_tile"
  %1 = "ttir.cos"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @rsqrt(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  // CHECK: emitc.call_opaque "init_sfpu"
  // CHECK: emitc.call_opaque "copy_tile_init"(%[[CB0:.+]]) :
  // CHECK-NEXT: emitc.call_opaque "copy_tile"(%[[CB0]], %{{.+}}, %{{.+}})
  // CHECK: emitc.call_opaque "rsqrt_tile_init"
  // CHECK-NEXT: emitc.call_opaque "rsqrt_tile"
  %1 = "ttir.rsqrt"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @neg(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  // CHECK: emitc.call_opaque "init_sfpu"
  // CHECK: emitc.call_opaque "copy_tile_init"(%[[CB0:.+]]) :
  // CHECK-NEXT: emitc.call_opaque "copy_tile"(%[[CB0]], %{{.+}}, %{{.+}})
  // CHECK: emitc.call_opaque "negative_tile_init"
  // CHECK-NEXT: emitc.call_opaque "negative_tile"
  %1 = "ttir.neg"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @sigmoid(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  // CHECK: emitc.call_opaque "init_sfpu"
  // CHECK: emitc.call_opaque "copy_tile_init"(%[[CB0:.+]]) :
  // CHECK-NEXT: emitc.call_opaque "copy_tile"(%[[CB0]], %{{.+}}, %{{.+}})
  // CHECK: emitc.call_opaque "sigmoid_tile_init"
  // CHECK-NEXT: emitc.call_opaque "sigmoid_tile"
  %1 = "ttir.sigmoid"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @ceil(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
  %0 = ttir.empty() : tensor<64x128xbf16>
  // CHECK: emitc.call_opaque "init_sfpu"
  // CHECK: emitc.call_opaque "copy_tile_init"(%[[CB0:.+]]) :
  // CHECK-NEXT: emitc.call_opaque "copy_tile"(%[[CB0]], %{{.+}}, %{{.+}})
  // CHECK: emitc.call_opaque "rounding_op_tile_init"
  // CHECK-NEXT: emitc.call_opaque "ceil_tile"
  %1 = "ttir.ceil"(%arg0, %0) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
  return %1 : tensor<64x128xbf16>
}

func.func @ceil_f32(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  // CHECK: emitc.call_opaque "init_sfpu"
  // CHECK: emitc.call_opaque "copy_tile_init"(%[[CB0:.+]]) :
  // CHECK-NEXT: emitc.call_opaque "copy_tile"(%[[CB0]], %{{.+}}, %{{.+}})
  // CHECK: emitc.call_opaque "rounding_op_tile_init"
  // CHECK-NEXT: emitc.call_opaque "ceil_tile_float32"
  %1 = "ttir.ceil"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
