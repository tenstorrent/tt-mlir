// RUN: ttmlir-opt --tt-register-device --ttir-lower-to-layout %s | FileCheck %s

#l1_ = #tt.memory_space<l1>
#layout = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<256x768xf32, #l1_>>
#layout1 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<8x24x!tt.tile<32x32, f32>, #l1_>>
#layout2 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <8x8>, memref<1x3x!tt.tile<32x32, f32>, #l1_>>

func.func @to_device(%arg0: tensor<256x768xf32>) -> tensor<256x768xf32, #layout> {
  %0 = ttir.empty() : tensor<256x768xf32, #layout>
  // CHECK: ttir.to_layout %arg0, %0 : tensor<256x768xf32> into tensor<256x768xf32, #layout> hostInfo = #layout -> tensor<256x768xf32, #layout>
  %1 = ttir.to_layout %arg0, %0 : tensor<256x768xf32> into tensor<256x768xf32, #layout> -> tensor<256x768xf32, #layout>
  return %1 : tensor<256x768xf32, #layout>
}

func.func @from_device(%arg0: tensor<256x768xf32, #layout>) -> tensor<256x768xf32> {
  %0 = ttir.empty() : tensor<256x768xf32>
  // CHECK: ttir.to_layout %arg0, %0 : tensor<256x768xf32, #layout> into tensor<256x768xf32> hostInfo = #layout -> tensor<256x768xf32>
  %1 = ttir.to_layout %arg0, %0 : tensor<256x768xf32, #layout> into tensor<256x768xf32> -> tensor<256x768xf32>
  return %1 : tensor<256x768xf32>
}

func.func @tilize(%arg0: tensor<256x768xf32, #layout>) -> tensor<256x768xf32, #layout1> {
  %0 = ttir.empty() : tensor<256x768xf32, #layout1>
  // CHECK: ttir.generic {grid = #tt.grid<1x1>
  // CHECK: ttir.tile_tilize_block
  %1 = ttir.to_layout %arg0, %0 : tensor<256x768xf32, #layout> into tensor<256x768xf32, #layout1> -> tensor<256x768xf32, #layout1>
  return %1 : tensor<256x768xf32, #layout1>
}

func.func @untilize(%arg0: tensor<256x768xf32, #layout1>) -> tensor<256x768xf32, #layout> {
  %0 = ttir.empty() : tensor<256x768xf32, #layout>
  // CHECK: ttir.generic {grid = #tt.grid<1x1>
  // CHECK: ttir.tile_untilize_block
  %1 = ttir.to_layout %arg0, %0 : tensor<256x768xf32, #layout1> into tensor<256x768xf32, #layout> -> tensor<256x768xf32, #layout>
  return %1 : tensor<256x768xf32, #layout>
}

func.func @reblock(%arg0: tensor<256x768xf32, #layout1>) -> tensor<256x768xf32, #layout2> {
  %0 = ttir.empty() : tensor<256x768xf32, #layout2>
  // CHECK: ttir.generic {grid = #tt.grid<8x8>
  // CHECK: ^datamovement0
  %1 = ttir.to_layout %arg0, %0 : tensor<256x768xf32, #layout1> into tensor<256x768xf32, #layout2> -> tensor<256x768xf32, #layout2>
  return %1 : tensor<256x768xf32, #layout2>
}

func.func @compound(%arg0: tensor<256x768xf32>) -> tensor<256x768xf32> {
  %0 = ttir.empty() : tensor<256x768xf32, #layout2>
  %1 = ttir.empty() : tensor<256x768xf32>
  // to_device
  // CHECK: ttir.to_layout {{.*}} : tensor<256x768xf32> into tensor<256x768xf32, [[device:#layout[0-9]*]]>
  // tilize
  // CHECK: ttir.generic {grid = #tt.grid<1x1>
  // CHECK-NEXT: ins(%{{.*}} : tensor<256x768xf32, [[device]]>)
  // CHECK-NEXT: outs(%{{.*}} : tensor<256x768xf32, [[tiled:#layout[0-9]*]]>)
  // reblock
  // CHECK: "ttir.view_layout"(%{{.*}}) <{reinterpretLayout = false}> : (tensor<256x768xf32, [[tiled]]>) -> tensor<256x768xf32, [[reblocked:#layout[0-9]*]]>
  // CHECK-NEXT: ttir.generic {grid = #tt.grid<8x8>
  %2 = ttir.to_layout %arg0, %0 : tensor<256x768xf32> into tensor<256x768xf32, #layout2> -> tensor<256x768xf32, #layout2>
  // undo reblock
  // CHECK: "ttir.view_layout"(%{{.*}}) <{reinterpretLayout = false}> : (tensor<256x768xf32, [[reblocked]]>) -> tensor<256x768xf32, [[tiled]]>
  // CHECK-NEXT: ttir.generic {grid = #tt.grid<1x1>
  // untilize
  // CHECK: ttir.generic {grid = #tt.grid<1x1>
  // CHECK-NEXT: ins(%{{.*}} : tensor<256x768xf32, [[tiled]]>)
  // CHECK-NEXT: outs(%{{.*}} : tensor<256x768xf32, [[device]]>)
  // to_host
  // CHECK: ttir.to_layout {{.*}} : tensor<256x768xf32, [[device]]> into tensor<256x768xf32>
  %3 = ttir.to_layout %2, %1 :  tensor<256x768xf32, #layout2> into tensor<256x768xf32> -> tensor<256x768xf32>
  return %3 : tensor<256x768xf32>
}
