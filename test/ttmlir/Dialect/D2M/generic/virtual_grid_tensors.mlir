#physLayout = #ttcore.metal_layout<
    logical_shape = 128x8192,
    dim_alignments = 32x32,
    collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>,
    undef,
    l1,
    sharded,
    index_map = (d0,d1) -> (0, 8 * d0 + d1)>

#viewLayout = #ttcore.metal_layout<
    logical_shape = 128x8192,
    dim_alignments = 32x32,
    collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>,
    undef,
    l1,
    sharded,
    index_map = (d0,d1,d2,d3) -> (d1 floordiv 8, d1 mod 8,d2,d3)>

!physTensorT = tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #physLayout>
!virtTensorT = tensor<1x64x4x4x!ttcore.tile<32x32, f32>, #viewLayout>

func.func @to_layout() -> !virtTensorT {
  %0 = d2m.empty() : !physTensorT
  %1 = "d2m.view_layout"(%0) : (!physTensorT) -> !virtTensorT
  return %1 : !virtTensorT
}

//func.func @view_layout() -> tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout4> {
//  %arg0 = d2m.empty() : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>
//  // CHECK: = d2m.view_layout
//  %view = "d2m.view_layout"(%arg0) : (tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout4>
//  return %view : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout4>
//}
