// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-grid-selection -o %t %s
// RUN: FileCheck %s --input-file=%t

!ttype = tensor<128x128xf32>
!ttype1d = tensor<128xf32>
!ttype1d_scalar = tensor<1xf32>
!ttype_row = tensor<1x128xf32>
!ttype_col = tensor<128x1xf32>
!ttype_scalar = tensor<1x1xf32>

!ttype4d = tensor<1x4x32x128xbf16>
!ttype4d_row = tensor<1x1x1x128xbf16>
!ttype4d_col = tensor<1x1x32x1xbf16>
!ttype4d_outer = tensor<1x1x32x128xbf16>
!ttype4d_multi_outer = tensor<2x4x32x128xbf16>
!ttype3d_outer_in = tensor<1x32x32xbf16>
!ttype3d_outer_out = tensor<4x32x32xbf16>

module {
  // CHECK-LABEL: func.func @explicit_1d_bcast
  func.func @explicit_1d_bcast(%arg0: !ttype1d_scalar) -> !ttype1d {
    // CHECK: d2m.generic
    // CHECK: linalg.generic
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type col>}>
    // CHECK-NOT: "ttir.broadcast"
    %0 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 128>}> : (!ttype1d_scalar) -> !ttype1d
    return %0 : !ttype1d
  }

  // CHECK-LABEL: func.func @explicit_bcast_row
  func.func @explicit_bcast_row(%arg0: !ttype_row) -> !ttype {
    // CHECK: d2m.generic
    // CHECK: linalg.generic
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type row>}>
    // CHECK-NOT: "ttir.broadcast"
    %0 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 128, 1>}> : (!ttype_row) -> !ttype
    return %0 : !ttype
  }

  // CHECK-LABEL: func.func @explicit_bcast_col
  func.func @explicit_bcast_col(%arg0: !ttype_col) -> !ttype {
    // CHECK: d2m.generic
    // CHECK: linalg.generic
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type col>}>
    // CHECK-NOT: "ttir.broadcast"
    %0 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 1, 128>}> : (!ttype_col) -> !ttype
    return %0 : !ttype
  }

  // CHECK-LABEL: func.func @explicit_bcast_scalar
  func.func @explicit_bcast_scalar(%arg0: !ttype_scalar) -> !ttype {
    // CHECK: d2m.generic
    // CHECK: linalg.generic
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type scalar>}>
    // CHECK-NOT: "ttir.broadcast"
    %0 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 128, 128>}> : (!ttype_scalar) -> !ttype
    return %0 : !ttype
  }

  // CHECK-LABEL: func.func @explicit_nd_bcast_col
  func.func @explicit_nd_bcast_col(%arg0: !ttype4d_col) -> !ttype4d {
    // CHECK: d2m.generic
    // CHECK: linalg.generic
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type col>}>
    // CHECK-NOT: "ttir.broadcast"
    %0 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 1, 4, 1, 128>}> : (!ttype4d_col) -> !ttype4d
    return %0 : !ttype4d
  }

  // CHECK-LABEL: func.func @explicit_nd_bcast_row
  func.func @explicit_nd_bcast_row(%arg0: !ttype4d_row) -> !ttype4d {
    // CHECK: d2m.generic
    // CHECK: linalg.generic
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type row>}>
    // CHECK-NOT: "ttir.broadcast"
    %0 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 1, 4, 32, 1>}> : (!ttype4d_row) -> !ttype4d
    return %0 : !ttype4d
  }

  // CHECK-LABEL: func.func @explicit_outer_bcast
  func.func @explicit_outer_bcast(%arg0: !ttype4d_outer) -> !ttype4d {
    // CHECK-NOT: d2m.generic
    // CHECK: d2m.view_layout
    // CHECK-NOT: d2m.composite_view
    // CHECK-NOT: "d2m.tile_bcast"
    // CHECK-NOT: "ttir.broadcast"
    %0 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 1, 4, 1, 1>}> : (!ttype4d_outer) -> !ttype4d
    return %0 : !ttype4d
  }

  // CHECK-LABEL: func.func @explicit_3d_outer_bcast
  func.func @explicit_3d_outer_bcast(%arg0: !ttype3d_outer_in) -> !ttype3d_outer_out {
    // CHECK-NOT: d2m.generic
    // CHECK: d2m.view_layout
    // CHECK-NOT: d2m.composite_view
    // CHECK-NOT: "d2m.tile_bcast"
    // CHECK-NOT: "ttir.broadcast"
    %0 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 4, 1, 1>}> : (!ttype3d_outer_in) -> !ttype3d_outer_out
    return %0 : !ttype3d_outer_out
  }

  // CHECK-LABEL: func.func @explicit_multi_outer_bcast
  func.func @explicit_multi_outer_bcast(%arg0: !ttype4d_outer) -> !ttype4d_multi_outer {
    // CHECK-NOT: d2m.generic
    // CHECK: d2m.view_layout
    // CHECK-NOT: d2m.composite_view
    // CHECK-NOT: "d2m.tile_bcast"
    // CHECK-NOT: "ttir.broadcast"
    %0 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 2, 4, 1, 1>}> : (!ttype4d_outer) -> !ttype4d_multi_outer
    return %0 : !ttype4d_multi_outer
  }

  // CHECK-LABEL: func.func @explicit_bcast_then_add
  func.func @explicit_bcast_then_add(%arg0: !ttype_col, %arg1: !ttype) -> !ttype {
    // CHECK: d2m.generic
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type col>}>
    // CHECK: d2m.generic
    // CHECK: d2m.tile_add
    // CHECK-NOT: "ttir.broadcast"
    %0 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 1, 128>}> : (!ttype_col) -> !ttype
    %1 = "ttir.add"(%0, %arg1) : (!ttype, !ttype) -> !ttype
    return %1 : !ttype
  }
}
