// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-grid-selection -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test implicit broadcasting in TTIRToD2M conversion.
// Broadcasting follows NumPy semantics: dimensions are aligned right-to-left,
// and lower-rank inputs are implicitly unsqueezed with leading 1s.

!ttype = tensor<128x96xf32>
!ttype_col = tensor<128x1xf32>
!ttype_row = tensor<1x96xf32>
!ttype_scalar = tensor<1x1xf32>

module {

  // Row broadcast: input with dim 1 in height broadcasts across rows.
  // Shape: [128x96] + [1x96] -> [128x96]
  // CHECK-LABEL: func @implicit_bcast_row
  func.func @implicit_bcast_row(%in0: !ttype, %in1: !ttype_row) -> (!ttype) {
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type row>}>
    // CHECK: d2m.tile_add
    %0 = "ttir.add"(%in0, %in1) : (!ttype, !ttype_row) -> (!ttype)
    return %0 : !ttype
  }

  // Col broadcast: input with dim 1 in width broadcasts across columns.
  // Shape: [128x96] + [128x1] -> [128x96]
  // CHECK-LABEL: func @implicit_bcast_col
  func.func @implicit_bcast_col(%in0: !ttype, %in1: !ttype_col) -> (!ttype) {
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type col>}>
    // CHECK: d2m.tile_add
    %0 = "ttir.add"(%in0, %in1) : (!ttype, !ttype_col) -> (!ttype)
    return %0 : !ttype
  }

  // Scalar broadcast: input with dims 1x1 broadcasts across both dimensions.
  // Shape: [128x96] + [1x1] -> [128x96]
  // CHECK-LABEL: func @implicit_bcast_scalar
  func.func @implicit_bcast_scalar(%in0: !ttype, %in1: !ttype_scalar) -> (!ttype) {
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type scalar>}>
    // CHECK: d2m.tile_add
    %0 = "ttir.add"(%in0, %in1) : (!ttype, !ttype_scalar) -> (!ttype)
    return %0 : !ttype
  }

  // Dual broadcast: both inputs need broadcasting in different dimensions.
  // Shape: [128x1] + [1x96] -> [128x96]
  // First input broadcasts along columns, second along rows.
  // CHECK-LABEL: func @implicit_bcast_dual
  func.func @implicit_bcast_dual(%in0: !ttype_col, %in1: !ttype_row) -> (!ttype) {
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type col>}>
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type row>}>
    // CHECK: d2m.tile_add
    %0 = "ttir.add"(%in0, %in1) : (!ttype_col, !ttype_row) -> (!ttype)
    return %0 : !ttype
  }

  // ---- Where op broadcast tests ----

  // Where with row broadcast on true_val.
  // Shape: where([128x96], [1x96], [128x96]) -> [128x96]
  // CHECK-LABEL: func @where_bcast_row
  func.func @where_bcast_row(%cond: !ttype, %true_val: !ttype_row, %false_val: !ttype) -> (!ttype) {
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type row>}>
    // CHECK: d2m.tile_where
    %0 = "ttir.where"(%cond, %true_val, %false_val) : (!ttype, !ttype_row, !ttype) -> (!ttype)
    return %0 : !ttype
  }

  // Where with col broadcast on false_val.
  // Shape: where([128x96], [128x96], [128x1]) -> [128x96]
  // CHECK-LABEL: func @where_bcast_col
  func.func @where_bcast_col(%cond: !ttype, %true_val: !ttype, %false_val: !ttype_col) -> (!ttype) {
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type col>}>
    // CHECK: d2m.tile_where
    %0 = "ttir.where"(%cond, %true_val, %false_val) : (!ttype, !ttype, !ttype_col) -> (!ttype)
    return %0 : !ttype
  }

  // Where with scalar broadcast on condition.
  // Shape: where([1x1], [128x96], [128x96]) -> [128x96]
  // CHECK-LABEL: func @where_bcast_scalar_cond
  func.func @where_bcast_scalar_cond(%cond: !ttype_scalar, %true_val: !ttype, %false_val: !ttype) -> (!ttype) {
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type scalar>}>
    // CHECK: d2m.tile_where
    %0 = "ttir.where"(%cond, %true_val, %false_val) : (!ttype_scalar, !ttype, !ttype) -> (!ttype)
    return %0 : !ttype
  }

  // Where with multiple broadcasts: row on true_val, col on false_val.
  // Shape: where([128x96], [1x96], [128x1]) -> [128x96]
  // CHECK-LABEL: func @where_bcast_dual
  func.func @where_bcast_dual(%cond: !ttype, %true_val: !ttype_row, %false_val: !ttype_col) -> (!ttype) {
    // CHECK: d2m.generic{{.+}}iterator_types = [#parallel, #parallel]
    // CHECK: linalg.generic
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type row>}>
    // CHECK: "d2m.tile_bcast"(%{{.*}}) <{bcast_type = #d2m<tile_bcast_type col>}>
    // CHECK: d2m.tile_where
    %0 = "ttir.where"(%cond, %true_val, %false_val) : (!ttype, !ttype_row, !ttype_col) -> (!ttype)
    return %0 : !ttype
  }
}
