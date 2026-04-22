// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-materialize-view-returns -o %t %s
// RUN: FileCheck %s --input-file=%t

!input_t = tensor<1x128xf32>
!values_t = tensor<1x128xf32>
!indices_t = tensor<1x128xi32>

module {
  // CHECK-LABEL: func.func @sort_1x128
  // CHECK-NOT: "ttir.sort"
  // CHECK: d2m.generic
  // CHECK: d2m.tile_topk_init
  // CHECK: d2m.tile_topk_local_sort
  // CHECK: d2m.tile_topk_merge
  // CHECK: d2m.tile_topk_rebuild
  func.func @sort_1x128(%arg0: !input_t) -> (!values_t, !indices_t) {
    %values, %indices = "ttir.sort"(%arg0) <{dim = -1 : si32, descending = false, stable = false}> : (!input_t) -> (!values_t, !indices_t)
    return %values, %indices : !values_t, !indices_t
  }
}
