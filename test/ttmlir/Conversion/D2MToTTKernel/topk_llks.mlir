// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s

module {
  // CHECK-LABEL: func.func private @topk_llks
  func.func private @topk_llks() attributes {d2m.thread = #d2m.thread<compute>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c32 = arith.constant 32 : index

    // CHECK: ttkernel.topk_tile_init
    // CHECK: ttkernel.topk_local_sort
    // CHECK-SAME: stable_sort = true
    "d2m.topk_local_sort"(%c0, %c1, %c5, %c0, %c0, %c0) <{stable_sort = true}> : (index, index, index, index, index, index) -> ()

    // CHECK: ttkernel.topk_merge
    // CHECK-SAME: sort_direction = true
    // CHECK-SAME: stable_sort = true
    "d2m.topk_merge"(%c0, %c0, %c32) <{sort_direction = true, stable_sort = true}> : (index, index, index) -> ()

    // CHECK: ttkernel.topk_rebuild
    // CHECK-SAME: stable_sort = true
    "d2m.topk_rebuild"(%c0, %c1, %c0, %c32, %c5, %c1) <{stable_sort = true}> : (index, index, index, index, index, index) -> ()

    return
  }
}
