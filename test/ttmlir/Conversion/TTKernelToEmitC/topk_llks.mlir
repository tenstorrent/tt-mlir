// RUN: ttmlir-opt --convert-ttkernel-to-emitc %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @topk_llks
  func.func @topk_llks() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c32 = arith.constant 32 : index

    // CHECK: emitc.call_opaque "topk_tile_init"()
    "ttkernel.topk_tile_init"() : () -> ()

    // CHECK: emitc.call_opaque "topk_local_sort"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {template_args = [#emitc.opaque<"true">]}
    "ttkernel.topk_local_sort"(%c0, %c1, %c5, %c0, %c0, %c0) <{stable_sort = true}> : (index, index, index, index, index, index) -> ()

    // CHECK: emitc.call_opaque "topk_merge"(%{{.*}}, %{{.*}}, %{{.*}}) {template_args = [#emitc.opaque<"true">, #emitc.opaque<"true">]}
    "ttkernel.topk_merge"(%c0, %c0, %c32) <{sort_direction = true, stable_sort = true}> : (index, index, index) -> ()

    // CHECK: emitc.call_opaque "topk_rebuild"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {template_args = [#emitc.opaque<"true">]}
    "ttkernel.topk_rebuild"(%c0, %c1, %c0, %c32, %c5, %c1) <{stable_sort = true}> : (index, index, index, index, index, index) -> ()

    return
  }
}
