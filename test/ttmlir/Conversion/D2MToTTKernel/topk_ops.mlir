// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s

// Tests the 1:1 lowering of d2m.tile_topk_* ops to ttkernel.topk_* ops.

module {
  // CHECK-LABEL: func.func @test_tile_topk_init
  func.func @test_tile_topk_init() attributes {d2m.thread = #d2m.thread<compute>} {
    // CHECK: ttkernel.topk_tile_init
    "d2m.tile_topk_init"() : () -> ()
    return
  }

  // CHECK-LABEL: func.func @test_tile_topk_local_sort_default
  func.func @test_tile_topk_local_sort_default() attributes {d2m.thread = #d2m.thread<compute>} {
    %idst = arith.constant 0 : index
    %idir = arith.constant 1 : i32
    %end_phase = arith.constant 4 : i32
    %start_phase = arith.constant 0 : i32
    %end_step = arith.constant 0 : i32
    %start_step = arith.constant 0 : i32
    // CHECK: ttkernel.topk_local_sort
    // CHECK-NOT: stable_sort = true
    "d2m.tile_topk_local_sort"(%idst, %idir, %end_phase, %start_phase, %end_step, %start_step) : (index, i32, i32, i32, i32, i32) -> ()
    return
  }

  // CHECK-LABEL: func.func @test_tile_topk_local_sort_stable
  func.func @test_tile_topk_local_sort_stable() attributes {d2m.thread = #d2m.thread<compute>} {
    %idst = arith.constant 0 : index
    %idir = arith.constant 1 : i32
    %end_phase = arith.constant 4 : i32
    %start_phase = arith.constant 0 : i32
    %end_step = arith.constant 0 : i32
    %start_step = arith.constant 0 : i32
    // CHECK: ttkernel.topk_local_sort
    // CHECK-SAME: stable_sort = true
    "d2m.tile_topk_local_sort"(%idst, %idir, %end_phase, %start_phase, %end_step, %start_step) <{stable_sort = true}> : (index, i32, i32, i32, i32, i32) -> ()
    return
  }

  // CHECK-LABEL: func.func @test_tile_topk_merge_default
  func.func @test_tile_topk_merge_default() attributes {d2m.thread = #d2m.thread<compute>} {
    %idst = arith.constant 0 : index
    %m_iter = arith.constant 1 : i32
    %k = arith.constant 32 : i32
    // CHECK: ttkernel.topk_merge
    // CHECK-NOT: idir = true
    // CHECK-NOT: stable_sort = true
    "d2m.tile_topk_merge"(%idst, %m_iter, %k) : (index, i32, i32) -> ()
    return
  }

  // CHECK-LABEL: func.func @test_tile_topk_merge_idir
  func.func @test_tile_topk_merge_idir() attributes {d2m.thread = #d2m.thread<compute>} {
    %idst = arith.constant 0 : index
    %m_iter = arith.constant 1 : i32
    %k = arith.constant 32 : i32
    // CHECK: ttkernel.topk_merge
    // CHECK-SAME: idir = true
    "d2m.tile_topk_merge"(%idst, %m_iter, %k) <{idir = true}> : (index, i32, i32) -> ()
    return
  }

  // CHECK-LABEL: func.func @test_tile_topk_rebuild
  func.func @test_tile_topk_rebuild() attributes {d2m.thread = #d2m.thread<compute>} {
    %idst = arith.constant 0 : index
    %idir = arith.constant true
    %m_iter = arith.constant 1 : i32
    %k = arith.constant 32 : i32
    %logk = arith.constant 5 : i32
    %skip_second = arith.constant 0 : i32
    // CHECK: ttkernel.topk_rebuild
    "d2m.tile_topk_rebuild"(%idst, %idir, %m_iter, %k, %logk, %skip_second) : (index, i1, i32, i32, i32, i32) -> ()
    return
  }
}
