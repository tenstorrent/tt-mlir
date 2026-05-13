// RUN: ttmlir-opt --split-input-file \
// RUN:   --d2m-distribute-compute-threads %s | FileCheck %s
// RUN: ttmlir-opt --split-input-file \
// RUN:   --d2m-distribute-compute-threads="num-compute-threads=4 split-dim=n" \
// RUN:   %s | FileCheck %s --check-prefix=SPLITN

// -----

// Default options (num=4, split-dim=m): matmul-shape linalg.generic split on
// the M iteration dim (size 8 -> 4 threads x tile 2). A and C get per-thread
// M-slices via affine.apply (d0 -> d0*2); B is untouched since N is not split
// and B has no M axis. The inner linalg.generic's indexing maps are
// preserved.

// CHECK: #[[$OFFMAP:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK: #[[$MM_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MM_B:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MM_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @distribute_matmul_m
// CHECK-SAME:  %[[A:[A-Za-z0-9_]+]]: memref<8x1x!ttcore.tile<32x32, bf16>, #l1>,
// CHECK-SAME:  %[[B:[A-Za-z0-9_]+]]: memref<1x8x!ttcore.tile<32x32, bf16>, #l1>,
// CHECK-SAME:  %[[C:[A-Za-z0-9_]+]]: memref<8x8x!ttcore.tile<32x32, bf16>, #l1>
// CHECK: d2m.generic
// CHECK: scf.forall (%[[TID:[A-Za-z0-9_]+]]) in (4) {
// CHECK-COUNT-3: affine.apply #[[$OFFMAP]](%[[TID]])
// CHECK: memref.subview %[[A]][%{{[A-Za-z0-9_]+}}, 0] [2, 1] [1, 1] : memref<8x1x!ttcore.tile<32x32, bf16>, #l1> to memref<2x1x!ttcore.tile<32x32, bf16>, strided<[1, 1], offset: ?>, #l1>
// CHECK: memref.subview %[[B]][0, 0] [1, 8] [1, 1] : memref<1x8x!ttcore.tile<32x32, bf16>, #l1> to memref<1x8x!ttcore.tile<32x32, bf16>, strided<[8, 1]>, #l1>
// CHECK: memref.subview %[[C]][%{{[A-Za-z0-9_]+}}, 0] [2, 8] [1, 1] : memref<8x8x!ttcore.tile<32x32, bf16>, #l1> to memref<2x8x!ttcore.tile<32x32, bf16>, strided<[8, 1], offset: ?>, #l1>
// CHECK: linalg.generic {indexing_maps = [#[[$MM_A]], #[[$MM_B]], #[[$MM_C]]], iterator_types = ["parallel", "parallel", "reduction"]}
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : memref<2x1x!ttcore.tile<32x32, bf16>, strided<[1, 1], offset: ?>, #l1>, memref<1x8x!ttcore.tile<32x32, bf16>, strided<[8, 1]>, #l1>)
// CHECK-SAME: outs(%{{.+}} : memref<2x8x!ttcore.tile<32x32, bf16>, strided<[8, 1], offset: ?>, #l1>)
// CHECK: d2m.tile_matmul
// CHECK: linalg.yield
// CHECK: } {mapping = [#d2m.compute_thread<num = 4>]}

func.func @distribute_matmul_m(
    %A: memref<8x1x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>,
    %B: memref<1x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>,
    %C: memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
      ins(%A, %B : memref<8x1x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>, memref<1x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>)
      outs(%C : memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%A, %B : memref<8x1x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>, memref<1x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>)
      outs(%C : memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
    ^bb0(%a: !ttcore.tile<32x32, bf16>, %b: !ttcore.tile<32x32, bf16>, %c: !ttcore.tile<32x32, bf16>):
      %r = "d2m.tile_matmul"(%a, %b, %c) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      linalg.yield %r : !ttcore.tile<32x32, bf16>
    }
  }
  return
}

// -----

// split-dim=n: distribute over the N iteration dim (size 8 -> 4 threads x
// tile 2). B and C get per-thread N-slices; A is untouched since A has no N
// axis. Indexing maps preserved.

// SPLITN: #[[$OFFMAP:.+]] = affine_map<(d0) -> (d0 * 2)>
// SPLITN: #[[$MM_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// SPLITN: #[[$MM_B:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// SPLITN: #[[$MM_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// SPLITN-LABEL: func.func @distribute_matmul_n
// SPLITN-SAME:  %[[A:[A-Za-z0-9_]+]]: memref<8x1x!ttcore.tile<32x32, bf16>, #l1>,
// SPLITN-SAME:  %[[B:[A-Za-z0-9_]+]]: memref<1x8x!ttcore.tile<32x32, bf16>, #l1>,
// SPLITN-SAME:  %[[C:[A-Za-z0-9_]+]]: memref<8x8x!ttcore.tile<32x32, bf16>, #l1>
// SPLITN: d2m.generic
// SPLITN: scf.forall (%[[TID:[A-Za-z0-9_]+]]) in (4) {
// SPLITN-COUNT-3: affine.apply #[[$OFFMAP]](%[[TID]])
// SPLITN: memref.subview %[[A]][0, 0] [8, 1] [1, 1] : memref<8x1x!ttcore.tile<32x32, bf16>, #l1> to memref<8x1x!ttcore.tile<32x32, bf16>, strided<[1, 1]>, #l1>
// SPLITN: memref.subview %[[B]][0, %{{[A-Za-z0-9_]+}}] [1, 2] [1, 1] : memref<1x8x!ttcore.tile<32x32, bf16>, #l1> to memref<1x2x!ttcore.tile<32x32, bf16>, strided<[8, 1], offset: ?>, #l1>
// SPLITN: memref.subview %[[C]][0, %{{[A-Za-z0-9_]+}}] [8, 2] [1, 1] : memref<8x8x!ttcore.tile<32x32, bf16>, #l1> to memref<8x2x!ttcore.tile<32x32, bf16>, strided<[8, 1], offset: ?>, #l1>
// SPLITN: linalg.generic {indexing_maps = [#[[$MM_A]], #[[$MM_B]], #[[$MM_C]]], iterator_types = ["parallel", "parallel", "reduction"]}
// SPLITN-SAME: ins(%{{.+}}, %{{.+}} : memref<8x1x!ttcore.tile<32x32, bf16>, strided<[1, 1]>, #l1>, memref<1x2x!ttcore.tile<32x32, bf16>, strided<[8, 1], offset: ?>, #l1>)
// SPLITN-SAME: outs(%{{.+}} : memref<8x2x!ttcore.tile<32x32, bf16>, strided<[8, 1], offset: ?>, #l1>)
// SPLITN: d2m.tile_matmul
// SPLITN: linalg.yield
// SPLITN: } {mapping = [#d2m.compute_thread<num = 4>]}

func.func @distribute_matmul_n(
    %A: memref<8x1x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>,
    %B: memref<1x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>,
    %C: memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
      ins(%A, %B : memref<8x1x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>, memref<1x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>)
      outs(%C : memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%A, %B : memref<8x1x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>, memref<1x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>)
      outs(%C : memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
    ^bb0(%a: !ttcore.tile<32x32, bf16>, %b: !ttcore.tile<32x32, bf16>, %c: !ttcore.tile<32x32, bf16>):
      %r = "d2m.tile_matmul"(%a, %b, %c) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      linalg.yield %r : !ttcore.tile<32x32, bf16>
    }
  }
  return
}

// -----

// A non-matmul linalg.generic (elementwise: only parallel iterators, single
// input) is left untouched in v1.

func.func @leave_elementwise(
    %A: memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>,
    %B: memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
  // CHECK-LABEL: func.func @leave_elementwise
  // CHECK: d2m.generic
  // CHECK-NOT: scf.forall
  // CHECK: linalg.generic
  // SPLITN-LABEL: func.func @leave_elementwise
  // SPLITN: d2m.generic
  // SPLITN-NOT: scf.forall
  // SPLITN: linalg.generic
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
      ins(%A : memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>)
      outs(%B : memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%A : memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>)
      outs(%B : memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
    ^bb0(%a: !ttcore.tile<32x32, bf16>, %b: !ttcore.tile<32x32, bf16>):
      linalg.yield %a : !ttcore.tile<32x32, bf16>
    }
  }
  return
}
