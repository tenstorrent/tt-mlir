// RUN: ttmlir-opt --split-input-file --d2m-fe-pipeline="enable-elementwise-fusion=true" %s | FileCheck %s

// Verify that elementwise chains containing implicit broadcasts (`tile_bcast`)
// can be fused with downstream eltwise ops. Previously `tile_bcast` carried
// `D2MSkipOpEltwiseFusionTrait` (TODO #5968) which forced each broadcasted
// op to lower as its own d2m.generic; this test now pins the relaxed
// behaviour so the fusion pass aggressively merges adjacent eltwise ops with
// implicit broadcasts. End-to-end correctness for these patterns is covered
// by `test/python/golden/d2m/test_eltwise_fusion.py` and
// `test/python/golden/d2m/test_binary.py` (both pass against the relaxation).
module {
  func.func @bcast_fusion_enabled(%arg1: tensor<32x1xbf16>, %arg2: tensor<1x32xbf16>, %arg3: tensor<1x1xbf16>) -> tensor<32x32xbf16> {
    %1 = "ttir.add"(%arg3, %arg2) : (tensor<1x1xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>
    %3 = "ttir.subtract"(%1, %arg3) : (tensor<1x32xbf16>, tensor<1x1xbf16>) -> tensor<1x32xbf16>
    %5 = "ttir.add"(%arg1, %arg3) : (tensor<32x1xbf16>, tensor<1x1xbf16>) -> tensor<32x1xbf16>
    %7 = "ttir.subtract"(%arg3, %5) : (tensor<1x1xbf16>, tensor<32x1xbf16>) -> tensor<32x1xbf16>
    %9 = "ttir.add"(%7, %3) : (tensor<32x1xbf16>, tensor<1x32xbf16>) -> tensor<32x32xbf16>
    return %9 : tensor<32x32xbf16>
  }
}

// CHECK-LABEL: func.func @bcast_fusion_enabled
// All five eltwise ops (4 broadcasted binary + 1 final 2D-broadcast add)
// must collapse into a SINGLE compute d2m.generic that contains every
// `tile_bcast`/`tile_add`/`tile_sub` in source order. Tilize/untilize
// boundary generics around input/output buffers are unrelated and may still
// appear; what we forbid is any *additional* compute generic between the
// fused tile_bcast/tile_add/tile_sub sequence.
// CHECK:           "d2m.tile_bcast"
// CHECK:           "d2m.tile_add"
// CHECK-NOT:       d2m.generic
// CHECK:           "d2m.tile_bcast"
// CHECK:           "d2m.tile_sub"
// CHECK-NOT:       d2m.generic
// CHECK:           "d2m.tile_bcast"
// CHECK:           "d2m.tile_add"
// CHECK-NOT:       d2m.generic
// CHECK:           "d2m.tile_bcast"
// CHECK:           "d2m.tile_sub"
// CHECK-NOT:       d2m.generic
// CHECK:           "d2m.tile_bcast"
// CHECK:           "d2m.tile_bcast"
// CHECK:           "d2m.tile_add"
