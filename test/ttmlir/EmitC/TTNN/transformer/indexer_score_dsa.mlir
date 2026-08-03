// REQUIRES: Blackhole
// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" -o %t.mlir %s
//
// RUN: ttmlir-opt --ttnn-common-to-runtime-pipeline -o %t_rt.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_rt.mlir
//
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %t.cpp %t2.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// The ttcore.composite "indexer_score_dsa" is promoted to ttnn.indexer_score_dsa by
// TTNNResolveComposites and then emitted to C++ as
// ttnn::experimental::indexer_score_dsa.

// seq_shard_axes is the 9th positional parameter, so the four intervening
// parameters are emitted at their ttnn defaults. An unset cluster_axis becomes
// std::nullopt, which is what the kernel treats as "flat device enumeration".

module {
  func.func @indexer_score_dsa(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x32x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16> {
    // CHECK: ttnn::experimental::indexer_score_dsa
    // CHECK-SAME: ::ttnn::operations::experimental::indexer_score::IndexerScoreProgramConfig{}
    // CHECK-SAME: ::std::nullopt, ::std::nullopt, ::std::nullopt, ::std::nullopt)
    %0 = "ttcore.composite"(%q, %k, %w) <{composite_name = "indexer_score_dsa", decomposition = @decomp, composite_attributes = {chunk_start_idx = 0 : ui32}}> : (tensor<1x8x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16>
    return %0 : tensor<1x1x32x32xbf16>
  }
  func.func private @decomp(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x32x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x32xbf16> {
    %0 = "ttir.slice_static"(%q) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 32 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x32x128xbf16>) -> tensor<1x1x32x32xbf16>
    return %0 : tensor<1x1x32x32xbf16>
  }

  // A named mesh axis lands in the last argument slot as a one-element vector.
  func.func @indexer_score_dsa_cluster_axis(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x64x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x64xbf16> {
    // CHECK: ttnn::experimental::indexer_score_dsa
    // CHECK-SAME: ::ttnn::operations::experimental::indexer_score::IndexerScoreProgramConfig{}
    // CHECK-SAME: ::std::nullopt, ::std::nullopt, ::std::nullopt, ::std::vector<uint32_t>{1})
    %0 = "ttcore.composite"(%q, %k, %w) <{composite_name = "indexer_score_dsa", decomposition = @decomp_cluster_axis, composite_attributes = {chunk_start_idx = 32 : ui32, cluster_axis = 1 : ui32}}> : (tensor<1x8x32x128xbf16>, tensor<1x1x64x128xbf16>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x64xbf16>
    return %0 : tensor<1x1x32x64xbf16>
  }
  func.func private @decomp_cluster_axis(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x64x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x64xbf16> {
    %0 = "ttir.slice_static"(%k) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 32 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x64x128xbf16>) -> tensor<1x1x32x64xbf16>
    return %0 : tensor<1x1x32x64xbf16>
  }
}
