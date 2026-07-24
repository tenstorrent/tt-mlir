// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 enable-greedy-optimizer=true disable-workarounds=true" -o %t %s --mlir-print-local-scope
// RUN: FileCheck %s --input-file=%t

// argmax is valid on a tiled input but runs single-core; a ROW_MAJOR input is
// what unlocks its multicore (fast) path (tt-metal #46340). With workarounds
// disabled, the only way the greedy optimizer reaches a row-major input is its
// RowMajor input siblings.

module {
  func.func @add_argmax(%a: tensor<64x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %b: tensor<64x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<64xi32> {
    // CHECK-LABEL: func.func @add_argmax
    // The sibling reshard is a to_layout (a real tile->row-major retile, NOT a
    // to_memory_config which cannot change page layout), targeting DRAM
    // interleaved so it leaves no L1 footprint the spill pass can't track. Its
    // output is a scalar-element (row-major) DRAM memref, not a tile.
    // CHECK: "ttnn.to_layout"{{.*}}memref<{{[0-9x]+}}xbf16, #ttnn.buffer_type<dram>
    // argmax then consumes that DRAM row-major input.
    // CHECK: "ttnn.argmax"{{.*}}memref<{{[0-9x]+}}xbf16, #ttnn.buffer_type<dram>
    %0 = "ttir.add"(%a, %b) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %1 = "ttir.argmax"(%0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<64x128xbf16>) -> tensor<64xi32>
    return %1 : tensor<64xi32>
  }
}
