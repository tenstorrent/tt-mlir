// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8 mock-system-desc-arch=wormhole_b0" -o %t %s
// RUN: FileCheck %s --input-file=%t --check-prefix=WH12
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8 mock-system-desc-arch=blackhole" -o %t2 %s
// RUN: FileCheck %s --input-file=%t2 --check-prefix=BH8

// The DS weight layout is width-sharded across exactly the DRAM banks the
// *device* has, and its shard width follows from that count.
//
// Wormhole exposes 12 banks, Blackhole 8. A hardcoded count produces a weight
// layout that cannot be allocated on a part with a different one: tensor
// creation aborts in get_dram_channel_from_logical_core ("Logical DRAM core ...
// is outside valid range"), and nothing before silicon catches it because
// validateTensorSpec deliberately skips the shard bounding-box check for DRAM
// buffers.
//
// The N padding differs per bank count, which is what makes the two cases
// distinct:
//   WH: pad 4096 up to a multiple of 32*12=384 -> 4224, /12 = 352 = 11 tiles
//   BH: pad 4096 up to a multiple of 32*8 =256 -> 4096, /8  = 512 = 16 tiles
// K is 4096 = 128 tiles in both.
//
// WH12-DAG: #ttnn.ttnn_layout<{{.*}}<1x12>, memref<128x11x!ttcore.tile<32x32, bfp_bf8>, #dram>, <width_sharded>
// BH8-DAG: #ttnn.ttnn_layout<{{.*}}<1x8>, memref<128x16x!ttcore.tile<32x32, bfp_bf8>, #dram>, <width_sharded>

module attributes {} {
  // WH12-LABEL: func.func @ds_matmul_bank_count
  // WH12: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config

  // BH8-LABEL: func.func @ds_matmul_bank_count
  // BH8: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config
  func.func @ds_matmul_bank_count(
      %act: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %weight: tensor<4096x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %other: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<32x4096xbf16> {
    %0 = "ttir.matmul"(%act, %weight) : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %1 = "ttir.multiply"(%0, %other) : (tensor<32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    return %1 : tensor<32x4096xbf16>
  }
}
