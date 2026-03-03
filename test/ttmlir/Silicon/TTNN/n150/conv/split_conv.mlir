// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// UNSUPPORTED: true

// TODO(jserbedzija): Failing with tt-metal error:
// "CB page size 64 should be greater than the config tensor page size 132"
// Disabled until fixed in tt-metal
// https://github.com/tenstorrent/tt-metal/issues/35207

module {
  func.func @test_conv_sliced_batch_group_count() -> tensor<1x768x768xbf16> {
    %0 = ttir.empty() : tensor<1x9x3072xbf16>
    %1 = ttir.empty() : tensor<1x9x768xbf16>
    %2 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 9 : i32, 768 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x9x3072xbf16>) -> tensor<1x9x768xbf16>
    %3 = "ttir.slice_static"(%1) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 9 : i32, 192 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x9x768xbf16>) -> tensor<1x9x192xbf16>
    %4 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32, 768 : i32], ends = [1 : i32, 9 : i32, 1536 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x9x3072xbf16>) -> tensor<1x9x768xbf16>
    %5 = "ttir.slice_static"(%1) <{begins = [0 : i32, 0 : i32, 192 : i32], ends = [1 : i32, 9 : i32, 384 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x9x768xbf16>) -> tensor<1x9x192xbf16>
    %6 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32, 1536 : i32], ends = [1 : i32, 9 : i32, 2304 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x9x3072xbf16>) -> tensor<1x9x768xbf16>
    %7 = "ttir.slice_static"(%1) <{begins = [0 : i32, 0 : i32, 384 : i32], ends = [1 : i32, 9 : i32, 576 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x9x768xbf16>) -> tensor<1x9x192xbf16>
    %8 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32, 2304 : i32], ends = [1 : i32, 9 : i32, 3072 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x9x3072xbf16>) -> tensor<1x9x768xbf16>
    %9 = "ttir.slice_static"(%1) <{begins = [0 : i32, 0 : i32, 576 : i32], ends = [1 : i32, 9 : i32, 768 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x9x768xbf16>) -> tensor<1x9x192xbf16>
    %10 = "ttir.reshape"(%2) <{shape = [1 : i32, 9 : i32, 768 : i32, 1 : i32]}> : (tensor<1x9x768xbf16>) -> tensor<1x9x768x1xbf16>
    %11 = "ttir.reshape"(%3) <{shape = [1 : i32, 9 : i32, 192 : i32, 1 : i32]}> : (tensor<1x9x192xbf16>) -> tensor<1x9x192x1xbf16>
    %12 = "ttir.permute"(%10) <{permutation = array<i64: 2, 1, 3, 0>}> : (tensor<1x9x768x1xbf16>) -> tensor<768x9x1x1xbf16>
    %13 = "ttir.permute"(%11) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<1x9x192x1xbf16>) -> tensor<192x1x9x1xbf16>
    %14 = "ttir.conv2d"(%12, %13) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<768x9x1x1xbf16>, tensor<192x1x9x1xbf16>) -> tensor<768x1x1x192xbf16>
    %15 = "ttir.permute"(%14) <{permutation = array<i64: 1, 0, 3, 2>}> : (tensor<768x1x1x192xbf16>) -> tensor<1x768x192x1xbf16>
    %16 = "ttir.reshape"(%15) <{shape = [1 : i32, 768 : i32, 192 : i32]}> : (tensor<1x768x192x1xbf16>) -> tensor<1x768x192xbf16>
    %17 = "ttir.reshape"(%4) <{shape = [1 : i32, 9 : i32, 768 : i32, 1 : i32]}> : (tensor<1x9x768xbf16>) -> tensor<1x9x768x1xbf16>
    %18 = "ttir.reshape"(%5) <{shape = [1 : i32, 9 : i32, 192 : i32, 1 : i32]}> : (tensor<1x9x192xbf16>) -> tensor<1x9x192x1xbf16>
    %19 = "ttir.permute"(%17) <{permutation = array<i64: 2, 1, 3, 0>}> : (tensor<1x9x768x1xbf16>) -> tensor<768x9x1x1xbf16>
    %20 = "ttir.permute"(%18) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<1x9x192x1xbf16>) -> tensor<192x1x9x1xbf16>
    %21 = "ttir.conv2d"(%19, %20) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<768x9x1x1xbf16>, tensor<192x1x9x1xbf16>) -> tensor<768x1x1x192xbf16>
    %22 = "ttir.permute"(%21) <{permutation = array<i64: 1, 0, 3, 2>}> : (tensor<768x1x1x192xbf16>) -> tensor<1x768x192x1xbf16>
    %23 = "ttir.reshape"(%22) <{shape = [1 : i32, 768 : i32, 192 : i32]}> : (tensor<1x768x192x1xbf16>) -> tensor<1x768x192xbf16>
    %24 = "ttir.reshape"(%6) <{shape = [1 : i32, 9 : i32, 768 : i32, 1 : i32]}> : (tensor<1x9x768xbf16>) -> tensor<1x9x768x1xbf16>
    %25 = "ttir.reshape"(%7) <{shape = [1 : i32, 9 : i32, 192 : i32, 1 : i32]}> : (tensor<1x9x192xbf16>) -> tensor<1x9x192x1xbf16>
    %26 = "ttir.permute"(%24) <{permutation = array<i64: 2, 1, 3, 0>}> : (tensor<1x9x768x1xbf16>) -> tensor<768x9x1x1xbf16>
    %27 = "ttir.permute"(%25) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<1x9x192x1xbf16>) -> tensor<192x1x9x1xbf16>
    %28 = "ttir.conv2d"(%26, %27) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<768x9x1x1xbf16>, tensor<192x1x9x1xbf16>) -> tensor<768x1x1x192xbf16>
    %29 = "ttir.permute"(%28) <{permutation = array<i64: 1, 0, 3, 2>}> : (tensor<768x1x1x192xbf16>) -> tensor<1x768x192x1xbf16>
    %30 = "ttir.reshape"(%29) <{shape = [1 : i32, 768 : i32, 192 : i32]}> : (tensor<1x768x192x1xbf16>) -> tensor<1x768x192xbf16>
    %31 = "ttir.reshape"(%8) <{shape = [1 : i32, 9 : i32, 768 : i32, 1 : i32]}> : (tensor<1x9x768xbf16>) -> tensor<1x9x768x1xbf16>
    %32 = "ttir.reshape"(%9) <{shape = [1 : i32, 9 : i32, 192 : i32, 1 : i32]}> : (tensor<1x9x192xbf16>) -> tensor<1x9x192x1xbf16>
    %33 = "ttir.permute"(%31) <{permutation = array<i64: 2, 1, 3, 0>}> : (tensor<1x9x768x1xbf16>) -> tensor<768x9x1x1xbf16>
    %34 = "ttir.permute"(%32) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<1x9x192x1xbf16>) -> tensor<192x1x9x1xbf16>
    %35 = "ttir.conv2d"(%33, %34) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<768x9x1x1xbf16>, tensor<192x1x9x1xbf16>) -> tensor<768x1x1x192xbf16>
    %36 = "ttir.permute"(%35) <{permutation = array<i64: 1, 0, 3, 2>}> : (tensor<768x1x1x192xbf16>) -> tensor<1x768x192x1xbf16>
    %37 = "ttir.reshape"(%36) <{shape = [1 : i32, 768 : i32, 192 : i32]}> : (tensor<1x768x192x1xbf16>) -> tensor<1x768x192xbf16>
    %38 = "ttir.concat"(%16, %23, %30, %37) <{dim = 2 : si32}> : (tensor<1x768x192xbf16>, tensor<1x768x192xbf16>, tensor<1x768x192xbf16>, tensor<1x768x192xbf16>) -> tensor<1x768x768xbf16>
    return %38 : tensor<1x768x768xbf16>
  }
  func.func @test_conv2d_sliced_batch_group_count() -> tensor<1x16x32x32xbf16> {
    %0 = ttir.empty() : tensor<2x4x32x32xbf16>
    %1 = ttir.empty() : tensor<16x4x3x3xbf16>
    %2 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 4 : i32, 32 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x4x32x32xbf16>) -> tensor<1x4x32x32xbf16>
    %3 = "ttir.slice_static"(%1) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [8 : i32, 4 : i32, 3 : i32, 3 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<16x4x3x3xbf16>) -> tensor<8x4x3x3xbf16>
    %4 = "ttir.slice_static"(%0) <{begins = [1 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 4 : i32, 32 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x4x32x32xbf16>) -> tensor<1x4x32x32xbf16>
    %5 = "ttir.slice_static"(%1) <{begins = [8 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [16 : i32, 4 : i32, 3 : i32, 3 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<16x4x3x3xbf16>) -> tensor<8x4x3x3xbf16>
    %6 = "ttir.permute"(%2) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x4x32x32xbf16>) -> tensor<1x32x32x4xbf16>
    %7 = "ttir.permute"(%3) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<8x4x3x3xbf16>) -> tensor<8x4x3x3xbf16>
    %8 = "ttir.conv2d"(%6, %7) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> : (tensor<1x32x32x4xbf16>, tensor<8x4x3x3xbf16>) -> tensor<1x32x32x8xbf16>
    %9 = "ttir.permute"(%4) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x4x32x32xbf16>) -> tensor<1x32x32x4xbf16>
    %10 = "ttir.permute"(%5) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<8x4x3x3xbf16>) -> tensor<8x4x3x3xbf16>
    %11 = "ttir.conv2d"(%9, %10) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> : (tensor<1x32x32x4xbf16>, tensor<8x4x3x3xbf16>) -> tensor<1x32x32x8xbf16>
    %12 = "ttir.concat"(%8, %11) <{dim = 3 : si32}> : (tensor<1x32x32x8xbf16>, tensor<1x32x32x8xbf16>) -> tensor<1x32x32x16xbf16>
    %13 = "ttir.permute"(%12) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x32x32x16xbf16>) -> tensor<1x16x32x32xbf16>
    return %13 : tensor<1x16x32x32xbf16>
  }
}
