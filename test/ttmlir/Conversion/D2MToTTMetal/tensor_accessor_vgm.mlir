// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="mock-system-desc-arch=wormhole_b0 disable-tolayout-folding=1 use-tensor-accessor-dma=0 force-compile-time-args=0" -o %t.wh_no_ta.mlir %s
// RUN: FileCheck %s --check-prefix=CHECK-WH-NO-TA --input-file=%t.wh_no_ta.mlir
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="mock-system-desc-arch=wormhole_b0 disable-tolayout-folding=1 use-tensor-accessor-dma=1 force-compile-time-args=0" -o %t.wh_ta.mlir %s
// RUN: FileCheck %s --check-prefix=CHECK-WH-TA --input-file=%t.wh_ta.mlir
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="mock-system-desc-arch=blackhole disable-tolayout-folding=1 use-tensor-accessor-dma=0 force-compile-time-args=0" -o %t.bh_no_ta.mlir %s
// RUN: FileCheck %s --check-prefix=CHECK-BH-NO-TA --input-file=%t.bh_no_ta.mlir
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="mock-system-desc-arch=blackhole disable-tolayout-folding=1 use-tensor-accessor-dma=1 force-compile-time-args=0" -o %t.bh_ta.mlir %s
// RUN: FileCheck %s --check-prefix=CHECK-BH-TA --input-file=%t.bh_ta.mlir

#layout = #ttcore.metal_layout<logical_shape = 128x384, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>

module {
  // CHECK-WH-NO-TA-LABEL: func.func @tensor_accessor_vgm
  // CHECK-WH-NO-TA-NOT: TensorAccessor
  // CHECK-WH-NO-TA: "ttmetal.create_buffer"() <{address = {{[0-9]+}} : i64, virtualGridForwardMapping = #map{{[0-9]*}}, virtualGridInverseMapping = #map{{[0-9]*}}}> : () -> memref<1x12x4x1x!ttcore.tile<32x32, f32>
  // CHECK-WH-NO-TA-NOT: TensorAccessor
  // CHECK-WH-NO-TA: #ttmetal.core_range<0x0, 3x4>
  // CHECK-WH-NO-TA-NOT: TensorAccessor
  // CHECK-WH-NO-TA: #ttmetal.core_range<0x0, 2x6>
  // CHECK-WH-NO-TA-NOT: TensorAccessor

  // CHECK-WH-TA-LABEL: func.func @tensor_accessor_vgm
  // CHECK-WH-TA: "ttmetal.create_buffer"() <{address = {{[0-9]+}} : i64, virtualGridForwardMapping = #map{{[0-9]*}}, virtualGridInverseMapping = #map{{[0-9]*}}}> : () -> memref<1x12x4x1x!ttcore.tile<32x32, f32>
  // CHECK-WH-TA: "ttmetal.enqueue_program"{{.*}}#ttmetal.core_range<0x0, 3x4>{{.*}}ct_args = [<tensor_accessor_args[0]>]
  // CHECK-WH-TA: "ttmetal.enqueue_program"{{.*}}#ttmetal.core_range<0x0, 2x6>{{.*}}ct_args = [<tensor_accessor_args[0]>]
  // CHECK-WH-TA: emitc.verbatim "constexpr auto {{.*}} TensorAccessorArgs
  // CHECK-WH-TA: const uint32_t page_id_{{[0-9]+}}

  // CHECK-BH-NO-TA-LABEL: func.func @tensor_accessor_vgm
  // CHECK-BH-NO-TA-NOT: TensorAccessor
  // CHECK-BH-NO-TA: "ttmetal.create_buffer"() <{address = {{[0-9]+}} : i64}> : () -> memref<1x12x4x1x!ttcore.tile<32x32, f32>
  // CHECK-BH-NO-TA-NOT: TensorAccessor
  // CHECK-BH-NO-TA: #ttmetal.core_range<0x0, 1x12>
  // CHECK-BH-NO-TA-NOT: TensorAccessor
  // CHECK-BH-NO-TA: #ttmetal.core_range<0x0, 2x6>
  // CHECK-BH-NO-TA-NOT: TensorAccessor

  // CHECK-BH-TA-LABEL: func.func @tensor_accessor_vgm
  // CHECK-BH-TA: "ttmetal.create_buffer"() <{address = {{[0-9]+}} : i64}> : () -> memref<1x12x4x1x!ttcore.tile<32x32, f32>
  // CHECK-BH-TA: "ttmetal.enqueue_program"{{.*}}#ttmetal.core_range<0x0, 1x12>{{.*}}ct_args = [<tensor_accessor_args[0]>]
  // CHECK-BH-TA: "ttmetal.enqueue_program"{{.*}}#ttmetal.core_range<0x0, 2x6>{{.*}}ct_args = [<tensor_accessor_args[0]>]
  // CHECK-BH-TA: emitc.verbatim "constexpr auto {{.*}} TensorAccessorArgs
  // CHECK-BH-TA: const uint32_t page_id_{{[0-9]+}}

  func.func @tensor_accessor_vgm(%arg0: tensor<128x384xf32>) -> tensor<128x384xf32> {
    %0 = ttir.empty() : tensor<1x1x4x12x!ttcore.tile<32x32, f32>, #layout>
    %1 = ttir.to_layout %arg0, %0 : tensor<128x384xf32> into tensor<1x1x4x12x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x4x12x!ttcore.tile<32x32, f32>, #layout>
    %2 = ttir.empty() : tensor<1x12x4x1x!ttcore.tile<32x32, f32>, #layout>
    %3 = ttir.to_layout %1, %2 : tensor<1x1x4x12x!ttcore.tile<32x32, f32>, #layout> into tensor<1x12x4x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x12x4x1x!ttcore.tile<32x32, f32>, #layout>
    %4 = ttir.empty() : tensor<2x6x2x2x!ttcore.tile<32x32, f32>, #layout>
    %5 = ttir.to_layout %3, %4 : tensor<1x12x4x1x!ttcore.tile<32x32, f32>, #layout> into tensor<2x6x2x2x!ttcore.tile<32x32, f32>, #layout> -> tensor<2x6x2x2x!ttcore.tile<32x32, f32>, #layout>
    %6 = ttir.empty() : tensor<128x384xf32>
    %7 = ttir.to_layout %5, %6 : tensor<2x6x2x2x!ttcore.tile<32x32, f32>, #layout> into tensor<128x384xf32> -> tensor<128x384xf32>
    return %7 : tensor<128x384xf32>
  }
}
