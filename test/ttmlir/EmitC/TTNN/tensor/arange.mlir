// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path% enable-const-eval=false" -o %t.mlir %s
//
// RUN: ttmlir-opt --ttnn-common-to-runtime-pipeline -o %t_rt.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_rt.mlir
//
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline="tuplify-input-if-empty=true" -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %t.cpp %t2.mlir
// RUN: FileCheck %s --input-file=%t.cpp

func.func @arange() -> tensor<1x1x1x32xf32> {
  // CHECK: ttnn::arange(0, 32, 1, ::ttnn::DataType::FLOAT32,{{.*}}::ttnn::Layout::TILE)
  %0 = "ttir.arange"() <{start = 0 : si64, end = 32 : si64, step = 1 : si64, arange_dimension = 3 : i64}> : () -> tensor<1x1x1x32xf32>
  return %0 : tensor<1x1x1x32xf32>
}
