// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path% enable-const-eval=true" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-flatbuffer-pipeline -o %t_fb.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_fb.mlir
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir
// RUN: FileCheck %s --input-file %t.mlir

module {
  func.func @forward(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: = ttcore.load_cached(@forward_const_eval_0, [%arg1, %arg2, %arg3])
    // CHECK: %[[TILED_INPUT:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK: = "ttnn.add"(%[[TILED_INPUT]], %arg1)
    %1 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %3 = "ttir.add"(%arg1, %arg2)  : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %5 = "ttir.add"(%arg2, %arg3)  : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %7 = "ttir.subtract"(%3, %5) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: = "ttnn.multiply"(%{{.*}}, %{{.*}})
    %9 = "ttir.multiply"(%1, %7) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %9 : tensor<32x32xbf16>
  }

  func.func @const_eval_no_input(%arg0: tensor<71x4x1xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<71x4x1xbf16> {
    // CHECK: = ttcore.load_cached(@const_eval_no_input_const_eval_0, [])
    %0 = "ttir.full"() <{fill_value = 32.0 : f32, shape = array<i32>}> : () -> tensor<bf16>
    %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1xbf16>
    %2 = "ttir.repeat"(%1) <{repeat_dimensions = array<i64: 71, 4>}> : (tensor<1x1xbf16>) -> tensor<71x4xbf16>
    %3 = "ttir.reshape"(%2) <{shape = [71 : i32, 4 : i32, 1 : i32]}> : (tensor<71x4xbf16>) -> tensor<71x4x1xbf16>
    %4 = "ttir.multiply"(%3, %arg0) : (tensor<71x4x1xbf16>, tensor<71x4x1xbf16>) -> tensor<71x4x1xbf16>
    return %4 : tensor<71x4x1xbf16>
  }
}
