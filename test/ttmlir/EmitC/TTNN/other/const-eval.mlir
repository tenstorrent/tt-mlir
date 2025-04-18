// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=true" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-modify-signatures-for-dylib --convert-ttnn-to-emitc --allow-unregistered-dialect %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp --allow-unregistered-dialect %t2.mlir > %basename_t.cpp

module {
  func.func @forward(%arg0: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<input>}, %arg1: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {tt.argument_type = #tt.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: = tt.load_cached(@forward_merge_const_eval_0, [%arg1, %arg2, %arg3])
    %0 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: = "ttnn.add"(%arg0, %arg1)
    %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = ttir.empty() : tensor<32x32xbf16>
    %3 = "ttir.add"(%arg1, %arg2, %2)  <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %4 = ttir.empty() : tensor<32x32xbf16>
    %5 = "ttir.add"(%arg2, %arg3, %4)  <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %6 = ttir.empty() : tensor<32x32xbf16>
    %7 = "ttir.subtract"(%3, %5, %6) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %8 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: = "ttnn.multiply"(%{{.*}}, %{{.*}})
    %9 = "ttir.multiply"(%1, %7, %8) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %9 : tensor<32x32xbf16>
  }
}
