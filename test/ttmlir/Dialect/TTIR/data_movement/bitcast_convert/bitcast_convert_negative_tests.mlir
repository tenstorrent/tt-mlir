// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// Verify that verification fails when bit widths do not match.
module {
  func.func @bitcast_convert_bit_width_mismatch(%arg0: tensor<32x32xf32>) -> tensor<32x32xui16> {
    %0 = "ttir.bitcast_convert"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xui16>
    // CHECK: error: 'ttir.bitcast_convert' op Input and output tensor element types must have the same bit width
    return %0 : tensor<32x32xui16>
  }
}

// -----

// Verify that verification fails when input type is not int or float.
module {
  func.func @bitcast_convert_non_int_float_input(%arg0: tensor<32x32xcomplex<f32>>) -> tensor<32x32xui64> {
    %0 = "ttir.bitcast_convert"(%arg0) : (tensor<32x32xcomplex<f32>>) -> tensor<32x32xui64>
    // CHECK: error: 'ttir.bitcast_convert' op Input tensor type must be an integer or floating point type
    return %0 : tensor<32x32xui64>
  }
}
