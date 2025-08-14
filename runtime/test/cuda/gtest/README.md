# CUDA Program Executor Tests

This directory contains comprehensive tests for the CUDA program executor that validate its functionality using MLIR compilation pipelines.

## Test Structure

### MLIR Compilation Tests
- **`CompileAndExecuteRealMlir`**: Tests compilation of `test_simple.mlir` (ReLU operation)
- **`CompileAndExecuteVectorAdd`**: Tests compilation of `test.mlir` (vector addition)
- **`CompileAndExecuteMultipleOperations`**: Tests compilation of `test_complex.mlir` (Simple module)

These tests use the following compiler pipeline:
1. `ttmlir-opt --convert-ttir-to-nvvm` - Converts TTIR to NVVM dialect
2. `ttmlir-translate --ptx-to-flatbuffer` - Converts NVVM to CUDA flatbuffer

### Mock Tests for Edge Cases
- **`ExecuteEmptyProgram`**: Tests handling of empty programs
- **`HandleEmptyFlatbuffer`**: Tests handling of empty flatbuffer
- **`HandleInvalidInput`**: Tests handling of invalid input arguments

## MLIR Test Files

### `test.mlir`
```mlir
module attributes {} {
  func.func @test_add(%arg0: tensor<5x2x2x2xf32>, %arg1: tensor<5x2x2x2xf32>) -> tensor<5x2x2x2xf32> {
    %0 = ttir.empty() : tensor<5x2x2x2xf32>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<5x2x2x2xf32>, tensor<5x2x2x2xf32>, tensor<5x2x2x2xf32>) -> tensor<5x2x2x2xf32>
    return %1 : tensor<5x2x2x2xf32>
  }
}
```

### `test_simple.mlir`
```mlir
module attributes {} {
  func.func @test_simple(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %0 = ttir.empty() : tensor<2xf32>
    %1 = "ttir.relu"(%arg0, %0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }
}
```

### `test_complex.mlir`
```mlir
module @MNISTLinear attributes {} {
  func.func @forward(%arg0: tensor<1x784xf32> {ttir.name = "input_1"}, %arg1: tensor<784x512xf32> {ttir.name = "linear_relu_stack.0.weight"}, %arg2: tensor<512xf32> {ttir.name = "linear_relu_stack.0.bias"}, %arg3: tensor<512x512xf32> {ttir.name = "linear_relu_stack.2.weight"}, %arg4: tensor<512xf32> {ttir.name = "linear_relu_stack.2.bias"}, %arg5: tensor<512x10xf32> {ttir.name = "linear_relu_stack.4.weight"}, %arg6: tensor<10xf32> {ttir.name = "linear_relu_stack.4.bias"}) -> (tensor<1x10xf32> {ttir.name = "MNISTLinear_350.output_add_981"}) {
    %0 = ttir.empty() : tensor<1x512xf32>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<1x784xf32>, tensor<784x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
    %2 = ttir.empty() : tensor<1x512xf32>
    %3 = "ttir.add"(%1, %arg2, %2) : (tensor<1x512xf32>, tensor<512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
    %4 = ttir.empty() : tensor<1x512xf32>
    %5 = "ttir.relu"(%3, %4) : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
    %6 = ttir.empty() : tensor<1x512xf32>
    %7 = "ttir.matmul"(%5, %arg3, %6) : (tensor<1x512xf32>, tensor<512x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
    %8 = ttir.empty() : tensor<1x512xf32>
    %9 = "ttir.add"(%7, %arg4, %8) : (tensor<1x512xf32>, tensor<512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
    %10 = ttir.empty() : tensor<1x512xf32>
    %11 = "ttir.relu"(%9, %10) : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
    %12 = ttir.empty() : tensor<1x10xf32>
    %13 = "ttir.matmul"(%11, %arg5, %12) : (tensor<1x512xf32>, tensor<512x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %14 = ttir.empty() : tensor<1x10xf32>
    %15 = "ttir.add"(%13, %arg6, %14) : (tensor<1x10xf32>, tensor<10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    return %15 : tensor<1x10xf32>
  }
}
```

## Running the Tests

### Prerequisites
1. Build tt-mlir with CUDA support (TT_RUNTIME_ENABLE_CUDA)
2. Ensure CUDA drivers and runtime are available
3. Build the tests

### Execution
```bash
# Run all tests
./build/runtime/test/cuda/gtest/test_cuda_program_executor

# Tests will automatically skip if:
# - CUDA hardware is not available
# - MLIR compilation fails (missing tools, etc.)
```

## Test Features

### Automatic CUDA Detection
Tests automatically detect if CUDA is available and skip gracefully if not.

### Compiler Integration
Tests use the `ttmlir-opt` and `ttmlir-translate` tools to create test scenarios.

### Comprehensive Error Handling
Tests validate that the program executor handles various error conditions gracefully:
- Empty programs
- Empty input buffer
- Invalid input arguments
