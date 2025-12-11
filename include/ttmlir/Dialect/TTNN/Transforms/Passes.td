// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TRANSFORMS_TTNNPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TRANSFORMS_TTNNPASSES_TD

include "mlir/Pass/PassBase.td"

def TTNNDeallocate: Pass<"ttnn-deallocate", "::mlir::ModuleOp"> {
  let summary = "Insert deallocate ops for tensors.";
  let description = [{
    This pass inserts deallocate ops after a tensor value's last use.
  }];
}

def TTNNDecomposeLayouts: Pass<"ttnn-decompose-layouts", "::mlir::ModuleOp"> {
  let summary = "Decompose ToLayoutOps to more granular memory ops.";
  let description = [{
    This pass decomposes ToLayoutOps to memory ops (e.g. toDevice, toMemoryConfig etc.).
  }];
}

def TTNNLayout : Pass<"ttnn-layout", "::mlir::ModuleOp"> {
  let summary = "Add layout information to tensors.";
  let description = [{
    This pass adds layout information to tensors.
  }];
}

def TTNNWorkarounds : Pass<"ttnn-workaround", "::mlir::ModuleOp"> {
  let summary = "Apply TTNN workarounds to the IR.";
  let description = [{
    This pass applies necessary TTNN workarounds to the IR in order to create
    a valid and functional IR that can be executed on the hardware.
  }];

  let options = [
      Option<"layoutWorkaroundsEnabled",
             "ttnn-enable-layout-workaround-pass",
             "bool", /*default=*/"true",
             "TTNN Layout Workarounds Pass">,
      Option<"decompositionWorkaroundsEnabled",
             "ttnn-enable-decomposition-workaround-pass",
             "bool", /*default=*/"true",
             "TTNN Decompsition Workarounds Pass">,
      Option<"optimizerEnabled",
             "ttnn-is-optimizer-enabled",
             "bool", /*default=*/"false",
             "flag specifying if optimizer is enabled">,
  ];
}

def TTNNCreateInputGenerators: Pass<"ttnn-create-input-gens", "::mlir::ModuleOp"> {
  let summary = "Create input generators for the forward functions.";
  let description = [{
    This pass creates input generators for the "forward" functions. It
    additionally creates a main function to run the forward function with the
    generated inputs.

    The pass is useful for EmitC path. By creating input generators before
    converting to Emitc Dialect, followed by transformation to C++ code, the
    resulting code won't require any edits to run.

    Given a forward function like this:

    ```mlir
    func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
      %0 = "ttnn.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      return %0 : tensor<32x32xbf16>
    }
    ```

    The pass will prepend `_` to the existing function name to avoid name collision
    and create two function like this:

    ```mlir
    func.func @create_inputs_for_add() -> (tensor<32x32xbf16>, tensor<32x32xbf16>) {
      %0 = "ttnn.empty"() <{shape = #ttnn.shape<32x32>}> : () -> tensor<32x32xbf16>
      %1 = "ttnn.empty"() <{shape = #ttnn.shape<32x32>}> : () -> tensor<32x32xbf16>
      return %0, %1 : tensor<32x32xbf16>, tensor<32x32xbf16>
    }

    func.func @main() -> i32 {
      %0:2 = call @create_inputs_for_add() : () -> (tensor<32x32xbf16>, tensor<32x32xbf16>)
      %1 = call @_add(%0#0, %0#1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %c0_i32 = arith.constant 0 : i32
      return %c0_i32 : i32
    }
    ```
  }];
}

def TTNNLoadInputTensors: Pass<"ttnn-load-input-tensors", "::mlir::ModuleOp"> {
  let summary = "Load input tensors from disk for the forward functions.";
  let description = [{
    This pass loads input tensors from disk using ttnn.load_tensor operations
    for the "forward" functions. It additionally creates a main function to run
    the forward function with the loaded inputs.

    This is an alternative to TTNNCreateInputGenerators that loads real tensor
    data from disk instead of generating synthetic inputs with ttnn.ones.

    Given a forward function like this:

    ```mlir
    func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
      %0 = "ttnn.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      return %0 : tensor<32x32xbf16>
    }
    ```

    The pass will prepend `_` to the existing function name to avoid name collision
    and create functions that load inputs from arg0.tensorbin, arg1.tensorbin, etc:

    ```mlir
    func.func @load_inputs_for_add() -> (tensor<32x32xbf16>, tensor<32x32xbf16>) {
      %0 = "ttnn.load_tensor"() <{file_path = "arg0.tensorbin"}> : () -> tensor<32x32xbf16>
      %1 = "ttnn.load_tensor"() <{file_path = "arg1.tensorbin"}> : () -> tensor<32x32xbf16>
      return %0, %1 : tensor<32x32xbf16>, tensor<32x32xbf16>
    }

    func.func @main() -> i32 {
      %0:2 = call @load_inputs_for_add() : () -> (tensor<32x32xbf16>, tensor<32x32xbf16>)
      %1 = call @_add(%0#0, %0#1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %c0_i32 = arith.constant 0 : i32
      return %c0_i32 : i32
    }
    ```
  }];
  let options = [
    Option<"tensorLoadDirectory", "tensor-load-directory", "std::string", /*default=*/"\"\"",
           "Directory path where input tensors are stored">,
    Option<"tensorLoadFilePrefix", "tensor-load-file-prefix", "std::string", /*default=*/"\"arg\"",
           "Prefix for input tensor files">,
  ];
}

def TTNNTuplifyTensors: Pass<"ttnn-tuplify-tensors", "::mlir::ModuleOp"> {
  let summary = "Tuplify tensors in all funcs within a module.";
  let description = [{
    This pass identifies all public functions in the module that exclusively use tensors as inputs and outputs. It rewrites
    each such function by packing all input arguments into a single tuple and wrapping the return value(s) into a tuple
    type as well.

    Given a forward function like this:

    ```mlir
    func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
      %0 = "ttnn.get_device"() : () -> !ttnn.device
      %1 = "ttnn.to_device"(%arg0, %0) : (tensor<32x32xbf16, !ttnn.device) -> tensor<32x32xbf16>
      %2 = "ttnn.to_device"(%arg1, %0) : (tensor<32x32xbf16, !ttnn.device) -> tensor<32x32xbf16>
      %3 = "ttnn.add"(%1, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      return %3 : tensor<32x32xbf16>
    }
    ```

    The pass will return:

    ```mlir
    func.func @add(%arg0: tuple<tensor<32x32xbf16>, tensor<32x32xbf16>>, %arg1: !ttnn.device) -> tuple<tensor<32x32xbf16>> {
      %0 = ttcore.get_tuple_element %arg0[0] : (tuple<tensor<32x32xbf16>, tensor<32x32xbf16>>) -> tensor<32x32xbf16>
      %1 = ttcore.get_tuple_element %arg0[1] : (tuple<tensor<32x32xbf16>, tensor<32x32xbf16>>) -> tensor<32x32xbf16>
      %2 = "ttnn.to_device"(%0, %arg1) : (tensor<32x32xbf16, !ttnn.device) -> tensor<32x32xbf16>
      %3 = "ttnn.to_device"(%1, %arg1) : (tensor<32x32xbf16, !ttnn.device) -> tensor<32x32xbf16>
      %4 = "ttnn.add"(%2, %3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %5 = ttcore.tuple %4 : tuple<tensor<32x32xbf16>>
      return %5 : tuple<tensor<32x32xbf16>>
    }
    ```
  }];

  let options = [
    Option<"tuplifyInputIfEmpty",
            "tuplify-input-if-empty",
            "bool", /*default=*/"false",
            "If input arg is originally empty, force create an empty tuple arg. This is useful for scenarios where "
            "signatures are required have tuples on the input, like the EmitC dylib path.">
  ];
}

def TTNNPrepareConv2dWeightsAndBias : Pass<"ttnn-prepare-conv2d-weights-and-bias", "::mlir::ModuleOp"> {
  let summary = "Optimize Conv2d ops by inserting PrepareConv2dWeights and PrepareConv2dBias ops before them.";
  let description = [{
    This pass inserts a PrepareConv2dWeights and a PrepareConv2dBias operation before each Conv2d op, which preprocess the weights and bias used by the Conv2d operations.
    These operations can then be const-evaluated, leading to improved performance.
    To use this pass, the project must be built with the op model library by setting -DTTMLIR_ENABLE_OPMODEL=ON during the build process.
  }];
}

def TTNNFusing: Pass<"ttnn-fusing", "::mlir::ModuleOp">
{
  let summary = "TTNN fusing pass.";
  let description = "This pass tries to fuse operations together with goal to reduce the number of operations in the graph.";

  let options = [
    Option<"enableOpConstraints",
            "enable-op-constraints",
            "bool", /*default=*/"false",
            "When enabled fusing pass includes patterns which check for op constraints.">
  ];
}

def TTNNTraceHoistTransform : Pass<"ttnn-trace-hoist-transform", "::mlir::ModuleOp"> {
  let summary = "TTNN Trace hoist transform";
  let description = [{
    Transform pass which runs an analysis pass to find traceable subgraphs and hoist them into separate trace functions.
    The ops within the trace functions are then captured and replayed during runtime using the metal trace infra.
    This is a performance optimization feature that enables batches of ops to be replayed with a single command instead
    of dispatching commands for each op individually. One hard requirement is that all ops within a trace must be device
    operations and cannot have any host involvement. This will get verified within the verifier of the `ttnn.capture_or_execute_trace`
    operation.

    Example:
    ```mlir
    func.func @single_add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x32xbf16> {
      %0 = ttir.empty() : tensor<32x32xbf16>
      %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      return %1 : tensor<32x32xbf16>
    }
    ```

    the pass will rewrite the IR, hoist the operations and generate the following functions:

    ```mlir
    // Trace function. Contains the ops that were hoisted to trace.
    func.func @trace_0_single_add(%arg0: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout> attributes {ttnn.trace} {
      %0 = "ttnn.add"(%arg0, %arg1) <{output_dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
      return %0 : tensor<32x32xbf16, #ttnn_layout>
    }

    // Capture trace function. Copies inputs into pre-allocated empty tensors on device.
    // Notice that constants/params will not be copied into pre-allocated tensors - they
    // will be used directly instead.
    // Then, runs the trace function once to gather the outputs and warm up the device.
    // Finally, runs the trace function again to capture the trace.
    // Returns the trace id, function outputs, and the trace input output tensors.
    func.func @run_and_capture_trace_0_single_add(%arg0: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> (tensor<ui32, #ttnn_layout1>, tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) attributes {ttnn.trace} {
      %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
      %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout>
      %2 = "ttnn.from_device"(%arg0) : (tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout2>
      "ttnn.write_tensor"(%2, %1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #ttnn_layout2>, tensor<32x32xbf16, #ttnn_layout>) -> ()
      "ttnn.deallocate"(%2) <{force = false}> : (tensor<32x32xbf16, #ttnn_layout2>) -> ()
      %3 = call @trace_0_single_add(%1, %arg1) : (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
      %4 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn_layout1>
      %5 = call @trace_0_single_add(%1, %arg1) : (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
      "ttnn.end_trace_capture"(%0, %4) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn_layout1>) -> ()
      return %4, %3, %1, %arg1, %5 : tensor<ui32, #ttnn_layout1>, tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>
    }
    ```

    These functions will get called conditionally within `ttnn.capture_or_execute_trace`:

    ```mlir
    func.func @single_add(%arg0: tensor<32x32xbf16, #ttnn_layout>, %arg1: tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout> {
      %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
      %1 = "ttnn.capture_or_execute_trace"(%0, %arg0, %arg1) <{capture_callee = @run_and_capture_trace_0_single_add, execute_callee = @execute_trace_0_single_add}> : (!ttnn.device, tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
      return %1 : tensor<32x32xbf16, #ttnn_layout>
    }
    ```

  }];
}

def TTNNUniqueLocations: Pass<"ttnn-unique-locations", "::mlir::ModuleOp">
{
  let summary = "Check if operations have unique locations.";
  let description = "This pass checks if all operations relevant to the optimizer have unique locations. If not, it will emit an error. This is necessary for the overrides to be applied correctly.";
}

def TTNNOperationValidationAndFallback: Pass<"ttnn-operation-validation-and-fallback", "::mlir::ModuleOp"> {
  let summary = "Validate operations and apply fallback strategies when needed.";
  let description = [{
    This pass validates TTNN operations by checking if their configurations
    will work at runtime. For operations that fail validation, it applies
    fallback strategies by transforming input operand layouts and data types
    to find working configurations.
  }];
  let options = [
    Option<"tensorL1UsageCap",
           "tensor-l1-usage-cap",
           "float", /*default=*/"1.0",
           "Limit L1 memory usage to this fraction of available space (0.0-1.0)">
  ];
}

def TTNNEmitPyWorkarounds: Pass<"ttnn-emitpy-workarounds", "::mlir::ModuleOp"> {
  let summary = "Apply TTNN workarounds specific to EmitPy pipeline.";
  let description = [{
    This pass applies TTNN workarounds that are specific to the EmitPy pipeline.
  }];
}

def TTNNAdjustDeallocs: Pass<"ttnn-adjust-deallocs", "::mlir::ModuleOp"> {
  let summary = "Adjust previously inserted tensor deallocation ops.";
  let description = [{
    This pass will adjust previously inserted tensor deallocation ops. It will
    remove deallocations of parameter and constant tensors.
  }];
}

def TTNNWeightBFP8Conversion: Pass<"ttnn-weight-bfp8-conversion", "::mlir::ModuleOp"> {
  let summary = "Convert weight tensors to BFP8 format in matmul and linear operations.";
  let description = [{
    This pass converts weight tensors in matrix multiplication and linear
    operations to BFP8 format while keeping activations and outputs in
    high precision (bf16/f32).

    For matmul/linear operations:
      - Inserts ttnn.typecast operations to convert weights to BFP8
      - Only converts weights that trace back to constant/parameter arguments
      - Keeps activation inputs and outputs in their original precision
  }];
}

def TTNNCollectPerfMetrics: Pass<"ttnn-collect-perf-metrics", "::mlir::ModuleOp"> {
  let summary = "Collect and export TTNN operation metrics.";
  let description = [{
    This pass collects metrics about TTNN operations and their memory placement,
    including information about sharded layouts, DRAM spills, and system memory usage.
    The metrics are serialized to JSON format for debugging, profiling, and external analysis.

    Usage:
    Enable performance metrics collection in the TTIR to TTNN backend pipeline using:
    --ttir-to-ttnn-backend-pipeline="ttnn-perf-metrics-enabled=true"

    Output file naming:
    - If --ttnn-perf-metrics-output-file is not specified or is the default, files are automatically
      named based on the module/function name and stored in perf_metrics/ directory
    - Example: module @MyModel produces perf_metrics/MyModel_perf_metrics.json
    - If custom output file is specified, that path is used directly

    Metrics collected:
    - total_ops: Total TTNN operations (excluding const-eval ops)
    - total_ops_with_output_tensor: Total TTNN operations that produce output tensors
    - total_shardable_ops: Total TTNN operations that can use sharded layouts
    - sharded_ops: Operations using sharded layouts (height/width/block sharded)
    - effectively_sharded_ops: Sharded ops that are NOT spilled to DRAM
    - sharded_and_spilled_ops: Sharded ops whose output is spilled to DRAM
    - dram_spilled_ops: Total operations spilled to DRAM (interleaved)
    - system_memory_ops: Operations with system_memory buffer type
    - Percentages: All counts as percentage of total_shardable_ops

    Verbose output (--ttnn-perf-metrics-verbose-output-enabled) includes:
    - Detailed per-operation analysis with sharding and memory layout information for shardable ops
    - Complete operation type breakdown with counts for all operation types (excluding const-eval functions)
  }];

  let options = [
    Option<"ttnnPerfMetricsOutputFile", "ttnn-perf-metrics-output-file", "std::string", /*default=*/"\"ttnn_perf_metrics.json\"",
           "Output file path for the performance metrics JSON">,
    Option<"ttnnPerfMetricsVerboseOutputEnabled", "ttnn-perf-metrics-verbose-output-enabled", "bool", /*default=*/"false",
           "Enable verbose output with per-operation details and operation type breakdown">,
    Option<"ttnnEnableTrace", "ttnn-enable-trace", "bool", /*default=*/"false",
           "Collect metrics only for the main traced function">
  ];
}

def TTNNIsolateOps: Pass<"ttnn-isolate-ops", "::mlir::ModuleOp"> {
  let summary = "Isolate TTNN operations into minimal separate functions.";
  let description = [{
    This pass isolates individual TTNN operations by extracting each target
    operation into its own minimal function. Each isolated function contains:
    - Only the target operation itself
    - Function arguments with the final tensor types (after any preprocessing)
    - A return operation for the result

    The pass does NOT include preprocessing operations (like reshape, to_layout,
    from_device, etc.) in the isolated functions. Instead, the function signature
    captures the tensor types after preprocessing has been applied.

    This transformation is useful for:
    - Generating parametrized unit tests for individual operations
    - Extracting operation configurations for test generation
    - Creating minimal reproducers for debugging
    - Analyzing individual operation behavior in isolation

    Example:
    ```mlir
    func.func @model(%arg0: tensor<32x32x32x64xbf16>, ...) {
      %0 = ttnn.get_device() : () -> !ttnn.device
      %1 = ttnn.reshape(%arg0) : (...) -> tensor<1x1x32768x64xbf16>
      %2 = ttnn.to_layout(%arg2) : (...) -> tensor<1x1x1x64xbf16>
      %3 = ttnn.from_device(%2) : (...) -> tensor<1x1x1x64xbf16>
      %4 = ttnn.to_layout(%1) : (...) -> tensor<1x1x32768x64xbf16>
      %5 = ttnn.conv2d(%4, %arg1, %3, %0) : (...) -> tensor<1x1x28800x64xbf16>
      %6 = ttnn.reshape(%5) : (...) -> tensor<32x30x30x64xbf16>
      return %6
    }
    ```

    After the pass with `--filter-ops="conv2d"`:
    ```mlir
    func.func private @conv2d_0(
      %arg0: tensor<1x1x32768x64xbf16>,  // Final type after preprocessing
      %arg1: tensor<64x1x3x3xbf16>,
      %arg2: tensor<1x1x1x64xbf16>,
      %arg3: !ttnn.device
    ) -> tensor<1x1x28800x64xbf16> {
      %0 = ttnn.conv2d(%arg0, %arg1, %arg2, %arg3) <{...}> : (...) -> (...)
      return %0
    }
    ```

    The isolated function arguments have the exact types needed by the operation,
    making it easy to parametrize tests from multiple operation instances.
  }];

  let options = [
    ListOption<"filterOps", "filter-ops", "std::string",
               "List of operation names to isolate (e.g., conv2d, matmul). "
               "If empty, isolates all non-utility operations.">,
    Option<"preserveOriginal", "preserve-original", "bool", /*default=*/"false",
           "Keep the original function in addition to isolated functions">
  ];
}

def TTNNAnnotateIsolatedInputs: Pass<"ttnn-annotate-isolated-inputs", "::mlir::ModuleOp"> {
  let summary = "Annotate isolated functions with input operand metadata for test generation.";
  let description = [{
    This pass annotates isolated functions (marked with the `isolated` attribute)
    with metadata about their input operands. This information can be used to
    generate parametrized Python tests that automatically create appropriate
    input tensors.

    For each function with the `isolated` attribute, the pass adds attributes:
    - `input_shapes`: Array of tensor shapes for each input operand
    - `input_dtypes`: Array of data types for each input operand
    - `input_layouts`: Array of layout information for each tensor operand
    - `op_name`: Name of the operation being isolated

    Example (input from TTNNIsolateOps pass):
    ```mlir
    func.func private @conv2d_0(%arg0: tensor<1x1x784x1xbf16, #layout1>,
                                 %arg1: tensor<16x1x3x3xbf16, #layout2>,
                                 %arg2: tensor<1x1x1x16xbf16, #layout3>)
        -> tensor<1x1x784x16xbf16, #layout4> attributes {isolated} {
      %0 = "ttnn.get_device"() : () -> !ttnn.device
      %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0) <{...}> : (...) -> (...)
      return %1
    }
    ```

    After the pass:
    ```mlir
    func.func private @conv2d_0(%arg0: tensor<1x1x784x1xbf16, #layout1>,
                                 %arg1: tensor<16x1x3x3xbf16, #layout2>,
                                 %arg2: tensor<1x1x1x16xbf16, #layout3>)
        -> tensor<1x1x784x16xbf16, #layout4>
        attributes {
          isolated,
          op_name = "ttnn.conv2d",
          input_shapes = [[1, 1, 784, 1], [16, 1, 3, 3], [1, 1, 1, 16]],
          input_dtypes = ["bf16", "bf16", "bf16"],
          input_layouts = [#layout1, #layout2, #layout3]
        } {
      %0 = "ttnn.get_device"() : () -> !ttnn.device
      %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0) <{...}> : (...) -> (...)
      return %1
    }
    ```

    This metadata enables Python test generators to create appropriate input
    tensors automatically.
  }];

  let options = [];
}

def TTNNExtractIsolatedToTestModule: Pass<"ttnn-extract-isolated-to-test-module", "::mlir::ModuleOp"> {
  let summary = "Extract isolated functions to a separate test module.";
  let description = [{
    This pass extracts all functions marked with the `isolated` attribute
    into a separate nested module for test generation. This enables:
    - Clear separation between production code and test code
    - Independent processing of test functions for pytest generation
    - Eventual emission to separate Python files

    The pass creates a new nested `ModuleOp` named "test_module" and moves
    all isolated functions from the original module into it. The isolated
    functions are removed from the original module.

    Example:
    ```mlir
    module {
      func.func @main(...) {...}
      func.func @reshape_0(...) attributes {isolated} {...}
      func.func @reshape_1(...) attributes {isolated} {...}
    }
    ```

    After the pass:
    ```mlir
    module {
      func.func @main(...) {...}
      module @test_module {
        func.func @reshape_0(...) attributes {isolated} {...}
        func.func @reshape_1(...) attributes {isolated} {...}
      }
    }
    ```
  }];

  let options = [];
}

def TTNNPrepareIsolatedForPytest: Pass<"ttnn-prepare-isolated-for-pytest", "::mlir::ModuleOp"> {
  let summary = "Transform isolated functions into parametrized pytest tests.";
  let description = [{
    This pass transforms isolated TTNN operations into parametrized pytest
    test functions. It groups isolated functions by operation type, identifies
    varying attributes, and merges functions with the same operation into a
    single parametrized test.

    For each group of functions with the same `op_name`:
    1. Identify varying attributes across functions (input_shapes, input_dtypes, call_opaque inline attrs)
    2. Create a merged test function named `test_{op_name}`
    3. Add function arguments for varying attributes
    4. Generate `@pytest.mark.parametrize` decorators
    5. Convert call_opaque inline attributes to function operands
    6. Delete original isolated functions

    Example transformation:

    Before:
    ```mlir
    func.func @reshape_5(%arg0: tensor) attributes {
      isolated, op_name = "ttnn.reshape",
      input_shapes = [[1,1,784,1]], input_dtypes = ["bf16"]
    } {
      %0 = emitpy.call_opaque "ttnn.reshape"(%arg0)
             {args = [0, #emitpy.opaque<"[1,28,28,1]">, ...]}
      return %0
    }
    func.func @reshape_6(%arg0: tensor) attributes {
      isolated, op_name = "ttnn.reshape",
      input_shapes = [[2,2,784,1]], input_dtypes = ["bf16"]
    } {
      %0 = emitpy.call_opaque "ttnn.reshape"(%arg0)
             {args = [0, #emitpy.opaque<"[2,32,32,1]">, ...]}
      return %0
    }
    ```

    After:
    ```mlir
    func.func @test_reshape(%input: tensor, %shape: opaque, %input_shape: opaque) attributes {
      emitpy.decorators = [
        #emitpy.decorator<"@pytest.mark.parametrize('shape', [[1,28,28,1], [2,32,32,1]])">,
        #emitpy.decorator<"@pytest.mark.parametrize('input_shape', [(1,1,784,1), (2,2,784,1)])">,
        #emitpy.decorator<"@pytest.mark.parametrize('dtype', ['bf16'])">,
      ]
    } {
      %0 = emitpy.call_opaque "ttnn.reshape"(%input, %shape) {args = [0, 1], ...}
      return %0
    }
    ```

    This generates Python code:
    ```python
    @pytest.mark.parametrize('shape', [[1,28,28,1], [2,32,32,1]])
    @pytest.mark.parametrize('input_shape', [(1,1,784,1), (2,2,784,1)])
    @pytest.mark.parametrize('dtype', ['bf16'])
    def test_reshape(input, shape, input_shape):
        result = ttnn.reshape(input, shape, ...)
        return result
    ```
  }];

  let options = [];
}

#endif
