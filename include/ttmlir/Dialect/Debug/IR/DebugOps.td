// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_DEBUG_DEBUGOPS_TD
#define TTMLIR_TTMLIR_DIALECT_DEBUG_DEBUGOPS_TD

include "ttmlir/Dialect/Debug/IR/DebugBase.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

//===----------------------------------------------------------------------===//
// Debug ops
//===----------------------------------------------------------------------===//

class Debug_BaseOp<string mnemonic, list<Trait> traits = []> :
    Debug_Op<mnemonic, [Pure, SameOperandsAndResultType]> {

    let summary = "Base class for debug op.";
    let description = [{
      Ensures that the operations that produce the `operand` are executed before any
      operations that depend on the `result` and prevents compiler transformations
      from moving operations across the barrier. Other than that, the operation is
      an identity, i.e. `result` = `operand`.
    }];
}

def Debug_AnnotateOp : Debug_BaseOp<"annotate", []> {
  let summary = "Annotate operation";
  let description = [{
    Attaches a user-defined debug annotation to a value without changing its semantics or runtime behavior.

    Example:
    ```mlir
    %1 = "debug.annotate"(%0) <{annotation = "This is from KV cache"}> : (tensor<32x32xf32>) -> tensor<32x32xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$operand,
    StrAttr:$annotation
  );

  let results = (outs
    AnyRankedTensor:$result
  );
}

def Debug_BreakpointOp : Op<Debug_Dialect, "breakpoint", [MemoryEffects<[MemWrite]>]> {
  let summary = "Breakpoint operation";
  let description = [{
    Inserts a breakpoint in runtime execution.

    Example:
    ```mlir
    "debug.breakpoint"(%0) : (tensor<32x32xf32>) -> ()
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$operand
  );
}

def Debug_PrintOp : Op<Debug_Dialect, "print", [MemoryEffects<[MemWrite]>]> {
  let summary = "Print operation";
  let description = [{
    Print a message during runtime.

    Example:
    ```mlir
    %1 = "debug.print"(%0) <{message = "This is from KV cache"}> : (tensor<32x32xf32>) -> tensor<32x32xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$operand,
    StrAttr:$message
  );
}

def Debug_DumpOp : Op<Debug_Dialect, "dump", [MemoryEffects<[MemWrite]>]> {
  let summary = "Dump operation";
  let description = [{
    Save a tensor to disk.

    Example:
    ```mlir
    %1 = "debug.dump"(%0) <{file_path = "system_tensor.tensorbin"}> : (tensor<32x32xf32>) -> tensor<32x32xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$operand,
    StrAttr:$file_path
  );
}

def Debug_MemorySnapshotOp : Op<Debug_Dialect, "memory_snapshot", [MemoryEffects<[MemWrite]>]> {
  let summary = "Memory snapshot operation";
  let description = [{
    Capture memory state of device.

    Example:
    ```mlir
    %1 = "debug.memory_snapshot"(%0) <{file_path = "/path/to/memory/file.txt"}> : (tensor<32x32xf32>) -> tensor<32x32xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$operand,
    StrAttr:$file_path
  );
}

def Debug_RegionStartOp : Debug_Op<"region_start",
  [Pure, SameOperandsAndResultType]> {
  let summary = "Region start operation";
  let description = [{
    Mark the start of a region.

    Example:
    ```mlir
    %0 = debug.region_start %arg0 <{region_id = "region0"}> : tensor<32x32xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$operand,
    StrAttr:$region_id
  );

  let results = (outs
    AnyRankedTensor:$result
  );
}

def Debug_RegionEndOp : Debug_Op<"region_end",
  [Pure, SameOperandsAndResultType]> {
  let summary = "Region end operation";
  let description = [{
    Mark the end of a region.

    Example:
    ```mlir
    %1 = debug.region_end %0 <{region_id = "region0"}> : tensor<32x32xf32>, tensor<32x32xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$operand,
    StrAttr:$region_id
  );

  let results = (outs
    AnyRankedTensor:$result
  );
}

def Debug_OverwriteOp : Debug_BaseOp<"overwrite",
  [Pure, SameOperandsAndResultType]> {
  let summary = "Overwrite operation";
  let description = [{
    Overwrite a tensor.

    Example:
    ```mlir
    %1 = debug.overwrite %0 %cst : tensor<32x32xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$dst,
    AnyRankedTensor:$src
  );

  let results = (outs
    AnyRankedTensor:$result
  );
}

def Debug_SnapshotOp : Debug_BaseOp<"snapshot",
  [Pure, SameOperandsAndResultType]> {
  let summary = "Snapshot operation";
  let description = [{
    Snapshot device state.

    Example:
    ```mlir
    %1 = "debug.snapshot"(%0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$operand
  );

  let results = (outs
    AnyRankedTensor:$result
  );
}

def Debug_BeginTraceOp : Debug_Op<"begin_trace",
  [Pure, SameOperandsAndResultType]> {
  let summary = "Begin trace operation";
  let description = [{
    Begin a graph trace.

    Example:
    ```mlir
    %0 = debug.begin_trace %arg0 <{trace_id = "trace0"}> : tensor<32x32xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$operand,
    StrAttr:$trace_id
  );

  let results = (outs
    AnyRankedTensor:$result
  );
}

def Debug_EndTraceOp : Debug_Op<"end_trace",
  [Pure, SameOperandsAndResultType]> {
  let summary = "End trace operation";
  let description = [{
    End a graph trace.

    Example:
    ```mlir
    %1 = debug.end_trace %0 <{trace_id = "trace0"}> : tensor<32x32xf32>, tensor<32x32xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$operand,
    StrAttr:$trace_id
  );

  let results = (outs
    AnyRankedTensor:$result
  );
}

def Debug_ProfilerStartOp : Debug_Op<"profiler_start",
  [Pure, SameOperandsAndResultType]> {
  let summary = "Profiler start operation";
  let description = [{
    Start profiler.

    Example:
    ```mlir
    %0 = debug.profiler_start %arg0 <{scope_id = "scope0"}> : tensor<32x32xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$operand,
    StrAttr:$scope_id
  );

  let results = (outs
    AnyRankedTensor:$result
  );
}

def Debug_ProfilerEndOp : Debug_Op<"profiler_end",
  [Pure, SameOperandsAndResultType]> {
  let summary = "Profiler end operation";
  let description = [{
    End profiler.

    Example:
    ```mlir
    %1 = debug.profiler_end %0 <{scope_id = "scope0"}> : tensor<32x32xf32>, tensor<32x32xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$operand,
    StrAttr:$scope_id
  );

  let results = (outs
    AnyRankedTensor:$result
  );
}

def Debug_TimestampOp : Debug_Op<"timestamp",
  [Pure, SameOperandsAndResultType]> {
  let summary = "Timestamp operation";
  let description = [{
    Capture current clock time.

    Example:
    ```mlir
    %1 = "debug.timestamp"(%0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$operand
  );

  let results = (outs
    AnyRankedTensor:$result
  );
}

def Debug_HookOp : Debug_Op<"hook",
  [Pure, SameOperandsAndResultType]> {
  let summary = "Hook operation";
  let description = [{
    Insert runtime hook.

    Example:
    ```mlir
    %1 = "debug.hook"(%0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$operand
  );

  let results = (outs
    AnyRankedTensor:$result
  );
}

#endif
