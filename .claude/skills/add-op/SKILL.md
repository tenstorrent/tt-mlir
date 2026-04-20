---
name: add-op
description: >
  How to add a new operation (op) to the tt-mlir compiler across all layers: TTIR/TTNN dialect
  definitions, StableHLO composite conversion, TTIR-to-TTNN conversion, EmitC/EmitPy conversions,
  flatbuffer schema and serialization, runtime implementation, OpModel, ttir_builder, golden
  functions, and all associated tests. Use this skill whenever the user asks to add an op, implement
  an op, create a new operation, add support for a TTNN op, or mentions adding an op to the compiler
  pipeline. Also trigger when the user wants to know what files to change for a new op, or asks about
  the op-adding workflow.
---

# Adding an Op to tt-mlir

Adding a new op touches ~15-30 files across the compiler, runtime, and test infrastructure. This
skill walks through each layer, starting from the TTNN device-level API and working upward. Use
the existing ops in each file as your primary reference — find the most similar op and follow its
pattern.

The implementation order is:

```
1. TTNN dialect definition (models the TTNN C++ API)
2. TTIR dialect definition (higher-level, device-agnostic)
3. TTIR → TTNN conversion
4. TTNN → EmitC / EmitPy conversions
5. StableHLO composite → TTIR conversion (if needed)
6. Flatbuffer schema and serialization
7. Runtime implementation
8. OpModel
9. TTIRBuilder, golden functions, tests
10. CPU-hoisted version (if applicable)
```

The key principle: **start from the TTNN C++ API** and work outward. The TTNN dialect op should
model the TTNN library function as closely as possible. The TTIR op is a simplified, device-agnostic
version derived from it.

Consult `references/ttnn_type_mapping.md` for the mapping between TTNN C++ types and their MLIR
tablegen equivalents. This covers tensor types, scalar attributes, TTNN-specific types (MemoryConfig,
DeviceComputeKernelConfig, Topology, etc.), and which C++ types have no MLIR equivalent and are
handled at runtime only.

Before starting, identify:
- **The TTNN C++ API** you're targeting — read the actual function signature in
  `third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/`. This is the source of truth for
  what parameters the op takes, their types, and which are optional.
- **A similar existing op** to use as a template (e.g., `RMSNormOp` for normalization ops,
  `MatmulOp` for ops with multiple tensor inputs, `ReduceScatterOp` for CCL ops)
- **Which arguments are optional** — this affects whether you need `AttrSizedOperandSegments`
- **Naming conflicts** — check if an op with the same name already exists (e.g., StableHLO `GatherOp`
  already exists, so torch.gather semantics was named `GatherDimOp` in TTIR). Search the existing
  tablegen definitions before choosing a name.
- **Data type requirements** — some TTNN metal ops require specific types (e.g., `ttnn::gather`
  requires UINT32/UINT16 index tensors, not INT32). Check the metal API docs or headers.
- **Tensor shape conventions** — some metal kernels expect specific tensor layouts (e.g., SDPA
  decode expects Q in `(S, B, H, D)` format, not `(B, H, S, D)`). Search
  `third_party/tt-metal/src/tt-metal/tests/` for existing unit tests of the TTNN op to find the
  exact tensor shapes, dtypes, and any required permutations. These tests are the ground truth for
  what shapes the metal kernel actually supports.

## Step 1: Define the Op in TTNN

### 1a. Tablegen definition

**File:** `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td`

The TTNN op should model the TTNN C++ API as closely as possible. Read the actual C++ function
signature and map each parameter to its MLIR equivalent using `references/ttnn_type_mapping.md`.

Refer to `references/ttnn_type_mapping.md` for the complete mapping of C++ types to tablegen
types, op interfaces, optional/default patterns, and which types have no MLIR equivalent.

```tablegen
def TTNN_YourOp : TTNN_Op<"your_op",
    [AttrSizedOperandSegments, TTNN_MemoryConfigOpInterface]> {
  let summary = "Your operation.";
  let description = [{...}];

  let arguments = (ins AnyRankedTensor:$input,
                       Optional<AnyRankedTensor>:$weight,
                       Optional<AnyRankedTensor>:$bias,
                       DefaultValuedAttr<F32Attr, "1e-12">:$epsilon,
                       OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}
```

### 1b. Verifier implementation

**File:** `lib/Dialect/TTNN/IR/TTNNOps.cpp`

The TTNN verifier enforces device-specific constraints (e.g., TTNN LayerNorm only supports
normalization over the last dimension, so weight/bias must be 1D).

## Step 2: Define the Op in TTIR

### 2a. Tablegen definition

**File:** `include/ttmlir/Dialect/TTIR/IR/TTIROps.td`

The TTIR op is a simplified, device-agnostic version of the TTNN op. It captures the mathematical
semantics without device-specific parameters. Key differences from TTNN:
- Drops device-specific attributes (memory_config, compute_config, sub_device_id, topology, etc.)
- Drops parameters that only matter at the hardware level
- Keeps only the essential mathematical/logical parameters

Choose the right base class:
- `TTIR_NamedOp` — most non-elementwise ops (normalization, matmul, etc.)
- `TTIR_ElementwiseUnaryOp` / `TTIR_ElementwiseBinaryOp` — elementwise ops
- `TTIR_DPSOp` — destination-passing style ops

If the op has optional operands, add the `AttrSizedOperandSegments` trait.

```tablegen
def TTIR_YourOp : TTIR_NamedOp<"your_op", [AttrSizedOperandSegments]> {
  let summary = "Your operation";
  let description = [{
    Describe what the op does, including the mathematical formula.
    Example: layer_norm(x, weight, bias, epsilon) =
      ((x - mean(x)) / sqrt(var(x) + epsilon)) * weight + bias
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       Optional<AnyRankedTensor>:$weight,
                       Optional<AnyRankedTensor>:$bias,
                       DenseI64ArrayAttr:$normalized_shape,
                       DefaultValuedAttr<F32Attr, "1e-05">:$epsilon);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}
```

See `references/ttnn_type_mapping.md` for the complete type mapping (tensor types, scalar
attributes, optional/default patterns, etc.). The same types apply to TTIR ops.

### 2b. Verifier implementation

**File:** `lib/Dialect/TTIR/IR/TTIROps.cpp`

Implement the verifier. TTIR verifiers validate the general mathematical semantics (not
device-specific constraints).

```cpp
::mlir::LogicalResult mlir::tt::ttir::YourOp::verify() {
  RankedTensorType inputType = getInput().getType();
  RankedTensorType outputType = getResult().getType();

  // Verify input/output shape compatibility
  if (inputType.getShape() != outputType.getShape()) {
    return emitOpError("input and output must have the same shape");
  }

  // Verify optional operand shapes if present
  if (getWeight()) {
    RankedTensorType weightType = getWeight().getType();
    // ... validate weight shape ...
  }

  return success();
}
```

### 2c. Operand layout workarounds (if needed)

Some TTNN metal ops require specific operands to be in `ROW_MAJOR` layout (e.g., page tables and
position tensors for attention ops require ROW_MAJOR, not TILE). If the compiler's layout pass tiles
an operand that the metal kernel expects as ROW_MAJOR, the runtime will fail with errors like
`Expect cur_pos to be ROW_MAJOR, got Layout::TILE`.

To fix this, implement a workaround that inserts `to_layout` ops before/after the op:

**Files:**
- `include/ttmlir/Dialect/TTNN/IR/TTNNWorkaroundsPass.h` — add factory method declaration
- `lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp` — add factory method implementation
- `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td` — add `extraClassDeclaration` to the op

1. Declare a factory method in `TTNNOperandsWorkaroundsFactory`:
```cpp
static TTNNOperandsWorkarounds createYourOpOperandsWorkarounds(Operation *op);
```

2. Implement it — add an input workaround for each operand (empty for operands that don't need
   fixing, `Layout::RowMajor` for operands that need ROW_MAJOR). For ops with optional operands,
   conditionally add workarounds only when the operand is present:
```cpp
TTNNOperandsWorkarounds
TTNNOperandsWorkaroundsFactory::createYourOpOperandsWorkarounds(Operation *op) {
  auto yourOp = cast<YourOp>(op);
  TTNNOperandWorkarounds empty;
  TTNNOperandWorkarounds rowMajor;
  rowMajor.tensorLayoutWorkaround = Layout::RowMajor;

  auto wa = TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds();
  wa = wa.addInputOperandWorkaround(empty);        // input: no workaround
  wa = wa.addInputOperandWorkaround(rowMajor);      // index: force ROW_MAJOR
  wa = wa.addOutputOperandWorkaround(empty);         // output: no workaround
  return wa;
}
```

3. Add `extraClassDeclaration` to the TTNN op in `TTNNOps.td`:
```tablegen
let extraClassDeclaration = [{
  wa::TTNNOperandsWorkarounds getOperandsWorkarounds() {
    return wa::TTNNOperandsWorkaroundsFactory::createYourOpOperandsWorkarounds(getOperation());
  }
}];
```

Look at existing examples like `PagedScaledDotProductAttentionDecodeOp` or `ScatterOp` for
reference patterns. Also add a forward declaration for your op in `TTNNWorkaroundsPass.h` if needed.

## Step 3: TTIR to TTNN Conversion

**File:** `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`

Add a conversion pattern class and register it in `populateTTIRToTTNNPatterns`.

```cpp
class YourOpConversionPattern
    : public OpConversionPattern<ttir::YourOp> {
public:
  using OpConversionPattern<ttir::YourOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::YourOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Validate any TTNN-specific constraints
    // (e.g., TTNN only supports last-dim normalization)

    rewriter.replaceOpWithNewOp<ttnn::YourOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), adaptor.getWeight(), adaptor.getBias(),
        adaptor.getEpsilon(), /*memoryConfig*/ nullptr);
    return success();
  }
};
```

Register in `populateTTIRToTTNNPatterns` by adding `YourOpConversionPattern` to the
`patterns.add<...>()` call.

## Step 4: TTNN to EmitC Conversion

**File:** `lib/Conversion/TTNNToEmitC/TTNNToEmitC.cpp`

Uses `EmitCTTNNEmitter` to generate C++ function call arguments. Arguments must match the order
of the TTNN C++ API.

```cpp
class YourOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<mlir::tt::ttnn::YourOp> {
public:
  using TTNNToEmitCBaseOpConversionPattern<
      mlir::tt::ttnn::YourOp>::TTNNToEmitCBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::YourOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ttnn_to_emitc::EmitCTTNNEmitter<mlir::tt::ttnn::YourOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getEpsilon()),
        emitter.emit(srcOp.getWeight()),
        emitter.emit(srcOp.getBias()),
        emitter.emit(std::nullopt) | emitter.getMemoryConfig(srcOp.getResult()),
    };

    emitter.replaceOp(*this, args);
    return success();
  }
};
```

Register in `populateTTNNToEmitCPatterns`.

Look at similar existing ops (e.g., `RMSNormOpConversionPattern` or `LayerNormOpConversionPattern`)
to see the exact argument ordering for the TTNN C++ API. The arguments must match the TTNN C++
function signature exactly.

## Step 5: TTNN to EmitPy Conversion

**File:** `lib/Conversion/TTNNToEmitPy/TTNNToEmitPy.cpp`

Similar to EmitC but uses `EmitPyTTNNEmitter` and keyword argument names (second parameter to
`emit`).

**IMPORTANT — Positional vs keyword argument ordering:** EmitPy generates Python function calls.
In Python, keyword arguments must come AFTER all positional arguments. If you emit an argument with
a keyword name (e.g., `emitter.emit(srcOp.getDim(), "dim")`) but a later argument is positional
(no keyword name), the generated Python will be invalid: `ttnn.op(input, dim=0, index, ...)` is a
syntax error. To fix this, either:
- Emit the argument **without** a keyword name (positional): `emitter.emit(srcOp.getDim())`
- Or ensure all subsequent arguments also use keyword names

Check the target Python function's signature to determine which args are positional vs keyword.

```cpp
class YourOpConversionPattern
    : public TTNNToEmitPyBaseOpConversionPattern<mlir::tt::ttnn::YourOp> {
public:
  using TTNNToEmitPyBaseOpConversionPattern<
      mlir::tt::ttnn::YourOp>::TTNNToEmitPyBaseOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::tt::ttnn::YourOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ttnn_to_emitpy::EmitPyTTNNEmitter<mlir::tt::ttnn::YourOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getEpsilon(), "epsilon"),
        emitter.emit(srcOp.getWeight(), "weight"),
        emitter.emit(srcOp.getBias(), "bias"),
        emitter.emit(srcOp.getMemoryConfig() |
                         emitter.getMemoryConfig(srcOp.getResult()),
                     "memory_config"),
    };

    emitter.replaceOp(*this, args);
    return success();
  }
};
```

Register in `populateTTNNToEmitPyPatterns`.

## Step 6: StableHLO Composite to TTIR Conversion

**File:** `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp`

If the op comes from a StableHLO composite (e.g., `tenstorrent.your_op`), add a conversion pattern.

For simple ops that map 1:1, use the generic template:
```cpp
patterns.add<StableHLOToTTIRCompositeOpConversionPattern<ttir::YourOp>>(
    context, "tenstorrent.your_op");
```

For ops with optional operands (`AttrSizedOperandSegments`) or attributes that need conversion
(e.g., `DenseIntElementsAttr` → `DenseI64ArrayAttr`), write a custom pattern:

```cpp
class TenstorrentYourOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompositeOp> {
public:
  TenstorrentYourOpConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                  mlir::stablehlo::CompositeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getName() != "tenstorrent.your_op") {
      return failure();
    }

    // Extract attributes from compositeAttributes
    DictionaryAttr compositeAttrs = srcOp.getCompositeAttributes();

    // Build named attributes for the TTIR op
    SmallVector<NamedAttribute> namedAttrs;
    // ... extract and convert attributes ...

    // Compute operandSegmentSizes for optional operands
    size_t numOperands = adaptor.getOperands().size();
    SmallVector<int32_t> segmentSizes = {1, ...};
    namedAttrs.push_back(rewriter.getNamedAttr(
        "operandSegmentSizes", rewriter.getDenseI32ArrayAttr(segmentSizes)));

    auto outputType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
    rewriter.replaceOpWithNewOp<ttir::YourOp>(
        srcOp, outputType, adaptor.getOperands(), namedAttrs);
    return success();
  }
};
```

Register in `populateStableHLOCompositeLegalizationPatterns`.

## Step 7: Flatbuffer Schema

### 7a. Define the flatbuffer table

**File:** `include/ttmlir/Target/TTNN/operations/<category>.fbs`

Add a table in the appropriate category file (e.g., `normalization.fbs`, `eltwise.fbs`). Or create
a new `.fbs` file if no existing category fits (and add it to the CMakeLists.txt).

```flatbuffers
table YourOp {
  input: tt.target.ttnn.TensorRef;
  weight: tt.target.ttnn.TensorRef;   // null if not provided
  bias: tt.target.ttnn.TensorRef;     // null if not provided
  epsilon: float;
  memory_config: tt.target.ttnn.MemoryConfig;
  out: tt.target.ttnn.TensorRef;
}
```

### 7b. Register in OpType union

**File:** `include/ttmlir/Target/TTNN/program.fbs`

Add `YourOp` to the `OpType` union (keep alphabetical order).

### 7c. Serialize to flatbuffer

**File:** `lib/Target/TTNN/TTNNToFlatbuffer.cpp`

Add a `createOp` overload and a dispatch entry in `emitTTNNOperation`.

```cpp
::flatbuffers::Offset<::tt::target::ttnn::YourOp>
createOp(FlatbufferObjectCache &cache, YourOp op) {
  auto input = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));

  // Handle optional operands (use offset 0 for absent)
  ::flatbuffers::Offset<::tt::target::ttnn::TensorRef> weight = 0;
  if (op.getWeight()) {
    weight = cache.at<::tt::target::ttnn::TensorRef>(
        getOperandThroughDPSOps(op.getWeight()));
  }

  ::flatbuffers::Offset<::tt::target::ttnn::TensorRef> bias = 0;
  if (op.getBias()) {
    bias = cache.at<::tt::target::ttnn::TensorRef>(
        getOperandThroughDPSOps(op.getBias()));
  }

  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer);
  auto memoryConfig = getMemoryConfigIfNeeded(cache, op);

  return ::tt::target::ttnn::CreateYourOp(
      *cache.fbb, input, weight, bias,
      op.getEpsilon().convertToFloat(), memoryConfig, output);
}
```

Add the dispatch in `emitTTNNOperation`:
```cpp
if (auto yourOp = dyn_cast<YourOp>(op); yourOp) {
  return createOperation(cache, createOp(cache, yourOp), debugString, locInfo);
}
```

## Step 8: Runtime Implementation

### 8a. Operation header

**File:** `runtime/lib/ttnn/operations/<category>/your_op.h` (new file)

```cpp
#ifndef RUNTIME_LIB_TTNN_OPERATIONS_CATEGORY_YOUR_OP_H
#define RUNTIME_LIB_TTNN_OPERATIONS_CATEGORY_YOUR_OP_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::your_op {
void run(const ::tt::target::ttnn::YourOp *op, ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::your_op

#endif
```

### 8b. Operation implementation

**File:** `runtime/lib/ttnn/operations/<category>/your_op.cpp` (new file)

```cpp
#include "operations/<category>/your_op.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::your_op {
void run(const ::tt::target::ttnn::YourOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());

  // Handle optional operands
  std::optional<::ttnn::Tensor> weight = std::nullopt;
  if (op->weight()) {
    weight = tensorPool.getTTNNTensorAndValidate(op->weight());
  }

  std::optional<::ttnn::Tensor> bias = std::nullopt;
  if (op->bias()) {
    bias = tensorPool.getTTNNTensorAndValidate(op->bias());
  }

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memory_config());

  // Call the TTNN library function
  ::ttnn::Tensor output = ::ttnn::your_op(
      input, op->epsilon(), weight, bias, memoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::your_op
```

### 8c. CMakeLists.txt

**File:** `runtime/lib/ttnn/operations/CMakeLists.txt`

Add the new source file to `TTNN_OPS_SRCS`:
```cmake
${CMAKE_CURRENT_SOURCE_DIR}/<category>/your_op.cpp
```

### 8d. Program executor dispatch

**File:** `runtime/lib/ttnn/program_executor.cpp`

Add include and dispatch case:
```cpp
#include "operations/<category>/your_op.h"

// In the switch statement:
case ::tt::target::ttnn::OpType::YourOp: {
  return operations::your_op::run(op->type_as_YourOp(), getContext());
}
```

### 8e. Runtime input/output refs

**File:** `runtime/lib/ttnn/runtime.cpp`

Add cases to both `getOpOutputRef` and `getOpInputRefs`:

```cpp
// In getOpOutputRef switch:
case ::tt::target::ttnn::OpType::YourOp: {
  tensorRef = opContext.type_as_YourOp()->out();
  break;
}

// In getOpInputRefs switch:
case ::tt::target::ttnn::OpType::YourOp: {
  tensorRefs = {opContext.type_as_YourOp()->input()};
  if (opContext.type_as_YourOp()->weight()) {
    tensorRefs.push_back(opContext.type_as_YourOp()->weight());
  }
  if (opContext.type_as_YourOp()->bias()) {
    tensorRefs.push_back(opContext.type_as_YourOp()->bias());
  }
  break;
}
```

### 8f. TTNN library header

**File:** `runtime/include/tt/runtime/detail/ttnn/ttnn.h`

Add the TTNN library include:
```cpp
#include "ttnn/operations/<category>/<header>.hpp"
```

## Step 9: OpModel

### 9a. Metal headers

**File:** `include/ttmlir/OpModel/TTNN/MetalHeaders.h`

Add the TTNN metal header:
```cpp
#include "ttnn/operations/<category>/<header>.hpp"
```

### 9b. OpModel declaration

**File:** `include/ttmlir/OpModel/TTNN/TTNNOpModel.h`

Add a template specialization of `OpModel<YourOp>` declaring `getOpConstraints` and `getOpRuntime`.
Follow the pattern of similar ops (e.g., `OpModel<LayerNormOp>` for ops with optional parameters).

### 9c. OpModel implementation

**File:** `lib/OpModel/TTNN/TTNNOpModel.cpp`

Implement `getOpConstraints` and `getOpRuntime`. Both are guarded by `#ifdef TTMLIR_ENABLE_OPMODEL`.

The pattern:
1. Convert inputs to `TensorSpec` using `detail::convertToTensorSpec` (required) or
   `detail::convertToOptionalTensorSpec` (optional)
2. Create a query lambda that calls `::ttnn::graph::query_op_constraints` /
   `::ttnn::graph::query_op_runtime` with the TTNN function
3. Return via `operation::getOpConstraints` / `operation::getOpRuntime`

### 9d. OpModel interface (REQUIRED for ALL TTNN ops)

**File:** `lib/Dialect/TTNN/Interfaces/TTNNOpModelInterface.cpp`

**IMPORTANT:** Every TTNN op inherits `TTNN_OpModelInterface` through the `TTNN_Op` base class.
This means every TTNN op MUST have `getOpConstraints` and `getOpRuntime` implementations in this
file, or the build will fail with undefined symbol errors. Even if you are not implementing full
OpModel support, you must add stub implementations:

```cpp
llvm::Expected<op_model::OpConstraints>
YourOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
}

llvm::Expected<size_t>
YourOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
}
```

Find the right alphabetical location in the file (search for similar ops like `ScatterOp`) and add
your stubs there.

For full OpModel support, implement the interface that bridges the MLIR op to the OpModel. For ops
with optional operands, create a helper struct and unpacking function (see `LayerNormOptionalArgs`
pattern).

## Step 10: TTIRBuilder

**File:** `tools/builder/ttir/ttir_builder.py`

Add three methods to the `TTIRBuilder` class:

### 10a. Builder method (tagged with `@tag`)

**IMPORTANT:** Match the MLIR attribute types exactly:
- `SI32Attr` in tablegen → `IntegerAttr.get(IntegerType.get_signed(32), value)` in Python
- `I32Attr` in tablegen → `IntegerAttr.get(IntegerType.get_signless(32), value)` in Python
- `UI32Attr` in tablegen → `IntegerAttr.get(IntegerType.get_unsigned(32), value)` in Python
- `F32Attr` in tablegen → `FloatAttr.get_f32(value)` in Python

Getting these wrong (e.g., using `get_signless` for `SI32Attr`) will cause the pass manager to
reject the op with an attribute constraint error.

```python
@tag(ttir.YourOp)
def your_op(
    self,
    in0: Operand,
    weight: Optional[Operand] = None,
    bias: Optional[Operand] = None,
    epsilon: float = 1e-5,
    output_type: Optional[torch.dtype] = None,
    loc: Optional[str] = None,
    unit_attrs: Optional[List[str]] = None,
) -> OpResult:
    ttir_op = self.get_opview_from_method(TTIRBuilder.your_op)
    epsilon_attr = FloatAttr.get_f32(epsilon)

    # Compute golden output
    input0 = self._get_golden_tensor(in0)
    weight0 = self._get_golden_tensor(weight) if weight else None
    bias0 = self._get_golden_tensor(bias) if bias else None
    op_golden_function = get_golden_function(ttir_op)
    golden_output = op_golden_function(input0, weight=weight0, bias=bias0, ...)

    result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)
    loc = Location.name(loc) if loc else self._get_location()

    op = ttir_op(result, in0, weight=weight, bias=bias, epsilon=epsilon_attr, loc=loc)
    op_result = op.result

    if unit_attrs is not None:
        for attr_name in unit_attrs:
            op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

    if not self._disable_golden_check:
        self._set_golden_tensor(op_result, golden_output)

    return op_result
```

### 10b. Parser method (tagged with `@parse`)

Reconstructs the op from an existing TTIR module by mapping old operands through `global_dict`.

### 10c. Split method (tagged with `@split`)

Creates a new `Module` containing just this op wrapped in a function. Handles building the input
type list dynamically based on which optional operands are present.

## Step 11: Golden Functions

**File:** `tools/golden/mapping.py`

Add golden (reference) implementations for both TTIR and TTNN versions.

**IMPORTANT — GoldenMapTensor limitations:** `GoldenMapTensor` wraps per-shard torch tensors and
supports `torch.*` functions via the `__torch_function__` protocol. However, it does NOT support
Python arithmetic operators like `*`, `+`, `-` directly. For example, `tensor * scale` will fail
with `unsupported operand type(s)`. Instead, use the torch function equivalents:
- `torch.mul(tensor, scale)` instead of `tensor * scale`
- `torch.add(tensor, bias)` instead of `tensor + bias`
- `torch.sub(a, b)` instead of `a - b`

**IMPORTANT — Parameter ordering:** The golden function's parameter order must match the order the
builder passes arguments. The builder calls the golden function with positional args in this order:
1. All tensor inputs (required and optional, in the order they appear in the tablegen definition)
2. All attribute args (head_dim_v, is_causal, scale, etc.)
3. `output_type_mlir` as the last positional arg

If the golden function's parameter order doesn't match, you'll get errors like `Unexpected attribute
type: GoldenMapTensor` (a tensor landing in an attribute parameter slot) or vice versa.

```python
def ttir_your_op_golden(
    input: GoldenMapTensor,
    weight: Optional[GoldenMapTensor] = None,
    bias: Optional[GoldenMapTensor] = None,
    epsilon: FloatAttr = None,
    output_type_mlir: Type = None,
    **kwargs,
) -> GoldenMapTensor:
    epsilon = unpack_mlir_attr(epsilon)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.nn.functional.your_op(
        input.float(), weight=weight, bias=bias, eps=epsilon
    ).to(output_dtype)
```

Register in the golden mapping dictionaries (search for the TTIR and TTNN mapping dicts and add
entries):
```python
ttir.YourOp: ttir_your_op_golden,
ttnn.YourOp: ttnn_your_op_golden,
```

## Step 12: Precompiled Headers

**File:** `tools/ttnn-standalone/ttnn-precompiled.hpp`

Add the TTNN operation header:
```cpp
#include "ttnn/operations/<category>/<header>.hpp"
```

## Step 13: Tests

### 13a. TTIR-to-TTNN conversion test

**File:** `test/ttmlir/Dialect/TTNN/<op_name>/simple_<op_name>.mlir` (new file)

Test that the TTIR op converts to TTNN correctly. Cover all operand combinations (e.g., with/without
weight, with/without bias).

```mlir
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @forward(%arg0: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
    // CHECK: "ttnn.your_op"
    %1 = "ttir.your_op"(%arg0) <{epsilon = 1.0e-05 : f32,
        operandSegmentSizes = array<i32: 1, 0, 0>}>
        : (tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    return %1 : tensor<512x1024xbf16>
  }
}
```

For ops with `AttrSizedOperandSegments`, test multiple combinations of optional operands. Use
`ttir.empty()` to create placeholder tensors for optional operands.

### 13b. StableHLO composite test

**File:** `test/ttmlir/Conversion/StableHLOToTTIR/composite/test_<op_name>.mlir` (new file)

Test that the StableHLO composite converts to TTIR.

### 13c. EmitC test

**File:** `test/ttmlir/EmitC/TTNN/<op_name>/<op_name>.mlir` (new file)

Test the full pipeline: TTIR → TTNN → Flatbuffer, and TTNN → EmitC → C++.

```mlir
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-to-emitc-device-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir
```

### 13d. Golden/builder tests

**File:** `test/python/golden/test_ttir_ops.py`

Add a parametrized test that exercises the TTIRBuilder method. Parametrize over:
- Different input shapes
- Optional operand combinations (has_weight, has_bias, etc.)
- Targets: `["ttnn", "emitpy", "emitc"]` — all three backends should be tested

**Index tensor types:** If your op takes index tensors (like gather/scatter), the TTNN metal op
may require unsigned integer types (`torch.uint32` → MLIR `ui32`), not signed (`torch.int32` →
MLIR `i32`). Using the wrong integer type will cause a runtime error like "Index tensor must be of
type UINT32 or UINT16". Check the metal API docs for the required types.

### 13e. OpModel tests (if OpModel was implemented)

**Files:**
- `test/unittests/OpModel/TTNN/Lib/TestOpModelLib.cpp` — unit tests for the OpModel functions
- `test/unittests/OpModel/TTNN/Op/TestOpModelInterface.cpp` — tests via the MLIR interface

## Step 14: CPU-Hoisted Version (if applicable)

CPU-hoisting moves selected TTIR ops off the device and executes them on the host CPU. This
improves numerical precision (host uses full f32/i32) and reduces peak DRAM/L1 usage by keeping
intermediate tensors in host memory. The hoisting passes run in two scenarios: **const-eval
hoisting** (automatically hoisting entire constant-evaluation subgraphs) and **manual hoisting**
(individual ops tagged with `ttir.should_hoist`). Both paths typically operate on model weights and
constants rather than activations, so if your op has strictly activation semantics it likely does
not need CPU-hoisting support.

Once ops are hoisted into the CPU module, they are lowered through **two independent compilation
paths** depending on the target:

- **Runtime target** (flatbuffer path): TTIR → Linalg/TOSA → LLVM IR → standalone `.so` dylib,
  embedded in the flatbuffer and loaded by the runtime via `dlopen()`.
- **EmitPy target**: TTIR → EmitPy `CallOpaqueOp("ttir_cpu.<op>")` → Python code that calls
  pure-torch implementations in the `ttir_cpu` module.

Both paths share the same hoisting infrastructure and both require TTIR → Linalg/TOSA support
(the runtime path uses it for actual lowering; the hoisting pass uses it as a validation gate via
`canLowerTTIRToLinalg()`). The EmitPy path additionally requires an EmitPy CPU conversion pattern
and a torch implementation.

### 14a. Decomposition alternative (CPUFallback mode)

Before implementing full CPU support for your op, check whether it can be decomposed into simpler
TTIR ops that already have CPU support. The hoisting validation runs
`TTIRToTTIRDecomposition(CPUFallback)` before the Linalg lowering check, so decomposed ops pass
validation automatically. This avoids implementing Linalg, EmitPy CPU, and torch code entirely.

**File:** `lib/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecompositionPass.cpp`

Add the op as illegal in the `CPUFallback` switch case:

```cpp
case DecompMode::CPUFallback:
  // ...existing legal/illegal ops...
  target.addIllegalOp<ttir::YourOp>();  // will be decomposed
  break;
```

Then add the decomposition pattern itself in the appropriate file under
`lib/Conversion/TTIRToTTIRDecomposition/`. The decomposed ops must all have existing Linalg and
EmitPy CPU support. Use this approach when the op has no natural Linalg/TOSA equivalent but
decomposes cleanly (e.g., `DotGeneralOp` decomposes to `MatmulOp`).

If decomposition is not viable, implement steps 14b–14f below.

### 14b. TTIR → Linalg/TOSA conversion pattern

**Files:**
- `lib/Conversion/TTIRToLinalg/TTIRToLinalg.cpp` — add the conversion pattern
- `lib/Conversion/TTIRToLinalg/EltwiseUnary.cpp` / `EltwiseBinary.cpp` / `Pooling.cpp` — for
  elementwise or pooling ops, add patterns in the appropriate category file

Register the pattern in `populateTTIRToLinalgPatterns` or `populateTTIRToTosaPatterns`:

```cpp
// In populateTTIRToTosaPatterns or populateTTIRToLinalgPatterns:
patterns.add<YourOpConversionPattern>(typeConverter, ctx);
```

The pattern lowers TTIR ops to equivalent Linalg, TOSA, Arith, or Math dialect ops. The conversion
target declares all TTIR ops as illegal, so every TTIR op in a CPU-hoisted function must have a
pattern or both the runtime pipeline (Linalg → LLVM) and the hoisting validation will fail. Look
at similar existing patterns — elementwise ops typically lower to `linalg.generic` or TOSA
equivalents; reductions, matmuls, and normalization ops have dedicated patterns.

### 14c. TTIR → Linalg/TOSA lit test

**File:** `test/ttmlir/Conversion/TTIRToLinalg/<op_name>.mlir` (new file)

Test that the TTIR op correctly lowers to Linalg/TOSA/Arith ops via `--convert-ttir-to-linalg`.
Cover the key shape and attribute variations (e.g., different ranks, index types, optional operands).
Use `CHECK-LABEL` per function and `CHECK` for the expected lowered ops.

```mlir
// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  // CHECK-LABEL: func.func @test_your_op_basic
  func.func @test_your_op_basic(%arg0: tensor<32x64xf32>, %arg1: tensor<4xi32>) -> tensor<4x64xf32> {
    // CHECK: tensor.empty()
    // CHECK: linalg.generic
    // CHECK: tensor.extract %arg0
    // CHECK: linalg.yield
    %0 = "ttir.your_op"(%arg0, %arg1) <{...}> : (tensor<32x64xf32>, tensor<4xi32>) -> tensor<4x64xf32>
    // CHECK: return %{{[0-9]+}} : tensor<4x64xf32>
    return %0 : tensor<4x64xf32>
  }
}
```

Look at the existing tests in `test/ttmlir/Conversion/TTIRToLinalg/` for patterns — ops that lower
to TOSA use `CHECK: tosa.<op>` (e.g., `layer_norm.mlir`), ops that lower to Linalg use
`CHECK: linalg.generic` or `CHECK: linalg.<named_op>` (e.g., `embedding.mlir`, `matmul.mlir`).
For ops with index-type conversions, check for `arith.index_cast`. For ops with multiple data type
inputs (e.g., i32 vs i64 indices), add separate test functions exercising each type path.

### 14e. TTIR → EmitPy CPU conversion pattern

**File:** `lib/Conversion/TTIRToEmitPy/TTIRCPUToEmitPyPass.cpp`

Add a conversion pattern that lowers the TTIR op to an `emitpy::CallOpaqueOp` targeting
`ttir_cpu.<op>`. Use `EmitPyCallBuilder` to construct the call:

```cpp
class TTIRYourOpToEmitPy : public OpConversionPattern<ttir::YourOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::YourOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    EmitPyCallBuilder b(op, getTypeConverter(), getCallee(op));
    b.addOperand(adaptor.getInput());
    b.addKwarg("epsilon", std::to_string(adaptor.getEpsilon().convertToFloat()));
    b.replaceOp(rewriter);
    return success();
  }
};
```

Register the pattern in the `ConvertTTIRToEmitPyCPUPass::runOnOperation` method:
```cpp
patterns.add<TTIRYourOpToEmitPy>(typeConverter, &getContext());
```

For elementwise ops, use the existing generic templates (`TTIRUnaryToEmitPy<ttir::YourOp>` or
`TTIRBinaryToEmitPy<ttir::YourOp>`) instead of writing a custom pattern. For reductions, use
`TTIRReductionToEmitPy<ttir::YourOp>`.

### 14f. torch implementation in `ttir_cpu`

**File:** `tools/tt-alchemist/templates/python/local/ttir_cpu.py`

Add a function that mirrors the TTIR op semantics using pure torch. This is only used by the
EmitPy target — the runtime target compiles through Linalg → LLVM instead.

```python
def your_op(input, *, epsilon=1e-5, **_):
    return torch.nn.functional.your_op(input.float(), eps=epsilon).to(input.dtype)
```

Key conventions:
- Use `**_` to absorb unused keyword arguments (the EmitPy call may pass extra kwargs)
- Use `builtins.*` references when a local name shadows a Python builtin (e.g., `builtins.max`)
- Operate in float32 for numerical precision when the op involves reductions or normalization
- Match the TTIR op semantics exactly — this is the reference implementation for const-eval

### 14g. Tests for CPU-hoisted ops

#### Lit test: EmitPy CPU-hoisted output

**File:** `test/ttmlir/EmitPy/cpu_hoisted_ops.mlir`

Add a validation function that tags the op with `{ttir.should_hoist}`, runs it alongside the
device version, and checks that the generated Python contains the expected `ttir_cpu.<op>` call:

```mlir
// CHECK-LABEL: def your_op_validation
// CHECK: cpu_hoisted_ttir_your_op_{{.*}}
// CHECK: ttnn.your_op(
func.func @your_op_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu = "ttir.your_op"(%arg0) <{epsilon = 1.0e-05 : f32}> {ttir.should_hoist}
      : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %dev = "ttir.your_op"(%arg0) <{epsilon = 1.0e-05 : f32}>
      : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu, %dev) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}
```

#### Builder test: CPU-hoisted golden

**File:** `test/python/golden/ttir_ops/<category>/test_<category>.py`

Add a parametrized test that exercises the builder method with `unit_attrs=["ttir.should_hoist"]`.
Target all three backends (`ttnn`, `ttmetal`, `emitpy`):

```python
@pytest.mark.parametrize("shape", hoisted_shapes, ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal", "emitpy"])
def test_cpu_hoistable_your_op(shape, dtype, target, request, device):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def hoisted_your_op(in0, builder, unit_attrs=None):
            return builder.your_op(in0, epsilon=1e-5, unit_attrs=["ttir.should_hoist"])

    compile_and_execute_ttir(module, test_base=request.node.name, target=target)
```

Look at the existing `test_cpu_hoistable_*` tests in the same file for the exact parametrization
and decorator patterns (e.g., `@x86_only`).

## Verification

After implementing all the code changes, you MUST verify they work.

### Build

```bash
source env/activate
cmake --build build
```

If the build fails, fix the errors and rebuild before proceeding to tests.

### Launch the review webserver

After the build succeeds, launch the review webserver. This runs all tests, collects the git diff,
generates emitted Python and C++ code, and serves a review page the user can inspect.

```bash
source env/activate
python .claude/skills/add-op/review/generate_review.py \
  --op-name <op_name> \
  --ttnn-test-dir test/ttmlir/Dialect/TTNN/<test_dir>/ \
  --emitc-test-dir test/ttmlir/EmitC/TTNN/<test_dir>/ \
  --pytest-filter <op_name> \
  --emitpy-input test/ttmlir/Dialect/TTNN/<test_dir>/simple_<op_name>.mlir \
  --emitc-input test/ttmlir/Dialect/TTNN/<test_dir>/simple_<op_name>.mlir
```

Replace `<op_name>` with the op name (e.g., `gather_dim`) and `<test_dir>` with the test
directory name (e.g., `gather`).

The review page has these tabs:
- **Tests** — Lit test and pytest results with pass/fail coloring
- **Code Changes** — Colored git diff (unified or side-by-side view)
- **Emitted Python** — Python code generated by the EmitPy pipeline
- **Emitted C++** — C++ code generated by the EmitC pipeline
- **tt-alchemist** — Placeholder for future content

Tell the user the URL (default: http://localhost:3118) and wait for them to review.

**IMPORTANT — Port Forwarding:** If the user is working on a remote machine via SSH (which is the
common case), they will need to set up VS Code port forwarding to access the review page. You MUST
notify the developer about this. Tell them:

> The review server is running at http://localhost:3118 on the remote machine.
> To access it, you need to forward the port in VS Code:
> 1. Open the **Ports** panel (View → Open View → Ports, or click "Ports" in the bottom panel)
> 2. Click **Forward a Port** and enter `3118`
> 3. Click the **Local Address** link (or the globe icon) to open the review page in your browser

Wait for the user to confirm they can see the review page before proceeding.

To generate a static HTML file instead of starting a server:
```bash
python .claude/skills/add-op/review/generate_review.py \
  --op-name <op_name> \
  ... \
  --static review.html
```

## Checklist

Use this to make sure you haven't missed anything:

- [ ] TTNN tablegen definition (`TTNNOps.td`)
- [ ] TTNN verifier (`TTNNOps.cpp`)
- [ ] TTIR tablegen definition (`TTIROps.td`)
- [ ] TTIR verifier (`TTIROps.cpp`)
- [ ] TTIR → TTNN conversion pattern (`TTIRToTTNN.cpp`)
- [ ] TTNN → EmitC conversion pattern (`TTNNToEmitC.cpp`)
- [ ] TTNN → EmitPy conversion pattern (`TTNNToEmitPy.cpp`)
- [ ] StableHLO composite → TTIR conversion (`StableHLOLegalizeCompositePass.cpp`)
- [ ] Flatbuffer table definition (`.fbs`)
- [ ] Flatbuffer OpType union entry (`program.fbs`)
- [ ] Flatbuffer serialization (`TTNNToFlatbuffer.cpp`)
- [ ] Runtime header and implementation (new `.h` and `.cpp`)
- [ ] Runtime CMakeLists.txt entry
- [ ] Runtime program_executor dispatch (`program_executor.cpp`)
- [ ] Runtime getOpOutputRef and getOpInputRefs (`runtime.cpp`)
- [ ] Runtime TTNN library include (`ttnn.h`)
- [ ] OpModel metal headers (`MetalHeaders.h`)
- [ ] OpModel declaration (`TTNNOpModel.h`)
- [ ] OpModel implementation (`TTNNOpModel.cpp`)
- [ ] OpModel interface (`TTNNOpModelInterface.cpp`)
- [ ] TTIRBuilder method, parser, split (`ttir_builder.py`)
- [ ] Golden functions (`mapping.py`)
- [ ] Precompiled headers (`ttnn-precompiled.hpp`)
- [ ] TTIR-to-TTNN conversion test
- [ ] StableHLO composite test
- [ ] EmitC test
- [ ] Builder/golden tests (`test_ttir_ops.py`)
- [ ] OpModel unit tests
- [ ] CPU-hoisting: CPUFallback decomposition (`TTIRToTTIRDecompositionPass.cpp`) — or steps below
- [ ] CPU-hoisting: TTIR → Linalg/TOSA pattern (`TTIRToLinalg.cpp`)
- [ ] CPU-hoisting: TTIR → Linalg lit test (`test/ttmlir/Conversion/TTIRToLinalg/<op_name>.mlir`)
- [ ] CPU-hoisting: TTIR → EmitPy CPU pattern (`TTIRCPUToEmitPyPass.cpp`)
- [ ] CPU-hoisting: torch implementation (`ttir_cpu.py`)
- [ ] CPU-hoisting: EmitPy lit test (`test/ttmlir/EmitPy/cpu_hoisted_ops.mlir`)
- [ ] CPU-hoisting: builder golden test (`test/python/golden/ttir_ops/`)
