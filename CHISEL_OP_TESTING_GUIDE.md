# Chisel Op Coverage Testing Guide

## Overview

This guide provides a structured approach to testing chisel's operation coverage using tt-mlir's device tests.

## Quick Start

```bash
cd /localdev/ndrakulic/tt-xla/third_party/tt-mlir/src/tt-mlir
source env/activate

# Setup
ttrt query --save-artifacts
export SYSTEM_DESC_PATH=$(pwd)/ttrt-artifacts/system_desc.ttsys
export TT_INJECT_TTNN2FB=1

# Run quick validation suite
./test_chisel_op_coverage.sh
```

## Operation Categories by Complexity

### Level 1: Element-wise Operations (Start Here ✓)

**Unary Operations** - Single input, single output:
```bash
# Test file: test/python/golden/ttir_ops/eltwise/test_ttir_unary.py
# Operations: exp, log, sqrt, abs, neg, sin, cos, tanh, sigmoid, relu, etc.

pytest test/python/golden/ttir_ops/eltwise/test_ttir_unary.py -v -k "exp"
pytest test/python/golden/ttir_ops/eltwise/test_ttir_unary.py -v -k "sqrt"
pytest test/python/golden/ttir_ops/eltwise/test_ttir_unary.py -v -k "log"
```

**Binary Operations** - Two inputs, single output:
```bash
# Test file: test/python/golden/ttir_ops/eltwise/test_ttir_binary.py
# Operations: add, sub, mul, div, maximum, minimum, etc.

pytest test/python/golden/ttir_ops/eltwise/test_ttir_binary.py -v -k "add"
pytest test/python/golden/ttir_ops/eltwise/test_ttir_binary.py -v -k "mul"
pytest test/python/golden/ttir_ops/eltwise/test_ttir_binary.py -v -k "sub"
```

**Ternary Operations** - Three inputs:
```bash
# Test file: test/python/golden/ttir_ops/eltwise/test_ttir_ternary.py
# Operations: where, select, etc.

pytest test/python/golden/ttir_ops/eltwise/test_ttir_ternary.py -v
```

**Why Start Here:**
- Simplest operations
- Fast to execute
- Clear input/output semantics
- Good for validating basic chisel infrastructure

---

### Level 2: Matrix Operations (Core ML)

**MatMul** - Matrix multiplication:
```bash
# Test file: test/python/golden/ttir_ops/matmul/test_matmul.py

pytest test/python/golden/ttir_ops/matmul/test_matmul.py -v --save-artifacts

# Smoke test (faster)
pytest test/ttnn-jit/test_matmul_smoketest.py -v
```

**Operations Tested:**
- Basic matmul (2D x 2D)
- Batch matmul (3D, 4D)
- Transpose variations

**Why Important:**
- Most critical operation for ML
- Tests tensor contractions
- Common in every neural network

---

### Level 3: Data Movement Operations

```bash
# Test file: test/python/golden/ttir_ops/data_movement/test_data_movement.py

# Individual operations
pytest test/python/golden/ttir_ops/data_movement/test_data_movement.py -v -k "transpose"
pytest test/python/golden/ttir_ops/data_movement/test_data_movement.py -v -k "reshape"
pytest test/python/golden/ttir_ops/data_movement/test_data_movement.py -v -k "concat"
pytest test/python/golden/ttir_ops/data_movement/test_data_movement.py -v -k "slice"
```

**Operations:**
- `transpose` - Permute tensor dimensions
- `reshape` - Change tensor shape
- `concat` - Concatenate tensors
- `slice` - Extract subtensor
- `squeeze` - Remove size-1 dimensions
- `unsqueeze` - Add size-1 dimensions
- `broadcast` - Broadcast to larger shape

**Additional Layout Tests:**
```bash
pytest test/python/golden/test_metal_layout.py -v
pytest test/python/golden/test_metal_tilize.py -v
pytest test/python/golden/test_metal_tms.py -v  # Tensor manipulation ops
```

---

### Level 4: Reduction Operations

```bash
# Test file: test/python/golden/ttir_ops/reduction/test_reduction.py

pytest test/python/golden/ttir_ops/reduction/test_reduction.py -v -k "sum"
pytest test/python/golden/ttir_ops/reduction/test_reduction.py -v -k "mean"
pytest test/python/golden/ttir_ops/reduction/test_reduction.py -v -k "max"

# Smoke test
pytest test/ttnn-jit/test_reduction_smoketest.py -v
```

**Operations:**
- `sum` - Sum reduction
- `mean` - Mean reduction
- `max` / `min` - Maximum/minimum reduction
- `argmax` / `argmin` - Index of max/min

**Why Important:**
- Used in pooling layers
- Required for normalization
- Tests dimension reduction logic

---

### Level 5: Convolution Operations

```bash
# Test file: test/python/golden/ttir_ops/convolution/test_conv2d.py

pytest test/python/golden/ttir_ops/convolution/test_conv2d.py -v --save-artifacts

# Transpose convolution
pytest test/python/golden/ttir_ops/convolution/test_conv_transpose2d.py -v
```

**Operations:**
- 2D Convolution (various stride, padding, dilation)
- Transpose Convolution (deconvolution)
- Depthwise convolution
- Grouped convolution

**Why Important:**
- Core operation in CNNs
- Complex memory access patterns
- Tests spatial operations

---

### Level 6: Pooling Operations

```bash
# Test file: test/python/golden/ttir_ops/pooling/test_pooling.py

pytest test/python/golden/ttir_ops/pooling/test_pooling.py -v -k "maxpool"
pytest test/python/golden/ttir_ops/pooling/test_pooling.py -v -k "avgpool"
```

**Operations:**
- MaxPool2D
- AvgPool2D
- AdaptiveAvgPool2D

---

### Level 7: Normalization Operations

```bash
# Test file: test/python/golden/ttir_ops/normalization/test_normalization.py

pytest test/python/golden/ttir_ops/normalization/test_normalization.py -v -k "layernorm"
pytest test/python/golden/ttir_ops/normalization/test_normalization.py -v -k "rmsnorm"
```

**Operations:**
- LayerNorm
- RMSNorm
- BatchNorm
- GroupNorm

**Why Important:**
- Used in transformers and modern architectures
- Composite operations (multiple ops fused)
- Tests complex patterns

---

### Level 8: Fusion Patterns

```bash
# Test file: test/python/golden/ttir_ops/fusing/

pytest test/python/golden/ttir_ops/fusing/test_mish_fusing.py -v
pytest test/python/golden/ttir_ops/fusing/test_rope_fusing.py -v
pytest test/python/golden/ttir_ops/fusing/test_sdpa_fusing.py -v  # Scaled dot-product attention
pytest test/python/golden/ttir_ops/fusing/test_permute_matmul_fusing.py -v
```

**Patterns Tested:**
- Mish activation fusion
- RoPE (Rotary Position Embedding) fusion
- SDPA (Scaled Dot-Product Attention) fusion
- Permute-MatMul fusion
- Reshape-Broadcast-Reshape to Repeat fusion

**Why Important:**
- Tests chisel's ability to recognize fused patterns
- Common in optimized models
- Multiple operations combined into one

---

## Comprehensive Op Coverage Test

Test all operation categories systematically:

```bash
#!/bin/bash
# Comprehensive op coverage test

source env/activate
export TT_INJECT_TTNN2FB=1
export SYSTEM_DESC_PATH=$(pwd)/ttrt-artifacts/system_desc.ttsys

OUTPUT_DIR="chisel_full_coverage_$(date +%Y%m%d_%H%M%S)"

# Level 1: Eltwise
pytest test/python/golden/ttir_ops/eltwise/ -v --save-artifacts --path=$OUTPUT_DIR/eltwise

# Level 2: MatMul
pytest test/python/golden/ttir_ops/matmul/ -v --save-artifacts --path=$OUTPUT_DIR/matmul

# Level 3: Data Movement
pytest test/python/golden/ttir_ops/data_movement/ -v --save-artifacts --path=$OUTPUT_DIR/data_movement

# Level 4: Reduction
pytest test/python/golden/ttir_ops/reduction/ -v --save-artifacts --path=$OUTPUT_DIR/reduction

# Level 5: Convolution
pytest test/python/golden/ttir_ops/convolution/ -v --save-artifacts --path=$OUTPUT_DIR/convolution

# Level 6: Pooling
pytest test/python/golden/ttir_ops/pooling/ -v --save-artifacts --path=$OUTPUT_DIR/pooling

# Level 7: Normalization
pytest test/python/golden/ttir_ops/normalization/ -v --save-artifacts --path=$OUTPUT_DIR/normalization

# Level 8: Fusion
pytest test/python/golden/ttir_ops/fusing/ -v --save-artifacts --path=$OUTPUT_DIR/fusing

echo "All artifacts saved to: $OUTPUT_DIR"
```

---

## TTNN-JIT Smoke Tests (Fast Validation)

For quick validation, use smoke tests:

```bash
# Run all smoke tests (fast, covers many ops)
pytest test/ttnn-jit/test_eltwise_smoketest.py -v
pytest test/ttnn-jit/test_eltwise_composite_smoketest.py -v
pytest test/ttnn-jit/test_matmul_smoketest.py -v
pytest test/ttnn-jit/test_reduction_smoketest.py -v

# Or run all smoke tests at once
pytest test/ttnn-jit/*smoketest*.py -v
```

---

## Analyzing Results with Chisel

After running tests with `--save-artifacts`, the flatbuffers will be saved with embedded TTNN MLIR:

```bash
# Find generated flatbuffers
find $OUTPUT_DIR -name "*.ttnn"

# Run chisel on a flatbuffer
cd runtime/tools/chisel
python -m chisel <path_to_flatbuffer.ttnn>

# Or if chisel is installed
chisel <path_to_flatbuffer.ttnn>
```

---

## Op Coverage Checklist

Use this checklist to track chisel testing progress:

### Basic Operations
- [ ] Unary eltwise (exp, log, sqrt, abs, neg, sin, cos, tanh, sigmoid, relu)
- [ ] Binary eltwise (add, sub, mul, div, maximum, minimum)
- [ ] Ternary eltwise (where, select)

### Core ML Operations
- [ ] MatMul (2D, batch, transpose variants)
- [ ] Convolution (2D, stride, padding, dilation, groups)
- [ ] Transpose Convolution

### Data Movement
- [ ] Transpose
- [ ] Reshape
- [ ] Concat
- [ ] Slice
- [ ] Squeeze/Unsqueeze
- [ ] Broadcast

### Aggregations
- [ ] Reduction (sum, mean, max, min, argmax, argmin)
- [ ] Pooling (MaxPool, AvgPool, AdaptiveAvgPool)

### Normalization
- [ ] LayerNorm
- [ ] RMSNorm
- [ ] BatchNorm
- [ ] GroupNorm

### Fusion Patterns
- [ ] Mish fusion
- [ ] RoPE fusion
- [ ] SDPA fusion
- [ ] Permute-MatMul fusion
- [ ] Reshape-Broadcast-Reshape fusion

---

## Tips for Testing

1. **Start Simple**: Begin with unary eltwise operations
2. **Save Artifacts**: Always use `--save-artifacts` to get flatbuffers
3. **Use -k flag**: Filter specific operations with `-k <pattern>`
4. **Check TTNN MLIR**: Verify `TT_INJECT_TTNN2FB=1` is set
5. **One at a time**: Test one operation category at a time initially
6. **Check failures**: Use `-x` flag to stop on first failure for debugging

---

## Common Issues

### Issue: Tests fail to find device
**Solution**:
```bash
ttrt query --save-artifacts
export SYSTEM_DESC_PATH=$(pwd)/ttrt-artifacts/system_desc.ttsys
```

### Issue: Flatbuffers don't have TTNN MLIR
**Solution**:
```bash
export TT_INJECT_TTNN2FB=1
```

### Issue: Tests are slow
**Solution**: Use smoke tests or filter with `-k`:
```bash
pytest test/ttnn-jit/test_eltwise_smoketest.py -v
# or
pytest test/python/golden/ttir_ops/eltwise/test_ttir_unary.py -v -k "exp or sqrt"
```

---

## Date
Created: 2026-02-15
