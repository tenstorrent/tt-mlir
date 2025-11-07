# TT-Boltz Tenstorrent-Specific Code JIT Analysis

## Overview
This document analyzes the Tenstorrent-specific implementation in `tt-boltz/src/boltz/model/modules/tenstorrent.py` to identify functions and operations that could benefit from JIT compilation using `ttnn-jit`.

## Current Implementation Characteristics

### Architecture
- All operations use `ttnn` operations (already compatible with JIT)
- Operations work directly with `ttnn.Tensor` objects
- Complex modules with nested loops and conditional logic
- Heavy use of chunking for memory management
- Dynamic program configs based on device type (Wormhole B0 vs Blackhole)

### Key Patterns
1. **Chunked Processing**: Many operations process data in chunks (e.g., `TRIANGLE_MULT_CHUNK_SIZE = 32`, `TRANSITION_CHUNK_SIZE = 64`)
2. **Conditional Logic**: Device-specific configurations and branching
3. **Nested Loops**: Head-wise processing, chunk-wise processing
4. **Composite Operations**: Sequences of norm → linear → activation → linear

## JIT Compilation Opportunities

### 🔴 HIGH PRIORITY - Excellent JIT Candidates

#### 1. **Transition Block Inner Function** (`Transition.__call__`)
**Location**: Lines 495-533

**Current Implementation**:
```python
def f(x):
    x_norm = ttnn.layer_norm(x, weight=self.norm_weight, bias=self.norm_bias, ...)
    x_1 = ttnn.linear(x_norm, self.fc1_weight, activation="silu", ...)
    x_2 = ttnn.linear(x_norm, self.fc2_weight, ...)
    x = ttnn.multiply(x_1, x_2, ...)
    x = ttnn.linear(x, self.fc3_weight, ...)
    return x
```

**JIT Benefits**:
- Called repeatedly in chunked loops (line 538-549)
- Simple, well-defined operation sequence
- Perfect for caching - same operation, different data
- No dynamic control flow within the function

**JIT Implementation**:
```python
@ttnn_jit.jit(
    max_grid=(7, 7),  # or device-specific
    enable_cache=True,
    graph_capture=True,
)
def jit_transition_chunk(
    x: ttnn.Tensor,
    norm_weight: ttnn.Tensor,
    norm_bias: ttnn.Tensor,
    fc1_weight: ttnn.Tensor,
    fc2_weight: ttnn.Tensor,
    fc3_weight: ttnn.Tensor,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig,
) -> ttnn.Tensor:
    x_norm = ttnn.layer_norm(
        x,
        weight=norm_weight,
        bias=norm_bias,
        epsilon=1e-5,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    x_1 = ttnn.linear(
        x_norm,
        fc1_weight,
        activation="silu",
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )
    x_2 = ttnn.linear(
        x_norm,
        fc2_weight,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )
    x = ttnn.multiply(x_1, x_2, memory_config=ttnn.L1_MEMORY_CONFIG)
    x = ttnn.linear(
        x,
        fc3_weight,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
    )
    return x
```

**Expected Impact**: High - Called in loops, benefits from caching

---

#### 2. **AdaLN (Adaptive Layer Norm)** (`AdaLN.__call__`)
**Location**: Lines 692-725

**Current Implementation**:
```python
def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor) -> ttnn.Tensor:
    # Layer norms, linear ops, sigmoid, multiply, add
    ...
```

**JIT Benefits**:
- Called in every diffusion transformer layer
- Well-defined sequence of operations
- No loops or complex conditionals
- Frequently called operation

**JIT Implementation**:
```python
@ttnn_jit.jit(
    max_grid=(7, 7),
    enable_cache=True,
    graph_capture=True,
)
def jit_adaln(
    a: ttnn.Tensor,
    s: ttnn.Tensor,
    s_norm_weight: ttnn.Tensor,
    s_scale_weight: ttnn.Tensor,
    s_scale_bias: ttnn.Tensor,
    s_bias_weight: ttnn.Tensor,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    memory_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    if not USE_FLOAT32:
        a = ttnn.clone(a, dtype=ttnn.float32, memory_config=memory_config)
        s = ttnn.clone(s, dtype=ttnn.float32, memory_config=memory_config)
    a = ttnn.layer_norm(a, epsilon=1e-5, compute_kernel_config=compute_kernel_config)
    s = ttnn.layer_norm(
        s,
        weight=s_norm_weight,
        epsilon=1e-5,
        compute_kernel_config=compute_kernel_config,
    )
    if not USE_FLOAT32:
        a = ttnn.clone(a, dtype=ttnn.bfloat16, memory_config=memory_config)
        s = ttnn.clone(s, dtype=ttnn.bfloat16, memory_config=memory_config)
    s_scale = ttnn.linear(s, s_scale_weight, bias=s_scale_bias, ...)
    s_scale = ttnn.sigmoid_accurate(s_scale)
    s_bias = ttnn.linear(s, s_bias_weight, ...)
    a = ttnn.multiply_(a, s_scale)
    a = ttnn.add_(a, s_bias)
    return a
```

**Expected Impact**: High - Core operation in diffusion transformers

---

#### 3. **Triangle Multiplication Chunk Processing**
**Location**: Lines 128-166 in `TriangleMultiplication.__call__`

**Current Implementation**:
```python
for chunk_start in range(0, W // 2, TRIANGLE_MULT_CHUNK_SIZE):
    a_chunk = ttnn.slice(...)
    a_chunk = ttnn.permute(...)
    a_chunk = ttnn.typecast(...)
    # ... similar for b_chunk
    x_chunk = ttnn.matmul(a_chunk, b_chunk, ...)
    # ... concat logic
```

**JIT Benefits**:
- Chunk processing loop - same operation on different chunks
- Could JIT the chunk processing function
- Benefits from caching when processing multiple chunks

**JIT Implementation**:
```python
@ttnn_jit.jit(
    max_grid=(7, 7),
    enable_cache=True,
    graph_capture=True,
)
def jit_triangle_mult_chunk(
    a_chunk: ttnn.Tensor,
    b_chunk: ttnn.Tensor,
    ending: bool,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    program_config: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig,
) -> ttnn.Tensor:
    a_chunk = ttnn.permute(
        a_chunk, (0, 3) + ((2, 1) if ending else (1, 2))
    )
    a_chunk = ttnn.typecast(a_chunk, ttnn.bfloat8_b)
    a_chunk = ttnn.reallocate(a_chunk)
    b_chunk = ttnn.permute(
        b_chunk, (0, 3) + ((1, 2) if ending else (2, 1))
    )
    b_chunk = ttnn.typecast(b_chunk, ttnn.bfloat8_b)
    b_chunk = ttnn.reallocate(b_chunk)
    x_chunk = ttnn.matmul(
        a_chunk,
        b_chunk,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        program_config=program_config,
    )
    x_chunk = ttnn.permute(x_chunk, (0, 2, 3, 1))
    return x_chunk
```

**Expected Impact**: Medium-High - Called in loops, but complex

---

#### 4. **Conditioned Transition Block** (`ConditionedTransitionBlock.__call__`)
**Location**: Lines 746-786

**Current Implementation**:
```python
def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor) -> ttnn.Tensor:
    a = self.adaln(a, s)  # Could JIT this too
    a_swish = ttnn.linear(...)
    # Split, silu, multiply operations
    ...
```

**JIT Benefits**:
- Called in every diffusion transformer layer
- Well-defined operation sequence
- Some conditional logic (atom_level) but manageable

**Expected Impact**: Medium-High - Frequently called

---

### 🟡 MEDIUM PRIORITY - Good JIT Candidates

#### 5. **Pair Weighted Averaging Head Loop** (`PairWeightedAveraging.__call__`)
**Location**: Lines 925-974

**Current Implementation**:
```python
for i in range(self.n_heads):
    b = ttnn.linear(z, self.z_weight[:, i : i + 1], ...)
    w = ttnn.softmax(b, ...)
    v = ttnn.linear(m, self.m_weight[:, i * self.head_dim : (i + 1) * self.head_dim], ...)
    o = ttnn.matmul(v, w, ...)
    g = ttnn.linear(m, self.g_weight[:, i * self.head_dim : (i + 1) * self.head_dim], ...)
    g = ttnn.sigmoid_accurate(g)
    o = ttnn.multiply(o, g)
    o = ttnn.linear(o, self.o_weight[i * self.head_dim : (i + 1) * self.head_dim, :], ...)
    # Accumulate
```

**JIT Benefits**:
- Loop with repeated operations
- Could JIT the per-head processing function
- Benefits from caching across heads

**Challenge**: Dynamic weight slicing - may need to pass full weights and slice inside JIT function

**Expected Impact**: Medium - Loop overhead, but operations are complex

---

#### 6. **Outer Product Mean** (`OuterProductMean.__call__`)
**Location**: Lines 993-1034

**Current Implementation**:
```python
def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
    m = ttnn.layer_norm(...)
    a = ttnn.linear(...)
    b = ttnn.linear(...)
    # Permute, reshape, matmul operations
    z = ttnn.matmul(a, b, ...)
    # Layout conversions, multiply, linear
```

**JIT Benefits**:
- Well-defined operation sequence
- Called in MSA layers
- No loops or complex conditionals

**Challenge**: Layout conversions (ROW_MAJOR ↔ TILE_LAYOUT) may complicate JIT

**Expected Impact**: Medium - Good candidate but layout conversions

---

#### 7. **Fourier Embedding in Diffusion** (`Diffusion.__call__`)
**Location**: Lines 1277-1296

**Current Implementation**:
```python
fourier = ttnn.linear(times, self.conditioner_fourier_embed_weight, ...)
fourier = ttnn.multiply(fourier, 2 * pi)
fourier = ttnn.cos(fourier)
fourier = ttnn.layer_norm(fourier, ...)
fourier = ttnn.linear(fourier, self.conditioner_fourier_single_weight, ...)
```

**JIT Benefits**:
- Simple, well-defined sequence
- Called in every diffusion step
- Pure elementwise/linear operations

**Expected Impact**: Medium - Simple but frequently called

---

### 🟢 LOW PRIORITY - Possible but Complex

#### 8. **Triangle Attention Head Loop** (`TriangleAttention.__call__`)
**Location**: Lines 265-287

**Current Implementation**:
```python
for head in range(0, q.shape[0]):
    head_q = q[head : head + 1, :, :, :]
    head_k = k[head : head + 1, :, :, :]
    head_v = v[head : head + 1, :, :, :]
    head_triangle_bias = triangle_bias[head : head + 1, :, :, :]
    head_o = ttnn.transformer.scaled_dot_product_attention(...)
    o.append(head_o)
```

**JIT Benefits**:
- Could JIT the per-head attention computation
- Benefits from caching

**Challenge**: 
- Uses `ttnn.transformer.scaled_dot_product_attention` which may not be JIT-compatible
- Complex operation with program configs

**Expected Impact**: Low-Medium - Complex operation, may not benefit much

---

#### 9. **Attention Pair Bias** (`AttentionPairBias.__call__`)
**Location**: Lines 337-477

**JIT Benefits**:
- Called frequently in attention layers
- Well-defined operations

**Challenge**:
- Complex conditional logic (atom_level vs not)
- Dynamic padding based on sequence length
- Uses `ttnn.transformer.scaled_dot_product_attention` in some paths

**Expected Impact**: Low-Medium - Complex conditionals may limit JIT benefits

---

#### 10. **Diffusion Transformer Layer** (`DiffusionTransformerLayer.__call__`)
**Location**: Lines 821-850

**JIT Benefits**:
- Called in loops (24 layers for token transformer)
- Could potentially JIT the entire layer

**Challenge**:
- Contains conditional logic (atom_level)
- Calls other modules (AdaLN, AttentionPairBias, ConditionedTransitionBlock)
- Dynamic attribute caching (`s_o`)

**Expected Impact**: Low - Too complex, better to JIT sub-components

---

## Implementation Strategy

### Phase 1: High-Impact, Low-Risk
1. **Transition Block** (`Transition.__call__` inner function)
   - Simple, well-tested
   - Clear performance benefit
   - Easy to implement

2. **AdaLN** (`AdaLN.__call__`)
   - Frequently called
   - Well-defined operations
   - Moderate complexity

### Phase 2: Medium-Impact
3. **Triangle Multiplication Chunk Processing**
   - More complex but high frequency
   - Requires careful testing

4. **Conditioned Transition Block**
   - Similar to Transition but with more logic

### Phase 3: Optimization
5. **Fourier Embedding**
   - Simple, low risk
   - Incremental improvement

6. **Outer Product Mean**
   - Good candidate if layout conversions work

## Challenges and Considerations

### 1. **Dynamic Program Configs**
Many operations use device-specific program configs:
```python
core_grid = (10, 13) if is_blackhole() else (8, 8)
program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(...)
```

**Solution**: Pass program configs as parameters to JIT functions, or create separate JIT functions for different device types.

### 2. **Memory Management**
Current code uses explicit `ttnn.deallocate()` calls:
```python
ttnn.deallocate(a_chunk)
ttnn.deallocate(b_chunk)
```

**Solution**: JIT functions should handle memory management, or deallocate outside JIT functions.

### 3. **Conditional Logic**
Some functions have device-specific or mode-specific branches:
```python
if is_blackhole():
    core_grid = ttnn.CoreGrid(y=10, x=13)
else:
    core_grid = None
```

**Solution**: JIT supports if/else, but may need separate JIT functions for different branches to maximize caching.

### 4. **Weight Slicing**
Some operations slice weights dynamically:
```python
self.z_weight[:, i : i + 1]
```

**Solution**: Pass full weights and slice inside JIT function, or create separate JIT functions for each head.

### 5. **Layout Conversions**
Operations that convert between layouts:
```python
x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
# ... operations ...
x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
```

**Solution**: Test if JIT handles layout conversions correctly.

### 6. **Experimental Operations**
Some operations use `ttnn.experimental`:
```python
ttnn.experimental.nlp_create_qkv_heads_boltz(...)
ttnn.experimental.nlp_concat_heads_boltz(...)
```

**Solution**: Verify JIT compatibility with experimental operations, may need to exclude from JIT.

## Recommended JIT Configuration

### For High-Frequency Operations
```python
@ttnn_jit.jit(
    max_grid=(7, 7),  # Or device-specific
    enable_cache=True,  # Enable for repeated calls
    graph_capture=True,  # Use graph capture compiler
    debug=False,  # Set to True for debugging
)
```

### For Chunked Operations
```python
@ttnn_jit.jit(
    max_grid=(7, 7),
    enable_cache=True,  # Important for loop iterations
    graph_capture=True,
)
```

### For Simple Operations
```python
@ttnn_jit.jit(
    max_grid=(7, 7),
    enable_cache=False,  # May not need caching for simple ops
    graph_capture=True,
)
```

## Testing Strategy

1. **Unit Tests**: Create tests for each JIT-compiled function
2. **Integration Tests**: Test within full model forward pass
3. **Performance Benchmarks**: Compare JIT vs non-JIT performance
4. **Correctness Tests**: Verify numerical equivalence
5. **Cache Tests**: Verify caching behavior

## Expected Performance Gains

### High Priority Candidates
- **Transition Block**: 20-40% speedup (frequent, cached)
- **AdaLN**: 15-30% speedup (frequent, simple)
- **Triangle Multiplication Chunk**: 10-25% speedup (looped, cached)

### Medium Priority Candidates
- **Conditioned Transition**: 10-20% speedup
- **Fourier Embedding**: 5-15% speedup
- **Outer Product Mean**: 5-15% speedup

### Overall Model Impact
- **Estimated**: 5-15% overall speedup from JIT compilation
- **Best Case**: 20-30% if all high-priority candidates are optimized
- **Worst Case**: 2-5% if only simple operations benefit

## Conclusion

The Tenstorrent-specific code has **excellent JIT compilation opportunities**, particularly:

1. ✅ **Transition blocks** - Perfect candidates (looped, simple, cached)
2. ✅ **AdaLN** - High frequency, well-defined
3. ✅ **Chunk processing functions** - Benefit from caching in loops
4. ⚠️ **Complex attention operations** - May have limited benefit due to complexity
5. ⚠️ **Operations with experimental APIs** - Need compatibility verification

**Recommendation**: Start with Phase 1 (Transition and AdaLN) to validate the approach, then expand to other candidates based on performance profiling results.

