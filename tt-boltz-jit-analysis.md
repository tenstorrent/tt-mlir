# TT-Boltz JIT Compilation Analysis

## Overview
This document analyzes whether functions in the tt-boltz model could be JIT compiled using the `ttnn-jit` approach demonstrated in `test/ttnn-jit`.

## How ttnn-jit Works

### Basic Pattern
```python
import ttnn_jit
import ttnn

def my_operation(input_tensor):
    return ttnn.abs(ttnn.exp(input_tensor))

# JIT compile the function
op_jit = ttnn_jit.jit(
    debug=True,
    max_grid=(7, 7),  # For sharded tensors
    enable_cache=False,
    graph_capture=True,
)(my_operation)

# Use the compiled function
output = op_jit(input_tensor)
```

### Supported Operations
- **Elementwise operations**: `ttnn.abs`, `ttnn.exp`, `ttnn.log`, `ttnn.sin`, `ttnn.cos`, `ttnn.tan`, `ttnn.gelu`, `ttnn.sigmoid`, etc.
- **Binary operations**: `ttnn.add`, `ttnn.multiply`, `ttnn.subtract`, etc.
- **Control flow**: if/else statements, for loops, nested control structures
- **Memory configs**: Both L1 (sharded) and DRAM (interleaved) tensors

### Key Requirements
1. Functions must use `ttnn` operations (not PyTorch operations)
2. Functions should operate on `ttnn.Tensor` objects
3. Control flow is supported but must use ttnn operations in branches

## Current TT-Boltz Implementation

### Tenstorrent Module (`tenstorrent.py`)
The model already has a Tenstorrent-specific implementation that:
- Uses `ttnn` operations directly
- Implements complex modules (Pairformer, Diffusion, MSA)
- Converts between PyTorch and ttnn tensors via `TorchWrapper` classes

### PyTorch-Based Layers
Most model layers use standard PyTorch operations:
- `nn.Linear`, `nn.LayerNorm`
- `torch` operations (add, multiply, etc.)
- Standard PyTorch tensor operations

## Potential JIT Candidates

### 1. **Elementwise Operations in Layers**
Functions that perform simple elementwise operations could be JIT compiled if converted to use ttnn:

**Example from layers:**
```python
# Current (PyTorch)
def apply_activation(x):
    return torch.relu(x * 0.5)

# JIT-able (ttnn)
def apply_activation_ttnn(x: ttnn.Tensor):
    return ttnn.relu(ttnn.multiply(x, 0.5))
```

**Potential locations:**
- Activation functions in transition layers
- Normalization operations
- Dropout masks (if converted to ttnn)

### 2. **Composite Operations**
Functions that combine multiple ttnn operations:

**Example:**
```python
@ttnn_jit.jit(max_grid=(7, 7), enable_cache=True)
def composite_operation(x: ttnn.Tensor, y: ttnn.Tensor):
    z = ttnn.add(x, y)
    z = ttnn.layer_norm(z, ...)
    z = ttnn.multiply(z, 0.5)
    return ttnn.relu(z)
```

**Potential locations:**
- Transition blocks (normalize → linear → activation → linear)
- Attention pre/post processing
- Residual connection operations

### 3. **Repeated Operations in Loops**
Functions called repeatedly that could benefit from caching:

**Example:**
```python
@ttnn_jit.jit(max_grid=(7, 7), enable_cache=True)
def process_chunk(chunk: ttnn.Tensor):
    return ttnn.layer_norm(ttnn.linear(chunk, weight), ...)

# In a loop
for chunk in chunks:
    result = process_chunk(chunk)  # Cache hit after first call
```

**Potential locations:**
- Chunked operations in triangular attention
- Window-based processing in diffusion
- Recycling iterations

### 4. **Control Flow Operations**
Functions with conditional logic using ttnn operations:

**Example:**
```python
@ttnn_jit.jit(max_grid=(7, 7))
def conditional_operation(x: ttnn.Tensor, use_exp: bool):
    if use_exp:
        return ttnn.exp(x)
    else:
        return ttnn.log(x)
```

**Potential locations:**
- Training vs inference paths
- Conditional feature processing
- Mode-dependent operations

## Challenges and Considerations

### 1. **Tensor Conversion Overhead**
- Current model uses PyTorch tensors
- JIT requires ttnn tensors
- Conversion overhead may negate JIT benefits for small operations

### 2. **Memory Layout Compatibility**
- JIT works best with consistent memory layouts
- Model uses various layouts (L1 sharded, DRAM interleaved)
- Need to ensure compatibility

### 3. **Complex Operations**
- Many operations are already optimized in `tenstorrent.py`
- JIT may not provide additional benefit for already-optimized paths
- Best for simple, repeated operations

### 4. **Cache Management**
- JIT cache can grow large
- Need to manage cache size for production use
- Cache keys based on tensor properties (shape, dtype, memory config)

## Recommended Approach

### High Priority Candidates

1. **Simple Activation Functions**
   - Convert frequently-used activation functions to ttnn
   - JIT compile with caching enabled
   - Example: `gelu`, `silu`, `sigmoid` operations

2. **Normalization Wrappers**
   - Create ttnn-based normalization functions
   - JIT compile for repeated use
   - Example: Layer norm operations in loops

3. **Composite Operations in Hot Paths**
   - Identify frequently-called composite operations
   - Convert to ttnn and JIT compile
   - Example: Residual connection + normalization patterns

### Implementation Strategy

1. **Start Small**: Begin with simple elementwise operations
2. **Profile First**: Identify hot paths before JIT compilation
3. **A/B Testing**: Compare JIT vs non-JIT performance
4. **Gradual Migration**: Convert operations incrementally
5. **Cache Management**: Monitor and manage JIT cache size

## Example Implementation

```python
# In a new file: tt-boltz/src/boltz/model/layers/jit_ops.py
import ttnn
import ttnn_jit

# Simple activation with JIT
@ttnn_jit.jit(max_grid=(7, 7), enable_cache=True, graph_capture=True)
def jit_gelu(x: ttnn.Tensor) -> ttnn.Tensor:
    """JIT-compiled GELU activation."""
    return ttnn.gelu(x)

# Composite operation with JIT
@ttnn_jit.jit(max_grid=(7, 7), enable_cache=True, graph_capture=True)
def jit_residual_norm(
    x: ttnn.Tensor,
    residual: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
) -> ttnn.Tensor:
    """JIT-compiled residual connection with layer norm."""
    x = ttnn.add(x, residual)
    x = ttnn.layer_norm(x, weight=weight, bias=bias, epsilon=1e-5)
    return x

# Conditional operation with JIT
@ttnn_jit.jit(max_grid=(7, 7), enable_cache=True, graph_capture=True)
def jit_conditional_activation(
    x: ttnn.Tensor,
    activation_type: str = "relu",
) -> ttnn.Tensor:
    """JIT-compiled conditional activation."""
    if activation_type == "relu":
        return ttnn.relu(x)
    elif activation_type == "gelu":
        return ttnn.gelu(x)
    else:
        return ttnn.silu(x)
```

## Conclusion

**Yes, ttnn-jit can be used for tt-boltz functions**, but with considerations:

✅ **Good candidates:**
- Simple elementwise operations
- Frequently-called composite operations
- Operations in hot loops that can benefit from caching

❌ **Not ideal:**
- Already-optimized Tenstorrent implementations
- Operations with significant PyTorch dependencies
- One-time operations where JIT overhead exceeds benefit

**Recommendation**: Start with profiling to identify hot paths, then selectively apply JIT compilation to simple, frequently-called operations that use ttnn operations.

