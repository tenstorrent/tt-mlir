# Option 6: Enhanced with Explicit API (Best of Both Worlds)

## Overview

The enhanced Option 6 implementation supports **BOTH** API styles:
- ✅ **Explicit API**: `builder.ttir.sigmoid(x)` - Like Option 1, crystal clear
- ✅ **Implicit API**: `builder.sigmoid(x)` - Automatic delegation, convenient
- ✅ **Shared State**: No synchronization overhead, all benefits of Option 6

## Implementation

```python
class MultiDialectBuilder(Builder):
    def __init__(self, ctx, location, dialects=["ttir", "ttnn"], ...):
        super().__init__(ctx, location, ...)

        # Create dialect builders with shared state
        self._dialect_builders = {}
        for dialect in dialects:
            builder = dialect_cls.__new__(dialect_cls)
            builder.__dict__ = self.__dict__  # Share ALL state
            self._dialect_builders[dialect] = builder

    # EXPLICIT API: Property accessors for clarity
    @property
    def ttir(self) -> TTIRBuilder:
        """Access TTIR dialect builder explicitly."""
        return self._dialect_builders["ttir"]

    @property
    def ttnn(self) -> TTNNBuilder:
        """Access TTNN dialect builder explicitly."""
        return self._dialect_builders["ttnn"]

    @property
    def stablehlo(self) -> StableHLOBuilder:
        """Access StableHLO dialect builder explicitly."""
        return self._dialect_builders["stablehlo"]

    # IMPLICIT API: Automatic delegation for convenience
    def __getattr__(self, name):
        """Automatically find the right dialect."""
        for builder in self._dialect_builders.values():
            if hasattr(builder, name):
                return getattr(builder, name)
        raise AttributeError(f"No builder has method {name}")
```

## Usage Examples

### Style 1: Explicit API (Maximum Clarity)

```python
builder = MultiDialectBuilder(ctx, loc, dialects=["ttir", "ttnn"])

@builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
def forward(in0, in1, builder):
    # Crystal clear which dialect each operation uses
    x = builder.ttir.sigmoid(in0)     # ✅ Explicitly TTIR
    y = builder.ttir.relu(in1)        # ✅ Explicitly TTIR
    z = builder.ttnn.multiply(x, y)   # ✅ Explicitly TTNN
    return z
```

**Benefits:**
- ✅ No ambiguity about which dialect is used
- ✅ Easy to understand at a glance
- ✅ Good for code review
- ✅ Self-documenting code

### Style 2: Implicit API (Maximum Convenience)

```python
builder = MultiDialectBuilder(ctx, loc, dialects=["ttir", "ttnn"])

@builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
def forward(in0, in1, builder):
    # Automatic dialect resolution
    x = builder.sigmoid(in0)      # Automatically finds ttir.sigmoid
    y = builder.relu(in1)         # Automatically finds ttir.relu
    z = builder.multiply(x, y)    # Automatically resolved
    return z
```

**Benefits:**
- ✅ Shorter code
- ✅ Less typing
- ✅ Works when you don't care about specific dialect
- ✅ Convenient for prototyping

### Style 3: Mixed API (Best of Both Worlds)

```python
builder = MultiDialectBuilder(ctx, loc, dialects=["ttir", "ttnn"])

@builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
def forward(in0, in1, builder):
    # Use EXPLICIT when clarity matters
    x = builder.ttir.sigmoid(in0)     # ✅ Clear: TTIR sigmoid

    # Use IMPLICIT when there's no ambiguity
    y = builder.relu(in1)             # OK: Only one relu available

    # Use EXPLICIT when semantic differences matter
    z = builder.ttnn.multiply(x, y)   # ✅ Clear: TTNN-specific semantics

    return z
```

**Benefits:**
- ✅ Explicit where it matters
- ✅ Concise where it doesn't
- ✅ Flexible and pragmatic
- ✅ Best developer experience

## Comparison with Options 1 and 6

### Option 1 (Composite - Separate State)

```python
class MultiDialectBuilder(Builder):
    def __init__(self, ctx, location, ...):
        super().__init__(ctx, location, ...)
        self.ttir = TTIRBuilder(ctx, location, ...)  # Separate instance
        self.ttnn = TTNNBuilder(ctx, location, ...)  # Separate instance

# Usage: Only explicit API
x = builder.ttir.sigmoid(input)
y = builder.ttnn.add(x, input)  # ❌ KeyError! TTNN doesn't have golden for x
```

**Problems:**
- ❌ Only supports explicit API
- ❌ Separate state requires synchronization
- ❌ O(N×M) performance overhead
- ❌ Cross-dialect operations fail

### Original Option 6 (Delegation Only)

```python
class MultiDialectBuilder(Builder):
    def __init__(self, ctx, location, ...):
        super().__init__(ctx, location, ...)
        # Create builders with shared state
        ...

    def __getattr__(self, name):
        # Only implicit delegation
        ...

# Usage: Only implicit API
x = builder.sigmoid(input)  # Works, but which dialect?
y = builder.add(x, input)   # Works, but unclear
```

**Issues:**
- ⚠️ Only supports implicit API
- ⚠️ Unclear which dialect is used
- ⚠️ Can be ambiguous for code review

### Enhanced Option 6 (This Implementation)

```python
class MultiDialectBuilder(Builder):
    def __init__(self, ctx, location, ...):
        super().__init__(ctx, location, ...)
        # Create builders with shared state
        ...

    @property
    def ttir(self):
        return self._dialect_builders["ttir"]

    def __getattr__(self, name):
        # Implicit delegation as fallback
        ...

# Usage: BOTH APIs supported!
x = builder.ttir.sigmoid(input)  # ✅ Explicit: Clear
y = builder.add(x, input)        # ✅ Implicit: Convenient
z = builder.ttnn.multiply(x, y)  # ✅ Explicit: Clear
```

**Benefits:**
- ✅ Supports explicit API (like Option 1)
- ✅ Supports implicit API (for convenience)
- ✅ Shared state (no synchronization)
- ✅ Zero performance overhead
- ✅ Best of all worlds!

## When to Use Each API Style

### Use Explicit API When:

1. **Multiple dialects have the same operation**
   ```python
   # Both TTIR and TTNN might have 'add'
   result = builder.ttir.add(x, y)    # Clear: using TTIR add
   ```

2. **Semantic differences matter**
   ```python
   # TTNN ops might have different semantics than TTIR
   result = builder.ttnn.matmul(x, y)  # Clear: TTNN-specific behavior
   ```

3. **Code clarity is paramount**
   ```python
   # In critical sections, be explicit
   x = builder.ttir.sigmoid(input)
   y = builder.ttir.relu(x)
   z = builder.ttir.matmul(y, weights)
   ```

4. **Code review and maintenance**
   ```python
   # Makes it obvious to reviewers which dialect is used
   output = builder.ttnn.some_specialized_op(input)
   ```

### Use Implicit API When:

1. **Operation is unambiguous**
   ```python
   # Only one dialect has this op
   result = builder.unique_op(input)
   ```

2. **Quick prototyping**
   ```python
   # Don't care about specific dialect yet
   x = builder.sigmoid(input)
   y = builder.add(x, input)
   ```

3. **Code brevity matters**
   ```python
   # Shorter, less verbose
   return builder.relu(builder.add(builder.matmul(x, w), b))
   ```

4. **Don't care about dialect**
   ```python
   # Common ops that work the same everywhere
   normalized = builder.divide(x, builder.sum(x))
   ```

## Addressing API Clarity Concerns

### Concern: "Implicit API is unclear"

**Solution:** Use explicit API when clarity matters!

```python
# Instead of:
result = builder.add(x, y)  # Which add?

# Write:
result = builder.ttir.add(x, y)  # Crystal clear! ✅
```

### Concern: "Hard to know which dialect is being used"

**Solution:** Combine with introspection!

```python
# Check which dialect provides a method
dialect = builder.get_method_dialect('sigmoid')
print(f"sigmoid comes from: {dialect}")  # Output: "ttir"

# Use explicit API based on that
result = getattr(builder, dialect).sigmoid(input)
```

### Concern: "Method name conflicts"

**Solution:** Explicit API disambiguates!

```python
# If both dialects have 'add':
ttir_result = builder.ttir.add(x, y)    # TTIR add
ttnn_result = builder.ttnn.add(x, y)    # TTNN add
```

## Best Practices

### 1. Default to Explicit for Production Code

```python
# Production code: Be explicit
@builder.func([...], [...])
def production_model(inputs, builder):
    x = builder.ttir.conv2d(inputs[0], inputs[1])
    x = builder.ttir.relu(x)
    x = builder.ttnn.specialized_op(x)
    return x
```

### 2. Use Implicit for Prototypes

```python
# Prototype code: Use implicit for speed
@builder.func([...], [...])
def quick_test(inputs, builder):
    x = builder.conv2d(inputs[0], inputs[1])
    x = builder.relu(x)
    return x
```

### 3. Mix Both Styles Pragmatically

```python
# Real code: Mix based on needs
@builder.func([...], [...])
def pragmatic_model(inputs, builder):
    # Explicit when it matters
    x = builder.ttir.conv2d(inputs[0], inputs[1])

    # Implicit when obvious
    x = builder.relu(x)
    x = builder.add(x, inputs[2])

    # Explicit for specialized ops
    output = builder.ttnn.specialized_fusion(x)
    return output
```

### 4. Document Your Conventions

```python
"""
Module conventions:
- Use explicit API (builder.dialect.op()) for:
  * Ops with semantic differences across dialects
  * Critical operations in production code
  * When multiple dialects provide the same op

- Use implicit API (builder.op()) for:
  * Common operations with no ambiguity
  * Prototyping and testing
  * When dialect doesn't matter
"""
```

## Performance Comparison

| Feature | Option 1 | Original Option 6 | Enhanced Option 6 |
|---------|----------|-------------------|-------------------|
| Explicit API | ✅ Yes | ❌ No | ✅ Yes |
| Implicit API | ❌ No | ✅ Yes | ✅ Yes |
| State sharing | ❌ Manual sync | ✅ Automatic | ✅ Automatic |
| Overhead | ❌ O(N×M) | ✅ O(1) | ✅ O(1) |
| API clarity | ✅ Always clear | ⚠️ Can be unclear | ✅ Clear when needed |
| Convenience | ⚠️ More typing | ✅ Less typing | ✅ Both options |
| Flexibility | ❌ One style | ⚠️ One style | ✅ Two styles |

**Winner:** Enhanced Option 6 ✅

## Code Example: Full Implementation

```python
# Create builder with multiple dialects
builder = MultiDialectBuilder(
    ctx,
    loc,
    dialects=["ttir", "ttnn", "stablehlo"]
)

# Build a complex model using both API styles
@builder.func([(128, 784), (784, 256), (256, 10)],
              [torch.float32, torch.float32, torch.float32])
def neural_network(x, w1, w2, builder):
    # Layer 1: Explicit TTIR ops
    h1 = builder.ttir.matmul(x, w1)
    h1 = builder.ttir.relu(h1)

    # Layer 2: Mix explicit and implicit
    h2 = builder.matmul(h1, w2)              # Implicit: only one matmul available
    h2 = builder.ttnn.batch_norm(h2)         # Explicit: TTNN-specific op

    # Output: Explicit for clarity
    output = builder.ttir.softmax(h2)
    return output
```

## Conclusion

Enhanced Option 6 provides **the best of both worlds**:

1. ✅ **Explicit API** for clarity (addresses your concern!)
2. ✅ **Implicit API** for convenience
3. ✅ **Shared state** - no synchronization overhead
4. ✅ **Flexible** - use the right style for each situation
5. ✅ **Simple** - same implementation as Option 6, just add property accessors

**Recommendation:** This is the ideal solution for multi-dialect builder support.

## Migration from Option 1

If you're currently thinking about Option 1, Enhanced Option 6 gives you the same explicit API style with none of the synchronization problems:

```python
# Option 1 code (with sync problems):
x = builder.ttir.sigmoid(input)
builder._sync_state()  # ❌ Required!
y = builder.ttnn.add(x, input)
builder._sync_state()  # ❌ Required!

# Enhanced Option 6 code (no sync needed):
x = builder.ttir.sigmoid(input)  # ✅ Works!
y = builder.ttnn.add(x, input)   # ✅ Works!
# No sync needed - state is shared automatically!
```

The API looks the same, but Enhanced Option 6 actually works correctly! ✅
