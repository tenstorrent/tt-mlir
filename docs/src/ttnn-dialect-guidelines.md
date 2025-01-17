# TTNN Dialect Contribution Guidelines

This document provides clear and consistent guidelines for contributing to the TTNN dialect, including operations, attributes, types, and other components. Following these ensures a streamlined development process, faster code reviews, and higher-quality code with fewer bugs.

## General Principle: Model TTNN Library Closely

The TTNN dialect should closely reflect the TTNN library wherever practical, serving as the **core guiding principle** when contributing to the dialect. Whenever there's a need to deviate from this principle, it should be discussed with stakeholders.

## Ops and Operands

### Signature Selection

Ops in TTNN may have multiple signatures available - it's important to choose the right one when creating its model in the TTNN dialect. Going through an example, these are the available signatures for the `ttnn::transpose` op:

```C++
struct ExecuteTranspose {
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const int64_t& dim1,
        const int64_t& dim2,
        const std::optional<MemoryConfig>& memory_config_arg,
        const std::optional<float>& pad_value = 0.0f);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const int64_t& dim1,
        const int64_t& dim2,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<float>& pad_value = 0.0f);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const int64_t& dim1,
        const int64_t& dim2,
        const std::optional<float>& pad_value = 0.0f);
};
```

The first and second signature differ only in the `queue_id` parameter - we don't model queues today, so the second signature has priority here. The second and third signature differ in `memory_config` parameter - the second signature is preferred as it is more robust: the parameter is optional so it can remain unused if it isn't needed.

Only one signature should be chosen. If the need would arise for more than one signature, it would be a precedent, and should be discussed with stakeholders.

### Operand ordering

Operands in the TTNN dialect ops should match the ordering of the signature of the op being modelled. For the chosen signature of the `ttnn::transpose` op, the operands should look like this:

```mlir
let arguments = (ins AnyRankedTensor:$input,
                     SI64Attr:$dim0,
                     SI64Attr:$dim1,
                     OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                     OptionalAttr<FloatAttr>:$pad_value);
```

Mixing types and attributes within the ordering is **not** an issue, this is valid:

```
let arguments = (ins TTNN_ShapeAttr:$shape,
                     OptionalAttr<TT_DataTypeAttr>:$dtype,
                     OptionalAttr<TTNN_LayoutAttr>:$layout,
                     Optional<TT_Device>:$device,
                     OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config);
```

Following this guideline provides consistency with the TTNN lib.

### Optional operands

If an operand is optional in the TTNN lib, it should be modelled as optional in the dialect.

### Default-valued operands

If an operand has a default value in the TTNN lib, it should have a default value in the dialect.

`ttnn::permute` as an example:

```C++
static ttnn::Tensor invoke(
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const int64_t> dims,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<float>& pad_value = 0.0f);
```

```mlir
let arguments = (ins AnyRankedTensor:$input,
                     DenseI64ArrayAttr:$permutation,
                     OptionalAttr<TTNN_MemoryConfigAttr>:$memory_config,
                     DefaultValuedOptionalAttr<F32Attr, "0.0f">:$pad_value);
```

### Numerical operands

Numerical operands should match in signedness and bit width. If an operand is a signed integer of width of 32 bits, `SI32Attr` should be used to model it.

### Pointers and references

Pointers and references should be ignored. We do not want to model this level of detail at this point in time.

There were very few issues with these previously, and they were caused by inconsistencies in TTNN lib APIs.

### Attrs vs Types

General guideline is that if a value is known at compile time, it should probably be an `Attr`. Example: dims in transpose op, pooling windows in a conv, etc. If the value is unknown at compile time (e.g. tensor) it should be a `Type`.

There's another consideration to account for: does the value need its own SSA? Remember, `Attr`s need something to latch onto, like an op or a `Type`, but `Type`s need to be constructed, i.e. have their own SSA, in order to exist. Let's look at `ttnn::Shape` for example - in TTNN lib, these need to be constructed, so it naturally follows that they should have their own SSA value within the IR, implying that they should be implemented as `Type`s. However, there are several downsides to this:
- More IR is produced
- Diminished readability as they're not attached to the object whose shape they're describing
- Not as easy to construct in code
- Runtime would need to keep track of all the Shape objects (it currently maps all SSAs, which are currently only tensors and devices)

One upside for implementing `ttnn::Shape` as a `Type` is that it would enable optimizing out multiple constructor calls for the same Shape.

It is agreed that we should prefer using `Attr`s in these scenarios. However, this guideline is not set in stone - stakeholders should be notified if anyone believes there's a need to implement an object as a `Type`.

### Destination-passing style (DPS)

If the op in TTNN lib has the destination tensor, is should be modelled as DPS op.

An example signature, where the last operand is a destination tensor:

```C++
static Tensor invoke(
    const Tensor& input_tensor,
    float exponent,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);
```

### Variadic operands

`Variadic<>` type constraint should only be used for operands that are variadic in nature, e.g. a vector of tensors, like in `ttnn::concat`:

```C++
static ttnn::Tensor invoke(
    const std::vector<ttnn::Tensor>& input_tensors,
    int dim,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt,
    unsigned int groups = 1);
```

### Operand naming

Operands should be named as they are in the TTNN lib. However, this guideline is not strict, and some reasonable deviations are acceptable.

### Operand namespaces

Some operands are defined in a namespace nested within the TTNN namespace, i.e. `ttnn::ccl::Topology`, and some are in other but related namespaces, i.e. `tt::tt_metal::MemoryConfig`. While it would be ideal to model these completely accurately, it doesn’t provide value and we should pretend they’re all in the `ttnn::` namespace for the sake of simplicity.
