# Builder Testing

The tests for builder compilability & semantic correctness all live in
`test/python/golden`. `pytest` is used to orchestrate these tests. The basic
stages of each test are as follows:

- Compilation
    - Builder Graph -> TTIR-MLIR
    - TTIR-MLIR -> TTNN-MLIR or TTMetal-MLIR (depending on `target` parameter)
    - {TTNN,TTMetal}-MLIR -> Executable Flatbuffer

## Test Structure
All op tests utilizing builder follow a similar structure. Here we use
`reshape` (from `test/python/golden/test_ttir_ops.py`) to demonstrate:

```python
@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        # [input_shape, output_shape]
        ((128, 128), (16384,)),             # Flatten 2D to 1D
        ((24,), (2, 3, 4)),                 # Unflatten 1D to 3D
        ((2, 3, 4), (6, 4)),                # 3D to 2D reshape
        ((128, 128), (64, 256)),            # 2D to 2D different arrangement
        ((1, 1, 1), (1,)),                  # Edge case: all dimensions are 1
        ((10,), (10,)),                     # Identity reshape
        ((64, 512), (64, 1, 512)),          # Common ML pattern: expand dims
        ((256, 256), (512, 128)),           # Power of 2 reshape
        ((32, 3, 224, 224), (32, 150528)),  # Large ML pattern: batch flatten
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32], ids=["f32", "i32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_reshape(input_shape, output_shape, dtype: torch.dtype, request):

    def reshape_wrapper(in0: Operand, builder: TTIRBuilder):
        return builder.reshape(in0, output_shape)

    compile_ttir_to_flatbuffer(
        reshape_wrapper,
        [input_shape],
        [dtype],
        target=target,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
```

As seen above, each test is broadly split into 3 parts:
1. Parametrizations to generate multiple tests for a single op, testing common
   patterns and edge cases.
2. The test function (`Callable`) (in this case, it is `reshape_wrapper`) that
   defines the builder graph to be compiled
3. A call to `compile_ttir_to_flatbuffer`. This is where the test is actually
   executed and the graph compiled. The docstring for
   `compile_ttir_to_flatbuffer` explains each parameter in detail

### Parametrization Rules

> NOTE: these rules are temporary, and may be relaxed completely upon
> completion of #4518

The builder tests make heavy utilization of `pytests` parametrization features
to reuse as much code as possible when just changing the inputs to different
ops. There are some special rules to follow when defining parametrizations for tests here:
- All tests _must_ be parametrized with the input shape of an op for reporting
  to pick up the shape, and it must be named one of  `shapes`, `shape`,
  `input_shape`, or `inputs_shapes`. This requirement is to allow these shapes
  to easily be dumped to a test report and ingested by some report analysis
  tool, so the dumper must know which parameters to look for.
- If input types are being parametrized, then they must be named one of
  `dtypes`, `dtype`, `inputs_dtypes` for the same reason as above. The arity of
  this must either match the arity of the shapes, or be exactly one element,
  which signals that all input tensors are to be marked as that type.
- If the test function itself is being parametrized (see `test_unary_ops` in
  `test/python/golden/test_ttir_ops.py`), then that parameter _must_ be named
  `test_fn`
- If you want to tie two semantic parameters together (e.g. stride and padding
  in a `conv2d` op), do _not_ parametrized them as tuples and unpack them in
  the function. This blinds the test reporter from the actual correct names of
  the parameters. Instead, use `pytest.mark.parametrize`'s built in feature for
  tying multiple parameters together via a comma separated string.

    Good:
    ```python
    @pytest.mark.parametrize(
        "stride,padding,dilation,groups", [([2, 1], [2, 1], [2, 1], 2)]
    )
    ```
    Bad:
    ```python
    @pytest.mark.parametrize(
        "stride_padding_dilation_groups", [([2, 1], [2, 1], [2, 1], 2)]
    )
    ...
    stride, padding, dilation, groups = stride_padding_dilation_groups
- If a different backend to `"ttnn"` is desired, then it must be parametrized
  under the name `"target"`. This is true even of test cases that don't need to
  be over multiple targets, since the `device` fixture must be able to read the
  target from the parameters to initialize the device for execution
- All tests must contain the `pytest.mark.frontend` mark, denoting either
  `"ttir"` or `"shlo"`. The easiest way to do this is by utilizing file wide
  marks (set the `pytestmark` variable to the mark or a list of marks you want
  to apply to every test in the file)

### Marks
There are a number of custom marks provided for this test suite, most of them
having to do with skipping tests on specific hardware or with specific
parameter configurations. They are as follows:
- `x86_only`: Skips a test if the host is not running x86 hardware
- `skip_config(...)`: Given a set of target and hardware params, this will skip
  the test if the specific config is met. For example, `skip_config(["ttmetal",
  "p150"])` will skip the test that targets `ttmetal` on all `p150` machines.
  This functions over set intersection, so something like
  `skip_config(["ttmetal"])` will skip all `ttmetal` tests that this test
  function generates
- `frontend(fe)`: `fe` is one of `"ttir"` or `"shlo"` and denotes for the test
  reporter which frontend builder this test is using
