# Quick Start: TTNN Unit Test Generation

## 🚀 Getting Started in 5 Minutes

### Prerequisites
- tt-mlir repository cloned and environment set up
- Python 3.8+ with pytest installed

### Step 1: Build tt-alchemist

```bash
# Activate environment
source env/activate

# Configure build
cmake -G Ninja -B build -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

# Build and install tt-alchemist
cmake --build build --target tt-alchemist
```

### Step 2: Create a test MLIR file

Create `test_ops.mlir`:

```mlir
module attributes {ttnn.device = #ttnn.device<0>} {
  func.func @main(%arg0: tensor<1x32x128x128xbf16>,
                  %arg1: tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16> {
    %0 = "ttnn.add"(%arg0, %arg1) : (tensor<1x32x128x128xbf16>, tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %1 = "ttnn.relu"(%0) : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    return %1 : tensor<1x32x128x128xbf16>
  }
}
```

### Step 3: Generate tests

```python
from tt_alchemist import generate_unit_tests

# Generate tests
generate_unit_tests(
    input_file="test_ops.mlir",
    output_dir="tests/",
    parametrized=True,
    verbose=True
)
```

### Step 4: Run the generated tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_ttnn_add.py -v
```

## 📝 Common Use Cases

### Generate tests for specific operations only

```python
generate_unit_tests(
    "model.mlir",
    "tests/",
    op_filter=["ttnn.add", "ttnn.matmul"],  # Only these ops
    parametrized=True
)
```

### Generate from existing TTNN model

```bash
# First, get your model in TTNN MLIR format
# (You might already have this from your compilation pipeline)

# Then generate tests
python -c "
from tt_alchemist import generate_unit_tests
generate_unit_tests('my_model_ttnn.mlir', 'model_tests/')
"
```

### Generate tests for a specific layer

If you have a large model and want to test specific layers:

```python
# Extract specific function or layer to separate MLIR file
# Then generate tests for that specific part
generate_unit_tests(
    "layer_3_conv.mlir",
    "layer_3_tests/",
    parametrized=True
)
```

## 🎯 Example Output

After generation, you'll have:

```
tests/
├── conftest.py            # Device setup
├── test_utils.py          # Helper functions
├── test_ttnn_add.py       # Tests for add operations
└── test_ttnn_relu.py      # Tests for relu operations
```

Example test file (`test_ttnn_add.py`):

```python
class TestTtnnAdd:
    @pytest.mark.parametrize("shape", [(1, 32, 128, 128)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_add_parametrized(self, shape, dtype, device):
        input0 = create_random_tensor(shape, dtype)
        input1 = create_random_tensor(shape, dtype)

        ttnn_input0 = ttnn.from_torch(input0, device=device)
        ttnn_input1 = ttnn.from_torch(input1, device=device)

        output = ttnn.add(ttnn_input0, ttnn_input1)
        expected = torch.add(input0, input1)

        actual = ttnn.to_torch(output)
        assert validate_output(actual, expected)
```

## 🔧 Command Line Usage (Future)

Once CLI integration is complete:

```bash
# Basic usage
tt-alchemist gen-tests -i model.mlir -o tests/

# With operation filter
tt-alchemist gen-tests -i model.mlir -o tests/ --ops ttnn.add,ttnn.relu

# Without parametrization
tt-alchemist gen-tests -i model.mlir -o tests/ --no-parametrize

# Verbose output
tt-alchemist gen-tests -i model.mlir -o tests/ --verbose
```

## 💡 Tips

1. **Start small**: Test the feature with a simple MLIR file first
2. **Use filtering**: For large models, generate tests incrementally using op_filter
3. **Check parametrization**: Review generated tests to ensure proper parameter coverage
4. **Custom validation**: Modify test_utils.py to add custom validation logic
5. **CI Integration**: Add generated tests to your CI pipeline

## 🐛 Troubleshooting

### Import Error
```
ImportError: cannot import name 'generate_unit_tests' from 'tt_alchemist'
```
**Solution**: Rebuild and reinstall tt-alchemist:
```bash
cmake --build build --target tt-alchemist
```

### No tests generated
**Check**:
- MLIR file contains TTNN operations (not TTIR or other dialects)
- Operations are not filtered out by op_filter
- Build completed successfully

### Tests fail with device errors
**Check**:
- Device is available and properly configured
- Environment variables are set correctly
- System descriptor is available

## 📚 More Examples

See the complete example:
```bash
python tools/tt-alchemist/examples/test_generation_example.py
```

This will walk you through all features with detailed explanations.