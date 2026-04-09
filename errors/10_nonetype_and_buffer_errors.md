# NoneType and Buffer Errors (17 failures)

## Error 1: NoneType (14 failures)
```
'NoneType' object has no attribute 'get_data_buffer'
```
**Representative test**: `test_sdpa_decode_in_models[ttnn-qwen_3_4b_decode]`

### Root Cause
In the `golden()` callback at `tools/builder/base/builder_runtime.py:480`, `op_output_tensor_map[device_id]` is `None`. The C++ `getOpOutputTensor()` (`runtime/lib/ttnn/runtime.cpp:959-1000`) produces a map entry that is `None`, likely because the SDPA decode op's output tensor is in a degenerate state when the post-op callback fires.

### Fix
Add guard at `builder_runtime.py:479-480`:
```python
if op_output_tensor is None:
    continue
```
Same guard needed in `tools/ttrt/common/callback.py:145-146`.

---

## Error 2: Buffer Length (3 failures)
```
both buffer length (0) and count (-1) must not be 0
```
**Representative test**: `test_reshape[ttnn-ui8-0x32x128-0]`

### Root Cause
`torch.frombuffer()` at `builder_runtime.py:482` is called with a zero-length buffer from tensors with a zero dimension (shape `0x32x128`). The golden callback has no zero-length check, unlike the output comparison path at lines 844-850 which correctly handles `if len(data_buffer) == 0`.

### Fix
Add before `torch.frombuffer` at line 482:
```python
if len(rt_buffer) == 0:
    output_tensor_torch = torch.empty(0, dtype=golden_tensor_torch.dtype)
```
Same fix needed in `tools/ttrt/common/callback.py:150`.

## Key Files
- `tools/builder/base/builder_runtime.py` lines 478-486
- `tools/ttrt/common/callback.py` lines 145-150
- `runtime/lib/ttnn/runtime.cpp` lines 958-1000
