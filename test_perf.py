import ttnn
import ttnn_jit
import torch

import pytest
# from tracy import Profiler

GRID=(0, 0)
H = 1024
W = 1024



def trace_op(input_tensor, bias, scale):
    x1 = ttnn.exp(input_tensor)
    x2 = ttnn.abs(x1)
    x3 = ttnn.exp(x2)
    x4 = ttnn.add(x3, bias)
    x5 = ttnn.log(x4)
    x6 = ttnn.multiply(x5, scale)
    x7 = ttnn.sin(x6)
    x8 = ttnn.log(x7)
    x9 = ttnn.add(x8, bias)
    out = ttnn.divide(x9, scale)
    return out

def test_trace():
    # from tracy import Profiler, signpost
    jit_run = True

    device = ttnn.open_device(device_id=0, trace_region_size=100000)
    # profiler = Profiler()
    # profiler.enable()

    h, w = 1024, 2048
    dtype = torch.bfloat16
    max_grid = (7, 7)
    torch_tensor = torch.randn((h, w), dtype=dtype)
    torch_bias = torch.full((h, w), 1.5, dtype=dtype)
    torch_scale = torch.full((h, w), 9.9, dtype=dtype)

    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(max_grid[0], max_grid[1]))
    core_range_set = ttnn.CoreRangeSet([core_range])
    tensor_spec = ttnn.TensorSpec(
        shape=(h, w), 
        dtype=ttnn.DataType.BFLOAT16, 
        layout=ttnn.TILE_LAYOUT, 
        buffer_type=ttnn.BufferType.L1
    ).block_sharded(core_range_set)

    host_tensor = ttnn.from_torch(
        torch_tensor,
        spec=tensor_spec,
    )
    bias = ttnn.from_torch(
        torch_bias,
        spec=tensor_spec,
    )
    scale = ttnn.from_torch(
        torch_scale,
        spec=tensor_spec,
    )

    input_tensor_1 = ttnn.allocate_tensor_on_device(tensor_spec, device)
    bias_tensor = ttnn.allocate_tensor_on_device(tensor_spec, device)
    scale_tensor = ttnn.allocate_tensor_on_device(tensor_spec, device)
    
    if jit_run:
        op_jit = ttnn_jit.jit(backend="ttnn", max_grid=max_grid, debug=False)(trace_op)
    # abs_jit = ttnn_jit.jit(backend="ttnn", max_grid=max_grid, debug=True)(abs)
    
    # Warmup caches
    ttnn.copy_host_to_device_tensor(host_tensor, input_tensor_1)
    ttnn.copy_host_to_device_tensor(bias, bias_tensor)
    ttnn.copy_host_to_device_tensor(scale, scale_tensor)
    if jit_run:
        output_tensor = op_jit(input_tensor_1, bias_tensor, scale_tensor)
    else:
        output_tensor = trace_op(input_tensor_1, bias_tensor, scale_tensor)
    # output_tensor = trace_op(input_tensor_1, 1.5, 9.9)
    ttnn.synchronize_device(device)

    # Capture trace
    ttnn.copy_host_to_device_tensor(host_tensor, input_tensor_1)
    ttnn.copy_host_to_device_tensor(bias, bias_tensor)
    ttnn.copy_host_to_device_tensor(scale, scale_tensor)
    tid = ttnn.begin_trace_capture(device)
    if jit_run:
        output_tensor = op_jit(input_tensor_1, bias_tensor, scale_tensor)
    else:
        output_tensor = trace_op(input_tensor_1, bias_tensor, scale_tensor)
    # output_tensor = trace_op(input_tensor_1, 1.5, 9.9)
    ttnn.end_trace_capture(device, tid)

    # Execute trace
    # ttnn.synchronize_device(device)
    ttnn.copy_host_to_device_tensor(host_tensor, input_tensor_1)
    ttnn.copy_host_to_device_tensor(bias, bias_tensor)
    ttnn.copy_host_to_device_tensor(scale, scale_tensor)
    ttnn.execute_trace(device, tid, blocking=False)
    ttnn.synchronize_device(device)

    ttnn.release_trace(device, tid)

    # profiler.disable()
    ttnn.close_device(device)
