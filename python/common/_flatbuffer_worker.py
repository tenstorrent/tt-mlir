# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Subprocess worker for running flatbuffers on device using _ttmlir_runtime.

Executed as subprocess (not multiprocessing) to ensure clean hardware shutdown.
"""

import json
import sys


def _torch_dtype_to_runtime_dtype(dtype, tt_runtime):
    import torch

    if dtype == torch.float32:
        return tt_runtime.runtime.DataType.Float32
    if dtype == torch.float16:
        return tt_runtime.runtime.DataType.Float16
    if dtype == torch.bfloat16:
        return tt_runtime.runtime.DataType.BFloat16
    if dtype == torch.uint32:
        return tt_runtime.runtime.DataType.UInt32
    if dtype == torch.uint16:
        return tt_runtime.runtime.DataType.UInt16
    if dtype == torch.uint8:
        return tt_runtime.runtime.DataType.UInt8
    if dtype == torch.int32:
        return tt_runtime.runtime.DataType.Int32
    if dtype == torch.float64:
        return tt_runtime.runtime.DataType.Float64
    if dtype == torch.int64:
        return tt_runtime.runtime.DataType.Int64
    if dtype == torch.uint64:
        return tt_runtime.runtime.DataType.UInt64
    if dtype == torch.int16:
        return tt_runtime.runtime.DataType.Int16
    if dtype == torch.int8:
        return tt_runtime.runtime.DataType.Int8
    if dtype == torch.bool:
        return tt_runtime.runtime.DataType.Bool
    raise ValueError(f"Torch dtype: {dtype} has no runtime DataType equivalent")


def _runtime_str_dtype_to_torch_dtype(dtype_str):
    import torch

    mapping = {
        "Float32": torch.float32,
        "Float16": torch.float16,
        "BFloat16": torch.bfloat16,
        "UInt32": torch.uint32,
        "UInt16": torch.uint16,
        "UInt8": torch.uint8,
        "Int32": torch.int32,
        "Float64": torch.float64,
        "Int64": torch.int64,
        "UInt64": torch.uint64,
        "Int16": torch.int16,
        "Int8": torch.int8,
        "Bool": torch.bool,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported runtime dtype string: {dtype_str}")
    return mapping[dtype_str]


def _build_inputs(fbb, program_index, device, tt_runtime):
    import torch

    input_dict = json.loads(fbb.get_program_inputs_as_json(program_index))

    inputs = []
    for i, spec in enumerate(input_dict):
        shape = spec["desc"]["shape"]
        dtype_str = spec["desc"]["layout"]["memory_desc"]["data_type"]
        torch_dtype = _runtime_str_dtype_to_torch_dtype(dtype_str)

        if torch_dtype.is_floating_point:
            host_tensor = torch.randn(shape, dtype=torch_dtype)
        else:
            host_tensor = torch.zeros(shape, dtype=torch_dtype)

        rt_tensor = tt_runtime.runtime.create_owned_host_tensor(
            host_tensor.data_ptr(),
            list(host_tensor.shape),
            list(host_tensor.stride()),
            host_tensor.element_size(),
            _torch_dtype_to_runtime_dtype(host_tensor.dtype, tt_runtime),
        )

        layout = tt_runtime.runtime.get_layout(fbb, program_index, i)
        rt_tensor = tt_runtime.runtime.to_layout(rt_tensor, device, layout, True)
        inputs.append(rt_tensor)

    return inputs


def _run_flatbuffer(flatbuffer_path):
    """Run flatbuffer on device. Returns (return_code, error_message)."""
    import _ttmlir_runtime as tt_runtime

    fbb = tt_runtime.binary.load_binary_from_path(flatbuffer_path)
    tt_runtime.runtime.set_compatible_device_runtime(fbb)

    options = tt_runtime.runtime.MeshDeviceOptions()
    device = tt_runtime.runtime.open_mesh_device(options)

    return_code = 0
    error_message = None
    try:
        for program_index in range(fbb.get_num_programs()):
            if fbb.is_program_private(program_index):
                continue

            inputs = _build_inputs(fbb, program_index, device, tt_runtime)

            outputs = tt_runtime.runtime.submit(device, fbb, program_index, inputs)
            tt_runtime.runtime.wait(outputs)

            for out in outputs:
                tt_runtime.runtime.to_host(out, untilize=True)
                tt_runtime.runtime.deallocate_tensor(out, force=True)
    except Exception as e:
        return_code = 1
        error_message = str(e)
    finally:
        tt_runtime.runtime.close_mesh_device(device)

    return return_code, error_message


def main():
    """Run flatbuffer and write result to output file."""
    if len(sys.argv) != 3:
        print(
            "Usage: python -m python.common._flatbuffer_worker <flatbuffer_path> <result_path>"
        )
        sys.exit(1)

    flatbuffer_path = sys.argv[1]
    result_path = sys.argv[2]

    try:
        return_code, error_message = _run_flatbuffer(flatbuffer_path)

        if return_code != 0:
            with open(result_path, "w") as f:
                json.dump(
                    {
                        "status": "error",
                        "error": error_message or f"{return_code}",
                    },
                    f,
                )
        else:
            with open(result_path, "w") as f:
                json.dump({"status": "success", "return_code": return_code}, f)

    except Exception as e:
        with open(result_path, "w") as f:
            json.dump({"status": "error", "error": str(e)}, f)


if __name__ == "__main__":
    main()
