import ttrt
import ttmlir
import torch
import pdb

from ttrt.common.api import API as RtApi
from ttrt.common.util import Logger as RtLogger
from ttrt.common.util import Artifacts as RtArtifacts
from ttrt.runtime import (
    DebugHooks,
    get_op_debug_str,
    get_op_output_tensor_ref,
    get_op_input_tensor_refs,
    get_tensor,
    update_tensor,
    memcpy,
    create_tensor,
)
from ttir2torch import DTYPE_TO_TORCH_DTYPE



def get_torch_tensor(tensor):
    rt_data_ptr = tensor.get_data_buffer()
    rt_dtype = tensor.get_dtype()
    dtype = DTYPE_TO_TORCH_DTYPE[rt_dtype]
    shape = tensor.get_shape()
    torch_tensor = torch.frombuffer(rt_data_ptr, dtype=dtype)
    torch_tensor = torch_tensor.reshape(shape)

    return torch_tensor

def update_device_tensor(program_context, tensor_ref, dst_tensor, src_tensor):
    data_ptr = src_tensor.data_ptr()
    shape = dst_tensor.get_shape()
    stride = dst_tensor.get_stride()
    dtype = dst_tensor.get_dtype()
    size = torch.numel(src_tensor)
    tensor = create_tensor(data_ptr, shape, stride, size, dtype)
    update_tensor(program_context, tensor_ref, tensor)

check_in_tensors = {
    0: torch.ones([10, 10]),
    1: torch.ones([10, 10]),
    2: torch.ones([10, 10]),
    3: torch.ones([10, 10]),
    4: torch.ones([10, 10]) * 10,
    5: torch.ones([10]),
    6: torch.ones([10, 10]) * 20,
    7: torch.ones([10]),
}

update_in_tensors = {
    4: torch.ones([10, 10]) * 20
}

in_counter = 0
def preop(binary, programContext, opContext):    
    global in_counter
    tensor_refs = get_op_input_tensor_refs(opContext, programContext)

    for ref in tensor_refs:
        tensor = get_tensor(programContext, ref)
        torch_tensor = get_torch_tensor(tensor)
        if in_counter in check_in_tensors:
            assert torch.all(torch_tensor == check_in_tensors[in_counter])
        if in_counter in update_in_tensors:
            update_device_tensor(programContext, ref, tensor, update_in_tensors[in_counter])
        in_counter += 1

check_out_tensors = {
    0: torch.ones([10, 10]) * 10,
    1: torch.ones([10, 10]) * 21
}
out_counter = 0
def postop(binary, programContext, opContext):
    global out_counter
    tensor_ref = get_op_output_tensor_ref(opContext, programContext)
    if tensor_ref is None:
        return
    tensor = get_tensor(programContext, tensor_ref)

    torch_tensor = get_torch_tensor(tensor)
    print(torch_tensor)
    if out_counter in check_out_tensors:
        assert torch.all(torch_tensor == check_out_tensors[out_counter])
    out_counter += 1
    
flatbuffer_path = "/localdev/ndrakulic/chisel/linear/fb.ttnn"
output_dir = "./dump"
args = {
    "binary": str(flatbuffer_path),
    "save-artifacts": True,
    "--init": "ones",
}
rt_logger = RtLogger()
rt_artifacts = RtArtifacts(
    logger=rt_logger, artifacts_folder_path=str(output_dir)
)
RtApi.initialize_apis()
rt_api = RtApi.Run(
    args=args, logger=rt_logger, artifacts=rt_artifacts
)

callback_env_pre = DebugHooks.get(preop, postop)
result_code, results = rt_api()