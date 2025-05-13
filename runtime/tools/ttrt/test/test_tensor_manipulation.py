import ttrt
import ttmlir
import torch
import pdb

from ttrt.common.api import API as RtApi
from ttrt.common.util import Logger as RtLogger
from ttrt.common.util import Artifacts as RtArtifacts
from ttrt.runtime import (
    DebugHooks,
    get_op_output_tensor_ref,
    get_tensor,
)
from ttir2torch import DTYPE_TO_TORCH_DTYPE

flatbuffer_path = "/localdev/ndrakulic/chisel/linear/fb.ttnn"
output_dir = "./dump"
args = {
    "binary": str(flatbuffer_path),
    "save-artifacts": True,
}
rt_logger = RtLogger()
rt_artifacts = RtArtifacts(
    logger=rt_logger, artifacts_folder_path=str(output_dir)
)
RtApi.initialize_apis()
rt_api = RtApi.Run(
    args=args, logger=rt_logger, artifacts=rt_artifacts
)

def print_tensor(tensor):
    pdb.set_trace()
    rt_data_ptr = tensor.get_data_buffer()
    rt_shape = tensor.get_shape()
    rt_dtype = tensor.get_dtype()

    torch_tensor = torch.frombuffer(rt_data_ptr, dtype=DTYPE_TO_TORCH_DTYPE[rt_dtype])
    torch_tensor = torch_tensor.reshape(rt_shape)
    print(torch_tensor)
    print(torch_tensor.shape)
    print(torch_tensor.dtype)
    print(torch_tensor.device)
    print(torch_tensor.size())


    
    

def preop(binary, programContext, opContext):
    
    pass

def postop(binary, programContext, opContext):
    tensor_ref = get_op_output_tensor_ref(opContext, programContext)
    tensor = get_tensor(programContext, tensor_ref)
    print_tensor(tensor)
    print(tensor)
    pass

callback_env_pre = DebugHooks.get(preop, postop)

result_code, results = rt_api()