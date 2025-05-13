import ttmlir
import torch
from ttrt.runtime import DataType

ttir_dtype_maps = {
    "i32": torch.int32,
    "i64": torch.int64,
    "f32": torch.float32,
    "f64": torch.float64,
    "i1": torch.bool,
    "bf16": torch.bfloat16,
    "f16": torch.float16,
}

DTYPE_TO_TORCH_DTYPE = {
    DataType.Float32: torch.float32,
    DataType.Float16: torch.float16,
    DataType.BFloat16: torch.bfloat16,
    # DataType.BFP_Float8 
    # DataType.BFP_BFloat8
    # DataType.BFP_Float4
    # DataType.BFP_BFloat4
    # DataType.BFP_Float2
    # DataType.BFP_BFloat2
    DataType.UInt32: torch.uint32,
    DataType.UInt16: torch.uint16,
    DataType.UInt8: torch.uint8,
    DataType.Int32: torch.int32,
}
