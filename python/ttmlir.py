from torch_mlir.dynamo import make_simple_dynamo_backend
import torch
import torch._dynamo as dynamo
import torch_mlir

output_type = "linalg-on-tensors"
# output_type = "tosa"
# output_type = "stablehlo"

torch._dynamo.reset()


@make_simple_dynamo_backend
def compile(gm, example_inputs):
    mlir_module = torch_mlir.compile(gm, example_inputs, output_type=output_type)
    print(mlir_module)
    return gm.forward
