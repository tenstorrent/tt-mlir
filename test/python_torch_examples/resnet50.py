from typing import List
from torch_mlir.dynamo import make_simple_dynamo_backend
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import make_boxed_compiler, get_aot_graph_name, set_model_name
import torch
import torch._dynamo as dynamo
import torch_mlir
from torchvision.models import resnet18
resnet50 = resnet18
#import transformers.models as trans_models

output_type = "linalg-on-tensors"
#output_type = "tosa"
#output_type = "stablehlo"

torch._dynamo.reset()

@make_simple_dynamo_backend
def compile(gm, example_inputs):
    mlir_module = torch_mlir.compile(gm, example_inputs, output_type=output_type)
    print(mlir_module)
    return gm.forward

@make_boxed_compiler
def fx_import_aot_autograd_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(gm.print_readable(False), flush=True)
    m = torch_mlir.compile(gm, example_inputs, output_type=output_type)
    print(m, flush=True)
    return gm

def test():
    fx_import_backend = aot_autograd(fw_compiler=fx_import_aot_autograd_backend)
    set_model_name("basic_forward")
    @torch._dynamo.optimize(backend=fx_import_backend)
    def basic_forward(x):
        return torch.tanh(x)

    basic_forward(torch.randn(3, 4))

resnet50 = resnet50(pretrained=True)
resnet50.eval()
resnet50 = resnet50.to(memory_format=torch.channels_last)
#module = torch.compile(resnet50, backend=fx_import_aot_autograd_backend)
fx_import_backend = aot_autograd(fw_compiler=fx_import_aot_autograd_backend)
module = torch._dynamo.optimize(backend=fx_import_backend)(resnet50)

x = torch.randn(1, 3, 224, 224)
x = x.to(memory_format=torch.channels_last)
module(x)
