from torch_mlir.dynamo import make_simple_dynamo_backend
import torch
import torch._dynamo as dynamo
import torch_mlir
from torchvision.models import resnet50
resnet18 = resnet50
#import transformers.models as trans_models

#output_type = "linalg-on-tensors"
output_type = "tosa"
#output_type = "stablehlo"

torch._dynamo.reset()

@make_simple_dynamo_backend
def compile(gm, example_inputs):
    mlir_module = torch_mlir.compile(gm, example_inputs, output_type=output_type)
    print(mlir_module)
    return gm.forward


resnet18 = resnet18(pretrained=True)
resnet18.eval()
resnet18 = resnet18.to(memory_format=torch.channels_last)
x = torch.randn(1, 3, 224, 224)
x = x.to(memory_format=torch.channels_last)
module = torch_mlir.compile(resnet18, x, output_type=output_type)
print(module)
