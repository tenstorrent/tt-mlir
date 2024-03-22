from torch_mlir.dynamo import make_simple_dynamo_backend
import torch
import torch._dynamo as dynamo
import torch_mlir
import torchvision.models as vis_models
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


if True:
    resnet18 = vis_models.resnet18(pretrained=True)
    resnet18.train(False)
    with torch.no_grad():
        module = torch.compile(resnet18, backend=compile)
        module(torch.randn(1, 3, 224, 224))
