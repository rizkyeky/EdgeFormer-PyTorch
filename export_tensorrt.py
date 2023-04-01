import torch
import torch.nn as nn
import torch_tensorrt
from torchvision import models

# model = models.mobilenet_v3_small(weights="IMAGENET1K_V1").cuda().half().eval()
model = models.detection.ssdlite320_mobilenet_v3_large(weights='DEFAULT').cuda().half().eval()
input_data = torch.randn((1,3,224,224)).cuda().half()

result = model(input_data)
# print(result.shape)

# model_traced = torch.jit.trace(model, input_data)

inputs = [
    torch_tensorrt.Input(
        min_shape=[1, 1, 64, 64],
        opt_shape=[1, 2, 128, 128],
        max_shape=[1, 3, 256, 256],
        dtype=torch.half,
    )
]
enabled_precisions = {torch.float, torch.half}  # Run with fp16

trt_ts_module = torch_tensorrt.compile(
    model, inputs=inputs, enabled_precisions=enabled_precisions
)
input_data = input_data.to(torch.device("cuda")).half()
result = trt_ts_module(input_data)
torch.jit.save(trt_ts_module, "pretrained/ssdlite320_mobilenet_v3_large.trt")