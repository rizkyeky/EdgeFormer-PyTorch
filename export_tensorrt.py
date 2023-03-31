import torch
import torchvision
import torch.nn as nn
import torch_tensorrt as torchtrt
import torch_tensorrt

model = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1").cuda().half().eval()

inputs = [
    torch_tensorrt.Input(
        min_shape=[1, 1, 16, 16],
        opt_shape=[1, 1, 32, 32],
        max_shape=[1, 1, 64, 64],
        dtype=torch.half,
    )
]
enabled_precisions = {torch.float, torch.half}  # Run with fp16

trt_ts_module = torch_tensorrt.compile(
    model, inputs=inputs, enabled_precisions=enabled_precisions
)
input_data = torch.randn((1,3,224,224))
input_data = input_data.to("cuda").half()
result = trt_ts_module(input_data)
torch.jit.save(trt_ts_module, "mobilenet_v3_small.trt")