import json
import time

import torch
import torchvision.transforms as transforms
from torchvision import models

from PIL import Image

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # model = models.quantization.
    model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')

    print(model)

    # model_scripted = torch.jit.script(model) # Export to TorchScript
    # model_scripted.save('model_scripted.pt') # Save
    # model = torch.jit.load('pretrained/edgeformer-cls_scripted.pt') # Load
    # model = torch.jit.load('pretrained/yolov8n-cls.torchscript') # Load

    model.to(device)
    model.eval()

    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=3),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     [0.485, 0.456, 0.406],
        #     [0.229, 0.224, 0.225]
        # )
    ])


    image = Image.open('images_test/basket.jpg')
    tensor = data_transforms(image).unsqueeze(0)
    tensor.to(device)

    start = time.time()

    outputs = model(tensor)
    index, preds = torch.max(outputs, dim=1)
    print(index, preds)
    
    with open('labels/imagenet_classes.json') as f:
        classes = json.load(f)
        classes = [v[1] for k, v in classes.items()]
    print(classes[preds.item()])

    end = time.time()
    print(str(round(end - start, 5)) + ' seconds')
    
    
    