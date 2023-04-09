import torch
from ultralytics import YOLO

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = 224

    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    model.to(device)
    model.eval()
    model_scripted = torch.jit.script(model)
    model_scripted.save('pretrained/detr.torchscript')
