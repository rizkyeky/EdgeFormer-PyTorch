import torch
from ultralytics import YOLO

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = 224

    model = YOLO('pretrained/yolov8m.pt')
    model.to(device)
    model.export(format='torchscript', imgsz=IMG_SIZE, nms=True)
