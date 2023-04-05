import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
import main_dec
from torchvision import models
import cv2

from parcnet.cvnets.models.detection.base_detection import DetectionPredTuple

if __name__ == '__main__':
    # torch.backends.quantized.engine = 'qnnpack'

    model = main_dec.init_model()
    # model = models.detection.ssdlite320_mobilenet_v3_large(weights='DEFAULT')
    img = cv2.imread('images_test/krsbi5.jpg')
    pred_labels, pred_scores, pred_boxes = main_dec.predict_image(model, img)
    model.to(torch.device('cpu'))
    model.eval()

    # print(pred_labels)
    # print(pred_scores)
    # print(pred_boxes)

    # print("="*20, "DONE PREDICT", "="*20)
    
    dummy_input = torch.rand(1, 3, 224, 224)
    # pred: DetectionPredTuple = model(dummy_input)
    
    # print(pred[0].keys())
    model_traced = torch.jit.trace(model, dummy_input)
    model_scripted = torch.jit.script(model_traced)

    model_optimized = optimize_for_mobile(model_scripted)
    model_optimized.save('pretrained/edgeformer-det.pt')
    # torch.onnx.export(model, dummy_input, "pretrained/edgeformer-det.onnx", verbose=True)

    # model = torch.load('pretrained/edgeformer-det.pt')
    # model.to(torch.device('cpu'))
    # model.eval()
    # result = model(dummy_input)
    # print(result)