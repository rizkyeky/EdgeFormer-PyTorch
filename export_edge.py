import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
import main_det
import cv2

if __name__ == '__main__':

    model = main_det.init_model()

    img = cv2.imread('images_test/krsbi3.jpg')
    img = main_det.img_transforms(img)
    img = img.unsqueeze(0)
    
    dummy_input = torch.rand(1, 3, 224, 224)
    try:
        model_traced = torch.jit.trace(model, img)
    except Exception as e:
        err = e.__str__().replace('\t', '')
        err = err.split('\n')
        with open('error_trace.txt', 'w') as f:
            f.write(e.__str__())
        print('Error tracing model')
        # exit()
    model_scripted = torch.jit.script(model)

    # model_optimized = optimize_for_mobile(model_scripted)
    model_scripted.save('pretrained/edgeformer-det_test.pt')