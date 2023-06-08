import torch
# from torch.utils.mobile_optimizer import optimize_for_mobile
import main_det
# import cv2
import torchprof

if __name__ == '__main__':

    model = main_det.init_model()
    model = model.cuda()
    
    x = torch.randn([1, 3, 224, 224]).cuda()

    with torchprof.Profile(model, use_cuda=torch.cuda.is_available()) as prof:
        model(x)

    print(prof)


    