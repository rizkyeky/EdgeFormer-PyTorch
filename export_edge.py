import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
import main_det

if __name__ == '__main__':

    model = main_det.init_model().cuda()
    
    # dummy_input = torch.rand(1, 3, 224, 224)
    model_traced = torch.jit.trace(model)
    model_scripted = torch.jit.script(model_traced)

    # model_optimized = optimize_for_mobile(model_scripted)
    model_scripted.save('pretrained/edgeformer-det.pt')