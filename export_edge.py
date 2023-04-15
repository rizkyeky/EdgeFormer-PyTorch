import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
import main_det

if __name__ == '__main__':

    model = main_det.init_model()
    
    dummy_input = torch.rand(1, 3, 224, 224)
    try:
        model_traced = torch.jit.trace(model, dummy_input)
    except Exception as e:
        err = e.__str__().split('\n')
        print(len(err))
        print(err[:10])
        # print('Error tracing model')
        exit()
    # model_scripted = torch.jit.script(model)

    # model_optimized = optimize_for_mobile(model_scripted)
    # model_scripted.save('pretrained/edgeformer-det.pt')