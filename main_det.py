import sys
import time

sys.argv.extend(['--common.config-file', 'config/edgeformer_det_infer.yaml'])

sys.path.append('edgeformer')
from edgeformer.cvnets.models.detection.ssd import *
from edgeformer.cvnets.models.classification.edgeformer import *
from edgeformer.options.opts import get_eval_arguments
from edgeformer.utils.common_utils import device_setup
from edgeformer.cvnets.models import get_model
# from edgeformer.engine.eval_detection import *
from edgeformer.utils.tensor_utils import tensor_size_from_opts

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
# from accelerate import Accelerator

opts = get_eval_arguments()
opts = device_setup(opts)
# accelerator = Accelerator(cpu=True, mixed_precision='fp16')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# device = accelerator.device

mixed_precision = getattr(opts, "common.mixed_precision", False)
n_classes: int = getattr(opts, "model.detection.n_classes", 4)
size_variance: float = getattr(opts, "model.detection.ssd.size_variance", 0.2)
center_variance: float = getattr(opts, "model.detection.ssd.center_variance", 0.1)
conf_threshold: float = getattr(opts, "model.detection.ssd.conf_threshold", 0.01)
top_k: int = getattr(opts, "model.detection.ssd.num_objects_per_class", 200)
nms_threshold: float = getattr(opts, "model.detection.ssd.nms_iou_threshold", 0.5)

res_h, res_w = tensor_size_from_opts(opts)
img_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((res_h, res_w)),
    transforms.ToTensor(),
])

def init_model(path = "") -> SingleShotDetector:
    if path == "":    
        print('From Repo')
        model: SingleShotDetector = get_model(opts)
        model.to(device)
        model.eval()

        if model.training:
            model.eval()
        
        return model
    else:
        print('From TorchScript File')
        model = torch.jit.load(path)
        model.to(device)
        model.eval()

        if model.training:
            model.eval()
        
        return model

def predict_image(model: SingleShotDetector, image: np.array) -> Tuple[np.array, np.array, np.array]:
    
    with torch.no_grad():
        orig_h, orig_w = image.shape[0], image.shape[1]
        
        image = img_transforms(image)
        image = image.unsqueeze(0)
        
        output_stride = 32
        curr_height, curr_width = image.shape[2:]
        new_h = (curr_height // output_stride) * output_stride
        new_w = (curr_width // output_stride) * output_stride

        if new_h != curr_height or new_w != curr_width:
            image = F.interpolate(input=image, size=(new_h, new_w), mode="bilinear", align_corners=False)
        
        image.to(device)

        if (torch.cuda.is_available()):
            with torch.cuda.amp.autocast(enabled=True):
                img = image.cuda()
                tensor = model(img)
        else:
            tensor = model(image)

        # result = SingleShotDetector.predict(tensor, 
        #                                     n_classes, 
        #                                     size_variance, 
        #                                     center_variance, 
        #                                     conf_threshold, 
        #                                     top_k, 
        #                                     nms_threshold)
        boxes = tensor[0].cpu().numpy()
        scores = tensor[1].cpu().numpy()
        labels = tensor[2].cpu().numpy()

        boxes[..., 0::2] = boxes[..., 0::2] * orig_w
        boxes[..., 1::2] = boxes[..., 1::2] * orig_h
        boxes[..., 0::2] = np.clip(a_min=0, a_max=orig_w, a=boxes[..., 0::2])
        boxes[..., 1::2] = np.clip(a_min=0, a_max=orig_h, a=boxes[..., 1::2])
        
        boxes = boxes.astype(np.int16)
        
    return boxes, scores, labels