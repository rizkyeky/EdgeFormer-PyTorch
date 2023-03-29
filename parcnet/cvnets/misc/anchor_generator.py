
# from itertools import product
import torch
from math import sqrt
import numpy as np
from typing import Dict, Optional, List

@torch.jit.script
class AnchorsDict(object):
    def __init__(self):
        self.data: Dict[str, torch.Tensor] = {}

    def __setitem__(self, key: str, value: torch.Tensor):
        self.data[key] = value

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

class SSDAnchorGenerator(torch.nn.Module):
    """
        Generate anchors (or priors) for Single Shot object detector:
            https://arxiv.org/abs/1512.02325

        Anchor boxes can be generated for any image size
    """
    def __init__(self,
                 output_strides: List,
                 aspect_ratios: List,
                 min_ratio: Optional[float] = 0.1,
                 max_ratio: Optional[float] = 1.05,
                 no_clipping: Optional[bool] = False
                 ):
        super(SSDAnchorGenerator, self).__init__()
        output_strides_aspect_ratio = dict()
        for k, v in zip(output_strides, aspect_ratios):
            output_strides_aspect_ratio[k] = v

        self.anchors_dict = AnchorsDict()

        scales = np.linspace(min_ratio, max_ratio, len(output_strides) + 1)
        self.sizes = dict()
        for i, s in enumerate(output_strides):
            self.sizes[s] = {
                "min": scales[i],
                "max": sqrt(scales[i] * scales[i+1])
            }
        self.output_strides_aspect_ratio = self.process_aspect_ratio(output_strides_aspect_ratio)

        self.clip = not no_clipping

    @staticmethod
    def process_aspect_ratio(output_strides_aspect_ratio: Dict[int, List]) -> Dict[int, List]:
        for os, curr_ar in output_strides_aspect_ratio.items():
            assert isinstance(curr_ar, list)
            new_ar = list(set(curr_ar)) # keep only unique values
            output_strides_aspect_ratio[os] = new_ar
        return output_strides_aspect_ratio

    def num_anchors_per_os(self):
        # Estimate num of anchors based on aspect ratios: 2 default boxes + 2 * aspect ratios in feature map.
        return [2 + 2 * len(ar) for os, ar in self.output_strides_aspect_ratio.items()]

    @torch.no_grad()
    def generate_anchors_center_form(self, height: int, width: int, output_stride: int):
        min_size_h = float(self.sizes[output_stride]["min"])
        min_size_w = float(self.sizes[output_stride]["min"])

        max_size_h = float(self.sizes[output_stride]["max"])
        max_size_w = float(self.sizes[output_stride]["max"])
        aspect_ratio = self.output_strides_aspect_ratio[output_stride]

        default_anchors_ctr: List[torch.Tensor] = []
        scale_x = float(1.0 / width)
        scale_y = float(1.0 / height)
        ls = torch.cartesian_prod(torch.tensor(list(range(height))), torch.tensor(list(range(width))))
        ls = list(ls)

        for res in ls:
            # [x, y, w, h] format
            x = float(res[1].item())
            y = float(res[0].item())
            cx = float((x + 0.5) * scale_x)
            cy = float((y + 0.5) * scale_y)

            # small size box
            a = torch.tensor([cx, cy, min_size_w, min_size_h])
            default_anchors_ctr.append(a)

            # big size box
            b = torch.tensor([cx, cy, max_size_w, max_size_h])
            default_anchors_ctr.append(b)

            # change h/w ratio of the small sized box based on aspect ratios
            for ratio in aspect_ratio:
                ratio = torch.sqrt(torch.tensor(ratio))
                ratio = float(ratio.item())
                c = torch.tensor([cx, cy, min_size_w * ratio, min_size_h / ratio])
                default_anchors_ctr.append(c)
                d = torch.tensor([cx, cy, min_size_w / ratio, min_size_h * ratio])
                default_anchors_ctr.append(d)
        default_anchors_ctr_tensor: torch.Tensor = torch.stack(default_anchors_ctr.copy())
        default_anchors_ctr.clear()
        if self.clip:
            default_anchors_ctr_tensor = torch.clamp(default_anchors_ctr_tensor, min=0.0, max=1.0)

        return default_anchors_ctr_tensor

    @torch.no_grad()
    def get_anchors(self, fm_height: int, fm_width: int, fm_output_stride: int) -> torch.Tensor:
        key = "h_{}_w_{}_os_{}".format(fm_height, fm_width, fm_output_stride)
        
        if key not in self.anchors_dict:
            default_anchors_ctr = self.generate_anchors_center_form(height=fm_height, width=fm_width, output_stride=fm_output_stride)
            self.anchors_dict[key] = default_anchors_ctr
            return default_anchors_ctr
        else:
            return self.anchors_dict[key]

    @torch.no_grad()
    def forward(self, fm_height: int, fm_width: int, fm_output_stride: int) -> torch.Tensor:
        return self.get_anchors(fm_height=fm_height, fm_width=fm_width, fm_output_stride=fm_output_stride)
    
    # def cart_product(self, seqs1, seqs2):
    #     return torch.cartesian_prod(seqs1, seqs2)
