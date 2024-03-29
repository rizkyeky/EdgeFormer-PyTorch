import torch
# import torchvision
from torch import nn, Tensor
from utils import logger
import argparse
from typing import Optional, Tuple
from torchvision.ops import nms as torch_nms

from . import register_detection_models
from .base_detection import BaseDetection
from ...layers import ConvLayer, SeparableConv, AdaptiveAvgPool2d
from ...modules import SSDHead
from ...models.classification import BaseEncoder
from ...misc.init_utils import initialize_conv_layer
from ...misc import box_utils
from ...misc.anchor_generator import SSDAnchorGenerator
from ...misc.profiler import module_profile


@register_detection_models("ssd")
class SingleShotDetector(BaseDetection):
    """
    This class implements Single Shot Object Detector
        https://arxiv.org/abs/1512.02325
    """
    coordinates = 4 # 4 coordinates (x, y, x1, y1) or (x, y, w, h)

    def __init__(self, opts, encoder: BaseEncoder):
        super(SingleShotDetector, self).__init__(opts=opts, encoder=encoder)

        # delete layers that are not required in detection network
        self.encoder.classifier = None
        self.encoder.conv_1x1_exp = None

        self.is_training = getattr(opts, "model.detection.is_training", False)
        self.device = getattr(opts, "dev.device", torch.device("cpu"))

        output_strides = getattr(opts, "model.detection.ssd.output_strides", [16, 32, 64, 128, 256, -1 ])
        n_os = len(output_strides)

        anchors_aspect_ratio = getattr(opts, "model.detection.ssd.anchors_aspect_ratio", [[2, 3]] * len(output_strides))
        proj_channels = getattr(opts, "model.detection.ssd.proj_channels", [512, 256, 256, 128, 128, 64])

        self.center_variance: float = getattr(self.opts, "model.detection.ssd.center_variance", 0.1)
        self.size_variance: float = getattr(self.opts, "model.detection.ssd.size_variance", 0.2)

        anchors_aspect_ratio = anchors_aspect_ratio + [[2]] * (n_os - len(anchors_aspect_ratio))
        proj_channels = proj_channels + [128] * (n_os - len(proj_channels))

        if len(output_strides) != len(anchors_aspect_ratio) != len(proj_channels):
            logger.error(
                "SSD model requires anchors to be defined for feature maps from each output stride. Also"
                "len(anchors_aspect_ratio) == len(output_strides) == len(proj_channels). "
                "Got len(output_strides)={}, len(anchors_aspect_ratio)={}, len(proj_channels)={}."
                " Please specify correct arguments using following arguments: "
                "\n--model.detection.ssd.anchors-aspect-ratio "
                "\n--model.detection.ssd.output-strides"
                "\n--model.detection.ssd.proj-channels".format(
                    len(output_strides),
                    len(anchors_aspect_ratio),
                    len(proj_channels)
                )
            )
        extra_layers = {}
        enc_channels_list = []
        in_channels = self.enc_l5_channels

        extra_proj_list = [256] * (len(output_strides) - len(proj_channels))
        proj_channels = proj_channels + extra_proj_list
        for idx, os in enumerate(output_strides):
            out_channels = proj_channels[idx]
            if os == 8:
                enc_channels_list.append(self.enc_l3_channels)
            elif os == 16:
                enc_channels_list.append(self.enc_l4_channels)
            elif os == 32:
                enc_channels_list.append(self.enc_l5_channels)
            elif os > 32 and os != -1:
                extra_layers["os_{}".format(os)] = SeparableConv(
                    opts=opts, in_channels=in_channels, out_channels=out_channels, kernel_size=3, use_act=True,
                    use_norm=True, stride=2
                )
                enc_channels_list.append(out_channels)
                in_channels = out_channels
            elif os == -1:
                extra_layers["os_{}".format(os)] = nn.Sequential(
                    AdaptiveAvgPool2d(output_size=1),
                    ConvLayer(opts=opts, in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                              use_act=True, use_norm=False)
                )
                enc_channels_list.append(out_channels)
                in_channels = out_channels
            else:
                raise NotImplementedError
        self.extra_layers = None if not extra_layers else nn.ModuleDict(extra_layers)
        if self.extra_layers is not None:
            self.reset_layers(module=self.extra_layers)

        # Anchor box related parameters
        self.conf_threshold = getattr(opts, "model.detection.ssd.conf_threshold", 0.01)
        self.nms_threshold = getattr(opts, "model.detection.ssd.nms_iou_threshold", 0.5)
        self.top_k = getattr(opts, "model.detection.ssd.num_objects_per_class", 200)

        self.anchor_box_generator = SSDAnchorGenerator(
            output_strides=output_strides,
            aspect_ratios=anchors_aspect_ratio,
            min_ratio=getattr(opts, "model.detection.ssd.min_box_size", 0.1),
            max_ratio=getattr(opts, "model.detection.ssd.max_box_size", 1.05)
        )

        anchors_aspect_ratio = self.anchor_box_generator.num_anchors_per_os()
        self.ssd_heads = nn.ModuleList()

        # Create SSD detection and classification heads
        for os, in_dim, proj_dim, n_anchors in zip(output_strides, enc_channels_list, proj_channels, anchors_aspect_ratio):
            self.ssd_heads += [
                SSDHead(opts=opts,
                        in_channels=in_dim,
                        n_classes=self.n_classes,
                        n_coordinates=self.coordinates,
                        n_anchors=n_anchors,
                        proj_channels=proj_dim,
                        kernel_size=3 if os != -1 else 1)
            ]

        self.anchors_aspect_ratio = anchors_aspect_ratio
        self.output_strides = output_strides

    @staticmethod
    def reset_layers(module):
        for layer in module.modules():
            if isinstance(layer, nn.Conv2d):
                initialize_conv_layer(module=layer, init_method='xavier_uniform')

    @staticmethod
    def process_anchors_ar(anchor_ar):
        assert isinstance(anchor_ar, list)
        new_ar = []
        for ar in anchor_ar:
            if ar in new_ar:
                continue
            new_ar.append(ar)
        return new_ar

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument('--model.detection.ssd.anchors-aspect-ratio', type=int, nargs="+", action="append",
                           default=[[2, 3]] * 5,
                           help="Anchors aspect ratio in each feature map obtained at different output strides.")
        group.add_argument('--model.detection.ssd.output-strides', type=int, nargs="+",
                           default=[16, 32, 64, 128, -1], help="Extract feature maps from these output strides.")
        group.add_argument('--model.detection.ssd.proj-channels', type=int, nargs="+",
                           default=[128, 256, 384, 512, 768],
                           help="Projection channels for feature map obtained at each output stride")

        # prior box arguments
        # SSD sample priors between min and max box sizes.
        # for example, if we use feature maps from three spatial levels (or output strides), then we
        # sample width and height for anchor boxes as:
        # scales = np.linspace(min_box_size, max_box_size, len(output_strides) + 1)
        # min_box dimensions for the first feature map is scales[0] * feature_map_dimensions
        # while the max_box dimensions will be sqrt(scales[0] * scales[1]) * feature_map dimensions. And so on
        group.add_argument('--model.detection.ssd.min-box-size', type=float, default=0.1,
                           help="Min. box size. Value between 0 and 1. Good default value is 0.1")
        group.add_argument('--model.detection.ssd.max-box-size', type=float, default=1.05,
                           help="Max. box size. Value between 0 and 1. Good default value is 1.05")

        #
        group.add_argument('--model.detection.ssd.center-variance', type=float, default=0.1,
                           help="Center variance.")
        group.add_argument('--model.detection.ssd.size-variance', type=float, default=0.2,
                           help="Size variance.")
        group.add_argument('--model.detection.ssd.iou-threshold', type=float, default=0.45,
                           help="IOU Threshold.")

        # inference related arguments
        group.add_argument('--model.detection.ssd.conf-threshold', type=float, default=0.05,
                           help="Confidence threshold. For evaluation on COCO, set to 0.01, so that we can compute mAP")
        group.add_argument('--model.detection.ssd.num-objects-per-class', type=int, default=200,
                           help="Keep only these many objects per class after NMS")
        group.add_argument('--model.detection.ssd.nms-iou-threshold', type=float, default=0.3,
                           help="NMS IoU threshold ")
        return parser

    def ssd_forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        
        enc_end_points = self.encoder.extract_end_points_all(x)

        locations = torch.empty(0).to(self.device)
        confidences = torch.empty(0).to(self.device)
        anchors = torch.empty(0).to(self.device)
        
        x = enc_end_points["out_l5"]

        i = 0
        for ssd_head in self.ssd_heads:
            os = self.output_strides[i]
            if os in (8, 16, 32):
                if os == 8:
                    point = enc_end_points["out_l3"]
                elif os == 16:
                    point = enc_end_points["out_l4"]
                else:
                    point = enc_end_points["out_l5"]
                fm_h, fm_w = point.shape[2:]
                loc, pred = ssd_head(point)
            elif os in (64, 128, 256):
                if os == 64:
                    x = self.extra_layers["os_64"](x)
                elif os == 128:
                    x = self.extra_layers["os_128"](x)
                else:
                    x = self.extra_layers["os_256"](x)
                fm_h, fm_w = x.shape[2:]
                loc, pred = ssd_head(x)
            else:
                x = self.extra_layers["os_-1"](x)
                fm_h, fm_w = x.shape[2:]
                loc, pred = ssd_head(x)

            locations = torch.cat((locations, loc), dim=1)
            confidences = torch.cat((confidences, pred), dim=1)
            
            anchors_fm_ctr = self.anchor_box_generator(
                fm_height=fm_h,
                fm_width=fm_w,
                fm_output_stride=os
            ).to(self.device)
            
            anchors = torch.cat((anchors, anchors_fm_ctr), dim=0)

            i += 1
        
        anchors = anchors.unsqueeze(dim=0).to(self.device)

        return confidences, locations, anchors

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        result = self.ssd_forward(x)
        # bsz, _, __, ___ = x.shape
        # if self.is_training:
        #     assert bsz != 1
        #     return self.ssd_forward(x)  
        # else:
        #     assert bsz == 1
        return self.predict(result,
                            n_classes=self.n_classes,
                            size_variance=self.size_variance,
                            center_variance=self.center_variance,
                            conf_threshold=self.conf_threshold,
                            top_k=self.top_k,
                            nms_threshold=self.nms_threshold
                            )

    @staticmethod
    def predict(tensors: Tuple[Tensor, Tensor, Tensor],
                n_classes: int = 4,
                size_variance: float = 0.2,
                center_variance: float = 0.1,
                conf_threshold: float = 0.5,
                top_k: int = 200,
                nms_threshold: float = 0.45,
                ) -> Tuple[Tensor, Tensor, Tensor]:
        cpu = torch.device('cpu')
        with torch.no_grad():
            confidences, locations, anchors = tensors
            confidences, locations, anchors = confidences.to(cpu), locations.to(cpu), anchors.to(cpu)
            scores = torch.nn.functional.softmax(confidences, dim=-1)
        # convert boxes in center form [c_x, c_y]
        # boxes = box_utils.convert_locations_to_boxes(
        #     pred_locations=locations,
        #     anchor_boxes=anchors,
        #     center_variance=self.center_variance,
        #     size_variance=self.size_variance,
        # )
        if anchors.dim() + 1 == locations.dim():
            anchors = anchors.unsqueeze(0)

        pred_size = torch.exp(locations[..., 2:] * size_variance) * anchors[..., 2:]
        pred_center = (locations[..., :2] * center_variance * anchors[..., 2:]) + anchors[..., :2]

        boxes = torch.cat((pred_center, pred_size), dim=-1)

        # convert boxes from center form [c_x, c_y] to corner form [x, y]
        # boxes = box_utils.center_form_to_corner_form(boxes)
        boxes = torch.cat((
            boxes[..., :2] - (boxes[..., 2:] * 0.5),
            boxes[..., :2] + (boxes[..., 2:] * 0.5)
        ), dim=-1)

        boxes = boxes[0]
        scores = scores[0]

        object_boxes = torch.empty(0).to(cpu)
        object_scores = torch.empty(0).to(cpu)
        object_labels = torch.empty(0).to(cpu)

        for class_index in range(1, n_classes):
            probs = scores[:, class_index]
            mask = probs > conf_threshold
            probs = probs[mask]
            masked_boxes = boxes[mask, :]

            # filtered_boxes, filtered_scores = nms(
            #     scores=probs.reshape(-1),
            #     boxes=masked_boxes,
            #     nms_threshold=self.nms_threshold,
            #     top_k=self.top_k
            # )
            reshape_probs = probs.reshape(-1)
            keep = torch.ops.torchvision.nms(masked_boxes, reshape_probs, nms_threshold)
            if top_k > 0: keep = keep[:top_k]            
            filtered_boxes, filtered_scores = masked_boxes[keep], reshape_probs[keep]

            filtered_labels = torch.full_like(filtered_scores, fill_value=class_index, dtype=torch.int8,)

            object_boxes = torch.cat((object_boxes, filtered_boxes), dim=0)
            object_scores = torch.cat((object_scores, filtered_scores), dim=0)
            object_labels = torch.cat((object_labels, filtered_labels), dim=0)

        return object_boxes, object_scores, object_labels

    def profile_model(self, input: Tensor) -> None:
        # Note: Model profiling is for reference only and may contain errors.
        # It relies heavily on the user to implement the underlying functions accurately.

        overall_params, overall_macs = 0.0, 0.0

        logger.log('Model statistics for an input of size {}'.format(input.size()))
        logger.double_dash_line(dashes=65)
        print('{:>35} Summary'.format(self.__class__.__name__))
        logger.double_dash_line(dashes=65)

        # profile encoder
        enc_str = logger.text_colors['logs'] + logger.text_colors['bold'] + 'Encoder  ' + logger.text_colors[
            'end_color']
        print('{:>45}'.format(enc_str))
        enc_end_points, encoder_params, encoder_macs = self.encoder.profile_model(input, is_classification=False)

        ssd_head_params = ssd_head_macs = 0.0
        x = enc_end_points["out_l5"]
        for os, ssd_head in zip(self.output_strides, self.ssd_heads):
            if os == 8:
                _, p, m = module_profile(module=ssd_head, x=enc_end_points["out_l3"])
                ssd_head_params += p
                ssd_head_macs += m
            elif os == 16:
                _, p, m = module_profile(module=ssd_head, x=enc_end_points["out_l4"])
                ssd_head_params += p
                ssd_head_macs += m
            elif os == 32:
                _, p, m = module_profile(module=ssd_head, x=enc_end_points["out_l5"])
                ssd_head_params += p
                ssd_head_macs += m
            else: # for all other feature maps with os > 32
                x, p1, m1 = module_profile(module=self.extra_layers["os_{}".format(os)], x=x)
                _, p2, m2 = module_profile(module=ssd_head, x=x)
                ssd_head_params += (p1 + p2)
                ssd_head_macs += (m1 + m2)

        overall_params += (encoder_params + ssd_head_params)
        overall_macs += (encoder_macs + ssd_head_macs)

        ssd_str = logger.text_colors['logs'] + logger.text_colors['bold'] + 'SSD  ' + logger.text_colors[
            'end_color']
        print('{:>45}'.format(ssd_str))

        print(
            '{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M'.format(
                self.__class__.__name__,
                'Params',
                round(ssd_head_params / 1e6, 3),
                'MACs',
                round(ssd_head_macs / 1e6, 3)
            )
        )

        logger.double_dash_line(dashes=65)
        print('{:<20} = {:>8.3f} M'.format('Overall parameters', overall_params / 1e6))
        # Counting Addition and Multiplication as 1 operation
        print('{:<20} = {:>8.3f} M'.format('Overall MACs', overall_macs / 1e6))
        overall_params_py = sum([p.numel() for p in self.parameters()])
        print('{:<20} = {:>8.3f} M'.format('Overall parameters (sanity check)', overall_params_py / 1e6))
        logger.double_dash_line(dashes=65)


def nms(boxes: Tensor, scores: Tensor, nms_threshold: float, top_k: Optional[int] = 200) -> Tuple[Tensor, Tensor]:
    """
    Args:
        boxes (N, 4): boxes in corner-form.
        scores (N): probabilities
        nms_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
         picked: Boxes and scores
    """
    keep = torch_nms(boxes, scores, nms_threshold)
    if top_k > 0:
        keep = keep[:top_k]
    return boxes[keep], scores[keep]