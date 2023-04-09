import torch
import torchvision
import cv2
import numpy as np
import json
import time
import sys

from typing import Optional, List

from torch import Tensor

def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list):
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def start():
    use_cuda = torch.cuda.is_available()
    
    file = ''
    is_torchscript = False
    if len(sys.argv) > 1 and sys.argv[1] == 'torchscript':
        is_torchscript = True
        file = 'pretrained/detr.torchscript'

    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print('Using CUDA')

    cap = cv2.VideoCapture('images_test/video_test2.mp4')
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    if (cap.isOpened() == False): 
        print("Error opening video stream or file")

    with open('labels/ms_coco_81_classes.json') as f:
        CLASSES = json.load(f)
        CLASSES = [CLASSES[str(i)] for i in range(len(CLASSES))]

    COLORS = np.random.randint(0, 255+1, size=(len(CLASSES), 3))
    COLORS = tuple(map(tuple, COLORS))
    IMG_SIZE = 224

    if is_torchscript:
        model = torch.jit.load(file)
        model.to(device)
        model.eval()
    else:
        model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
        model.to(device)
        model.eval()

    fps_list = []
    times_list = []

    frames_count = 0
    start_time = time.time()

    while (cap.isOpened()):

        ret, frame = cap.read()

        if ret:

            orig = frame
            frame = cv2.resize(frame, (IMG_SIZE,IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1).float()
            frame /= 225
            if use_cuda:
                frame = frame.cuda()
            frame.to(device)

            start_infer = time.time()
            
            if is_torchscript:
                outputs = model(nested_tensor_from_tensor_list([frame]))
            else:
                if use_cuda:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        outputs = model(frame)
                        torch.cuda.synchronize()
                else:
                    outputs = model(frame)
                
            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > 0.7

            bboxes_scaled = box_cxcywh_to_xyxy(outputs['pred_boxes'][0, keep])

            times_list.append(time.time() - start_infer)

            scores = probas[keep]
            boxes = bboxes_scaled
        
            for box, score in zip(boxes.tolist(), scores):
                idx = torch.argmax(score)
                conf = score[idx]
                if conf > 0.25:
                    label = "{}: {:.2f}%".format(CLASSES[idx], conf * 100)
                    startX = int(round(box[0] * orig.shape[1]))
                    startY = int(round(box[1] * orig.shape[0]))
                    endX = int(round(box[2] * orig.shape[1]))
                    endY = int(round(box[3] * orig.shape[0]))
                    color = COLORS[idx]
                    color = (int(color[0]), int(color[1]), int(color[2]))
                    cv2.rectangle(orig,
                        (startX, startY), (endX, endY),
                        color, 3
                    )
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            frames_count += 1
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frames_count / elapsed_time
            fps_list.append(fps)

            cv2.putText(orig,'FPS: {:.2f}'.format(fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            cv2.imshow('Frame', orig)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else: 
            break
        
    cap.release()
    cv2.destroyAllWindows()

    print('Averange fps: {:.2f}'.format(np.average(fps_list)))
    print('Averange infer time: {:.3f}'.format(np.average(times_list)))

if __name__ == '__main__':
    start()
