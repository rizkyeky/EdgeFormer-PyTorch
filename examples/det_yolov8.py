import torch
import cv2
import numpy as np
import json
import time
import sys
from ultralytics import YOLO

def start():

    file = 'pretrained/yolov8m.pt'
    is_torchscript = False
    if len(sys.argv) > 1 and sys.argv[1] == 'torchscript':
        is_torchscript = True
        file = 'pretrained/yolov8m.torchscript'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture('images_test/video_test2.mp4')
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    with open('labels/ms_coco_91_classes.json') as f:
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
        model = YOLO(file)
        model.to(device)
        model.fuse()

    fps_list = []
    times_list = []

    frames_count = 0
    start_time = time.time()

    while (cap.isOpened()):

        ret, frame = cap.read()

        if ret == True:

            orig = frame
            frame = cv2.resize(frame, (IMG_SIZE,IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if is_torchscript:
                frame = torch.from_numpy(frame).permute(2, 0, 1).float()
                frame /= 225
                frame = frame.half().unsqueeze(0)
                if torch.cuda.is_available():
                    frame = frame.cuda()
                frame.to(device)


            start_infer = time.time()
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    if is_torchscript:
                        outputs = model(frame)[0]
                    else:
                        outputs = model.predict([frame], iou=0.5, imgsz=IMG_SIZE, half=True)[0]
                    torch.cuda.synchronize()
            else:
                if is_torchscript:
                    outputs = model(frame)[0]
                else:
                    outputs = model([frame], iou=0.5, imgsz=IMG_SIZE, half=True)[0]
            times_list.append(time.time() - start_infer)
            
            for box in outputs.boxes:
                score = box.conf.item()
                idx = int(box.cls.item())
                if score > 0.25:
                    label = "{}: {:.2f}%".format(CLASSES[idx], score * 100)
                    xyxy = box.xyxy.detach().cpu().numpy().astype(np.int16)[0]
                    startX = int(xyxy[0] * orig.shape[1] / IMG_SIZE)
                    startY = int(xyxy[1] * orig.shape[0] / IMG_SIZE)
                    endX = int(xyxy[2] * orig.shape[1] / IMG_SIZE)
                    endY = int(xyxy[3] * orig.shape[0] / IMG_SIZE)
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

            cv2.putText(orig,'FPS: {:.2f}'.format(fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)

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
