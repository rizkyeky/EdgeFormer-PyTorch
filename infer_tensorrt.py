import torch
import torchvision
import torch.nn as nn
# import torch_tensorrt
import cv2
import main_dec
import numpy as np
import json
import time

def start():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture('images_test/video_test2.mp4')
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # CLASSES = ['_', 'robot', 'ball', 'goal']
    with open('labels/ms_coco_81_classes.json') as f:
        CLASSES = json.load(f)
        CLASSES = [CLASSES[str(i)] for i in range(len(CLASSES))]

    # COLORS = [(0,0,0), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
    COLORS = np.random.randint(0, 255+1, size=(len(CLASSES), 3))
    COLORS = tuple(map(tuple, COLORS))

    model = torch.jit.load('pretrained/ssdlite320_mobilenet_v3_large.pt')
    model.to(device)
    model.eval()

    if model.training:
        model.eval()

    fps_list = []
    times_list = []

    frames_count = 0
    start_time = time.time()

    while (cap.isOpened()):

        ret, frame = cap.read()

        if ret == True:

            orig = frame
            frame = cv2.resize(frame, (224,224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(frame).permute(2, 0, 1).float()
            image /= 255
            image.to(device)

            start_infer = time.time()
            outputs = model([image])[1][0]
            times_list.append(time.time() - start_infer)
            
            for box, idx, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
                if score > 0.5:
                    label = "{}: {:.2f}%".format(CLASSES[idx], score * 100)
                    box = box.cpu().numpy().astype("int")
                    startX = int(box[0] * orig.shape[1] / 255)
                    startY = int(box[1] * orig.shape[0] / 255)
                    endX = int(box[2] * orig.shape[1] / 255)
                    endY = int(box[3] * orig.shape[0] / 255)
                    color = COLORS[idx]
                    color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))
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
    print('Averange infer time: {:.2f}'.format(np.average(times_list)))

if __name__ == '__main__':
    start()