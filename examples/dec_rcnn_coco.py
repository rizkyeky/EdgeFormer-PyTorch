import json
from torchvision.models import detection
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2
import multiprocessing as mp

# from PIL import Image, ImageDraw

data_transforms = transforms.Compose([
    transforms.ToTensor(),
])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_model():
    # model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(
    #     weights='COCO_V1',
    #     progress=True,
    #     weights_backbone='IMAGENET1K_V1',
    # )
    model = detection.ssdlite320_mobilenet_v3_large(
        weights='COCO_V1',
        progress=True,
        weights_backbone='IMAGENET1K_V1',
    )
    model.forward()

    # model = torch.jit.load('examples/ssdlite320_mobilenet_v3_large.pt')

    model.to(DEVICE)
    model.eval()
    return model

def predict_image(model, image):

    image = np.array(image)
    
    image = data_transforms(image)
    # image = image.unsqueeze(0)
    image.to(DEVICE)

    outputs = model([image])[0]
    
    return outputs["labels"], outputs["scores"], outputs["boxes"]

# Define a function to read frames from video
def read_frames(queue):
    cap = cv2.VideoCapture("images_test/video_test2.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            queue.put(frame)
        else:
            break
    cap.release()

# Define a function to process frames and display them
def process_frames(queue):

    with open('labels/ms_coco_91_classes.json') as f:
        CLASSES = json.load(f)
    
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    COLORS = COLORS.astype(np.uint8).tolist()

    model = init_model()

    fps_list = []

    print('Starting to process frames...')

    while True:
        t1 = cv2.getTickCount()
        frame = queue.get()
        
        # Do some object detection on frame
        labels, scores, boxes = predict_image(model, frame)

        for box, label, score in zip(boxes, labels, scores):
    
            if score > 0.5:
                
                idx = int(label)-1
                box = box.detach().cpu().numpy()
                (startX, startY, endX, endY) = box.astype("uint16")
                
                text = "{}: {:.2f}%".format(CLASSES[str(idx)], score * 100)
                cv2.rectangle(frame,
                    (startX, startY), (endX, endY),
                    tuple(COLORS[idx]), 3
                )
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, tuple(COLORS[idx]), 2)

        t2 = cv2.getTickCount()
        time_diff = (t2 - t1) / cv2.getTickFrequency()
        fps = 1 / time_diff
        fps_list.append(fps)
        print('{:.2f}'.format(np.mean(fps_list)))

        cv2.putText(frame,'FPS: {:.2f}'.format(fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
        
        cv2.imshow("Frame", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

if __name__ == "__main__":

    # Create a queue to store frames
    queue = mp.Queue(maxsize=10)

    # Create two processes for reading and processing frames
    p1 = mp.Process(target=read_frames, args=(queue,))
    p2 = mp.Process(target=process_frames, args=(queue,))

    # Start the processes
    p1.start()
    p2.start()

    # Wait for the processes to finish
    p1.join()
    p2.join()

    # Destroy all windows
    cv2.destroyAllWindows()

    # cap = cv2.VideoCapture('images_test/video_test.mp4')

    # if (cap.isOpened()== False): 
    #     print("Error opening video stream or file")

    # with open('labels/ms_coco_91_classes.json') as f:
    #     CLASSES = json.load(f)
    
    # COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    # COLORS = COLORS.astype(np.uint8).tolist()

    # model = init_model()

    # model_script = torch.jit.script(model)
    # model_script.save('ssdlite320_mobilenet_v3_large.pt')
    # model = torch.jit.load('examples/ssdlite320_mobilenet_v3_large.pt')

    # fps_list = []

    # while (cap.isOpened()):

    #     t1 = cv2.getTickCount()
    #     ret, frame = cap.read()

    #     if ret:

    #         # orig = frame
    #         labels, scores, boxes = predict_image(model, frame)

    #         for box, label, score in zip(boxes, labels, scores):
        
    #             if score > 0.5:
    #                 idx = int(label)-1
    #                 box = box.detach().cpu().numpy()
    #                 (startX, startY, endX, endY) = box.astype("uint16")
                    
    #                 text = "{}: {:.2f}%".format(CLASSES[str(idx)], score * 100)
    #                 cv2.rectangle(frame,
    #                     (startX, startY), (endX, endY),
    #                     tuple(COLORS[idx]), 3
    #                 )
    #                 y = startY - 15 if startY - 15 > 15 else startY + 15
    #                 cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, tuple(COLORS[idx]), 2)

    #         t2 = cv2.getTickCount()
    #         time_diff = (t2 - t1) / cv2.getTickFrequency()
    #         fps = 1 / time_diff
    #         fps_list.append(fps)
    #         print('{:.2f}'.format(np.mean(fps_list)))

    #         cv2.putText(frame,'FPS: {:.2f}'.format(fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)

    #         cv2.imshow('Frame', frame)
            
    #         if cv2.waitKey(25) & 0xFF == ord('q'):
    #             break

    #     else: 
    #         break
        
    # cap.release()
    # cv2.destroyAllWindows()