# import sys
import cv2
import main_det
import numpy as np
import time

def start():

    # file = 'pretrained/edgeformer-det.pt'
    file = ''

    cap = cv2.VideoCapture('images_test/video_test.mp4')
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    CLASSES = ['_', 'robot', 'ball', 'goal']
    COLORS = [(0,0,0), (0, 0, 255), (0, 255, 0), (255, 0, 0)]

    model = main_det.init_model(file)

    fps_list = []

    frames_count = 0
    start_time = time.time()

    while (cap.isOpened()):

        ret, frame = cap.read()

        if ret == True:

            orig = frame
            labels, scores, boxes = main_det.predict_image(model, frame)
            for idx, score, coords in zip(labels, scores, boxes):
                if score > 0.0:
                    label = "{}: {:.2f}%".format(CLASSES[idx], score * 100)
                    startX, startY, endX, endY = coords
                    cv2.rectangle(orig,
                        (startX, startY), (endX, endY),
                        COLORS[idx], 3
                    )
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, tuple(COLORS[idx]), 2)

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
    print('Averange fps: {:.2f}'.format(np.mean(fps_list)))

if __name__ == '__main__':
    start()