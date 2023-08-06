# import sys
import cv2
import main_det
import numpy as np
import time

def start():

    file = 'pretrained/edgeformer_raw.pt'
    # file = ''

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

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280,  720))

    while (cap.isOpened()):

        ret, frame = cap.read()

        if ret:

            boxes, scores, labels = main_det.predict_image(model, frame)
            for idx, score, coords in zip(labels, scores, boxes):
                idx = int(idx)
                if score > 0.0:
                    label = "{}: {:.2f}%".format(CLASSES[idx], score * 100)
                    startX, startY, endX, endY = coords
                    cv2.rectangle(frame,
                        (startX, startY), (endX, endY),
                        COLORS[idx], 3
                    )
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, tuple(COLORS[idx]), 2)

            frames_count += 1
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frames_count / elapsed_time
            fps_list.append(fps)

            cv2.putText(frame,'FPS: {:.2f}'.format(fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)

            # cv2.imshow('Frame', frame)
            out.write(frame)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else: 
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('Averange fps: {:.2f}'.format(np.mean(fps_list)))

if __name__ == '__main__':
    start()