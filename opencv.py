import cv2
import numpy as np
import skimage.feature as skim
import time

def polygon_stier(image):
    # image_mask = image.copy()
    height, width = image.shape[:2]
    padding = 0
    vpadding = 80
    hpadding = 125
    alpha = 0.5
    poly = np.array([(padding, height), (320-hpadding, vpadding), (320+hpadding, vpadding), (width-padding, height)])
    # layer = np.zeros((height, width, 3), dtype=np.uint8)
    # # layer = cv2.fillPoly(layer, [poly], (0,0,255))    
    image = cv2.fillPoly(image, [poly], (0,0,0))
    # image_mask = cv2.addWeighted(layer, alpha, image_mask, 1-alpha, 0)

    return image

def line_len(line):
    x1, y1, x2, y2 = line[0]
    point1 = np.array([x1, y1])
    point2 = np.array([x2, y2])
    distance = np.linalg.norm(point2 - point1)
    return distance

def count_nearest_neigbors(lines: list[np.array], line: np.array, k=5):
    
    x1, y1, x2, y2 = line[0]
    main_line_center = np.array([(x1+x2)/2, (y1+y2)/2])
    
    count_neigbor = 0
    for neigbor in lines:
        x1, y1, x2, y2 = neigbor[0]
        neigbor_center1 = np.array([(x1+x2)*1/4, (y1+y2)*1/4])
        neigbor_center2 = np.array([(x1+x2)*2/4, (y1+y2)*2/4])
        neigbor_center3 = np.array([(x1+x2)*3/4, (y1+y2)*3/4])
        distance1 = np.linalg.norm(neigbor_center1 - main_line_center)
        distance2 = np.linalg.norm(neigbor_center2 - main_line_center)
        distance3 = np.linalg.norm(neigbor_center3 - main_line_center)
        if distance1 <= k or distance2 <= k or distance3 <= k:
            count_neigbor += 1
    
    # print(line[0], count_neigbor)
    return count_neigbor

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FPS, 12)
    
    max_line = 20

    frames_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
    # frame = cv2.imread('test.png')
    # frame = cv2.resize(frame, (640, 480))
        if ret:

            cropped = frame[230:, :]

            stier = polygon_stier(cropped)
            
            gray1 = cv2.cvtColor(stier, cv2.COLOR_BGR2GRAY)
            
            hist1 = cv2.equalizeHist(gray1)
            
            # edges1 = cv2.Canny(hist1, 300, 300)
            edges1 = cv2.Sobel(hist1, cv2.CV_64F, 1, 0, ksize=5)
            edges1 = cv2.Laplacian(hist1, cv2.CV_64F)
            # edges1 = skim.canny(hist1, sigma=3)*255
            # edges1 = edges1.astype(np.uint8)
            
            # blur1 = cv2.GaussianBlur(edges1, (9,9), 0)
            
            lines1 = cv2.HoughLinesP(edges1, 1, np.pi/180, 20, maxLineGap=25, minLineLength=10)
            
            if lines1 is not None:
                # filtered_lines = list(filter(lambda x: count_nearest_neigbors(lines1, x, 15) <= 2, lines1))
                # filtered_lines = sorted(filtered_lines, key=lambda x: line_len(x), reverse=True)
                mx = min(max_line, len(lines1)) if len(lines1) > max_line else len(lines1)
                for line in lines1[:mx]:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(cropped, (x1,y1), (x2,y2), (255,0,0), 2)

            frames_count += 1
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frames_count / elapsed_time
            
            cv2.putText(cropped,'{:.2f}fps'.format(fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow('frame1', cropped)
            # cv2.imshow('frame2', original2)
            # cv2.imshow('frame3', original3)
            # cv2.imshow('frame4', stier)
            # cv2.imshow('frame5', mask)
        
            if cv2.waitKey(1) == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()