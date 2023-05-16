import cv2
import numpy as np

def triangle_stir(image):
    padding = 10
    poly = np.array([(padding, 240), (320-60, padding), (320+60, padding), (640-padding, 240)])
    image = cv2.fillPoly(image, [poly], (0,0,0))
    return image

if __name__ == '__main__':

    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FPS, 12)
    
    max_line = 10
    
    # while cap.isOpened():
    #     ret, frame = cap.read()
    frame = cv2.imread('test.png')
    frame = cv2.resize(frame, (640, 480))
        # if ret:
    cropped = frame[240:, :]
    
    cropped1 = cropped[:, :320]
    cropped2 = cropped[:, 320:]

    cropped = triangle_stir(cropped)
    
    gray1 = cv2.cvtColor(cropped1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(cropped2, cv2.COLOR_BGR2GRAY)
    
    hist1 = cv2.equalizeHist(gray1)
    hist2 = cv2.equalizeHist(gray2)

    blur1 = cv2.GaussianBlur(hist1, (13,13), 0)
    blur2 = cv2.GaussianBlur(hist2, (13,13), 0)

    edges1 = cv2.Canny(blur1, 50, 200)
    edges2 = cv2.Canny(blur2, 50, 200)

    lines1 = cv2.HoughLinesP(edges1, 1, np.pi/180, 35, maxLineGap=200, minLineLength=100)
    lines2 = cv2.HoughLinesP(edges2, 1, np.pi/180, 35, maxLineGap=200, minLineLength=100)
    
    if lines1 is not None:
        # lines1 = sorted(lines1, key=lambda x: np.gradient([x[0][1], x[0][2]], [x[0][0], x[0][3]]))
        mx = max_line if len(lines1) > max_line else len(lines1)
        for line in lines1[:mx]:
            x1, y1, x2, y2 = line[0]
            cv2.line(cropped1, (x1,y1), (x2,y2), (0,255,0), 3)
    
    if lines2 is not None:
        # lines2 = sorted(lines2, key=lambda x: np.gradient([x[0][1], x[0][2]], [x[0][0], x[0][3]]))
        mx = max_line if len(lines2) > max_line else len(lines2)
        for line in lines2[:mx]:
            x1, y1, x2, y2 = line[0]
            cv2.line(cropped2, (x1,y1), (x2,y2), (0,255,0), 3)
    
    cv2.imshow('frame1', cropped)
    cv2.imshow('frame2', edges1)
    cv2.imshow('frame3', edges2)
        
    cv2.waitKey(0) == ord('q')
        # break
    
    # cap.release()
    cv2.destroyAllWindows()