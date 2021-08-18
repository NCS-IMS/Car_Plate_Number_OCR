import cv2
import imutils
import numpy as np
import pytesseract
import threading
from PIL import Image
from picamera.array import PiRGBArray
from picamera import PiCamera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))
pts = np.zeros((4,2), dtype=np.float32)


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,              cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
          screenCnt = approx
          break
    if screenCnt is None:
      detected = 0
      print ("No contour detected")
      continue
    else:
      detected = 1

    if detected == 1:
      cv2.drawContours(image, [screenCnt], -1, (255, 255, 255), 2)

    mask = np.zeros(gray.shape,np.uint8)
    try:
        new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    except:
        print("Error")
        continue
    
    new_image = cv2.bitwise_and(image,image,mask=mask)
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]
    
    for i in range(0,4):
      pts[i] = [screenCnt[i][0][0], screenCnt[i][0][1]]

    sm = pts.sum(axis=1)
    diff = np.diff(pts, axis = 1)

    topLeft = pts[np.argmin(sm)]
    bottomRight = pts[np.argmax(sm)]
    topRight = pts[np.argmin(diff)]
    bottomLeft = pts[np.argmax(diff)]

    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])

    width = max([w1, w2])
    height = max([h1, h2])

    pts2 = np.float32([[0,0], [width-1, 0], [width-1, height-1], [0,height-1]])

    mtrx = cv2.getPerspectiveTransform(pts1, pts2)

    result = cv2.warpPerspective(image, mtrx, (width, height))
    
    r_height, r_width, r_channel = result.shape
    
    dst = cv2.resize(result, None, fx = 5.0, fy = 4.0, interpolation=cv2.INTER_CUBIC)
    dst = cv2.GaussianBlur(dst, ksize=(3,3), sigmaX=0)
    _, dst = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)
    
    #cv2.imshow("Frame", image)
    #cv2.imshow('Cropped',Cropped)
    #cv2.imshow('result', result)
    #cv2.imshow('dst', dst)
    #cv2.imwrite('result.jpg',dst)
    
    text = pytesseract.image_to_string(dst, lang='kor',config='--psm 7 --oem 3')
    text=text.replace(" ","")
    
    result_text = ""
    
    for i in range(len(text)):
        if i == 0 and text[i] == '0':
            continue        
        if text[i].isalpha() == True or text[i].isdigit():
            result_text += text[i]
    
    print(result_text)
        
cv2.destroyAllWindows()