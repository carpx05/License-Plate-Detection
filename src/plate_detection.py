import cv2
import numpy as np




def detect_plate(img, text=''):
    plate_cascade = cv2.CascadeClassifier('E:\LPR\indian_license_plate.xml')
    plate_img = img.copy()
    roi = img.copy()
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.3, minNeighbors = 7) # detects numberplates and returns the coordinates and dimensions of detected license plate's contours.
    if plate_img.dtype != np.uint8:
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    for (x,y,w,h) in plate_rect:
        roi_ = roi[y:y+h, x:x+w, :] # extracting the Region of Interest of license plate for blurring.
    
        plate = roi[y:y+h, x:x+w, :]
        cv2.rectangle(plate_img, (x+2,y), (x+w-3, y+h-5), (51,181,155), 3) # finally representing the detected contours by drawing rectangles around the edges.
    if text!='':
        plate_img = cv2.putText(plate_img, text, (x-w//2,y-h//2), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.5, (51,181,155), 1, cv2.LINE_AA)
        
    return plate_img, plate # returning the processed image.
