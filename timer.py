import cv2
import numpy as np
import pytesseract

cap = cv2.VideoCapture('/Users/nishimurakai/lab/Presentation_timer/data/ex.mov')
 
# ret, frame = cap.read()
# cv2.imwrite('first_frame.png',frame)
 
while True:
    
    ret, frame = cap.read()
    if ret:
        number = pytesseract.image_to_string(frame)
        print(number)
    else:
        break
 
cap.release()
cv2.destroyAllWindows()
