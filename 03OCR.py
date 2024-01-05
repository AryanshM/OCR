import cv2 
from PIL import Image
import pytesseract
import numpy as np

# img_file="./threecolumn.jpg"
# img = Image.open(img_file)

# ocr_result =  pytesseract.image_to_string(img)
# print(ocr_result)

# using open cv to create bounding boxes

# we will blur the image to find structures
image=cv2.imread("./threecolumn.jpg")
gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("./index_gray.png", gray)
# cv2.imshow("GrayScale", gray)
# cv2.waitKey(2000)
# to find the structure of the cokumn
blur=cv2.GaussianBlur(gray, (7,7), 0)
cv2.imwrite("./index_blur.jpg", blur)
thresh=cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
cv2.imwrite("./index_thresh.jpg", thresh)
# cv2.imshow("threshold", thresh)
# cv2.waitKey(2000)
kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3,13))
cv2.imwrite("./index_kernel.jpg",kernel)
# cv2.imshow("threshold", kernel)
# cv2.waitKey(2000)
dilate=cv2.dilate(thresh, kernel, iterations=1)
cv2.imwrite("./index_dilate.jpg", dilate)
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts=sorted(cnts,key=lambda x:cv2.boundingRect(x)[0])

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(image, (x,y), (x+w, y+h), (36,255,12), 2)
cv2.imwrite("./index_bBox.jpg", image)

