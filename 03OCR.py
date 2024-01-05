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

results=[]

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    # For each contour, this line calculates the bounding rectangle around it using the cv2.boundingRect() function. The function returns four values: x and y are the coordinates of the top-left corner of the rectangle, and w and h are its width and height.
    if h>200 and w >20:    
        # This line checks whether the height (h) is greater than 200 and the width (w) is greater than 20. This condition is used to filter out small contours that might not be significant.
        roi = image[y:y+h, x:x+h]
        # If the contour passes the height and width conditions, this line extracts a region of interest (ROI) from the original image. It uses array slicing to define the region based on the bounding rectangle coordinates.
        cv2.rectangle(image, (x,y), (x+w, y+h), (36,255,12), 2)
        # This line draws a green rectangle around the detected region of interest on the original image. The rectangle is defined by two points: (x, y) for the top-left corner and (x+w, y+h) for the bottom-right corner. The color of the rectangle is represented by the tuple (36, 255, 12) (BGR values), and the thickness is set to 2 pixels.
        ocr_result = pytesseract.image_to_string(roi)
        ocr_result=ocr_result.split("\n")
        for item in ocr_result:
            results.append(item)
cv2.imwrite("./index_bBox.jpg", image)
# cv2.imshow("Bounding Boxes", image)
# cv2.waitKey(2000)
# print(ocr_result)

# print(results)

for item in results:
    item=item.strip()
    item=item.split(" ")[0]
    # we are extracting the first word of each line
    print(item)