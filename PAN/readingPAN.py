import cv2 
from PIL import Image
import pytesseract
import numpy as np
import easyocr
                

image=cv2.imread("PAN/PAN3.jpg")


combined_allowlist = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("PAN/01index_gray.jpg", gray)

reader = easyocr.Reader(['hi', 'en'])
results1=[]
results2=[]

# OCR STARTS

# gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("PAN/01index_gray.png", gray)
# to find the structure of the cokumn
blur=cv2.GaussianBlur(gray, (7,7), 0)
cv2.imwrite("PAN/02index_blur.jpg", blur)
thresh=cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
cv2.imwrite("PAN/03index_thresh.jpg", thresh)
# cv2.imshow("threshold", thresh)
# cv2.waitKey(2000)
kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (30,10))
cv2.imwrite("PAN/04index_kernel.jpg",kernel)
# cv2.imshow("threshold", kernel)   
# cv2.waitKey(2000)
dilate=cv2.dilate(thresh, kernel, iterations=1)
cv2.imwrite("PAN/05index_dilate.jpg", dilate)
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts=sorted(cnts,key=lambda x:cv2.boundingRect(x)[0])

    
roi_counter = 1  # Counter for naming the ROI images

for c in cnts:
    x, y, w, h = cv2.boundingRect(c)

    if h < 150 and h > 50 and x > 150 and x < 1450 and y > 150 and y < 800:
        roi = gray[y:y + h, x:x + w]
        cv2.rectangle(gray, (x, y), (x + w, y + h), (36, 255, 12), 2)

        # Save each ROI as a separate image
        roi_filename = f"PAN/roi_{roi_counter}.jpg"
        cv2.imwrite(roi_filename, roi)
        roi_counter += 1

        ocr_result_easyocr = reader.readtext(roi, detail=0, paragraph=False)
        ocr_result_pytesseract = pytesseract.image_to_string(roi, lang='eng+hin', config='--psm 6')
        for item in ocr_result_easyocr:
            results1.append(item)
        for item in ocr_result_pytesseract:
            results2.append(item)

# OCR ENDS

def listToString(results):
 
    # initialize an empty string
    str1 = ""
 
    # traverse in the string
    for ele in results:
        str1 += ele
 
    # return string
    return str1
results2 = list(listToString(results2).split(" "))
# Driver code
s = ['Geeks', 'for', 'Geeks']
print(listToString(s))




cv2.imwrite("PAN/panB_BOX.jpg", gray)
print("EasyOCR: ",results1)
print("Pytesseract: ",results2)
