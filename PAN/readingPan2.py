import cv2 
from PIL import Image
import pytesseract
import numpy as np
import easyocr

def ocr(gray,results,रीडर ):

    # gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("PAN/01index_gray.png", gray)
    # to find the structure of the cokumn
    blur=cv2.GaussianBlur(gray, (7,7), 0)
    cv2.imwrite("PAN/02index_blur.jpg", blur)
    thresh=cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    cv2.imwrite("PAN/03index_thresh.jpg", thresh)
    # cv2.imshow("threshold", thresh)
    # cv2.waitKey(2000)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (11,23))
    cv2.imwrite("PAN/04index_kernel.jpg",kernel)
    # cv2.imshow("threshold", kernel)   
    # cv2.waitKey(2000)
    dilate=cv2.dilate(thresh, kernel, iterations=1)
    cv2.imwrite("PAN/05index_dilate.jpg", dilate)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts=sorted(cnts,key=lambda x:cv2.boundingRect(x)[0])

    
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)

        if h <150 and h>50 and x>150 and x<1450 and y>150 and y<800:
            roi = gray[y:y+h, x:x+h]
            cv2.rectangle(gray, (x,y), (x+w, y+h), (36,255,12), 2)
            ocr_result = pytesseract.image_to_string(roi, lang='hin', config='--psm 6')
            for item in ocr_result:
                if item in combined_allowlist:
                    results.append(item)

                

image=cv2.imread("PAN/PAN2.jpg")
combined_allowlist = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 अ आ इ ई उ ऊ ऋ ए ऐ ओ औ क ख ग घ च छ ज झ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल व श ष स ह । ॥ ० १ २ ३ ४ ५ ६ ७ ८ ९'

gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("PAN/01index_gray.png", gray)

reader = easyocr.Reader(['hi', 'en'])
results=[]

ocr(gray,results,reader)


cv2.imwrite("PAN/panB_BOX.jpg", gray)
print(results)
