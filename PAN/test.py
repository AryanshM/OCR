import cv2 
from PIL import Image
import pytesseract
import numpy as np
import easyocr
                

image=cv2.imread("PAN/preprocessed.jpg")

results=[]
    
reader = easyocr.Reader(['hi', 'en'])

ocr_result = reader.readtext(image, detail=0, paragraph=False)
for item in ocr_result:
    results.append(item)

# OCR ENDS

print(results)
