import cv2
from PIL import Image
import ocr

im_file = "./image.jpg"
im=Image.open(im_file)
print(im.size)
im.rotate(90)
im.show()
im.save("imageCopy.jpg")

