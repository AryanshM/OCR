import cv2 
from PIL import Image
import ocr
image_file = "./image.jpg"
img = cv2.imread(image_file)
# cv2.imshow("Original Image", img)
# cv2.waitKey(2000)


# INVERTED IMAGES - not necessary

image_file2="./bookpage.jpg"
img2=cv2.imread(image_file2)

inverted_image=cv2.bitwise_not(img2)
cv2.imwrite("./invertedBookPage.jpg", inverted_image)
def display(title ,image):
    cv2.imshow(title, image)
    cv2.waitKey(2000)
# display(inverted_image)
# RESCALING - not neccessary

#Binarization 
# first convert to grayscale
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

grey_image=grayscale(img2)
# display(grey_image)

# grayscaled
# now binarize the image

thresh, im_bw = cv2.threshold(grey_image, 200,230,cv2.THRESH_BINARY)
cv2.imwrite("./bookpage_bw.jpg",im_bw)
# display("Black and white", im_bw)

#NOISE Removal
def noise_removal(image):
    import numpy as np 
    kernel = np.ones((1,1), np.uint8)
   
    Image=cv2.dilate(image, kernel,iterations=1)

    image=cv2.erode(image,kernel,iterations=1)
    # erode thins the pixel down

    #image = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
   
    # image=cv2.medianBlur(image, 3)
    return(image)

no_noise = noise_removal(im_bw)
cv2.imwrite("./bookNoNoise.jpg", no_noise)
# display("Without noise", no_noise)

# Dilation and Erosion
# thinning and thickening the font

def thin_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((1,1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image=cv2.bitwise_not(image)
    return(image)
# eroded_image = thin_font(no_noise)
# cv2.imwrite("temp/eroded_image.jpg",eroded_image)
# display("eroded_image", eroded_image)

# rotated images / skewed images
# OCR only works with vertically aligned text
# we can guess the rotation og the bounding boxes slope 

def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print (len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    cv2.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle
# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage


# Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)

new = cv2.imread("./bookskewed.jpg")
fixed=deskew(new)
cv2.imwrite("fixed.jpg",fixed)
# display("before",cv2.imread("bookskewed.jpg"))
# display("fixed",fixed)

# removing Borders

def remove_borders(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours in the grayscale image
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and get the largest one
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = cntsSorted[-1]

    # Get bounding box coordinates
    x, y, w, h = cv2.boundingRect(cnt)

    # Crop the image using the bounding box
    crop = image[y:y+h, x:x+w]

    return crop
borderImage= "./borders.jpg"
borders=cv2.imread(borderImage)
no_borders = remove_borders(borders)
cv2.imwrite("./no_borders.jpg", no_borders)
display("No Borders", no_borders)




   