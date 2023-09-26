import numpy as np
import cv2
import imutils
from skimage.filters import threshold_local
from glob import glob
def scan(input_path):
    my_path = sorted(glob(f'{input_path}/input/photo/*'))
    args_image = my_path[0]
    #args_image ="C:/Users/afrod/MusicAssistant/d.jpg"

    image = cv2.imread(args_image)
    image=cv2.resize(image,(500,500))
    orig = image.copy()

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayImageBlur = cv2.blur(grayImage,(2,2))
    edgedImage = cv2.Canny(grayImageBlur, 100, 300, 3)


    allContours = cv2.findContours(edgedImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    allContours = imutils.grab_contours(allContours)
    allContours = sorted(allContours, key=cv2.contourArea, reverse=True)[:1]

    perimeter = cv2.arcLength(allContours[0], True)
    ROIdimensions = cv2.approxPolyDP(allContours[0], 0.02*perimeter, True)
    cv2.drawContours(image, [ROIdimensions], -1, (0,255,0), 2)

    ROIdimensions = ROIdimensions.reshape(4,2)
    rect = np.zeros((4,2), dtype="float32")
    s = np.sum(ROIdimensions, axis=1)
    rect[0] = ROIdimensions[np.argmin(s)]
    rect[2] = ROIdimensions[np.argmax(s)]
    diff = np.diff(ROIdimensions, axis=1)
    rect[1] = ROIdimensions[np.argmin(diff)]
    rect[3] = ROIdimensions[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    widthA = np.sqrt((tl[0] -tr[0])**2 + (tl[1] - tr[1])**2 )
    widthB = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
    heightB = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0,0],
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]], dtype="float32")

    transformMatrix = cv2.getPerspectiveTransform(rect, dst)
    scan = cv2.warpPerspective(orig, transformMatrix, (maxWidth, maxHeight))

    scanGray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)


    T = threshold_local(scanGray, 9, offset=8, method="gaussian")
    scanBW = (scanGray > T).astype("uint8") * 255
    cv2.imwrite(f'{input_path}/output/photo/scanned.jpg', scanBW)
