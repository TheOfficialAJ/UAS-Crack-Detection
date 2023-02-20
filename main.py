import cv2
import sys
import numpy as np


def detectCrack(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return canny, thresh


if __name__ == '__main__':
    path = sys.argv[1]
    img = cv2.imread(path)
    cv2.imshow("image", img)
    canny, thresh = detectCrack(img)
    canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    canny *= np.array((0, 1, 0), dtype=np.uint8)
    thresh *= np.array((0, 1, 0), dtype=np.uint8)

    overlay = cv2.addWeighted(thresh, 0.5, img, 0.5, 0, canny)
    cv2.imshow("Detected Cracks", canny)
    cv2.imshow("Overlay", overlay)
    cv2.waitKey(0)
