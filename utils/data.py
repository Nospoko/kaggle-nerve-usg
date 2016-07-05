import cv2 as cv
import numpy as np

def get_mask_center(mask):
    """ Extract center of an ellipse fit to the mask shape """
    ret, thresh = cv.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, 1, 2)
    ellipse = cv.fitEllipse(contours[0])

    cx, cy = ellipse[0]

    return cx, cy

