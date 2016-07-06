import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_mask_center(mask):
    """ Extract center of an ellipse fit to the mask shape """
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    ellipse = cv2.fitEllipse(contours[0])

    cy, cx = ellipse[0]

    return cy, cx

def show_nine(pictures):
    """ Quick plot wrapper for showing a grid of 9 images """
    f, a = plt.subplots(3, 3, figsize=(9, 9))

    # pictures must be a list of 9 (at least)
    for it in range(3):
        for jt in range(3):
            idx = 3*it + jt
            a[it][jt].imshow(pictures[idx])
    plt.show()

def twist_image(pic):
    """ Perform some random perspective transformation on the image """
    # Make 'before' and 'after' corners
    before = [[0, 0], [0, pic.shape[1]],
               [pic.shape[0], 0], [pic.shape[0], pic.shape[1]] ]

    # Add some random twists
    after = [[p + np.random.randint(15) for p in pt] for pt in before]

    # OpenCV needs numpy
    before = np.float32(before)
    after  = np.float32(after)

    # Get transformation matrix (from before to after)
    M = cv2.getPerspectiveTransform(before, after)

    out = cv2.warpPerspective(pic, M, (pic.shape[1], pic.shape[0]))

    return out
