"""
    basic
    ****

    Set of basic data manipulative functions,
    set creators and data breeders
"""

import cv2 as cv
import numpy as np
from glob import glob
from utils import data as ud

def get_records():
    """ Read all avaiable patients """
    # There are 2 files per record
    paths = glob('data/train/*mask*')

    # Make path general for image and mask
    paths = [path[:-9] for path in paths]

    return paths

def get_image(record, color = False):
    """ Load only the image """
    img_path = record + '.tif'

    if color:
	img = cv.imread(img_path)
    else:
	img = cv.imread(img_path, -1)

    return img

def get_mask(record, color = False):
    """ Load only the mask for given sample """
    mask_path = record + '_mask.tif'
    # Color are useful if You'd like to draw a contour on them
    if color:
	mask = cv.imread(mask_path)
    else:
	mask = cv.imread(mask_path, -1)

    return mask

def get_sample(record, color = False):
    """ Read picture and mask for given patient """
    mask = get_mask(record, color)
    img  = get_image(record, color)

    return img, mask

def masked_records():
    """ Most of the records have no mask as indication of no nerve visible (probably often mis-labeled) """
    paths = get_records()

    masked = []

    for path in paths:
	mask = get_mask(path)
	if mask.max() > 0.01:
	    masked.append(path)

    return masked

def chop_image(path):
    """ Dissect single record into easier to process tiles """
    # Load data
    img, mask = get_sample(path)

    # Get nerve center
    nerve_x, nerve_y = ud.get_mask_center(mask)

    print 'Nerve coordinates: ({}, {})'.format(nerve_y, nerve_x)

    # Prepare tiles (width was guessed)
    tile_width = 170
    half = tile_width/2

    # FIXME as a test just take some middlish point
    new_y, new_x = 122, 160

    # Get spans
    x_left, x_right = new_x - half, new_x + half
    y_left, y_right = new_y - half, new_y + half

    # Get angle?
    dx = nerve_x - new_x
    dy = nerve_y - new_y

    print dx, dy

    # dafuq, which way is that
    phi = np.arctan2(dx, dy)%(np.pi*2)

    return phi, mask[y_left : y_right, x_left : x_right]
