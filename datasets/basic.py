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

def radians2compass(rad):
    """ Change continuous angle into discrete directions """
    out = rad * 8 / (2 * np.pi)

    return np.floor(out)

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

    dataset = []
    # Sample image randomly
    howmany = 500
    x_points = np.random.randint(half, mask.shape[1] - half, howmany)
    y_points = np.random.randint(half, mask.shape[0] - half, howmany)
    for new_y, new_x in zip(y_points, x_points):
	# Get spans
	x_left, x_right = new_x - half, new_x + half
	y_left, y_right = new_y - half, new_y + half

	# Get angle?
	dx = nerve_x - new_x
	dy = nerve_y - new_y

	# TODO Just make sure the same transformation is used throughout
	phi = np.arctan2(dx, dy)%(np.pi*2)
	# +4 compensates for the first 3 labels
	label = radians2compass(phi) + 0

	dataset.append((label, img[y_left : y_right, x_left : x_right]))

    return dataset

def make_location_label(a, b, c, d):
    """ Nerve pos and window center into nerve location transformation """
	# FIXME Test that later, begin with just the angles
	# If nerve is inside the view label with section numbber (1-4? 1-9?)
	# if x_left < nerve_x < x_right and y_left < nerve_y < y_right:
	    # x_on_left = abs(x_left-nerve_x) > abs(x_right-nerve_x)
	    # y_on_left = abs(y_left-nerve_y) > abs(y_right-nerve_y)
	    # if x_on_left and y_on_left:
	    #     label = 0
	    # if not x_on_left and y_on_left:
	    #     label = 1
	    # if x_on_left and not y_on_left:
	    #     label = 2
	    # if not x_on_left and not y_on_left:
	    #     label = 3
	    # print 'nerve in view!, at spot: ', label
	# else:
	    # If nerve is out of the view label it 
	    # with angle point at it (more or less)
            
    pass

