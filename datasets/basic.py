"""
    basic
    ****

    Set of basic data manipulative functions,
    set creators and data breeders
"""

import h5py
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
    # Let's say that there are 12 directions in which we
    # should be looking for the nerve
    howmany = 12
    label = rad * howmany/ (2 * np.pi)
    label = int(np.floor(label))

    one_hot = np.zeros(howmany)
    one_hot[label] = 1

    return one_hot

def chop_image(path, howmany = 10000):
    """ Dissect single record into easier to process tiles """
    # Load data
    img, mask = get_sample(path)

    # Get nerve center
    nerve_x, nerve_y = ud.get_mask_center(mask)

    print 'Chop chop chop', path

    # Prepare tiles (width was guessed)
    tile_width = 170
    half = tile_width/2

    # Prepare dataset container
    dataset = []
    # Copy image to be prepared for transformations
    picture = img[:]
    # Sample image randomly
    for it in range(howmany):
	new_x = np.random.randint(half, mask.shape[1] - half)
	new_y = np.random.randint(half, mask.shape[0] - half)

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

	# Change into a vector
	signal = picture[y_left : y_right, x_left : x_right]
	signal = signal.reshape((-1, ))

	dataset.append((signal, label))

	# Change image a little every so often
	if (it+1) % 50 == 0:
	    picture = ud.twist_image(img)

    return dataset

def prepare_data():
    """ Create in-ram dataset """
    # Get interesting images (label rather properly)
    paths = masked_records()

    # Prepare container for the full dataset
    dataset = []
    
    # FIXME Do not run this for every path or crash
    for path in paths[:10]:
	dataset += chop_image(path)

    return dataset

# TODO consider this
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

def make_hdf_set():
    """ Prepare storage for real usg images """
    dataset = prepare_data()

    # Make sure path exists
    filepath = 'data/augmented.storage'

    # We need numpy arrays of signals, labels 
    signals = np.array([dat[0] for dat in dataset])
    labels  = np.array([dat[1] for dat in dataset])

    # Number of records
    howmany = signals.shape[0]

    # Prepare training and validation separately
    perm = np.arange(howmany)
    np.random.shuffle(perm)

    # Split all of the arrays (make validation set 5000 long
    vids = perm[:5000]
    tids = perm[5000:]

    t_signals = signals[tids]
    v_signals = signals[vids]

    t_labels = labels[tids]
    v_labels = labels[vids]

    with h5py.File(filepath, 'w') as db:
	# Validation
	v_group = db.create_group('validation')

	v_group.create_dataset('signals', data = v_signals)
	v_group.create_dataset('labels',  data = v_labels)

	# Training
	t_group = db.create_group('training')

	t_group.create_dataset('signals', data = t_signals)
	t_group.create_dataset('labels',  data = t_labels)

    print 'Prepare data-file :', filepath

def make_huge_hdf_set():
    """ Same as below but with different module (hdf) """
    # Get interesting images (label rather properly)
    paths = masked_records()
    # Decide how many tile You want to generate from one image
    chunksize = 500

    # Calculate number of samples about to be generated
    howmany = len(paths) * chunksize
    print 'Making {} fake signal samples'.format(howmany)

    filepath = 'data/augmented_b.storage'

    with h5py.File(filepath, 'w') as db:
	# Prepare expandable datasets for signals and labels
	sigsize = 170**2
	sigset = db.create_dataset('training/signals',
				   shape = (howmany, sigsize), 
				   maxshape = (None, sigsize))

	# FIXME This must be negotiated with several places
	labsize = 12
	labset = db.create_dataset('training/labels',
				   shape = (howmany, labsize), 
				   maxshape = (None, labsize))

	# Add data in chunks
	sta = 0
	end = chunksize

	for path in paths:
	    print 'Stashing data {} >< {}, from file {}'.format(sta, end, path)

	    # For each image generate a miniset of *chunksize* samples
	    local_set = chop_image(path, chunksize)
	    signals = np.array([dat[0] for dat in local_set])
	    labels  = np.array([dat[1] for dat in local_set])

	    sigset[sta : end, :] = signals
	    labset[sta : end, :] = labels

	    sta += chunksize
	    end += chunksize
