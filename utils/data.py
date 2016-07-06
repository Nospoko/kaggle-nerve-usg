import cv2
import h5py
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

class HdfDataset():
    """ h4py based dataset """
    def __init__(self, datapath):
	""" Please make sure file exists """
	self.path = datapath

	# Iteration helpers
	self._epochs_completed = 0
	self._index_in_epoch = 0

	# TODO Consider 'info' data-sub-set
	with h5py.File(datapath) as db:
	    # Useful information cout
	    print 'Accessing hdf file with fields:'
	    def visitor(arg): print arg
	    db.visit(visitor)
	    self._nof_examples = db['training/signals'].shape[0]

	print 'File with {} records'.format(self._nof_examples)

	# Prepare randomized data accessing iterators
	self._ids = np.arange(self._nof_examples)
	# np.random.shuffle(self._ids)

    def next_batch(self, batch_size, shuffle = False):
	""" Use this for training only """
	start = self._index_in_epoch
	self._index_in_epoch += batch_size

	if self._index_in_epoch > self._nof_examples:
	    # Finished epoch
	    self._epochs_completed += 1
	    print 'Data epochs done:', self._epochs_completed

	    # Shuffle the data accessing iterators (will fail for veery big)
	    # np.random.shuffle(self._ids)

	    # Start next epoch
	    start = 0
	    self._index_in_epoch = batch_size
	    assert batch_size <= self._nof_examples

	end = self._index_in_epoch

	# Get random row numbers
	ids = self._ids[start : end]

	# h5py only accepts sorted lists
	ids.sort()

	with h5py.File(self.path) as db:
	    signals = db['training/signals'][ids.tolist()]
	    labels = db['training/labels'][ids.tolist()]

	return signals, labels

    def validation_batch(self):
	""" Return a batch never used for training """
	with h5py.File(self.path) as db:
	    # Magical operator of reading from file straight into np.array
	    signals = db['validation/signals'][()]
	    labels = db['validation/labels'][()]
	    infos = db['validation/infos'][()]

	return signals, labels, infos
