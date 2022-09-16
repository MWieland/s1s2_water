import json
import numpy as np
import random


def split_samples(samples_file, img_key, msk_key, split_key, split_value, exclude_substr=[], seed=0):
    """Splits samples into train, val and test subsets based on samples geojson file."""
    with open(samples_file) as f:
        splits = json.load(f)
        img_splits = [
            split["properties"][img_key] + ".tif"
            for split in splits["features"]
            if split_value in split["properties"][split_key]
            and split["properties"][img_key] is not None
            and not any(ids in split["properties"][img_key] for ids in exclude_substr)
        ]
        msk_splits = [
            split["properties"][msk_key] + ".tif"
            for split in splits["features"]
            if split_value in split["properties"][split_key]
            and split["properties"][msk_key] is not None
            and not any(ids in split["properties"][img_key] for ids in exclude_substr)
        ]

    # random shuffle images and masks equally
    z = list(zip(img_splits, msk_splits))
    random.seed(seed)
    random.shuffle(z)
    img_splits, msk_splits = zip(*z)

    return img_splits, msk_splits


def scale_min_max(array, min, max):
    """Scales an image from range [min, max] to [0,1] with fixed values for min and max.
    Example values for 8bit image: min=0, max=255; 11bit image: min=0, max=2047; 16bit image: min=0, max=65535
    NOTE: Check the original image radiometric resolution to choose the correct values, NOT the dtype. For Ikonos for
    example dtype is uint16 but actual radiometric resolution is only 11bit, which requires min=0, max=2047."""
    bands = []
    for i in range(array.shape[2]):
        bands.append(((np.clip(array[:, :, i], min, max).astype(np.float32) - min) / (max - min + 1e-8)))
    return np.dstack(bands)


def rolling_window(array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True):
    """This method applies a rolling (moving) window to an ndarray.

    :param array: array to which the rolling window is applied (array_like).
    :param window: Either a single integer to create a window of only the last axis or a
        tuple to create it for the last len(window) axes. 0 can be used as a to ignore a
        dimension in the window (int or tuple).
    :param asteps: aligned at the last axis, new steps for the original array, ie. for
        creation of non-overlapping windows (tuple).
    :param wsteps: steps for the added window dimensions. These can be 0 to repeat values
        along the axis (int or tuple (same size as window)).
    :param axes: if given, must have the same size as window. In this case window is
        interpreted as the size in the dimension given by axes. IE. a window
        of (2, 1) is equivalent to window=2 and axis=-2 (int or tuple)
    :param toend: if False, the new dimensions are right after the corresponding original
        dimension, instead of at the end of the array. Adding the new axes at the
        end makes it easier to get the neighborhood, however toend=False will give
        a more intuitive result if you view the whole array (bool).
    :returns: a view on `array` which is smaller to fit the windows and has windows added
        dimensions (0s not counting), ie. every point of `array` is an array of size
        window. (ndarray).

    Examples: \n
    >>> a = np.arange(9).reshape(3,3)
    >>> rolling_window(a, (2,2))
    array([[[[0, 1],
             [3, 4]],
            [[1, 2],
             [4, 5]]],
           [[[3, 4],
             [6, 7]],
            [[4, 5],
             [7, 8]]]])

    Or to create non-overlapping windows, but only along the first dimension: \n
    >>> rolling_window(a, (2,0), asteps=(2,1))
    array([[[0, 3],
            [1, 4],
            [2, 5]]])

    Note that the 0 is discared, so that the output dimension is 3: \n
    >>> rolling_window(a, (2,0), asteps=(2,1)).shape
    (1, 3, 2)

    This is useful for example to calculate the maximum in all (overlapping) \n
    2x2 submatrixes: \n
    >>> rolling_window(a, (2,2)).max((2,3))
    array([[4, 5],
           [7, 8]])

    Or delay embedding (3D embedding with delay 2): \n
    >>> x = np.arange(10)
    >>> rolling_window(x, 3, wsteps=2)
    array([[0, 2, 4],
           [1, 3, 5],
           [2, 4, 6],
           [3, 5, 7],
           [4, 6, 8],
           [5, 7, 9]])
    """
    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int)

    if axes is not None:
        axes = np.atleast_1d(axes)
        w = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            w[axis] = size
        window = w

    # Check if window is legal:
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if len(array.shape) < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension.")

    _asteps = np.ones_like(orig_shape)
    if asteps is not None:
        asteps = np.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError("`asteps` must be either a scalar or one dimensional.")
        if len(asteps) > array.ndim:
            raise ValueError("`asteps` cannot be longer then the `array` dimension.")
        # does not enforce alignment, so that steps can be same as window too.
        _asteps[-len(asteps) :] = asteps

        if np.any(asteps < 1):
            raise ValueError("All elements of `asteps` must be larger then 1.")
    asteps = _asteps

    _wsteps = np.ones_like(window)
    if wsteps is not None:
        wsteps = np.atleast_1d(wsteps)
        if wsteps.shape != window.shape:
            raise ValueError("`wsteps` must have the same shape as `window`.")
        if np.any(wsteps < 0):
            raise ValueError("All elements of `wsteps` must be larger then 0.")

        _wsteps[:] = wsteps
        _wsteps[window == 0] = 1
    wsteps = _wsteps

    # Check that the window would not be larger then the original:
    if np.any(orig_shape[-len(window) :] < window * wsteps):
        raise ValueError("`window` * `wsteps` larger then `array` in at least one dimension.")

    new_shape = orig_shape

    # For calculating the new shape 0s must act like 1s:
    _window = window.copy()
    _window[_window == 0] = 1

    new_shape[-len(window) :] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    # make sure the new_shape is at least 1 in any "old" dimension (ie. steps
    # is (too) large, but we do not care.
    new_shape[new_shape < 1] = 1
    shape = new_shape

    strides = np.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window) :] * wsteps

    # The full new shape and strides:
    if toend:
        new_shape = np.concatenate((shape, window))
        new_strides = np.concatenate((strides, new_strides))
    else:
        _ = np.zeros_like(shape)
        _[-len(window) :] = window
        _window = _.copy()
        _[-len(window) :] = new_strides
        _new_strides = _

        new_shape = np.zeros(len(shape) * 2, dtype=int)
        new_strides = np.zeros(len(shape) * 2, dtype=int)

        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides

    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]

    return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)


def tile_array(array, xsize=512, ysize=512, overlap=0.1, padding=True):
    """This method splits an ndarray into equally sized tiles
    with overlap.

    :param array: Numpy array of shape (rows, cols, bands) (Ndarray).
    :param xsize: Xsize of tiles (Integer).
    :param ysize: Ysize of tiles (Integer).
    :param overlap: Overlap of tiles between 0.0 and 1.0 (Float).
    :param padding: Pad array before tiling it to ensure that the whole array is used (Boolean).
    :returns: Numpy array of shape(tiles, rows, cols, bands) (Ndarray)
    """
    # get dtype, rows, cols, bands and dtype from first file
    dtype = array.dtype
    rows = array.shape[0]
    cols = array.shape[1]
    if array.ndim == 3:
        bands = array.shape[2]
    elif array.ndim == 2:
        bands = 1

    # get steps
    xsteps = int(xsize - (xsize * overlap))
    ysteps = int(ysize - (ysize * overlap))

    if padding is True:
        # pad array on all sides to fit all tiles.
        # replicate values here instead of filling with nan.
        # nan padding would cause issues for standardization and classification later on.
        ypad = ysize + 1
        xpad = xsize + 1
        array = np.pad(
            array,
            (
                (int(ysize * overlap), ypad + int(ysize * overlap)),
                (int(xsize * overlap), xpad + int(xsize * overlap)),
                (0, 0),
            ),
            mode="symmetric",
        )

    # tile the data into overlapping patches
    # this skips any tile at the end of row and col that exceeds the shape of the input array
    # therefore padding the input array is needed beforehand
    X_ = rolling_window(array, (xsize, ysize, bands), asteps=(xsteps, ysteps, bands))

    # access single tiles and write them to file and/or to ndarray of shape (tiles, rows, cols, bands)
    X = []
    for i in range(X_.shape[0]):
        for j in range(X_.shape[1]):
            X.append(X_[i, j, 0, :, :, :])

    return np.asarray(X, dtype=dtype)
