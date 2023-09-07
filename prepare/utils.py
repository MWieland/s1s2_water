import numpy as np


def scale_min_max(array, min=0, max=10000):
    bands = []
    for i in range(array.shape[2]):
        bands.append(array[:, :, i].astype(np.float32) / max)
    return np.dstack(bands)


def rolling_window(array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True):
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
    # get dtype, rows, cols, bands and dtype from first file
    dtype = array.dtype
    rows = array.shape[0]
    cols = array.shape[1]
    if array.ndim == 2:
        array = np.expand_dims(array, 2)
    bands = array.shape[2]

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
