#nputil_v3.py
#python scripts for manipulating Numpy arrays
# XY-pair array is [[x0,y0], [x1,y1], ...]
# 20160223 - ljr - code cleanup
# 20160329 - ljr - fixed get_half_max routine to exit with NaN when not found

import sys
import numpy as np

# Define global variables for accessing x and y values inside arrays
__X = 0 
__Y = 1

################################################################################
def swap_columns(array,i,j):
    """Returns a copy of the Numpy array with columns i and j being swapped
    Arguments:
       array(numpy 2d array): input array 
       i(int): column number to swap
       j(int): column number to swap
    """
    temp = np.copy(array)
    temp[:, [i, j]] = temp[:, [j, i]]
    return temp

def runsum(array, dir=0):
    """Returns a Numpy array containing the running sum in direction dir
    0 (default) along a row and 1 along a column

    Arguments:
       array(numpy 2d array): input array

    """
    if dir == 0:
        # along a row
        return np.cumsum(array).reshape(np.shape(array))
    else:
        # along a column
        return np.cumsum(array.T).reshape(np.shape(array.T)).T

def remove_zeros(xy):
    """Returns new Numpy array of same shape 
    where zeros have been removed from 1D array 
    or x=y=0 pairs have been removed from xy-pair Numpy array
    
    Arguments:
       xy(numpy array): input array
    """
    if np.ndim(xy) == 1:
        mask = np.where(xy != 0.0)[0]
        return xy[mask]
    elif np.ndim(xy) == 2:
        mask = np.where((xy[:, __X] != 0.0) & (xy[:, __Y] != 0.0))[0]
        return xy[mask, :]
    else:
        print "nputil.remove_zeros can't process input array"
        exit()

def remove_zero_pairs(xy):
    """Returns new xy-pair Numpy array where x=y=0 pairs have been removed

    Arguments:
       xy(numpy array): input array
    """
    mask = np.where((xy[:, __X] != 0.0) & (xy[:, __Y] != 0.0))[0]
    return xy[mask, :]

def apply_threshold_old(xy, thres):
    """Returns new Numpy array of same shape
    where values are above threshold in 1D array 
    xy-pair Numpy array where y-value >= thres

    Arguments:
       xy(numpy array): input array
       thres(int or double): threshold value
    """
    if np.ndim(xy) == 1:
        mask = np.where(xy >= thres)[0]
        return xy[mask]
    elif np.ndim(xy) == 2:
        mask = np.where(xy[:, __Y] >= thres)[0]
        return xy[mask, :]
    else:
        print "nputil.apply_threshold can't process input array"
        exit()

def apply_threshold(xy, thres):
    """Returns new Numpy array of same shape where values below threshold are 
    replaced with zeros 1D array and xy-pair Numpy array where y-value >= thres.

    Arguments:
       xy(numpy array): input array
       thres(int or double): threshold value
    """
    temp = np.zeros_like(xy)
    if np.ndim(xy) == 1:
        mask = np.where(xy >= thres)[0]
        temp[mask] = xy[mask]
        return temp
    elif np.ndim(xy) == 2 and np.shape(xy)[1] == 2:
        # xy pairs, not x, y vectors
        mask = np.where(xy[:, __Y] >= thres)[0]
        temp[mask, :] = xy[mask, :]
        return temp
    else:
        print "nputil.apply_threshold can't process input array"
        exit()

def apply_limits(xy, limits):
    """Returns new Numpy array of same shape 
    where limits[0] <= x value <= limits[1]
    for either 1D or xy-pair Numpy array

    Arguments:
       xy(numpy array): input array
       limits(list double): [xmin, xmax]

    """
    if limits:
        if np.ndim(xy) == 1:
            mask = np.where((xy >= limits[0]) & (xy <= limits[1]))[0]
            return xy[mask]
        elif np.ndim(xy) == 2:
            mask = np.where((xy[:, __X] >= limits[0]) & (xy[:, __X] <= limits[1]))[0]
            return xy[mask, :]
        else:
            print "nputil.apply_limits can't process input array"
            exit()
    else:
        return xy

def get_x(xy):
    """Returns the x-vector from xy-pair Numpy array.
    Arguments:
       xy(numpy array): input array
    """
    return xy[:,__X]

def get_y(xy):
    """Returns the y-vector from xy-pair Numpy array.

    Arguments:
       xy(numpy array): input array
    """
    return xy[:,__Y]

def get_x_y(xy):
    """Returns two Numpy vectors taken from the xy-pair Numpy array.
    Arguments:
       xy(numpy array): input array
    """
    return get_x(xy), get_y(xy)

def get_xy(x,y):
    """Returns the ordered xy-pairs in Numpy array.
    Arguments:
       xy(numpy array): input array
    """
    return np.vstack((x, y)).T

def get_max_pair(xy):
    """Returns an xy-pair from a Numpy array where y has the max value.

    Arguments:
       xy(numpy array): input array
    """
    return xy[np.argmax(xy[:, __Y])]

def get_max_index(xy):
    """Returns the Numpy array index of xy-pair with maximum y.

    Arguments:
       xy(numpy array): input array
    """
    return np.argmax(xy[:, __Y])

def get_max_val(xy):
    """Returns the maximum y value in the xy-pair array.
    Arguments:
       xy(numpy array): input array
    """
    return np.max(xy[:, __Y])

def get_min_pair(xy):
    """Returns an xy-pair from a Numpy array where y is minimum.
    Arguments:
       xy(numpy array): input array
    """
    return xy[np.argmin(xy[:, __Y])]

def get_min_index(xy):
    """Returns the Numpy array index of xy-pair with maximum y.
    Arguments:
       xy(numpy array): input array
    """
    return np.argmin(xy[:, __Y])

def get_min_val(xy):
    """Returns the minimum y value in the xy-pair array.
    Arguments:
       xy(numpy array): input array
    """
    return np.min(xy[:, __Y])

def get_pvr(xy):
    """Returns the y Peak to valley ratio for xy-pair Numpy array
    The minimum must be > 0.
    Arguments:
       xy(numpy array): input array
    """
    if np.min(xy[:, __Y]) > 0.0:
        return np.max(xy[:, __Y]) / np.min(xy[:, __Y])
    else:
        return float('nan')

def get_fwhm(xy):
    """Returns FWHM (deg) of xy-pair Numpy array.

    Arguments:
       xy(numpy array): input array

    Returns:
       phase diff between upper and lower half-max points
    """
    xlo = get_halfmax_x(xy, 'lower')
    xup = get_halfmax_x(xy, 'upper')
    return xup-xlo
    
def get_halfmax_x(xy, half):
    """Returns the interpolated x coord value corresponding to the halfmax_val
       in either the lower or upper half of the xy-pair Numpy array.

    Arguments:
       xy(numpy array): input array
       half (str): 'left' or 'lower' or 'right' or 'upper'

    Returns:
       Interpolated x coord corresponding to half_max val
    """
    max_index = get_max_index(xy)
    if half in  ["lower", "left"]:
        inc = -1
        stop_ind = 0
    elif half in ["upper", "right"]:
        inc = +1
        stop_ind = len(xy)
    else:
        print "half argument must be either lower/left or upper/right"
        sys.stop()
    halfmax = 0.5 * get_max_val(xy)
    for i in range(max_index, stop_ind, inc):
        if xy[i][__Y] <= halfmax:
            j=i
            break
    else: #loop fell through without finding halfmax
        print "could not find " + half + "half max point"
        return float("nan")

    x1 = xy[j][__X]
    x2 = xy[j-inc][__X]
    y1 = xy[j][__Y]
    y2 = xy[j-inc][__Y]
    xcoor = (x1-x2)/(y1-y2)*(halfmax-y2) + x2
    return xcoor
