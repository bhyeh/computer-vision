import numpy as np
from PIL import Image


def imread(filename):
    """
    Reads an image and converts to floating point array with values normalized between 0 and 1
    
    Parameters
    ----------
    filename : str

    Returns
    -------
    img_mtrx : ndarray

    """
    img = Image.open(filename).convert('RGB')
    img_mtrx = np.array(img)/255
    return img_mtrx

### TODO 2: Convolve an image (m x n x 3 or m x n) with a filter(l x k). Perform "same" filtering. 
###         Apply the filter to each channel if there are more than 1 channels
def convolve(img, filt):
    """
    Convolves an image with a filter

    Parameters
    ----------
    img : ndarray
        (m x n x ch) 

    filter : ndarray
        (l x k) 

    Returns
    -------
    convolved_img : ndarray
        (m x n x ch)
    
    """
    # Reshape to be (m x n x c)
    img = img.reshape(img.shape[0],img.shape[1], -1)
    filt = filt.reshape(filt.shape[0], filt.shape[1], -1)
    # Dim
    m, n, c =img.shape
    l, k, _ = filt.shape
    # Flip flip filter! 
    filt = np.flip(np.flip(filt, axis=0),axis=1)
    # Padding
    pad_width = int((k-1)/2)
    pad_height = int((l-1)/2)
    img_padded = np.pad(img, pad_width=((pad_height, pad_height), (pad_width, pad_width), (0, 0)))
    # Convolve
    convolved_img = np.zeros((m,n,c))
    for i in np.arange(m): # iterate down each row
        for j in np.arange(n):# iterate across each column            
            # neighborhood
            neighborhood = img_padded[i:i+l, j:j+k,:]
            # element wise multiplication 
            product = np.multiply(filt, neighborhood)
            # convolved s(f)[i,j] 
            convolved_img[i,j,:] = np.sum(product, axis=(0,1))
    # squeeze image if mxnx1; else return mxnxc
    if (convolved_img.shape[-1] == 1): 
        convolved_img = np.squeeze(convolved_img, axis=-1)
    return convolved_img

### TODO 3: Create a gaussian filter of size k x k and with standard deviation sigma
def gaussian_filter(k, sigma):
    """
    Creates gaussian (k x k) filter with standard deviation sigma

    Parameters
    ----------
    k : int

    sigma : float

    Returns
    -------
    kernel : ndarray
    
    """
    width = (k-1)/2
    interval = np.arange(-width, width+1, 1)
    # x and y
    x, y = np.meshgrid(interval, interval)
    # gaussian constant
    gaussian_const = 1 /(2*np.pi*sigma**2)
    # kernel
    kernel = (-1*(x**2 + y**2))/(2*sigma*sigma)
    kernel = gaussian_const*np.exp(kernel)
    kernel /= np.abs(kernel).sum()
    return kernel

def gradient(img):
    """
    Computes the gradient of image

    Parameters
    ----------
    img : ndarray

    Returns
    -------
    gradmag, gradori : ndarray

    """
    # grayscale image
    img = (0.2125 * img[:,:,0]) + (0.7154 * img[:,:,1]) + (0.0721 * img[:,:,2])
    # gaussian convolve 
    filt = gaussian_filter(5, 1)
    img = convolve(img, filt)
    # X and Y derivative
    xfilt = np.array([[0.50, 0, -0.50]])
    yfilt = np.array([[0.50],[0],[-0.50]])
    dx = convolve(img, xfilt)
    dy = convolve(img, yfilt)
    # gradient magnitude
    gradmag = np.sqrt(dx**2 + dy**2)
    gradori = np.arctan2(dy, dx)
    return gradmag, gradori

def check_distance_from_line(x, y, theta, c, thresh):
    """
    Checks the distance of a point, (x, y), to the line parametrized by theta and c

    Parameters
    ----------
    x, y : ndarray of int or float
        Input x and y are arrays representing the x and y coordinates of each pixel
    
    theta, c, thresh : int or float
        The equation of the line is: x cos(thheta) + y sin(theta) + c = 0

    Returns
    -------
    _ : ndarray of bool
        Boolean array indicating True for pixels whose distance is less than the threshold

    """
    # line: x cos(theta) + y sin(theta) + c = 0
    # points: [x1,x2,...,xn]; [y1,y2,...,yn]
    dist_fun = lambda x, y : abs(x*np.cos(theta) + y*np.sin(theta) + c)    
    dist = dist_fun(x, y)
    return dist < thresh

def draw_lines(img, lines, thresh):
    """
    Draws a set of lines on image

    Parameters
    ----------
    img : ndarray 

    lines : list of tuple
        List of (theta, c) pairs

    thresh : int or float
        Value to threshold distance from pixels and lines

    Returns
    -------
    img : ndarray
        Image with each line within threshold appearing as red
    
    """
    # img dim
    img = img.reshape(img.shape[0],img.shape[1], -1)
    m, n, _, = img.shape
    # indices
    row, col = np.indices((m,n))
    # lines
    for line in lines:
        theta, c = line
        bool_idx = check_distance_from_line(col.flatten(), row.flatten(), theta, c, thresh).reshape((m,n)) # mxn bool matrix
        img[bool_idx, 0] = 1
        img[bool_idx, 1:] = 0
    return img


def hough_voting(gradmag, gradori, thetas, cs, thresh1, thresh2, thresh3):
    """
    Performs Hough line voting

    Parameters
    ----------
    gradmag, gradori : ndarray

    thetas, cs : list of float

    thresh1, thresh2, thresh3 : float

    Returns
    -------
    voting_mtrx : ndarray of int
    
    """
    # dim
    gradmag = gradmag.reshape(gradmag.shape[0], gradmag.shape[1], -1)
    gradori = gradori.reshape(gradori.shape[0], gradori.shape[1], -1)
    m, n, _ = gradmag.shape # gradmag and gradori have same shape
    # voting mtrx
    T = np.size(thetas)
    C = np.size(cs)
    voting_mtrx = np.zeros((T,C)) # TxC possible candidate lines
    # condition (a)
    a_idx = gradmag > thresh1 # m x n x 1 boolean matrix
    # reduce pixel coordinates to search over
    row, col, _ = np.where(a_idx)
    for i in np.arange(T): # iterate down thetas
        theta = thetas[i]
        # condition (c)
        c_idx = abs(theta - gradori) < thresh3 
        for j in np.arange(C): # iterate across cs
            # temp matrix; mxnx1
            b_mtrx = np.full((m,n,1), False)
            # conditions
            c = cs[j]
            # condition (b)
            b_idx = check_distance_from_line(col, row, theta, c, thresh2)
            # fill bool
            b_mtrx[row[b_idx], col[b_idx], 0] = True            
            # count votes
            votes = np.sum(a_idx & b_mtrx & c_idx) 
            voting_mtrx[i,j] = votes # fill votes for (theta, c) candidate line
    return voting_mtrx

def localmax(votes, thetas, cs, thresh, nbhd):
    """
    Finds the local maxima in array of votes in neighborhood. Counts (theta, c) pair as local maxima if:
        (a) votes are greater than thresh, and
        (b) value is maximum in a nbhd x nbhd neighborhood in voting array

    Parameters
    ----------
    votes : ndarray

    thetas, cs : list of float

    thresh : int or float

    nbhd : int
        Describes neighborhood size (nbhd x nbhd)

    Returns
    -------
    pairs : list of tuple (theta, c)
    
    """
    # theta x cs dim
    T, C = votes.shape
    rows, cols = np.indices((T,C)) # coordinates of votes
    # border padding
    pad = int((nbhd-1)/2)
    votes_padded = np.pad(votes, pad_width=pad)
    # max (theta,c) pairs
    pairs = []
    # condition (a); reduce search space size
    a_idx = votes > thresh # TxC boolean matrix
    # (theta, c) rows and cols meeting condition (a)
    rows = np.unique(rows[a_idx]) 
    cols = np.unique(cols[a_idx])
    for i in rows: 
        theta = thetas[i]
        for j in cols:
            c = cs[j]
            vote = votes[i,j]
            # nbhd x nbhd neighborhood
            neighborhood = votes_padded[i:i+nbhd, j:j+nbhd]
            # condition (b)
            if (vote == np.amax(neighborhood)) & (vote > thresh):
                pairs.append((theta, c))
    return pairs

def do_hough_lines(filename):
    """
    Identifies lines using Hough transform

    Parameters
    ----------
    filename : str

    Returns
    -------
    result_img : float array

    lines : list of tuple (theta, c)

    """

    # Read image in
    img = imread(filename)

    # Compute gradient
    gradmag, gradori = gradient(img)

    # Possible theta and c values
    thetas = np.arange(-np.pi-np.pi/40, np.pi+np.pi/40, np.pi/40)
    imgdiagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    cs = np.arange(-imgdiagonal, imgdiagonal, 0.5)

    # Perform Hough voting
    votes = hough_voting(gradmag, gradori, thetas, cs, 0.1, 0.5, np.pi/40)

    # Identify local maxima to get lines
    lines = localmax(votes, thetas, cs, 20, 11)

    # Visualize: draw lines on image
    result_img = draw_lines(img, lines, 0.5)

    # Return visualization and lines
    return result_img, lines

