import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


### TODO 1: Read an Image and convert it into a floating point array with values between 0 and 1. 
###         You can assume a color image
def imread(filename):
    img = Image.open(filename).convert('RGB')
    img_mtrx = np.array(img)/255
    return img_mtrx

### TODO 2: Convolve an image (m x n x 3 or m x n) with a filter(l x k). Perform "same" filtering. 
###         Apply the filter to each channel if there are more than 1 channels
def convolve(img, filt):
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
    # 
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


# ## Image gradients

### TODO 4: Compute the image gradient. 
### First convert the image to grayscale by using the formula:
### Intensity = Y = 0.2125 R + 0.7154 G + 0.0721 B
### Then convolve with a 5x5 Gaussian with standard deviation 1 to smooth out noise. 
### Convolve with [[0.5, 0, -0.5]] to get the X derivative on each channel
### convolve with [[0.5],[0],[-0.5]] to get the Y derivative on each channel
### Return the gradient magnitude and the gradient orientation (use arctan2)
def gradient(img):
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

### TODO 5: Write a function to check the distance of a set of pixels from a line parametrized by theta and c. 
### The equation of the line is: x cos(thheta) + y sin(theta) + c = 0
### The input x and y are arrays representing the x and y coordinates of each pixel
### Return a boolean array that indicates True for pixels whose distance is less than the threshold
def check_distance_from_line(x, y, theta, c, thresh):
    # line: x cos(theta) + y sin(theta) + c = 0
    # points: [x1,x2,...,xn]; [y1,y2,...,yn]
    dist_fun = lambda x, y : abs(x*np.cos(theta) + y*np.sin(theta) + c)    
    dist = dist_fun(x, y)
    return dist < thresh


### TODO 6: Write a function to draw a set of lines on the image. The `lines` input is a list of (theta, c) pairs. Each line must appear as red on the final image
### where every pixel which is less than thresh units away from the line should be colored red
def draw_lines(img, lines, thresh):
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


### TODO 7: Do Hough voting. You get as input the gradient magnitude and the gradient orientation, as well as a set of possible theta values and a set of possible c
### values. If there are T entries in thetas and C entries in cs, the output should be a T x C array. Each pixel in the image should vote for (theta, c) if:
### (a) Its gradient magnitude is greater than thresh1
### (b) Its distance from the (theta, c) line is less than thresh2, and
### (c) The difference between theta and the pixel's gradient orientation is less than thresh3
def hough_voting(gradmag, gradori, thetas, cs, thresh1, thresh2, thresh3):
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


### TODO 8: Find local maxima in the array of votes. A (theta, c) pair counts as a local maxima if 
### (a) its votes are greater than thresh, and 
### (b) its value is the maximum in a nbhd x nbhd neighborhood in the votes array.
### Return a list of (theta, c) pairs

def localmax(votes, thetas, cs, thresh, nbhd):
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


# Final product: Identify lines using the Hough transform    
def do_hough_lines(filename):

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

