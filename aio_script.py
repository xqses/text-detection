## Notebook typically runs out of memory.
# However, it can be very hard to follow what's going on over several Python scripts
# Therefore, this is an All-In-One code script that can be run individually.

##############################################
################ Please note: ################
#
# This is a rather long script.
# It may take a long time to follow along completely.
# The general structure is to define all "shared" methods first.
# Afterwards it will define each part as a function and below this the functions used in that part.
# Below this, the scripts that actually load images and call parts are placed.
# Typically, these do not need to be altered much.
# At the very end, various helpful functions are placed in case they are needed or desired to make changes.
#
# Sometimes, using the show_img method can be useful to spot problems on-the-fly.
# However, for some purposes the images displayed in this way may be inadequate (they might be too small to spot details).
# You are then advised to write the images to disk and evaluate them that way. Another helpful function for this is the OpenCV imshow function.
# This is not implemented in a short-hand way, so you will have to do this manually where you need it, or define this function yourself.
#
##############################################
##############################################

### Retrieve imports
import cv2
import numpy as np
import os
from scipy import ndimage as ndi
from morphology import get_morph
from matplotlib import pyplot as plt
from skimage.filters import gabor
from skimage.feature import peak_local_max
from skimage.morphology import white_tophat
from line_cleaning import line_cleaning
from copy import deepcopy
from PST import PST

# Define your OS here. Currently, accepted values are "win" or "linux"
# This value is only used to define path names for loading images,
# and if paths do not work properly, you can edit these calls yourself
os_selected = "win"

##########################################################################################

## Define various image retrieval methods
class ImageHelper:
    def __init__(self):
        self.id = 0

    # Avoid using arrays of images unless specific reasons require this
    def set_img_array(self, name, n_img, img_format):
        self.img_array = [None] * n_img
        for i in range(n_img):
            self.img_array[i] = cv2.imread(name + str(i + 1) + img_format)

    def get_img_generator(self, name, img_format, n_img):
        for i in range(n_img):
            yield cv2.imread(name + str(i + 1) + img_format)

    def get_img_array(self, n_img):
        return self.img_array

    def get_img(self, name):
        return cv2.imread(name)

def write_img(name, img):
    status = cv2.imwrite(name, img)
    return status

def open_file(name: str, multiple: bool, n_img: int):
    if multiple:
        imgs_obj = ImageHelper()
        imgs_obj = imgs_obj.get_img_generator(name=name, n_img=n_img, img_format=".JPG")
        return imgs_obj
    else:
        img_obj = ImageHelper()
        img_obj.img = img_obj.get_img(name=name)
        return img_obj

def show_img(img, title):
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.title(title)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

##########################################################################################

#### Different image resizing methods (both are used)

def pyramid(img, direction: str, iterations: int):
    ## The pyramid iterates by factors of two in either direction
    # Thus it retains its width to height proportion
    has_color = True
    if len(img.shape) < 3:
        has_color = False
    if direction == "down":
        for i in range(iterations):
            if has_color:
                rows, cols, _channels = map(int, img.shape)
                ## Alternative syntax:
                # rows, cols, _channels = img.shape[0], img.shape[1], img.shape[2]
            else:
                ## If it is a grayscale image, there's no such thing as _channels
                rows, cols = map(int, img.shape)
            img = cv2.pyrDown(img, dstsize=(cols // 2, rows // 2))
    else:
        for i in range(iterations):
            if has_color:
                rows, cols, _channels = map(int, img.shape)
            else:
                rows, cols = map(int, img.shape)
            img = cv2.pyrUp(img, dstsize=(cols * 2, rows * 2))
    return img

def resize(img, dim):
    return cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

##########################################################################################

### Define booleans

# Provided images are already scaled and have shading corrected
## Moreover, shading correction can go wrong - while it is highly useful on photographs,
## it may corrupt scanned images. Therefore, be careful in using it.
scale = False
correct_shading = False

# Provided images do not have book cropped
seg_book = True

# Be careful with the noise clearing algorithm
# It runs slowly and may corrupt the photographed images
# The noise removal algorithm seems to work well on scanned images, however
noise = False

# Sharpening and normalization are used to increase the contrast between text / lines and background
sharpen = True
normalize = True

# PST is a patented algorithm which is restricted for use in commercial applications.
# An acceptable alternative is Canny. Declare whether to use PST.
uses_PST = True

def part_one(img):
    ## Book segmentation
    scale_copy, segment_copy, sharp_copy = None, None, None

    blurred = cv2.GaussianBlur(img, (5, 5), 5 / 6)
    if scale:
        if seg_book:
            unscaled_copy = deepcopy(blurred)
            scale_copy, img = call_scale(blurred, direction="down", iters=2)
        else:
            scale_copy, img = call_scale(blurred, direction="down", iters=2)

    if seg_book:
        if correct_shading:
            img = correct_shading(img).astype(np.uint8)
        # padded = add_padding(img)
        # sharp_copy, img = call_sharpen(img)
        gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ret, thresh = cv2.threshold(gs, 0, 255, cv2.THRESH_OTSU)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([0, 0, 100])
        higher_blue = np.array([60, 60, 255])

        lower_magenta = np.array([100, 0, 100])
        higher_magenta = np.array([255, 40, 255])

        b_mask = cv2.inRange(hsv_img, lower_blue, higher_blue)
        m_mask = cv2.inRange(hsv_img, lower_magenta, higher_magenta)

        final = (gs & b_mask) | (gs & m_mask)
        thresh = get_morph(
            get_morph(final, morph="close", sz=(7,7), kernel = "ones", threshold=True),
            morph="erode", sz=(3,3), kernel="ones")
        # show_img(thresh.astype(np.uint8), "thresh in main")
        # padded = add_padding(thresh)
        # show_img(padded, "second padding")
        img, book_x, book_y, book_w, book_h = segmentation(thresh.astype(np.uint8), img)
        segment_copy = deepcopy(img)
        # show_img(img, "img")
    else:
        segment_copy = deepcopy(img)
        # show_img(img, "img")

    if noise:
        img = remove_noise(img)

    if sharpen:
        if normalize:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2]
            # Calculate mean and STD
            # print(img)
            mean, STD = cv2.meanStdDev(img.astype(np.uint8))
            # Clip frame to lower and upper STD
            offset = 0.5
            clipped = np.clip(img, mean - offset * STD, mean + offset * STD).astype(np.uint8)

                # Normalize to range
            img = cv2.normalize(clipped, clipped, 0, 255, norm_type=cv2.NORM_MINMAX)
                # show_img(img, "normalized")
            sharp_copy, img = call_sharpen(img)
                # show_img(sharp_copy, "sharpened")
        else:
            sharp_copy, img = call_sharpen(img)
            # show_img(sharp_copy, "sharpened")

    # regardless of accepted preprocessing, we want to work in grayscale
    try:
        assert len(img.shape) == 2
    except AssertionError:
        print("exception")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return scale_copy, segment_copy, sharp_copy, img

def add_padding(img):
    # We add a border region around the image s.t.,
    # the book does not clip the edges of the image, thereby ruining the segmentation
    # Presumably, the image is in RGB when this function is called
    #show_img(img, "img")
    if len(img.shape) < 3:
        new = np.zeros((img.shape[0] + 20, img.shape[1] + 20), dtype=np.uint8)
        new[10:new.shape[0] - 10, 10:new.shape[1] - 10] = img
    else:
        new = np.zeros((img.shape[0] + 20, img.shape[1] + 20, 3), dtype=np.uint8)
        new[10:new.shape[0] - 10, 10:new.shape[1] - 10] = img
    #show_img(new, "new")
    return img

def call_scale(img, direction, iters):
    scaled_img = pyramid(img, direction=direction, iterations=iters)
    img = add_padding(scaled_img)
    copy_img = deepcopy(img)
    return copy_img, img

def call_sharpen(img):
    sharpened = sharpen(img)
    img = sharpened.astype(np.uint8)
    copy_img = deepcopy(img)
    img = cv2.GaussianBlur(img, ksize=(9,9), sigmaX=9/6)
    return copy_img, img


### Note that a non-learning segmentation method of the book is actually really hard to achieve
# This is partly due to high levels of information and noise in the image
# Likewise, Illumination and color levels varies between images
# A major problem is that sometimes the crease separating pages separates the book halves completely

# There is presumably not really any way to get around this,
# outside of preparing a bunch of cropped images,
# and running an object detection model NN on them

def find_contours(threshold_img, book_idx=0):
    # show_img(threshold_img, "threshold in fc")
    contours, hierarch = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    max_w = 0
    max_nwp = 0
    max_area = None
    k = -1
    for i, cont in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cont)
        # print(x,y,w,h)
        new_img = np.zeros(threshold_img.shape, dtype='uint8')
        cont_img = new_img[y:y + h, x:x + w]
        cv2.drawContours(new_img, contours, i, (255, 255, 255), -1)
        nwp = len(np.where(new_img == 255)[0])
        ## Demand only contours above a certain size.
        # Depending on image size, may be changed.
        # Anyway, this function is designed to find the book contour, not the text cell contours
        if (w > 200 and w < 1400) and (h > 100):
            # print(nwp)
            # plt.imshow(cont_img, cmap='gray', interpolation = 'bicubic')
            # plt.xticks([]), plt.yticks([])
            # plt.show()
            cv2.imwrite("test_img\manip_imgs\book_" + str(book_idx) + "_cont_" + str(i) + ".JPG", new_img)
            if (w >= max_w) and nwp > max_nwp:
                max_area = cont_img
                max_w = w
                max_nwp = nwp
                ## remember the index of the book
                k = i
                cont_x, cont_y, cont_w, cont_h = x, y, w, h
            # print(np.where(cont_img == 255), "pts =", cont_pts)
    return max_area, cont_x, cont_y, cont_w, cont_h


def segmentation(thresh_img, img):
    gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gs, 0, 255, cv2.THRESH_OTSU)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 100])
    higher_blue = np.array([60, 60, 255])

    lower_magenta = np.array([100, 0, 100])
    higher_magenta = np.array([255, 40, 255])

    b_mask = cv2.inRange(hsv_img, lower_blue, higher_blue)
    m_mask = cv2.inRange(hsv_img, lower_magenta, higher_magenta)

    final = (gs & b_mask) | (gs & m_mask)
    thresh = get_morph(
        get_morph(final, morph="close", sz=(7, 7), kernel="ones", threshold=True),
        morph="erode", sz=(3, 3), kernel="ones")
    # padded = add_padding(thresh)
    # show_img(padded, "second padding")
    img, book_x, book_y, book_w, book_h = segmentation(thresh.astype(np.uint8), img)
    segment_copy = deepcopy(img)

    book_cont, cont_x, cont_y, cont_w, cont_h = find_contours(thresh_img, 0)
    cropped_img = img[cont_y:cont_y + cont_h, cont_x:cont_x + cont_w]
    # show_img(book_cont, "cropped_img")

    if len(cropped_img.shape) > 2:
        cropped_img[book_cont[0:cont_h, 0:cont_w] == 0] = (0, 0, 0)
    else:
        cropped_img[book_cont[0:cont_h, 0:cont_w] == 0] = 0
    return cropped_img.astype(np.uint8), cont_x, cont_x, cont_w, cont_h

def get_kernel(size, kernel):
    if kernel == "ones":
        return cv2.getStructuringElement(cv2.MORPH_RECT, size)
    if kernel == "gaussian":
        if size[0] == size[1]:
            sigma = int(size[0] // 6)
            size = size[0]
            return cv2.getGaussianKernel(size, sigma)
        else:
            sigma = int((size[0] + size[1]) / 12)
            return cv2.getGaussianKernel(ksize=int((size[0]+size[1])//2), sigma=sigma)

def white_top(img, kern):
    return white_tophat(img, kern)

def blackhat(img, kern, iters):
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kern, iterations=iters)

def gradient(img, kern, iters):
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kern, iterations=iters)

def tophat(img, kern, iters):
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kern, iterations=iters)

def opening(img, kern, iters):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kern, iterations=iters)

def close(img, kern, iters):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kern, iterations=iters)

def dilation(img, kern, iters):
    return cv2.morphologyEx(img, cv2.MORPH_DILATE, kern, iterations=iters)

def erosion(img, kern, iters):
    return cv2.morphologyEx(img, cv2.MORPH_ERODE, kern, iterations=iters)

def get_morph(img, morph, sz, kernel="ones", iterations=1, ttype="otsu", threshold=False):
    kern = get_kernel(sz, kernel)
    switcher = {
        "tophat": tophat,
        "blackhat": blackhat,
        "gradient": gradient,
        "white_tophat": white_top,
        "open": opening,
        "close": close,
        "dilate": dilation,
        "erode": erosion
    }
    if threshold:
        get_func = lambda arg: switcher.get(arg, lambda: "No such morph defined")
        f_morph = get_func(morph)
        img = f_morph(img, kern, iterations)
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if ttype=="otsu":
            ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
            return thresh.astype(np.uint8)
        if ttype=="triangle":
            img = 255 - img
            ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_TRIANGLE)
            return thresh.astype(np.uint8)
        if ttype=="inverse":
            ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)
            return thresh.astype(np.uint8)
        if ttype=="adaptive":
            return cv2.adaptiveThreshold(img.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV ,11,2).astype(np.uint8)
    else:
        get_func = lambda arg: switcher.get(arg, lambda: "No such morph defined")
        f_morph = get_func(morph)
        if morph == "white_tophat":
            morphed = f_morph(img, kern)
        else:
            morphed = f_morph(img, kern, iterations)
        return morphed.astype(np.uint8)

from cv2 import ximgproc as xim

def remove_noise(img):
    return xim.edgePreservingFilter(img, d=15, threshold=20)

def correct_shading(img):
    return shaCorr(img)

## A Python rewrite of shaCorr.m authored by Anders Hast
# This code follows the license in the original implementation
# Written by: Olle Dahlstedt 2021-03-04

######### Copyright and permission notice: ##########
##  Copyright (c) 2011-2019 Anders Hast
##  Uppsala University
##  http://www.cb.uu.se/~aht
##
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## The Software is provided "as is", without warranty of any kind.

def shaCorr(inp_img, edge_preserving = False, f = 0.5, bright = 0.2, dark = 1.0, d = 6 , iter = 10, contrast = 8, corr = 0, pfact = 0, msize=100):
    if len(inp_img.shape) > 2:
        final = cv2.cvtColor(inp_img, cv2.COLOR_BGR2HSV)
        if edge_preserving:
            cols, rows, _channels = map(int, inp_img.shape)
            img = final
            placeholder = img[:,:,2]
        else:
            cols, rows, _channels = map(int, inp_img.shape)
            img = final[:,:,2]
            placeholder = img
    else:
        cols, rows = map(int, inp_img.shape)
        img = inp_img

    ## We should convolve the kernel with our chosen window size

    # Lowpass filter setup
    N, sigma, Sret = mask([cols, rows], msize)
    S = Sret
    krn = np.ones(S)
    div2 = cv2.GaussianBlur(krn, krn.shape, sigma)

    if f > 0:
        # Highpass filter setup
        N, sigma, _ = mask(np.ceil(N * f), 0)
        div1 = cv2.GaussianBlur(krn, krn.shape, sigma)

    k = 0
    while k < iter:
        if edge_preserving:
            ## Added by OD 2021-03-05

            # Division of subregions
            ## Resizing the image to roughly kern shape x 2
            ## Let subregion around each pixel be ceil(kern shape / 10)
            # I.e kern shape = 101, 143 -> subregion = ceil((202, 146) / 10) = (21, 15)
            # We would create our subwindow around each pixel thus from pixel at coord (21, 15)
            # Importantly, this means we will not blur the border of the image, which is fine in our material
            resized = resize(img, (krn.shape[1], krn.shape[0]))
            print(resized.shape)
            d = 30
            window_y, window_x = int(np.ceil(resized.shape[0] // d)), int(np.ceil(resized.shape[1] // d))
            # We cannot guarantee that either y or x region will be odd, so add if clause
            window_y += 1 if window_y % 2 == 0 else 0
            window_x += 1 if window_x % 2 == 0 else 0
            kern = np.ones((window_y*2, window_x*2, 3))

            ### Determine the number of subwindows ###
            # Since we will not blur the borders,
            # we will have one subregion around each pixel except the border pixels

            _, sigma, _ = mask(np.max([window_y,window_x]), 0)
            div = cv2.GaussianBlur(kern, (window_y, window_x), sigma)

            inits = {}
            # Using a nested dictionary for faster lookup
            # But it saves runtime, so that's nice

            # As we need a window around each pixel, this cannot be run outside two loops
            for y in range(window_y, resized.shape[0]-window_y):
                inits[y] = {}
                # print("y =", y, "window y =", window_y, "y - window_y = ", (y - window_y))
                for x in range(window_x, resized.shape[1]-window_x):
                    inits[y][x] = {}
                    # print("x =", x, "window x =", window_x, "x - window_x = ", (x - window_x))
                    subwindow = resized[y-window_y:y+window_y, x-window_x:x+window_x]

                    ### Blur each window ###
                    # Note: in order to avoid too strong blurring (erasing all color information),
                    # Blur with kernel size half of each window

                    p = cv2.GaussianBlur(subwindow, (window_y, window_x), sigma) / div
                    # Color mean within each (now blurred) window
                    color_means = np.array([np.mean(p[:, :, 0]), np.mean(p[:, :, 1]), np.mean(p[:, :, 2])])
                    inits[y][x]["cmeans"] = color_means


                    # pixelwise distances (5)
                    pd = np.array([np.linalg.norm(np.array([resized[y,x][0], color_means[0]]), ord=2),
                                  np.linalg.norm(np.array([resized[y,x][1], color_means[1]]), ord=2),
                                  np.linalg.norm(np.array([resized[y,x][2], color_means[2]]), ord=2)])

                    inits[y][x]["pixelwise_distance"] = pd

                    # Mean pixelwise distance (6)
                    mpd = np.mean(pd)
                    inits[y][x]["mean_pixelwise_distamce"] = mpd

            t = 20
            im = np.zeros(resized.shape)
            wts = np.ones(resized.shape)
            for y in range(window_y, resized.shape[0]-window_y):
                for x in range(window_x, resized.shape[1] - window_x):
                    im[y,x] = im[y,x] + (resized[y,x] * ((t - inits[y][x]["pixelwise_distance"])**2) * inits[y][x]["cmeans"])
                    wts[y,x] = wts[y,x] + (resized[y,x] * (t - inits[y][x]["pixelwise_distance"])**2)

            p = im / wts
            p = p[:,:,2]

        else:
            if f>0:
                imq = resize(img, (krn.shape[1], krn.shape[0]))
                # Compute the bandpass filter, highpass -> lowpass
                p1 = cv2.GaussianBlur(imq, krn.shape, sigma) / div1
                # print(p1)
                p2 = cv2.GaussianBlur(imq, krn.shape, sigma) / div2

                im2 = p2 - p1
                q = resize(im2, (img.shape[1], img.shape[0]))

                q[q < 0] *= bright
                q[q > 0] *= dark
                img = img + q


                N, sigma, _ = mask(np.ceil(np.max(S) / d), 0)
                div = cv2.GaussianBlur(krn, krn.shape, sigma)
                resized = resize(img, (krn.shape[1], krn.shape[0]))
                p = cv2.GaussianBlur(resized, krn.shape, sigma) / div

        if len(resized.shape) > 2:
            resized = resized[:,:,2]
        pm = np.mean(np.mean(p))
        s = np.abs(resized-p)
        ky, kx = s.shape[0]//2, s.shape[1]//2
        ky += 1 if ky % 2 == 0 else 0
        kx += 1 if kx % 2 == 0 else 0
        w = cv2.GaussianBlur(s, (int(ky), int(kx)), sigma)
        if corr > 0:
            wb = resize(w, (rows, cols))
        pb = resize(p, (rows, cols))
        placeholder = placeholder - pb

        mm = np.mean(np.mean(w))
        if corr > 1:
            wb[wb > mm] = mm

        if corr > 2:
            wb[wb < mm * pfact] = mm * pfact

        if corr > 0:
            wb = (wb - mm) * contrast + mm
            placeholder = mm * (placeholder / (wb))

        placeholder = placeholder + pm
        if edge_preserving:
            img[:,:,2] = placeholder
        else:
            img = placeholder
        if d > 1:
            d = d / 2

        k += 1

    if len(inp_img.shape) > 2:
        if edge_preserving:
            final = img
            print(final.shape)
            return cv2.cvtColor(final, cv2.COLOR_HSV2BGR)
        else:
            final[:,:,2] = img
            return cv2.cvtColor(final, cv2.COLOR_HSV2BGR)
    else:
        return img


def mask(So, sz):
    if sz != 0:
        scale = min(So) / sz
        S = np.ceil(np.divide(So, scale)).astype(np.uint8)
        N = np.ceil(max(S))
        S[0] += 1 if S[0] % 2 == 0 else 0
        S[1] += 1 if S[1] % 2 == 0 else 0
        N += 1 if N % 2 == 0 else 0

    else:
        N = So
        S = 0


    sigma = N / 6

    return N, sigma, S


def part_two(scale_copy, segment_copy, sharp_copy, img, normalize=True, clear_lines=True):
    ## Clean lines and preprocess image for text box detection

    try:
        assert scale_copy is not None
        assert segment_copy is not None
        assert sharp_copy is not None
        assert img is not None
    except:
        print("Part two will not work if we do not have copies of the images")

    if uses_PST:
        img, PST_Kernel = PST(img)

    if clear_lines:
        try:
            assert len(segment_copy.shape) < 3
        except:
            print("Image is not in grayscale, converting it to grayscale")
            segment_copy = cv2.cvtColor(correct_shading(segment_copy).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        print("before line cleaning")

        if uses_PST:
            # show_img(img, "img")
            # show_img(segment_copy, "copy")
            img = line_cleaning(img, segment_copy, use_Canny=False).astype(np.uint8)
            # show_img(img, "lines cleared")
        else:
            img = line_cleaning(img, segment_copy, use_Canny=True).astype(np.uint8)

    # markers(img)
    # markers(bez)
    show_img(img, "lines cleared")
    # print("lines cleared")
    # print(img)

    # edge_two, kern = PST(img.astype(np.uint8))
    # show_img(edge_two, "edges")

    real, imag = gabor(img, frequency=0.5)
    fltred = np.sqrt(real ** 2 + imag ** 2).astype(np.uint8)
    show_img(real, "gabor")
    # blur = cv2.GaussianBlur(img, (21,21), 21/6)
    # show_img(blur, "blur")
    # show_img(blur, "blurred")
    # grad = get_morph(img, "gradient", (5, 5), kernel="gaussian", ttype="otsu", threshold=True)
    # _,  = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    # show_img(grad, "thresholded")
    # box_img, boxes = get_text_boxes(grad, segment_copy)
    # show_img(box_img, boxes)
    # for box in boxes:
    #     print(box)
    # show_img(grad, "grad")
    # print("Image has been converted to grayscale")
    img_sift, img_mser = deepcopy(fltred), deepcopy(segment_copy)
    sift = cv2.SIFT_create()
    kp_sift = sift.detect(sharp_copy, None)
    # print("Keypoints detected")
    img_sift = cv2.drawKeypoints(segment_copy, kp_sift, segment_copy,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS).astype(np.uint8)
    mser = cv2.MSER_create()
    kp_mser, boxes = mser.detectRegions(segment_copy)
    # img_mser = cv2.drawKeypoints(img_mser, kp_mser, img_mser, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS).astype(np.uint8)
    for box in boxes:
        x, y, w, h = box;
        cv2.rectangle(img_mser, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # print("Keypoints drawn")
    # show_img(img, "keypoints image")
    return img_mser, img_sift
    # get_max_peaks(grad)s)

# img_row_sum = np.sum(img, axis=1).tolist()
# plt.plot(img_row_sum)
# plt.show()
# img_column_sum = np.sum(img, axis=0).tolist()

## Use this method to evaluate the algorithm on a single image
# This is useful to make quick changes and try new things
# Remember that changes that work for a single images do not necessarily work for all images
# Prefereably, use a "worst-case" scenario image here
def single():
    ## EXAMPLE USAGE OF 1 IMAGE
    if os_selected == "linux":
        f_name = 'test_img/IMG_3.JPG'
    if os_selected == "win":
        f_name = 'test_img\IMG_3.JPG'

    print("entered single")
        # f_name = 'img_test.jpeg'
        # f_name = 'ideal_grid.jpg'
        # f_name = 'sharpened_img.jpg'

    f_obj = open_file(name=f_name, multiple=False, n_img=1)
    print("f_obj opened")

    img = f_obj.img
    print("initializing part one")
    scale_copy, segment_copy, sharp_copy, img = part_one(img)

    return img


# Use this method to run the algorithm on multiple images
# It is useful to write new copies of the altered images to evaluate the results post-processing
# You can certainly do this using the "show_img"-method, but doing this repeatedly gets rather tiresome

def multi(n_img):
    if os_selected == "linux":
        f_name = 'image_folder/IMG_'
    if os_selected == "win":
        f_name = 'image_folder\IMG_'

    # We can retrieve less images than we load to the array
    print("####" * 10)
    print("Retrieving image generator. This may take a while, depending on image size and number of images")
    print("####" * 10)
    img_generator = open_file(name=f_name, multiple=True, n_img=n_img)
    arr = []
    print("Starting the script")
    for i in range(n_img):
        print("Starting image:", i + 1, "/", n_img)
        # The image generator is a yield function.
        # Calling it with the next keyword will iterate the function and return one image from the generator loop
        # Saves overhead memory costs (in comparison to keeping a long array of large images in memory)
        img = next(img_generator)

        ########### Important! ###########
        # When running the pipeline on many images (> 100),
        # it is advised that you wrap the calls within a try-except block
        # In some cases the pipeline fails in scenarios where fixes are possible but not implemented
        # This does not mean its results are affected much.
        # The try-except block will simply skip writing invalid images.
        # When running the pipeline on smaller sets of images for testing purposes,
        # avoid using the try-except block,
        # since it will obscure the true error messages and lead to painstaking debugging.
        ##################################

        # try:
        blurred = cv2.GaussianBlur(img, (3, 3), 3 / 6)
        print("Image has been scaled")
        scale_copy, segment_copy, sharp_copy, img = part_one(blurred)
        img_mser, img_sift = part_two(scale_copy, segment_copy, sharp_copy, img)
        # show_img(img, "title")
        _ = cv2.imwrite(str('sift_keypoints' + str(i) + '.JPG'), img_sift)
        _ = cv2.imwrite(str('mser_keypoints' + str(i) + '.JPG'), img_mser)
        # print(img.shape)
        # if img.shape == (500,750,3):
        #    print("Image to be added to array")
        #    arr.append(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2].flatten())
        # else: pass
    #     except:
    # print('Image {} does not exist'.format(i+1))
    # final = exec(img, scale=scale, noise=remove_noise, corr_shading=corr_shading, seg_book=seg_book, sharpen_img=sharpen, do_nothing=do_nothing)
    # show_img(final, "Final image")
    # write_img("final_image_" + str(i) + ".jpg", final)

single()
# multi(n_img = 20)