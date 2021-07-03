import cv2
import numpy as np
from skimage.morphology import white_tophat

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
            ret, thresh = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY_INV)
            return thresh.astype(np.uint8)
        if ttype=="adaptive":
            return cv2.adaptiveThreshold(img.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY ,11,2).astype(np.uint8)
    else:
        get_func = lambda arg: switcher.get(arg, lambda: "No such morph defined")
        f_morph = get_func(morph)
        if morph == "white_tophat":
            morphed = f_morph(img, kern)
        else:
            morphed = f_morph(img, kern, iterations)
        return morphed.astype(np.uint8)