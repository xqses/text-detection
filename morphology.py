import cv2
import numpy as np

def get_kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_RECT, size)

def blackhat(img, kern):
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kern)

def gradient(img, kern):
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kern)

def tophat(img, kern):
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kern)

def opening(img, kern):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kern)

def close(img, kern):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kern)

def dilation(img, kern):
    return cv2.morphologyEx(img, cv2.MORPH_DILATE, kern)

def erosion(img, kern):
    return cv2.morphologyEx(img, cv2.MORPH_ERODE, kern)

def get_morph(img, morph, sz, threshold=False):
    kern = get_kernel(sz)
    switcher = {
        "tophat": tophat,
        "blackhat": blackhat,
        "gradient": gradient,
        "open": opening,
        "close": close,
        "dilate": dilation,
        "erosion": erosion
    }
    if threshold:
        get_func = lambda arg: switcher.get(arg, lambda: "No such morph defined")
        f_morph = get_func(morph)
        morphed = f_morph(img, kern)
        gs_img = cv2.cvtColor(morphed, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gs_img,0,255, cv2.THRESH_OTSU)
        return thresh.astype(np.uint8)
        # return cv2.adaptiveThreshold(gs_img.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY ,11,2)
    else:
        get_func = lambda arg: switcher.get(arg, lambda: "No such morph defined")
        f_morph = get_func(morph)
        morphed = f_morph(img, kern)
        return morphed.astype(np.uint8)