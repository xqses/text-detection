import cv2
import numpy as np

def get_kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_RECT, size)

def blackhat(img):
    # define size (tuple)
    sz = (200,200)
    kern = get_kernel(sz)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kern)
    return blackhat
# thresh = cv2.adaptiveThreshold(tophat.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY ,11,2)
# _, thresh = cv2.threshold(eq_hist_two.astype(np.uint8),0, 255, cv2.THRESH_OTSU)

def gradient(img):
    sz = (200,200)
    kern = get_kernel(sz)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kern)
    return gradient

def tophat(img):
    sz = (200,200)
    kern = get_kernel(sz)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kern)
    return tophat

def get_morph(img, morph, threshold=False):
    switcher = {
        "tophat": tophat,
        "blackhat": blackhat,
        "gradient": gradient
    }
    if threshold:
        get_func = lambda func: switcher.get(morph, lambda: "No such morph defined")
        morphed = get_func(img)
        return cv2.adaptiveThreshold(morphed.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY ,11,2)
    else:
        get_func = lambda func: switcher.get(morph, lambda: "No such morph defined")
        morphed = get_func(img)
        return morphed.astype(np.uint8)