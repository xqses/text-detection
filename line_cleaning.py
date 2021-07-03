import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import corner_peaks, corner_harris, corner_subpix, corner_moravec
from skimage.morphology import binary_closing
from morphology import get_morph

from basics import show_img

def find_lines(img, ld):
    try:
        assert img.dtype == np.uint8
    except AssertionError:
        img = img.astype(np.uint8)
    lines = ld.detect(img)
    return lines

def clean_lines(gs_img, mask, lines, is_binary):
    # show_img(mask, "Mask")
    gs_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    thresh = np.zeros_like(gs_mask)
    thresh[gs_mask != 0] = 255
    # show_img(thresh, "mask")
    # show_img(thresh, "thresh mask")
    cleaned = gs_img
    dilate = get_morph(thresh.astype(np.uint8), sz=(6, 6), morph="dilate", kernel="ones")
    if not is_binary:
        non_zero = np.where(dilate != 0)
        for i in range(len(non_zero[0])):
            # cleaned[non_zero[0][i], non_zero[1][i]] = med
            cleaned[non_zero[0][i], non_zero[1][i]] = np.median(cleaned[non_zero[0][i]-8:non_zero[0][i]+8, non_zero[1][i]-8:non_zero[1][i]+8])
    else:
        # show_img(cleaned, "cleaned")
        # print(cleaned)
        cleaned[dilate != 0] = 255
        # print(cleaned)
        # show_img(cleaned, "cleaned")

    # cleaned[mask != 0] = median
    # cleaned[mask != 0] = median_pixel
    # show_img(cropped_img.astype(np.uint8), "title")
    # seg_pts = np.array(list(zip(np.where((eq_hist_two == mask_two).all(axis = 2)))), dtype=np.uint8)
    # eq_hist_two[np.where(mask_two == eq_hist_two).all(axis=2)] = (255,255,255)
    return cleaned

def draw_lines(img, lines, ld):
    return ld.drawSegments(img, lines)


def line_cleaning(mask, gs_img, use_Canny=True, is_binary=True):
    # show_img(sharpened_img, "entry image linecleaning")
    mask = binary_closing(mask)
    # show_img(mask, "binary closed entry image")
    if use_Canny:
        fld = cv2.ximgproc.createFastLineDetector(_length_threshold=10, _canny_th1=40, _canny_th2=40, _canny_aperture_size=5)
    else:
        fld = cv2.ximgproc.createFastLineDetector(_length_threshold=25, _canny_aperture_size=0, _do_merge=True)
    lines = find_lines(mask, fld)
    mask = np.zeros_like(mask).astype(np.uint8)
    mask = draw_lines(mask, lines, fld)

    cleaned_img = clean_lines(gs_img, mask, lines, is_binary)
    return cleaned_img