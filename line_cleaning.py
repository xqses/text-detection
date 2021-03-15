import cv2
import numpy as np
from basics import show_img

def find_lines(img, ld):
    lines = ld.detect(img)
    return lines

def clean_lines(gs_img, mask, lines):
    # show_img(mask, "Mask")
    gs_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cleaned = gs_img
    m_img = gs_mask & gs_img
    mask = cv2.dilate(m_img, None).astype(np.uint8)
    # show_img(mask, "Mask")
    median_pixel = np.median(cleaned.flatten())
    cleaned[mask != 0] = median_pixel
    # show_img(cropped_img.astype(np.uint8), "title")
    # seg_pts = np.array(list(zip(np.where((eq_hist_two == mask_two).all(axis = 2)))), dtype=np.uint8)
    # eq_hist_two[np.where(mask_two == eq_hist_two).all(axis=2)] = (255,255,255)
    return cleaned

def draw_lines(img, lines, ld):
    return ld.drawSegments(img, lines)


def line_cleaning(sharpened_img, gs_img):
    fld = cv2.ximgproc.createFastLineDetector(_length_threshold=20, _canny_th1=30, _canny_th2=30, _canny_aperture_size=5)
    lines = find_lines(sharpened_img, fld)
    mask = np.zeros_like(sharpened_img).astype(np.uint8)
    mask = draw_lines(mask, lines, fld)
    cleaned_img = clean_lines(gs_img, mask, lines)
    return cleaned_img