from cv2 import dilate, ximgproc as xim
import numpy as np

def find_lines(img, ld):
    lines = ld.detect(img)
    return lines

def clean_lines(img, mask, lines):
    cleaned = img
    mask = dilate(mask, None).astype(np.uint8)
    img = mask & img
    # cropped_img[eq_hist_two != 0] = 255
    # show_img(cropped_img.astype(np.uint8), "title")
    # seg_pts = np.array(list(zip(np.where((eq_hist_two == mask_two).all(axis = 2)))), dtype=np.uint8)
    # eq_hist_two[np.where(mask_two == eq_hist_two).all(axis=2)] = (255,255,255)
    return cleaned

def draw_lines(img, lines, ld):
    return ld.drawSegments(img, lines)


def line_cleaning(img, sharpened_img):
    fld = xim.createFastLineDetector()
    lines = find_lines(sharpened_img, fld)
    mask = np.zeros_like(img).astype(np.uint8)
    mask = draw_lines(mask, lines, fld)
    cleaned_img = clean_lines(img, mask, lines)
    return draw_lines(img, lines, fld)