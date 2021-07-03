import numpy as np
import cv2
from scipy.ndimage import label


def morphologize(prep_img):
    # Typical watershed preprocessing algorithm
    # Open -> dilate -> bg = dilated opening - eroded dilated opening
    # foreground = normalized distance transformed opening
    # may be thresholded if desired
    opening = cv2.morphologyEx(prep_img, cv2.MORPH_OPEN, np.ones((3, 3), np.int), iterations=10)
    sure_bg = cv2.dilate(opening, None)
    sure_bg = sure_bg - cv2.erode(sure_bg, None)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    sure_fg = ((dist_transform - dist_transform.min()) / (dist_transform.max() - dist_transform.min()) * 255).astype(
        np.uint8)
    # ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    return sure_fg, sure_bg


def morph_two(img):
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    # second deriv kernel (use gaussian?)
    imgLaplacian = cv2.filter2D(img, cv2.CV_32F, kernel)
    ft32 = np.float32(img)
    imgResult = ft32 - imgLaplacian
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)

    opening = cv2.morphologyEx(imgResult, cv2.MORPH_OPEN, np.ones((3, 3), np.int), iterations=10)
    sure_bg = cv2.dilate(opening, None, iterations=5)
    sure_bg = sure_bg - cv2.erode(sure_bg, None)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    sure_fg = ((dist_transform - dist_transform.min()) / (dist_transform.max() - dist_transform.min()) * 255).astype(
        np.uint8)
    # _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    return sure_bg, sure_fg


def get_markers(sure_fg, sure_bg):
    labels, ncc = label(sure_fg)
    labels = labels * labels * (255 / (ncc + 1))
    labels[sure_bg == 255] = 255
    labels = labels.astype(np.int32)
    return labels


def watershed(img, markers):
    cv2.watershed(img, markers)
    return img, markers


def dilate(labels, img_3C):
    labels[labels == -1] = 0
    # reformat labels
    labels = labels.astype(np.uint8)
    result = 255 - labels
    # threshold result
    result[result != 255] = 0
    result = cv2.dilate(result, None)
    ret, thresh_res = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY_INV)
    ## Set the marked regions border to red
    # This is useful to understand what findContours will find.
    img_3C[result == 255] = (0, 0, 255)
    return result, img_3C
