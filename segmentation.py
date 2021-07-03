from copy import deepcopy

import numpy as np
import scipy as scp
import cv2
from basics import show_img
from matplotlib import pyplot as plt
from watershed import morphologize, get_markers, watershed, dilate
from morphology import get_morph
from shading_correction import shaCorr
from skimage.feature import peak_local_max
from sharpening import sharpen
import skimage.segmentation as sk_seg


# Find contours
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

def segmentation(hsv_img, img):
    from scipy.ndimage import label

    ## Save for non-edge segmentation justification
    ## I.e. too blurry
    # gauss = cv2.GaussianBlur(img, (5,5), 5/6)
    # p = cv2.Canny(gauss, 10,10)
    # show_img(gauss, "gauss")
    # show_img(p, "p")
    # _, tt = cv2.threshold(p,0,255, cv2.THRESH_OTSU)
    # opening = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, np.ones((3, 3), np.int), iterations=2)
    # show_img(opening, "opened")

    # book_cont, cont_x, cont_y, cont_w, cont_h = find_contours(thresh_img, 0)
    # show_img(book_cont, "contour")
    # print(cont_x, cont_y, cont_w, cont_h)

    # cropped_img = img[cont_y:cont_y + cont_h, cont_x:cont_x + cont_w]
    # print(cropped_img.shape)
    # show_img(book_cont, "cropped_img")

    # if len(cropped_img.shape) > 2:
    #     cropped_img[book_cont[0:cont_h, 0:cont_w] == 0] = (0, 0, 0)
    # else:
    #     cropped_img[book_cont[0:cont_h, 0:cont_w] == 0] = 0
    # return cropped_img.astype(np.uint8), cont_x, cont_x, cont_w, cont_h
    gs, m = get_book(hsv_img)
    # show_img(gs, "HSV to grayscale thresholding")
    _, y = cv2.threshold(gs,0,255, cv2.THRESH_OTSU)
    # show_img(y, "Otsu thresholded")

    sure_bg = cv2.dilate(y, None, iterations=5)
    sure_bg = sure_bg - cv2.erode(sure_bg, None)
    dist_transform = cv2.distanceTransform(y, cv2.DIST_L2, 5)
    dist_transform = ((dist_transform - dist_transform.min()) / (dist_transform.max() - dist_transform.min()) * 255).astype(np.uint8)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    lbl, ncc = label(dist_transform)
    lbl = lbl * lbl * (255 / (ncc + 1))
    lbl[sure_bg == 255] = 255
    lbl = lbl.astype(np.int32)
    img_3C = deepcopy(img)

    cv2.watershed(img_3C, lbl)
    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)
    result = 255 - lbl
    # threshold result
    result[result != 255] = 0
    result = get_morph(result,morph="dilate", sz=(3,3))
    # ret, thresh_res = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY_INV)
    ## Set the marked regions border to red
    # This is useful to understand what findContours will find.
    img_3C[result == 255] = (255, 0, 0)
    mask = np.zeros_like(result)
    mask[result == 255] = 255
    # show_img(img_3C, "Tricolor image with contour borders")
    # show_img(mask, "Contour border mask")

    # result[result != 255] = 0
    # result = cv2.dilate(result, None)
    # ret, thresh_res = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY_INV)
    # cv2.imwrite('5_thresholded_watershed_regions.jpg', thresh_res)
    # img_3C[result == 255] = (0, 0, 255)
    # show_img(img_3C, "im3c")
    book_cont, cont_x, cont_y, cont_w, cont_h = find_contours(result, 0)
    # show_img(book_cont, "contour")
    # print(cont_x, cont_y, cont_w, cont_h)

    cropped_img = img[cont_y:cont_y + cont_h, cont_x:cont_x + cont_w]
    # print(cropped_img.shape)
    # show_img(book_cont, "cropped_img")

    # if len(cropped_img.shape) > 2:
    #     cropped_img[book_cont[0:cont_h, 0:cont_w] == 0] = (0,0,0)
    # else:
    #     cropped_img[book_cont[0:cont_h, 0:cont_w] == 0] = 0
    # print(cropped_img.shape)
    # show_img(cropped_img,"Final cropped image")
    # show_img(cropped_img, "cropped")
    return cropped_img.astype(np.uint8)


def get_book(hsv_img):
    img_gray = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2GRAY)

    lower_blue = np.array([0, 0, 60])
    higher_blue = np.array([60, 60, 255])

    lower_magenta = np.array([50, 0, 50])
    higher_magenta = np.array([255, 40, 255])

    b_mask = cv2.inRange(hsv_img, lower_blue, higher_blue)
    m_mask = cv2.inRange(hsv_img, lower_magenta, higher_magenta)

    final = (img_gray & b_mask) | (img_gray & m_mask)
    return cv2.GaussianBlur(final, (5, 5), 0), img_gray