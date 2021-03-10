import numpy as np
import cv2
from matplotlib import pyplot as plt
from watershed import morphologize, get_markers, watershed, dilate


# Find contours
def find_contours(threshold_img, book_idx):
    contours, hierarch = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_w = 0
    max_area = None
    k = -1
    for i, cont in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cont)
        new_img = np.zeros(threshold_img.shape, dtype='uint8')
        cont_img = new_img[y:y + h, x:x + w]
        ## Demand only contours above a certain size.
        # Depending on image size, may be changed.
        # Anyway, this function is designed to find the book contour, not the text cell contours
        if w > 200 and h > 100:
            cv2.drawContours(new_img, contours, i, (255, 255, 255), -1)
            plt.imshow(cont_img, cmap='gray', interpolation = 'bicubic')
            plt.xticks([]), plt.yticks([])
            plt.show()
            cv2.imwrite("test_img\manip_imgs\book_" + str(book_idx) + "_cont_" + str(i) + ".JPG", new_img)
            if w >= max_w:
                max_area = cont_img
                max_w = w
                ## remember the index of the book
                k = i
                cont_x, cont_y, cont_w, cont_h = x, y, w, h
            # print(np.where(cont_img == 255), "pts =", cont_pts)
    return max_area, cont_x, cont_y, cont_w, cont_h

def segmentation(img, hsv_img):
    book_img, gray_img = get_book(hsv_img)
    res, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
    sure_fg, sure_bg = morphologize(thresh)
    markers = get_markers(sure_fg, sure_bg)
    img, markers = watershed(img, markers)
    res, img_dilated = dilate(markers, img)
    _, thresh = cv2.threshold(cv2.cvtColor(img_dilated, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU)
    book_cont, cont_x, cont_y, cont_w, cont_h = find_contours(thresh, 0)

    ## Warning!
    # Recall that the perimeter given book_cont.shape is (necessarily) lower than the perimeter of the input image
    # Solution: return the bounding box coordinates to crop the input image as well
    cropped_img = img[cont_y:cont_y + cont_h, cont_x:cont_x + cont_w]
    cropped_img[book_cont[0:cont_h, 0:cont_w] == 0] = (0, 0, 0)
    return cropped_img


def get_book(hsv_img):
    img_gray = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2GRAY)

    lower_blue = np.array([0, 0, 100])
    higher_blue = np.array([60, 60, 255])

    lower_magenta = np.array([100, 0, 100])
    higher_magenta = np.array([255, 40, 255])

    b_mask = cv2.inRange(hsv_img, lower_blue, higher_blue)
    m_mask = cv2.inRange(hsv_img, lower_magenta, higher_magenta)

    final = (img_gray & b_mask) | (img_gray & m_mask)
    return cv2.GaussianBlur(final, (5, 5), 0), img_gray