import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from pythonRLSA import rlsa
from scipy.ndimage import label
import os
from sklearn import ensemble


def train_classifier():
    dataset = load_data()
    data, target = sort_data(dataset)
    clf = ensemble.RandomForestClassifier(n_estimators=1000)
    clf = clf.fit(data, target)
    return clf


def load_data():
    return np.array(np.genfromtxt('samples.txt'), dtype='float')


def sort_data(data):
    x = []
    label = []
    for item in data:
        new_list = [item[0], item[1], item[2], item[3], item[4]]
        x.append(new_list)
        label.append(item[5])

    return np.array(x, dtype='float32'), np.array(label, dtype='int')


def detect_text(image, contour):
    cimg = np.zeros_like(image)
    cv2.drawContours(cimg, [contour], -1, color=255, thickness=-1)
    cont_pts = np.where(cimg == 255)
    cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)

    intersect = cimg & image

    # print(calc_hist)
    blk_cnt_pts = np.where(intersect != 0)

    percentage_black = np.divide(len(blk_cnt_pts[0]), len(cont_pts[0]))
    sorted_blk_y, sorted_blk_x = sorted(blk_cnt_pts[0]), sorted(blk_cnt_pts[1])
    if (len(sorted_blk_y) > 0) & (len(sorted_blk_x) > 0):
        vertical_range, horiz_range = sorted_blk_y[-1] - sorted_blk_y[0], sorted_blk_x[-1] - sorted_blk_x[0]
        ratio_hz_vt = np.divide(horiz_range, vertical_range)
    else:
        return False
    if (np.isnan(vertical_range)) or (np.isnan(horiz_range)) or (np.isnan(percentage_black)) or (
            np.isnan(ratio_hz_vt)) or (np.isinf(ratio_hz_vt)) or (np.isnan(len(blk_cnt_pts[0]))):
        print(vertical_range, horiz_range, percentage_black, ratio_hz_vt, len(blk_cnt_pts[0]))
        return False
    print(vertical_range, horiz_range, percentage_black, ratio_hz_vt, len(blk_cnt_pts[0]))
    # if (percentage_black < 0.87) & (percentage_black > 0.4) & (len(blk_cnt_pts[0]) > 8000) & (ratio_hz_vt > 0.3) & (
    #         ratio_hz_vt < 10):
    #     print('detected')
    # return True
    # ## return True, tree_count, human_count
    pred_label = tree.predict([[vertical_range, horiz_range, percentage_black, ratio_hz_vt, len(blk_cnt_pts[0])]])
    if pred_label == 1:
        return True
    else:
        return False

# images = []
# counter = 0
# root = tk.Tk()
# root.withdraw()
# file_path = filedialog.askdirectory()
# print(file_path)
# count_human_true = 0
# count_tree_true = 0
tree = train_classifier()

## for file in os.listdir(file_path):
## file_name = os.path.join(file_path, file)
## counter += 1
## print("cleaning image " + str(counter) + " to path "+ file_path + '_output_' + file)
## open_file = open("samples.txt", "r+")
## beginning
file_name = '104.jpg'
img = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
img_rlsa_horizontal = rlsa.rlsa(thresh, True, False, 10)
img_rlsa_vertical = rlsa.rlsa(img_rlsa_horizontal, False, True, 15)

opening = cv2.morphologyEx(img_rlsa_vertical, cv2.MORPH_OPEN, np.ones((3, 3), np.int), iterations=2)

cv2.imwrite('placeholder2.jpg', opening)

sure_bg = cv2.dilate(opening, None, iterations=5)
sure_bg = sure_bg - cv2.erode(sure_bg, None)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
dist_transform = ((dist_transform - dist_transform.min()) / (dist_transform.max() - dist_transform.min()) * 255).astype(
    np.uint8)
cv2.imwrite('place6.jpg', sure_bg)
cv2.imwrite('placeholder3.jpg', dist_transform)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

lbl, ncc = label(dist_transform)
lbl = lbl * lbl * (255 / (ncc + 1))
lbl[sure_bg == 255] = 255
lbl = lbl.astype(np.int32)

img_3C = cv2.imread(file_name)
cv2.watershed(img_3C, lbl)
lbl[lbl == -1] = 0
lbl = lbl.astype(np.uint8)
result = 255 - lbl

# print(ncc)
# print(lbl)

result[result != 255] = 0
result = cv2.dilate(result, None)
ret, thresh_res = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite('placeholder4.jpg', thresh_res)
img_3C[result == 255] = (0, 0, 255)
cv2.imwrite('placeholder5.jpg', img_3C)

# find contours
contours, hierarch = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cont_counter = 0
list_of_bubbles = []
for cont in contours:
    new_cont = np.zeros(img_3C.shape, dtype='uint8')
    imgH = new_cont.shape[0]
    imgW = new_cont.shape[1]
    x, y, w, h = cv2.boundingRect(cont)
    if not (not (w > 30) or not (h > 50)) and h < 0.75 * imgH and w < 0.75 * imgW:
        is_text = detect_text(thresh, cont)
        if is_text:
            cont_counter += 1
            cv2.drawContours(new_cont, [cont], -1, (0, 255, 0), 3)
            cv2.imwrite('written_cont' + str(cont_counter) + '.jpg', new_cont)
            list_of_bubbles.append(cont)
            print(cont_counter)
            ## print(x,y,w,h)

# img_3C = cv2.imread(file_name)

# for bubble in list_of_bubbles:
#    cv2.drawContours(img_3C, [bubble], -1, (255,255,255), thickness = -1)
# status = cv2.imwrite(file_path + '_output_' + file, img_3C)
# print("written status " + str(status))
## open_file.close()

# print("total bubbles predicted by human values: " + str(count_human_true))
# print("total bubbles predicted by decision tree: " + str(count_tree_true))
