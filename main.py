from copy import deepcopy

import cv2
import networkx as nx
import numpy as np
import pandas as pd
import pickle

from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

from PST import PST
from basics import open_file, pyramid, show_img
from bgrem import bgremoval
from get_text_boxes import get_text_boxes
from line_cleaning import line_cleaning
from morphology import get_morph
from segmentation import segmentation
from sharpening import sharpen
from classifier import train_classifier


## Image name format should be "IMG_+{# of img}+.JPG"
# For an object with reference to one image, use explicit file name and set multiple=False
# For an object with reference to an array of images, use name="IMG_", set multiple=True
# See function multi() where n_img = desired # of images


#########################################################
##                                                     ##
##                                                     ##
##             Image processing pipeline               ##
##          for the master thesis project of           ##
##      Automatic Handwritten Text Cell Detection      ##
##              in Historical Documents                ##
##                                                     ##
##              Author: Olle Dahlstedt                 ##
##            Last VCS Update: 2021-03-10              ##
##                                                     ##
##                                                     ##
#########################################################

## The following code is not verified to work in every single use case on all types of images of books containing handwriting.


def generate_visualizations(gs_img, tricolor_img, iteration):
    ### This function really really grew super long. Sorry about that.
    ### This function will reproduce all deterministic results showcased in the essay.
    ### One may adjust the number of images freely.
    # The rest of the code has been / will be refactored to make readability better,
    # however this function kind of sort of needs to be super long so that one does not have to ping pong back and forth.

    tc_copy1, tc_copy2, tc_copy3, tc_copy4, tc_copy5 = deepcopy(tricolor_img), deepcopy(tricolor_img), deepcopy(
        tricolor_img), deepcopy(tricolor_img), deepcopy(tricolor_img)
    # histg = cv2.calcHist([img], [0], None, [256], [0, 256])
    # plt.plot(histg)
    # plt.show()
    hard_mask = np.zeros_like(gs_img).astype(np.uint8)
    hard_mask[PST(gs_img)[0]] = 255
    _ = cv2.imwrite("edge_images\\no_processing_" + str(iteration) + ".JPG", hard_mask)

    ### Generate results from first preprocessing procedure
    ################################################################################
    mser = cv2.MSER_create(_delta=10, _max_variation=0.20)
    s1 = sharpen(gs_img, type="nct").astype(np.uint8)
    _ = cv2.imwrite("sharpened_images\sharpened_img_" + str(iteration) + ".JPG", s1)
    s_copy = deepcopy(s1)
    _, s1 = cv2.threshold(s1, 0, 255, cv2.THRESH_OTSU)

    grad = get_morph(s1, morph="gradient", sz=(5, 5))
    img_sift, img_color = deepcopy(grad), deepcopy(tc_copy1)
    _ = cv2.imwrite("gradient_images\s1_gradient" + str(iteration) + "+.JPG", img_sift)
    color_boxes, list_of_features = get_text_boxes(grad, img_color)
    _ = cv2.imwrite("box_images\s1_ccl\s1_ccl_boxes_" + str(iteration) + "+.JPG", color_boxes)
    _ = cv2.imwrite("cropped_box_images\s1_ccl\s1_ccl_boxes_" + str(iteration) + "+.JPG", color_boxes[200:600, 500:900])
    # show_img(color_boxes)
    s1 = 255 - s1
    s1 = get_morph(s1, morph="dilate", sz=(5, 5))
    mboxes1, _ = mser.detectRegions(s1)
    # print(len(mboxes1))
    # show_img(tc_copy1)
    for box in mboxes1:
        x, y, w, h = cv2.boundingRect(box)
        cv2.rectangle(tc_copy1, (x, y), (x + w, y + h), (255, 0, 255), 1)
    # show_img(tricolor_img)
    _ = cv2.imwrite("box_images\s1_mser\s1_mser_boxes_" + str(iteration) + "+.JPG", tc_copy1)
    _ = cv2.imwrite("cropped_box_images\s1_mser\s1_mser_boxes_" + str(iteration) + "+.JPG", tc_copy1[200:600, 500:900])
    # print("Keypoints detected")
    mask1 = np.zeros_like(gs_img).astype(np.uint8)
    mask1[PST(s_copy)[0]] = 255
    _ = cv2.imwrite("edge_images\edge_image_" + str(iteration) + ".JPG", mask1)
    s1 = deepcopy(s_copy)
    # img_sift, img_color = deepcopy(grad), deepcopy(s1)
    # color_boxes, list_of_features = get_text_boxes(grad, img_color)
    ################################################################################

    ### Generate results from background-removal preprocessing procedure
    ################################################################################
    mser = cv2.MSER_create(_delta=20, _max_variation=0.20)
    rem_bg = (bgremoval(s_copy, f=3, g=3, th=2.0, ct=1) * 255).astype(np.uint8)
    # rem_bg = cv2.divide(s_copy, rem_bg, scale=255)
    _ = cv2.imwrite("p_normalized\\bgremoval_proc" + str(iteration) + ".JPG", rem_bg)
    grad = get_morph(rem_bg, morph="gradient", sz=(3, 3), threshold=True)
    img_sift, img_color = deepcopy(grad), deepcopy(tc_copy2)
    color_boxes, list_of_features = get_text_boxes(grad, img_color)
    _ = cv2.imwrite("box_images\\bgremoval_ccl\\bgremoval_binary_ccl_boxes_" + str(iteration) + ".JPG", color_boxes)
    _ = cv2.imwrite("cropped_box_images\\bgremoval_ccl\\bgremoval_binary_ccl_boxes_" + str(iteration) + ".JPG",
                    color_boxes[200:600, 500:900])

    # grad = 255 - grad
    s2 = normalization(s_copy, k=0.5)
    s_copy = deepcopy(s2)
    # show_img(s2)
    # _, s2 = cv2.threshold(s2, 0, 255, cv2.THRESH_OTSU)
    grad = get_morph(s2, morph="gradient", sz=(3, 3), threshold=True)
    img_sift, img_color = deepcopy(grad), deepcopy(tc_copy2)
    _ = cv2.imwrite("gradient_images\s2_gradient" + str(iteration) + "+.JPG", img_sift)
    color_boxes, list_of_features = get_text_boxes(grad, img_color)
    _ = cv2.imwrite("box_images\k05_s2_ccl\s2_ccl_boxes_" + str(iteration) + ".JPG", color_boxes)
    _ = cv2.imwrite("cropped_box_images\k05_s2_ccl\s2_ccl_boxes_" + str(iteration) + ".JPG",
                    color_boxes[200:600, 500:900])
    bg_copy = deepcopy(rem_bg)
    bg_copy = 255 - bg_copy
    bg_copy = get_morph(bg_copy, morph="dilate", sz=(5, 5))
    copy2 = deepcopy(tc_copy2)
    mboxes2, _ = mser.detectRegions(bg_copy)
    for box in mboxes2:
        x, y, w, h = cv2.boundingRect(box)
        cv2.rectangle(copy2, (x, y), (x + w, y + h), (255, 0, 255), 1)
    _ = cv2.imwrite("box_images\\bgremoval_mser\\non_bin_boxes_mser" + str(iteration) + ".JPG", copy2)
    _ = cv2.imwrite("cropped_box_images\\bgremoval_mser\\non_bin_boxes_mser" + str(iteration) + ".JPG",
                    copy2[200:600, 500:900])

    copy2 = deepcopy(tc_copy2)
    s_copy = 255 - s_copy
    mboxes2, _ = mser.detectRegions(s_copy)
    # print(len(mboxes2))
    for box in mboxes2:
        x, y, w, h = cv2.boundingRect(box)
        cv2.rectangle(copy2, (x, y), (x + w, y + h), (255, 0, 255), 1)
    _ = cv2.imwrite("box_images\k05_s2_mser\s2_mser_boxes_" + str(iteration) + ".JPG", copy2)
    _ = cv2.imwrite("cropped_box_images\k05_s2_mser\s2_mser_boxes_" + str(iteration) + ".JPG", copy2[200:600, 500:900])
    # show_img(tricolor_img)
    _, thresh = cv2.threshold(rem_bg, 0, 255, cv2.THRESH_OTSU)

    mask2 = np.zeros_like(gs_img).astype(np.uint8)
    # show_img(s_copy)
    mask2[PST(s1)[0]] = 255
    _ = cv2.imwrite("edge_images\edge_image_k05_sharpen" + str(iteration) + ".JPG",
                    mask2)

    b_img = line_cleaning(mask2, thresh,
                          use_Canny=False, is_binary=True).astype(np.uint8)
    _ = cv2.imwrite("lines_cleared\\rem_bg\cleared_lines_PST_binary" + str(iteration) + ".JPG",
                    b_img)

    # _, b_img = cv2.threshold(b_img, 0, 255, cv2.THRESH_OTSU)
    # show_img(b_img)
    grad = get_morph(b_img, morph="gradient", sz=(3, 3), threshold=True)
    # show_img(grad)
    # show_img(grad)
    img_sift, img_color = deepcopy(grad), deepcopy(tc_copy2)
    _ = cv2.imwrite("gradient_images\\bgremoved_gradient" + str(iteration) + "+.JPG", img_sift)
    color_boxes, list_of_features = get_text_boxes(grad, img_color)
    _ = cv2.imwrite("box_images\\bgrem_lc\\lc_boxes_ccl_" + str(iteration) + ".JPG",
                    color_boxes)
    _ = cv2.imwrite("cropped_box_images\\bgrem_lc\\lc_boxes_ccl_" + str(iteration) + ".JPG",
                    color_boxes[200:600, 500:900])

    g_img = line_cleaning(hard_mask, rem_bg,
                          use_Canny=False, is_binary=False).astype(np.uint8)
    _ = cv2.imwrite("lines_cleared\\rem_bg\cleared_lines_PST_nonbinary" + str(iteration) + ".JPG",
                    g_img)

    # show_img(s_copy))
    g_img = 255 - g_img
    g_img = get_morph(g_img, morph="dilate", sz=(5, 5))

    mboxes4, _ = mser.detectRegions(g_img)
    # show_img(tc_copy4)
    # for box in mboxes:
    #    x, y, w, h = cv2.boundingRect(box)
    #    coords = (x, y, x + w, y + h)
    copy2 = deepcopy(tc_copy2)

    #    box_coords.append(coords)
    for box in mboxes4:
        x, y, w, h = cv2.boundingRect(box)
        cv2.rectangle(copy2, (x, y), (x + w, y + h), (255, 0, 255), 1)
        # show_img(tricolor_img)
    _ = cv2.imwrite("box_images\\bgrem_mser\\mser_lc_" + str(iteration) + ".JPG", copy2)
    _ = cv2.imwrite("cropped_box_images\\bgrem_mser\mser_lc_boxes_" + str(iteration) + ".JPG",
                    copy2[200:600, 500:900])

    ################################################################################

    ### Generate results from second preprocessing procedure
    ################################################################################
    mser = cv2.MSER_create(_delta=35, _max_variation=0.20)
    s2 = normalization(s_copy, k=0.7)
    # show_img(s2)
    s_copy = deepcopy(s2)
    _ = cv2.imwrite("p_normalized\k_normalized_img_" + str(iteration) + "+.JPG", s2)
    # show_img(s2)
    # _, s2 = cv2.threshold(s2, 0, 255, cv2.THRESH_OTSU)
    grad = get_morph(s2, morph="gradient", sz=(3, 3), threshold=True)
    img_sift, img_color = deepcopy(grad), deepcopy(tc_copy2)
    _ = cv2.imwrite("gradient_images\k07_gradient" + str(iteration) + "+.JPG", img_sift)
    color_boxes, list_of_features = get_text_boxes(grad, img_color)
    _ = cv2.imwrite("box_images\k07_s2_ccl\s2_ccl_boxes_" + str(iteration) + ".JPG", color_boxes)
    _ = cv2.imwrite("cropped_box_images\k07_s2_ccl\k07_s2_ccl_boxes_" + str(iteration) + ".JPG",
                    color_boxes[200:600, 500:900])
    # show_img(color_boxes)
    # box_coords = []
    mboxes2, _ = mser.detectRegions(s2)
    # print(len(mboxes2))
    # show_img(tc_copy2)
    for box in mboxes2:
        x, y, w, h = cv2.boundingRect(box)
        cv2.rectangle(tc_copy2, (x, y), (x + w, y + h), (255, 0, 255), 1)

    # show_img(tricolor_img)
    # for box in mboxes:
    #     x, y, w, h = cv2.boundingRect(box)
    #     cv2.rectangle(tricolor_img, (x, y), (x + w, y + h), (255, 0, 255), 1)
    _ = cv2.imwrite("box_images\k07_s2_mser\k07_s2_mser_boxes_" + str(iteration) + "+.JPG", tc_copy2)
    _ = cv2.imwrite("cropped_box_images\k07_s2_mser\k07_s2_mser_boxes__" + str(iteration) + ".JPG",
                    tc_copy2[200:600, 500:900])
    # show_img(tricolor_img)
    mask2 = np.zeros_like(gs_img).astype(np.uint8)
    # show_img(s_copy)
    s_copy = 255 - s_copy
    mask2[PST(s_copy)[0]] = 255
    _ = cv2.imwrite("edge_images\edge_image_k07_sharpen" + str(iteration) + ".JPG",
                    mask2)
    ################################################################################

    ### Generate results from third preprocessing procedure
    ################################################################################
    mser = cv2.MSER_create(_delta=25, _max_variation=0.20)
    s_copy, n_copy = preproc(gs_img)
    _ = cv2.imwrite("p_normalized\sharpened_normalized_img_" + str(iteration) + "+.JPG", n_copy)
    _, s2 = cv2.threshold(s2, 0, 255, cv2.THRESH_OTSU)
    grad = get_morph(s2, morph="gradient", sz=(5, 5))
    img_sift, img_color = deepcopy(grad), deepcopy(tc_copy3)
    _ = cv2.imwrite("gradient_images\sharp_normal_gradient" + str(iteration) + "+.JPG", img_sift)
    color_boxes, list_of_features = get_text_boxes(grad, img_color)
    _ = cv2.imwrite("box_images\s3_ccl\s3_ccl_boxes_" + str(iteration) + "+.JPG", color_boxes)
    _ = cv2.imwrite("cropped_box_images\s3_ccl\s3_ccl_boxes__" + str(iteration) + ".JPG",
                    color_boxes[200:600, 500:900])
    # show_img(tc_copy3)
    # show_img(color_boxes)
    # show_img(n_copy)
    normal_copy = deepcopy(n_copy)
    n_copy = 255 - n_copy
    mboxes3, _ = mser.detectRegions(n_copy)
    # print(len(mboxes3))
    for box in mboxes3:
        x, y, w, h = cv2.boundingRect(box)
        cv2.rectangle(tc_copy3, (x, y), (x + w, y + h), (255, 0, 255), 1)

        # supp_boxes = box_nms(np.array(box_coords), overlapThresh=0.1)
    # for box in supp_boxes:
    # show_img(tricolor_img)
    _ = cv2.imwrite("box_images\s3_mser\s3_mser_boxes_" + str(iteration) + "+.JPG", tc_copy3)
    _ = cv2.imwrite("cropped_box_images\s3_mser\s3_mser_boxes__" + str(iteration) + ".JPG",
                    tc_copy3[200:600, 500:900])
    mask3 = np.zeros_like(gs_img).astype(np.uint8)
    mask3[PST(s_copy)[0]] = 255
    _ = cv2.imwrite("edge_images\edge_image_norm" + str(iteration) + ".JPG",
                    mask3)
    ################################################################################

    ### Generate results from fourth preprocessing procedure
    ################################################################################
    mser = cv2.MSER_create(_delta=40, _max_variation=0.20, _min_diversity=2)
    uses_PST = True
    _, thresh = cv2.threshold(normal_copy, 0, 255, cv2.THRESH_OTSU)
    if uses_PST:
        # show_img(img, "img")
        # show_img
        # show_img(binary_copy, "bin")
        b_img = line_cleaning(mask2, thresh,
                              use_Canny=False, is_binary=True).astype(np.uint8)

        _ = cv2.imwrite("lines_cleared\cleared_norm_lines_binary" + str(iteration) + ".JPG",
                        b_img)
        # _, b_img = cv2.threshold(b_img, 0, 255, cv2.THRESH_OTSU)
        # show_img(b_img)
        grad = get_morph(b_img, morph="gradient", sz=(3, 3))
        # show_img(grad)
        # show_img(grad)
        img_sift, img_color = deepcopy(grad), deepcopy(tc_copy4)
        _ = cv2.imwrite("gradient_images\\norm_lines_cleared" + str(iteration) + "+.JPG", img_sift)
        color_boxes, list_of_features = get_text_boxes(grad, img_color)
        # show_img(color_boxes)
        _ = cv2.imwrite("box_images\s4_ccl\\boxes_ccl_" + str(iteration) + ".JPG",
                        color_boxes)
        _ = cv2.imwrite("cropped_box_images\s4_ccl\\boxes_ccl_" + str(iteration) + ".JPG",
                        color_boxes[200:600, 500:900])

        g_img = line_cleaning(hard_mask, normal_copy,
                              use_Canny=False, is_binary=False).astype(np.uint8)
        _ = cv2.imwrite("lines_cleared\cleared_norm_lines_nonbinary" + str(iteration) + ".JPG",
                        g_img)
        # show_img(s_copy)
        mboxes4, _ = mser.detectRegions(g_img)
        # show_img(tc_copy4)
        # for box in mboxes:
        #    x, y, w, h = cv2.boundingRect(box)
        #    coords = (x, y, x + w, y + h)

        #    box_coords.append(coords)
        for box in mboxes4:
            x, y, w, h = cv2.boundingRect(box)
            cv2.rectangle(tc_copy4, (x, y), (x + w, y + h), (255, 0, 255), 1)
            # show_img(tricolor_img)
        _ = cv2.imwrite("box_images\s4_mser\s4_mser_boxes_" + str(iteration) + ".JPG", tc_copy4)
        _ = cv2.imwrite("cropped_box_images\s4_mser\s4_mser_boxes__" + str(iteration) + ".JPG",
                        tc_copy4[200:600, 500:900])

        # supp_boxes = box_nms(np.array(box_coords), overlapThresh=0.1)
        # for box in supp_boxes:
        #     x1, y1, x2, y2 = box
        #     cv2.rectangle(tricolor_img, (x1, y1), (x2, y2), (255, 0, 255), 1)

        # _ = cv2.imwrite("box_images\s4_boxes_mser_" + str(iteration) + ".JPG",
        #                color_boxes)

        # show_img(img, "lines cleared")
    else:
        sharp_copy = 255 - s1.astype(np.uint8)
        # show_img(sharp_copy, "sharp")
        img = line_cleaning(sharp_copy, n_copy, use_Canny=True).astype(np.uint8)
        _ = cv2.imwrite("lines_cleared\cleared_lines_Canny" + str(iteration) + ".JPG",
                        mask3)
    ################################################################################

    ### Generate results from special preprocessing procedure
    ################################################################################
    mser = cv2.MSER_create(_delta=5)
    s1 = sharpen(gs_img, type="nct").astype(np.uint8)
    _ = cv2.imwrite("sharpened_images\sharpened_img_" + str(iteration) + ".JPG", s1)
    grad = get_morph(s1, morph="gradient", sz=(3, 3), threshold=True)
    img_sift, img_color = deepcopy(grad), deepcopy(tc_copy5)
    color_boxes, list_of_features = get_text_boxes(grad, img_color)
    _ = cv2.imwrite("box_images\special_ccl\special_ccl_boxes_" + str(iteration) + ".JPG", color_boxes)
    _ = cv2.imwrite("cropped_box_images\special_ccl\special_ccl_boxes__" + str(iteration) + ".JPG",
                    color_boxes[200:600, 500:900])
    # show_img(color_boxes)
    gs_img = cv2.GaussianBlur(gs_img, (5, 5), 5 / 6)
    mboxes5, _ = mser.detectRegions(gs_img)
    # print(len(mboxes1))
    # show_img(tc_copy1)
    for box in mboxes5:
        x, y, w, h = cv2.boundingRect(box)
        cv2.rectangle(tc_copy5, (x, y), (x + w, y + h), (255, 0, 255), 1)
    # show_img(tricolor_img)
    _ = cv2.imwrite("box_images\special_mser\special_mser_boxes_" + str(iteration) + ".JPG", tc_copy1)
    _ = cv2.imwrite("cropped_box_images\special_mser\special_mser_boxes__" + str(iteration) + ".JPG",
                    tc_copy1[200:600, 500:900])
    # print("Keypoints detected")
    ################################################################################


def add_padding(img):
    # We add a border region around the image s.t.,
    # the book does not clip the edges of the image, thereby ruining the segmentation
    # Presumably, but not necessarily, the image is in RGB when this function is called
    if len(img.shape) < 3:
        new = np.zeros((img.shape[0] + 20, img.shape[1] + 20), dtype=np.uint8)
        new[10:new.shape[0] - 10, 10:new.shape[1] - 10] = img
    else:
        new = np.zeros((img.shape[0] + 20, img.shape[1] + 20, 3), dtype=np.uint8)
        new[10:new.shape[0] - 10, 10:new.shape[1] - 10] = img
    # show_img(new, "new")
    return img



def preproc(img):
    # img = get_morph(img, morph="erode", sz=(3, 3))
    # show_img(normalized, "normal")
    # normalized = 255 - normalized
    sharpened = sharpen(img, type="nct").astype(np.uint8)
    sharp_copy = deepcopy(sharpened)

    rem_bg = (bgremoval(sharp_copy, f=9, g=3, th=0.5, ct=1) * 255).astype(np.uint8)
    rem_bg = cv2.divide(sharpened, rem_bg, scale=255)
    # show_img(rem_bg)
    normalized = normalization(sharpened, k=0.5)
    # filt_real, filt_imag = gabor(normalized, frequency=0.1)
    # normalized = cv2.divide(normalized, filt_real, scale=255)
    # show_img(normalized, "normalized")
    normalized_copy = deepcopy(normalized)

    # show_img(cv2.dilate(median_blur, None), "median")
    return sharp_copy, normalized_copy


def call_scale(img, direction, iters):
    scaled_img = pyramid(img, direction=direction, iterations=iters)
    # print(scaled_img.shape)
    img = add_padding(scaled_img)
    copy_img = deepcopy(img)
    # gs = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
    return copy_img, img


def exec(img, noise=False, scale=True, corr_shading=False, seg_book=True, sharpen_img=True, descriptor_list=None, feature_list=None,
         do_nothing=False,
         uses_PST=True, iteration=0):
    color_copy = deepcopy(img)
    # _ = cv2.imwrite('HSV_IMG_'+str(iteration)+".JPG",cv2.cvtColor(color_copy, cv2.COLOR_BGR2HSV))
    iteration = train_classifier(img)
    print("Current label count:", iteration)
    return iteration

    ### Canny edges
    # edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 200)
    # _ = cv2.imwrite('edge_images\canny_edges\img_' + str(iteration) + ".JPG", edges)

    # print("hej")
    # generate_visualizations(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img, iteration)
    if do_nothing:
        return


    return img


def subspace_visualizer(box):
    t_sne = skl_m.TSNE(n_components=2)
    subspace_box = t_sne.fit_transform(box)
    return


def single(os="win", scale=True, remove_noise=False, corr_shading=True, seg_book=True, sharpen=True, do_nothing=False):
    os = "win"
    ## EXAMPLE USAGE OF 1 IMAGE
    if os == "linux":
        f_name = 'test_img/IMG_3.JPG'
    if os == "win":
        f_name = 'image_folder\IMG_70.JPG'
        # f_name = 'img_test.jpeg'
        # f_name = 'ideal_grid.jpg'
        # f_name = 'sharpened_img.jpg'

    f_obj = open_file(name=f_name, multiple=False, n_img=1)

    img = f_obj.img
    return exec(img, scale=scale, noise=remove_noise, corr_shading=corr_shading, seg_book=seg_book, sharpen_img=sharpen,
                do_nothing=do_nothing)


def multi(n_img, os="win", scale=True, remove_noise=False, corr_shading=True, seg_book=True, sharpen=True,
          do_nothing=False):
    if os == "linux":
        f_name = 'image_folder/IMG_'
    if os == "win":
        f_name = 'image_folder\\'

    # init BoW cluster
    # BOW = cv2.BOWKMeansTrainer(4)
    # We can retrieve less images than we load to the array
    print("####" * 10)
    print("Retrieving image generator. This may take a while, depending on image size and number of images")
    print("####" * 10)
    img_generator = open_file(name=f_name, multiple=True, n_img=n_img)
    descriptor_list = []

    # t_sne = skl_m.TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    print("Starting the script")
    for i in range(1,n_img):

        print("Starting image:", i + 1, "/", n_img)
        # The image generator is a yield function.
        # Calling it with the next keyword will iterate the function and return one image from the generator loop
        # Saves overhead memory costs (in comparison to keeping a long array of large images in memory)
        img = next(img_generator)
        # try:
        #     scaled, scaled_copy = call_scale(img, direction="down", iters=2)
        # blurred = cv2.GaussianBlur(img, (3, 3), 3 / 6)
        # print("Image has been scaled")
        iteration = exec(img,
                 seg_book=True,
                 sharpen_img=sharpen,
                 descriptor_list= descriptor_list,
                 scale=False,
                 )

        get_cancel_input = input("Enter 0 to break labeling, else enter any: ")
        if get_cancel_input == "0":
            break
        else:
            continue


        # except:
        #     print('Image {} does not exist'.format(i + 1))

        # final = exec(img, scale=scale, noise=remove_noise, corr_shading=corr_shading, seg_book=seg_book, sharpen_img=sharpen, do_nothing=do_nothing)
        # show_img(final, "Final image")
        # write_img("final_image_" + str(i) + ".jpg", final)


def main(os: str, multiple_images=False, n_img=1, scale=True, remove_noise=False, corr_shading=True, seg_book=True,
         sharpen=True, do_nothing=False):
    # For each step of this pipeline, please see the corresponding functions of each script
    # The main script has been cleaned extensively to avoid a huge mess of code
    if multiple_images:
        final_img = multi(n_img, os=os, scale=scale, remove_noise=remove_noise, corr_shading=corr_shading,
                          seg_book=seg_book, sharpen=sharpen, do_nothing=do_nothing)
    else:
        # final_img = single(os)
        final_img = single(os=os, scale=scale, remove_noise=remove_noise, corr_shading=corr_shading, seg_book=seg_book,
                           sharpen=sharpen, do_nothing=do_nothing)


# main(os="win", scale=True, remove_noise=False, corr_shading=True, seg_book=False, sharpen=True, do_nothing=False)
# main(os="win", remove_noise=False, corr_shading=False, seg_book=False, sharpen=True, multiple_images=True, n_img=15)
main(os="win", multiple_images=True, n_img=200, scale=False, do_nothing=False)


### Storage space ###

# import pandas as pd
# pca_50 = PCA(n_components=50)
# pca_components = pca_50.fit_transform(X=arr)
# print('Cumulative explained variation for 50 principal components: {}'.format(
#    np.sum(pca_50.explained_variance_ratio_)))
# transformator = t_sne.fit_transform(X=arr)

# import seaborn as sns
# subset = pd.DataFrame(data=[transformator[0,:], transformator[1,:]], columns=["tsne-2d-one", "tsne-2d-two"])
# print(subset)
# sns.scatterplot(
#   x="tsne-2d-one", y="tsne-2d-two",
#   hue=y,
#   palette=sns.color_palette("hls", 10),
#   data=subset,
#   legend="full",
#   alpha=0.3
# )

# bez = second_order_deriv(blur, fltr="bezier")
# bez_edt = distance_transform_edt(bez).astype(np.uint8)
# get_max_peaks(bez_edt)
# coords = peak_local_max(bez_edt, footprint=np.ones((5, 5)), labels=bez)
# mask = np.zeros(bez_edt.shape, dtype=bool)
# mask[tuple(coords.T)] = True
# markers, _ = ndi.label(mask)
# plt.imshow(markers, cmap=plt.cm.nipy_spectral)
# plt.title("Markers")
# plt.show()
# labels = watershed(-distance, markers, mask=image)

# bez_cdt = distance_transform_cdt(bez, metric='chessboard').astype(np.uint8)
# show_img(bez, "bezier")
# show_img(bez_edt, "euclidian dist. tform, bezier")
# show_img(bez_cdt, "chamfer dist. tform bezier")
# show_img(img, "lines cleared")
# grad = get_morph(img, "erode", (5, 5), kernel="gaussian", ttype="otsu", threshold=True)
# grad_edt = distance_transform_edt(grad).astype(np.uint8)
# grad_cdt = distance_transform_cdt(grad, metric='chessboard').astype(np.uint8)
# show_img(grad, "grad")
# show_img(grad_edt, "euclidian dist. tform, grad")
# show_img(grad_cdt, "chamfer dist. tform, grad")


def call_segment_book(img):
    # hsv_img = preprocessing(img).astype(np.uint8)
    # shading_corrected = correct_shading(img).astype(np.uint8)
    # show_img(shading_corrected, "shacorr hsv")
    segmented_img, book_x, book_y, book_w, book_h = segmentation(img)
    copy_img = deepcopy(segmented_img)
    return copy_img, segmented_img, book_x, book_y, book_w, book_h
    # show_img(img, "img")


def get_max_peaks(img):
    image_max = ndi.maximum_filter(img, size=20, mode='constant')
    coordinates = peak_local_max(img, min_distance=20)
    print(coordinates)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.autoscale(False)
    plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    plt.axis('off')
    plt.show()
    G = nx.Graph()
    G.add_nodes_from(coordinates)


# if do_nothing:
# shading_corrected = correct_shading(img).astype(np.uint8)
# gs = cv2.cvtColor(shading_corrected, cv2.COLOR_BGR2GRAY)
# filt_real, filt_imag = gabor(gs, frequency=0.1)
# gs[filt_real != 0] = 0
# non_zero = np.where(filt_real != 0)
# for i in range(len(non_zero[0])):
#    gs[non_zero[0][i], non_zero[1][i]] = np.median(
#        gs[non_zero[0][i] - 21:non_zero[0][i] + 21, non_zero[1][i] - 21:non_zero[1][i] + 21])
# show_img(gs, "gabor filter")
# _, thresh = cv2.threshold(gs, 0, 255, cv2.THRESH_OTSU)
# show_img(thresh, "img")
# return gs

def markers(image):
    from skimage.morphology import disk
    from skimage.filters import rank

    # denoise image
    denoised = rank.median(image, disk(2))

    # find continuous region (low gradient -
    # where less than 10 for this image) --> markers
    # disk(5) is used here to get a more smooth image
    markers = rank.gradient(denoised, disk(5)) < 10
    markers = ndi.label(markers)[0]

    # local gradient (disk(2) is used to keep edges thin)
    gradient = rank.gradient(denoised, disk(2))
    # display results
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title("Original")

    ax[1].imshow(gradient, cmap=plt.cm.nipy_spectral)
    ax[1].set_title("Local Gradient")

    ax[2].imshow(markers, cmap=plt.cm.nipy_spectral)
    ax[2].set_title("Markers")
    return gradient


def seg_unused(img):
    # padded = add_padding(img)
    # sharp_copy, img = call_sharpen(img)
    gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gs, 0, 255, cv2.THRESH_OTSU)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 100])
    higher_blue = np.array([60, 60, 255])

    lower_magenta = np.array([100, 0, 100])
    higher_magenta = np.array([255, 40, 255])

    b_mask = cv2.inRange(hsv_img, lower_blue, higher_blue)
    m_mask = cv2.inRange(hsv_img, lower_magenta, higher_magenta)

    final = (gs & b_mask) | (gs & m_mask)
    thresh = get_morph(get_morph(final, morph="close", sz=(7, 7), kernel="ones", threshold=True), morph="erode",
                       sz=(3, 3), kernel="ones")
    # show_img(thresh.astype(np.uint8), "thresh in main")
    # padded = add_padding(thresh)
    # show_img(padded, "second padding")
    img, book_x, book_y, book_w, book_h = segmentation(thresh.astype(np.uint8), img)
    segment_copy = deepcopy(img)
    # show_img(img, "img")
    # show_img(img, "img")

    ## scale up again
    # thresh = np.zeros_like(img)
    # thresh[img != (0,0,0)] = 255
    # upscaled = pyramid(img, direction="up", iterations=2)
    # downscaled = resize(img=unscaled_copy, dim=(upscaled.shape[1], upscaled.shape[0])).astype(np.uint8)
    # downscaled[upscaled == (0,0,0)] = 0
    # show_img(downscaled, "downscaled"


def calc_hist(img):
    # Visualization function for the report
    try:
        assert len(img.shape) > 2
    except:
        print("Either image is non-existant or is not tricolor")
        print("No histogram of color sums will be done")
    s_red = np.sum(img[:, :, 2])
    s_green = np.sum(img[:, :, 1])
    s_blue = np.sum(img[:, :, 0])
    print(s_red, s_green, s_blue)
    objects = ('Red', 'Green', 'Blue')
    y_pos = np.arange(len(objects))
    performance = [s_red, s_green, s_blue]

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Total ')
    plt.title('Bar chart over RGB pixel values')

    plt.show()
    show_img(img, "RGB image")
    show_img(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), "HSV image")

    def part_one(img):
        ## Book segmentation and preprocessing
        scale_copy, binary_copy, segment_copy, sharp_copy = None, None, None, None

        # if seg_book:
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # show_img(hsv, "hsv")
        # segmented = segmentation(hsv, img)
        # segment_copy = deepcopy(segmented)
        # if (segment_copy.shape[1] > 1000) and (segment_copy.shape[0] > 700):
        #     c1+=1
        # elif (segment_copy.shape[1] < 1000) and (segment_copy.shape[0] > 700):
        #     c2 += 1
        # else:
        #     c3 += 1

        # print(segment_copy)

        try:
            assert len(img.shape) == 2
        except AssertionError:
            print("exception")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return sharp_copy, binary_copy, segment_copy

    # scale_copy, segment_copy, sharp_copy, img = part_one(img)
    c = deepcopy(img)
    s = part_one(img)
    # status = cv2.imwrite("segmented_images\segmented_book"+str(iteration)+".JPG", seg)
    # print("Image written:",status)
    # print(mean_gs)

    # show_img(segment_copy, "segmented")
    # show_img(img, "edge image")
    # show_img(img.astype(np.uint8), "edge image")

    # show_img(gs, "Grayscale image"
    # get_max_peaks()

    def part_two(color_copy, sharp_copy, normalized_copy, normalize=True, clear_lines=True):
        ## Clean lines and preprocess image for text box detection

        # img_sift, img_color = deepcopy(grad), deepcopy(color_copy)
        # color_boxes, list_of_features = get_text_boxes(grad, img_color, feature_list)

        # mser = cv2.MSER_create()
        # kp_mser, boxes = mser.detectRegions(sharp_no_lines)
        # for box in boxes:
        # print(kp_sift)
        # show_img(box, "box")
        # print("Keypoints detected")

        copy = deepcopy(color_copy)

        # print("Keypoints drawn")
        # show_img(img, "keypoints image")
        # return list_of_features
        # get_max_peaks(grad)s)

    # img = part_two(c, s, n, clear_lines=True)

    def part_three():
        # show_img(second_order_deriv(img=white_tophat, fltr="cubic"), "cubic spline deriv")

        return
        # resized = resize(grad, (grad.shape[1]//4, grad.shape[0]//4))
        # show_img(grad, "gradient")
        # erode_grad = get_morph(grad, "erode", (7,1), threshold=False)
        # show_img(grad, "eroded gradient")

        # for box in boxes:
        # todo - label boxes and fit to TSNE
        # show_img(box, "boxbox")

        # subspace_visualizer(boxes)

        # show_img(box_img, "title me")

