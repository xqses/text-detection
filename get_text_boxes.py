import cv2
import numpy as np
from morphology import get_morph
from copy import deepcopy
from basics import show_img
from matplotlib import pyplot as plt
from box_nms import box_nms


def get_text_boxes(thresh_img, im3c, list_of_features=None, do_watershed=False):
    copy3c = deepcopy(im3c)
    sift = cv2.SIFT_create()
    output = deepcopy(im3c)
    out = cv2.connectedComponentsWithStats(thresh_img, connectivity=8)
    n_labels, labels, stats, centroids = out
    # print("n labels:", n_labels)
    # show_img(thresh_img, "grad")

    boxes = []
    box_coords = []
    final = []
    # rint(im3c.shape)
    print(n_labels)

    for i in range(1, n_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        # print(area)
        (cX, cY) = centroids[i]
        if 50 < area < 30000:
            # copy_clean = deepcopy(copy3c)
            # box_coords.append((x,y,x+w,y+h))
            # keypoints1, descriptors1 = brisk.detectAndCompute(thresh_img[y:y + h, x:x + w], None)
            # kp_sift = sift.detect(copy_clean[y:y + h, x:x + w], None)
            # print(kp_sift)
            # hp, des = sift.compute(copy_clean[y:y + h, x:x + w], kp_sift)  # note: no mask here!
            # try:
            #    list_of_features.add(des)
            # mboxes, _ = mser.detectRegions(output[y:y+h, x:x+w])

            # cv2.drawKeypoints(copy3c[y:y + h, x:x + w], kp_sift, copy_clean[y:y + h, x:x + w],
            #                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS).astype(np.uint8)
            cv2.rectangle(copy3c, (x, y), (x + w, y + h), (255, 0, 255), 1)
            # cv2.drawKeypoints(copy3c[y:y + h, x:x + w], kp_fast, copy3c[y:y + h, x:x + w],
            #                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS).astype(np.uint8)

            # bounding_boxes = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in mboxes]
            # show_img(im3c[y:y + h, x:x + w], "LABEL ME")
            # if type(des) != None:
            #     x_start, y_start = (x // 2), (y // 2)
            #     plt.imshow(copy_clean[y_start:y_start + y + h, x_start:x_start + x + w])
            #     plt.pause(0.05)
            #     label = input('category: ')


                # print(kp_fast)
            #     final.append([copy3c[y:y+h, x:x+w], kp_sift, label])

    # show_img(im3c, "im3c")

    # print(box_coords)
    # supp_boxes = box_nms(np.array(box_coords), overlapThresh=0.2)
    # for box in supp_boxes:
        # (box)
    #     x1,y1,x2,y2 = box
    #    cv2.rectangle(im3c, (x1, y1), (x2, y2), (255, 0, 255), 1)
    # show_img(output, "title")
    # show_img(im3c, "sift boxes")
    # show_img(copy3c, "fast boxes")
    # show_img(copy3c)

    return copy3c.astype(np.uint8), list_of_features


def FDT(im):
    ## A manual implementation of the fuzzy distance transform

    # For each pixel y,x we slide a 3x3 window calculating the FD of the centre pixel
    window_y, window_x = 3, 3
    # We cannot guarantee that either y or x region will be odd, so add if clause

    final = {}
    # Using a nested dictionary for faster lookup
    # This makes things slightly complicated-looking
    # But it saves runtime, so that's nice
    print(im.shape)

    change_made = True
    while change_made:
        ## Notably, this iterative process should continue until no update is made
        change_made = False
        looped = {}

        # Forward pass loop
        for y in range(window_y, im.shape[0] - window_y + 1):
            looped[y] = {}
            for x in range(window_x, im.shape[1] - window_x + 1):
                subwindow = im[y - window_y:y + window_y, x - window_x:x + window_x]

                # calculating the forward pass fuzzy distances to the pixel
                fp_dist = np.array([np.int(np.linalg.norm(subwindow[1, 1])-subwindow[0, 0]),
                                    np.int(np.linalg.norm(subwindow[1, 1])-subwindow[0, 1]),
                                    np.int(np.linalg.norm(subwindow[1, 1])-subwindow[0, 2]),
                                    np.int(np.linalg.norm(subwindow[1, 1])-subwindow[1, 0])])

                # Store the minimum distance of the fp
                looped[y][x] = np.min(fp_dist)

        # Backward pass loop
        for y in range(im.shape[0] - window_y, window_y, -1):
            # print("y =", y, "window y =", window_y, "y - window_y = ", (y - window_y))
            for x in range(im.shape[1] - window_x, window_x, -1):

                subwindow = im[y - window_y:y + window_y, x - window_x:x + window_x]

                # calculating the backward pass fuzzy distances to the pixel
                bp_dist = np.array([np.int(np.linalg.norm(subwindow[1, 1])-subwindow[2, 2]),
                                    np.int(np.linalg.norm(subwindow[1, 1])-subwindow[2, 1]),
                                    np.int(np.linalg.norm(subwindow[1, 1])-subwindow[2, 0]),
                                    np.int(np.linalg.norm(subwindow[1, 1])-subwindow[1, 2])])

                # Evaluate the minimum distance of the bp
                bp_dist = min(bp_dist)
                fp_dist = looped[y][x]
                if min(fp_dist, bp_dist) < im[y, x]:
                    im[y, x] = min(fp_dist, bp_dist)
                    change_made = True
    im = cv2.normalize(im, im, 0, 1.0, cv2.NORM_MINMAX)
    return im
