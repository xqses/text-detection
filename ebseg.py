import numpy as np
from matplotlib import pyplot as plt
import cv2


def e_b_segmentation(img):
    # send through an image

    ## to get the probability of colors p1 p2
    # t = approx. mean color value of the image
    n_pixels = img.shape[0] * img.shape[1]
    histg = cv2.calcHist([img], [0], None, [256], [0, 256])
    p = np.zeros((histg.shape[0]))
    plt.plot(histg)
    plt.show()
    for i in range(len(p)):
        p[i] = histg[i]

    t = np.where(p == np.max(p))[0][0]

    p1 = [None] * t
    p2 = [None] * (256-t)


    for i in range(len(p1)):
        p1[i] = p[i] / (n_pixels)
    y = t
    for i in range(len(p2)):
        p2[i] = p[y] / (n_pixels)
        y += 1

    p1 += np.finfo(np.float).eps
    p2 += np.finfo(np.float).eps

    base = 255

    hb = -1 * np.sum(p1 * (np.log(p1) / np.log(base)))
    hw = -1 * np.sum(p2 * (np.log(p2) / np.log(base)))

    e = hw + hb
    print(e)

    if e < 0.7:
        mw, mb = 2, 3

    if  0.7 <= e <= 0.85:
        mw, mb = 1, 2.6

    if e > 0.85:
        mw, mb = 1, 1

    lower = mw*hw / 10
    higher = mb*hb / 5
    print(lower)
    print(higher)
    #  print("th:", th)
    print(img)
    print(img / 256)
    img[(img / 256) >= higher] = 255
    img[(img / 256) <= lower] = 0

    return img
