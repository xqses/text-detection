import cv2
import numpy as np
import scipy.signal as scp_signal
from basics import resize, show_img

## A Python rewrite of shaCorr.m authored by Anders Hast
# This code follows the license in the original implementation
# Written by: Olle Dahlstedt 2021-03-04

######### Copyright and permission notice: ##########
##  Copyright (c) 2011-2019 Anders Hast
##  Uppsala University
##  http://www.cb.uu.se/~aht
##
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## The Software is provided "as is", without warranty of any kind.

def shaCorr(inp_img, edge_preserving = True, f = 0.5, bright = 0.8, dark = 0.3, d = 6 , iter = 10, contrast = 2, corr = 0, pfact = 0, msize=100):
    if len(inp_img.shape) > 2:
        final = cv2.cvtColor(inp_img, cv2.COLOR_BGR2HSV)
        if edge_preserving:
            cols, rows, _channels = map(int, inp_img.shape)
            img = final
            placeholder = img[:,:,2]
        else:
            cols, rows, _channels = map(int, inp_img.shape)
            img = final[:,:,2]
            placeholder = img
    else:
        cols, rows = map(int, inp_img.shape)
        img = inp_img

    ## We should convolve the kernel with our chosen window size

    # Lowpass filter setup
    N, sigma, Sret = mask([cols, rows], msize)
    S = Sret
    krn = np.ones(S)
    div2 = cv2.GaussianBlur(krn, krn.shape, sigma)

    if f > 0:
        # Highpass filter setup
        N, sigma, _ = mask(np.ceil(N * f), 0)
        div1 = cv2.GaussianBlur(krn, krn.shape, sigma)

    k = 0
    while k < iter:
        if edge_preserving:
            ## Added by OD 2021-03-05

            # Division of subregions
            ## Resizing the image to roughly kern shape x 2
            ## Let subregion around each pixel be ceil(kern shape / 10)
            # I.e kern shape = 101, 143 -> subregion = ceil((202, 146) / 10) = (21, 15)
            # We would create our subwindow around each pixel thus from pixel at coord (21, 15)
            # Importantly, this means we will not blur the border of the image, which is fine in our material
            resized = resize(img, (krn.shape[1], krn.shape[0]))
            print(resized.shape)
            d = 30
            window_y, window_x = int(np.ceil(resized.shape[0] // d)), int(np.ceil(resized.shape[1] // d))
            # We cannot guarantee that either y or x region will be odd, so add if clause
            window_y += 1 if window_y % 2 == 0 else 0
            window_x += 1 if window_x % 2 == 0 else 0
            kern = np.ones((window_y*2, window_x*2, 3))

            ### Determine the number of subwindows ###
            # Since we will not blur the borders,
            # we will have one subregion around each pixel except the border pixels

            _, sigma, _ = mask(np.max([window_y,window_x]), 0)
            div = cv2.GaussianBlur(kern, (window_y, window_x), sigma)

            inits = {}
            # Using a nested dictionary for faster lookup
            # This makes things slightly complicated-looking (and coming up with)
            # This also means most operations are done within the loop
            # But it saves runtime, so that's nice

            # As we need a window around each pixel, this cannot be run outside two loops
            for y in range(window_y, resized.shape[0]-window_y):
                inits[y] = {}
                # print("y =", y, "window y =", window_y, "y - window_y = ", (y - window_y))
                for x in range(window_x, resized.shape[1]-window_x):
                    inits[y][x] = {}
                    # print("x =", x, "window x =", window_x, "x - window_x = ", (x - window_x))
                    subwindow = resized[y-window_y:y+window_y, x-window_x:x+window_x]

                    ### Blur each window ###
                    # Note: in order to avoid too strong blurring (erasing all color information),
                    # Blur with kernel size half of each window

                    p = cv2.GaussianBlur(subwindow, (window_y, window_x), sigma) / div
                    # Color mean within each (now blurred) window
                    color_means = np.array([np.mean(p[:, :, 0]), np.mean(p[:, :, 1]), np.mean(p[:, :, 2])])
                    inits[y][x]["cmeans"] = color_means


                    # pixelwise distances (5)
                    pd = np.array([np.linalg.norm(np.array([resized[y,x][0], color_means[0]]), ord=2),
                                  np.linalg.norm(np.array([resized[y,x][1], color_means[1]]), ord=2),
                                  np.linalg.norm(np.array([resized[y,x][2], color_means[2]]), ord=2)])

                    inits[y][x]["pixelwise_distance"] = pd

                    # Mean pixelwise distance (6)
                    mpd = np.mean(pd)
                    inits[y][x]["mean_pixelwise_distamce"] = mpd

            t = 20
            im = np.zeros(resized.shape)
            wts = np.ones(resized.shape)
            for y in range(window_y, resized.shape[0]-window_y):
                for x in range(window_x, resized.shape[1] - window_x):
                    im[y,x] = im[y,x] + (resized[y,x] * ((t - inits[y][x]["pixelwise_distance"])**2) * inits[y][x]["cmeans"])
                    wts[y,x] = wts[y,x] + (resized[y,x] * (t - inits[y][x]["pixelwise_distance"])**2)

            p = im / wts
            p = p[:,:,2]

        else:
            if f>0:
                imq = resize(img, (krn.shape[1], krn.shape[0]))
                # Compute the bandpass filter, highpass -> lowpass
                p1 = cv2.GaussianBlur(imq, krn.shape, sigma) / div1
                # print(p1)
                p2 = cv2.GaussianBlur(imq, krn.shape, sigma) / div2

                im2 = p2 - p1
                q = resize(im2, (img.shape[1], img.shape[0]))
                s1 = q[q < 0]
                s2 = q[q > 0]
                q[q < 0] *= bright
                q[q > 0] *= dark
                img = img + q


                N, sigma, _ = mask(np.ceil(np.max(S) / d), 0)
                div = cv2.GaussianBlur(krn, krn.shape, sigma)
                resized = resize(img, (krn.shape[1], krn.shape[0]))
                p = cv2.GaussianBlur(resized, krn.shape, sigma) / div

        if len(resized.shape) > 2:
            resized = resized[:,:,2]
        pm = np.mean(np.mean(p))
        s = np.abs(resized-p)
        ky, kx = s.shape[0]//2, s.shape[1]//2
        ky += 1 if ky % 2 == 0 else 0
        kx += 1 if kx % 2 == 0 else 0
        w = cv2.GaussianBlur(s, (int(ky), int(kx)), sigma)
        if corr > 0:
            wb = resize(w, (rows, cols))
        pb = resize(p, (rows, cols))
        placeholder = placeholder - pb

        mm = np.mean(np.mean(w))
        if corr > 1:
            wb[wb > mm] = mm

        if corr > 2:
            wb[wb < mm * pfact] = mm * pfact

        if corr > 0:
            wb = (wb - mm) * contrast + mm
            placeholder = mm * (placeholder / (wb))

        placeholder = placeholder + pm
        if edge_preserving:
            img[:,:,2] = placeholder
        else:
            img = placeholder
        if d > 1:
            d = d / 2

        k += 1

    if len(inp_img.shape) > 2:
        if edge_preserving:
            final = img
            print(final.shape)
            return cv2.cvtColor(final, cv2.COLOR_HSV2BGR)
        else:
            final[:,:,2] = img
            return cv2.cvtColor(final, cv2.COLOR_HSV2BGR)
    else:
        return img


def mask(So, sz):
    if sz != 0:
        scale = min(So) / sz
        S = np.ceil(np.divide(So, scale)).astype(np.uint8)
        N = np.ceil(max(S))
        S[0] += 1 if S[0] % 2 == 0 else 0
        S[1] += 1 if S[1] % 2 == 0 else 0
        N += 1 if N % 2 == 0 else 0

    else:
        N = So
        S = 0


    sigma = N / 6

    return N, sigma, S