import cv2
import numpy as np
import scipy.signal as scp_signal
from basics import resize, show_img

## A Python rewrite of shaCorr.m authored by Anders Hast
# This code follows the license in the original implementation
# Python rewrite by: Olle Dahlstedt 2021-03-04

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

def shaCorr(inp_img, edge_preserving = False, f = 0.5, bright = 0.5, dark = 2.0, d = 6 , iter = 5, contrast = 2, corr = 0, pfact = 0, msize=100):
    resized = None
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
    print(sigma)
    S = Sret
    krn = np.ones(S)
    div2 = cv2.GaussianBlur(krn, krn.shape, sigma)

    if f > 0:
        # Highpass filter setup
        N, sigma, _ = mask(np.ceil(N * f), 0)
        print(sigma)
        div1 = cv2.GaussianBlur(krn, krn.shape, sigma)

    k = 0
    while k < iter:

        if f>0:
            imq = resize(img, (krn.shape[1], krn.shape[0]))
            # Compute the bandpass filter, highpass -> lowpass
            p1 = cv2.GaussianBlur(imq, krn.shape, sigma) / div1
            # print(p1)
            p2 = cv2.GaussianBlur(imq, krn.shape, sigma) / div2

            im2 = p2 - p1
            q = resize(im2, (img.shape[1], img.shape[0]))

            q[q < 0] *= bright
            q[q > 0] *= dark
            img = img + q


            N, sigma, _ = mask(np.ceil(np.max(S) / d), 0)
            div = cv2.GaussianBlur(krn, krn.shape, sigma)
            resized = resize(img, (krn.shape[1], krn.shape[0]))
            p = cv2.GaussianBlur(resized, krn.shape, sigma) / div

        try:
            assert resized is not None
            if len(resized.shape) > 2:
                resized = resized[:,:,2]
        except AssertionError:
            # Only lowpass filtering
            resized = resize(img, (krn.shape[1], krn.shape[0]))
            p = cv2.GaussianBlur(resized, krn.shape, sigma) / div2
        pm = np.mean(np.mean(p))
        # s = abs(h(x,y) - mu_I(x,y))
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