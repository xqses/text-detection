import cv2
import numpy as np
import skimage.filters as sk_filt

def ent_thresh(img):
    # Compute Otsu's gray level threshold selection value
    # send through an image

    ## to get the probability of colors p1 p2
    # t = approx. mean color value of the image
    n_pixels = img.shape[0] * img.shape[1]
    histg = cv2.calcHist([img], [0], None, [256], [0, 256])
    p = np.zeros((histg.shape[0]))
    # plt.plot(histg)
    # plt.show()
    for i in range(len(p)):
        p[i] = histg[i]

    t = np.where(p == np.max(p))[0][0]

    p1 = [None] * t
    p2 = [None] * (256 - t)

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
    mb = 2 * e * 0.753491
    mw = e * 0.753491

    return e

def bgremoval(img, f, g, th, ct):
    try:
        assert len(img.shape) < 3
    except AssertionError:
        print("Image needs to be grayscale for background removal")
        print("Converting to grayscale")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float64)

    s = img.shape
    # s[0] += 1 if s[0] % 2 == 0 else 0
    # s[1] += 1 if s[1] % 2 == 0 else 0
    krn = np.ones(s, dtype=np.float64)

    if g > 1:
        N, sigma, _ = mask(g, 0)
        # print(sigma, "sigma")
        div1 = cv2.GaussianBlur(krn, (N,1), sigma)
        div1 = div1.astype(np.float64)
        p1 = np.divide(cv2.GaussianBlur(img, (N,1), sigma), div1)
        # print(p1)
    else:
        p1 = img

    N, sigma, _ = mask(s, 300)
    try:
        assert type(N) == int
    except:
        N = int(N)

    div2 = cv2.GaussianBlur(krn, (N,1), sigma).astype(np.float64)
    p2 = np.divide(cv2.GaussianBlur(img, (N,1), sigma),div2)
    im2 = p2-p1

    # ts = sk_filt.threshold_yen(im2)
    im2 = 255 * im2  # Now scale by 255
    im2 = im2.astype(np.uint8)
    ts = ent_thresh(im2)
    im2 = im2.astype(np.float64)

    # print(entropic_thresh)

    # Control the entropic thresh value using th
    nim1 = (im2>(ts*th))


    if f > 1:
        N, sigma, _ = mask(f, 0)
        try:
            assert type(N) == int
        except:
            N = int(N)
        div1 = cv2.GaussianBlur(krn, (N,1), sigma)
        p1 = cv2.GaussianBlur(img, (N,1), sigma) / div1
    else:
        p1 = img

    N, sigma, _ = mask(s, 100)
    try:
        assert type(N) == int
    except:
        N = int(N)
    # print(krn)
    div2 = cv2.GaussianBlur(krn, (N,1), sigma)
    p2 = cv2.GaussianBlur(img, (N,1), sigma) / div2
    im2 = p2 - p1
    # print(im2.shape, "im2")


    nim2 = (im2 > 0) * im2
    try:
        assert type(N) == int
    except:
        N = int(N)

    if ct:
        nim2 = nim2 - np.min(np.min(nim2))
        nim2 = nim2/np.max(np.max(nim2))

    nim = 1- nim1*nim2


    return nim

def mask(So, sz):
    if sz != 0:
        scale = min(So) / sz
        # print(scale, "scale")
        S = np.ceil(np.divide(So, scale)).astype(np.uint8)
        # print(S, "S")
        N = np.ceil(max(S))
        # print(N, "N")
        S[0] += 1 if S[0] % 2 == 0 else 0
        S[1] += 1 if S[1] % 2 == 0 else 0
        N += 1 if N % 2 == 0 else 0

    else:
        N = So
        # print("sz 0, N:", N)
        S = 0


    sigma = N / 6

    return N, sigma, S