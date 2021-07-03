import numpy as np
import scipy as scp
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
# import skimage

from basics import show_img


def eq_hist_sharpen(img, convolve: bool, type="nct", n_segs = 0):
    seg_array = [None] * n_segs
    cdf_array = [None]
    start_x, start_y = 0, 0
    ## Primitively, we can segment the image before sharpening
    # The commented code will segment uniformly, i.e. "slicing the image evenly"
    # May be adjusted to deal with prior segmentation

    # If n_segs is passed with anything other than 0, no sharpening will be done.
    if n_segs != 0:
        return
        # for i in range(n_segs):
        # start_y, start_x = i*img.shape[0]//n_segs, i*img.shape[1]//n_segs
        # end_y, end_x = (i+1)*img.shape[0]//n_segs, (i+1)*img.shape[1]//n_segs
        # seg_array[i] = img[start_y:end_y, start_x:end_x]

        # cdf_array = [None] * len(seg_array)
        # for i, seg in enumerate(seg_array):
        # laplace_seg = cv2.Laplacian(seg, ddepth = cv2.CV_16S, ksize = 3)
        # hist, bins = np.histogram(laplace_seg.flatten(), 256, density=True)
        # plt.hist(laplace_seg, bins=256)
        # cdf = hist.cumsum()
        # Normalize cdf to interval [0, 255]
        # cdf = 255 * cdf / cdf[-1]
        # cdf_array[i] = cdf
        # for i in range(len(seg_array)):
        # start_y, start_x = i*img.shape[0]//n_segs, i*img.shape[1]//n_segs
        # end_y, end_x = (i+1)*img.shape[0]//n_segs, (i+1)*img.shape[1]//n_segs
        # img[i,y] = img[start_y:end_y, start_x:end_x]
        # sharpened = np.interp(img.flatten(),bins[:-1],cdf)
    else:
        hist, bins = np.histogram(img.flatten(), 256, density=True)
        if type=="t":
            t_cdf = scp.stats.t.cdf(hist, df=255, loc=np.median(hist), scale=np.var(hist))
            t_cdf = t_cdf.cumsum()
            cdf = ((t_cdf - t_cdf.min()) / (t_cdf.max() - t_cdf.min()) * 255)
        elif type=="emp":
            # print("emp")
            cdf = hist.cumsum()
            cdf = ((cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255)
        else:
            # print("nct")
            nct_cdf = scp.stats.nct.cdf(hist, nc=6, df=256, scale=np.var(hist))
            nct_cdf = nct_cdf.cumsum()
            cdf = ((nct_cdf - nct_cdf.min()) / (nct_cdf.max() - nct_cdf.min()) * 255)


    if convolve:
        exp_interv = np.linspace(0, 5.54, num=256)
        exp_signal = np.exp(exp_interv) / 20
        conv = scp.signal.convolve(exp_signal, cdf, mode="same")
        inv_conv = scp.ifft(conv)
        cdf = np.sqrt((np.real(inv_conv) ** 2) + (np.imag(inv_conv) ** 2))

    # Interpolate the image from the CDF
        #print("hello")
    # xnew = np.linspace(bins.min(), bins.max(), 256)
    # spl = make_interp_spline(bins[:-1], hist, k=1)  # type: BSpline
    sharpened = np.interp(img.flatten(), bins[:-1], cdf)
    # show_img(sharpened.reshape(img.shape), "sharpen")

    # show_img(power_smooth.reshape(img.shape), "i")
    # print(sharpened)

    # plt.figure()
    # plt.hist(bins[:-1], bins, weights=cdf)
    # plt.show()
        # print(img.shape)
    reshaped = sharpened.reshape(img.shape)
    # reshaped_float = float_sharpen.reshape(img.shape)
    # show_img(img, "not sharp")
    # show_img(reshaped_float.astype(np.uint8), "normal eqhist")
    # show_img(spline_float.astype(np.uint8), "spline eqhist")

    return reshaped

def sharpen(img, type="nct", convolve=False):
    # proxy name
    return eq_hist_sharpen(img, convolve,  type=type)