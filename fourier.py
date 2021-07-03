import scipy.fft as scp_fourier
import numpy as np
from matplotlib import pyplot as plt

def fourier(img):
    try:
        assert img.dtype == np.float32
    except AssertionError:
        img = img.astype(np.float32)

    print(img.shape)
    # Computes the 2D fourier transform of the image and shifts the result
    H, W = img.shape
    centerW = np.int(np.fix(0.5 * W))
    centerH = np.int(np.fix(0.5 * H))
    ft_img = scp_fourier.fft2(img) / (W*H)
    # ft_shift = scp_fourier.fftshift(ft_img)
    abs_shift = np.abs(ft_img)
    indices = np.where(abs_shift == np.max(abs_shift))
    maxY = (indices[0][0] - centerH) / H+np.finfo(np.float).eps
    maxX = (indices[1][0] - centerW) / W+np.finfo(np.float).eps
    alpha = np.arctan(maxY / maxX) * 180 / np.pi
    print(alpha)
    r = 400
    plt.figure()
    # plt.imshow(20*np.log10(np.abs(ft_shift)))
    plt.imshow(np.log(1+abs_shift[centerH-r:centerH+r,centerW-r:centerW+r]), extent=[-r,r,-r,r])
    plt.show()
    rows, cols = ft_img.shape
    return rows, cols, ft_img
