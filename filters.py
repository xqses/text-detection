import numpy as np
import skimage.filters as sk_filt
import scipy as scp
import cv2
from fourier import fourier
from basics import show_img, write_img
# from pythonRLSA import rlsa


class SplineFilter:
    def __init__(self):
        self.M = None
        self.u = None
        self.up = None
        self.upp = None

    def cubic_spline_filter(self):
        self.M = np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-2, -3, 6, 1], [0, 6, 0, 0]]) * (1/6)
        self.u = np.array([0.125, 0.250, 0.5, 1]).T
        self.up = np.array([0.75, 1, 1, 0]).T
        self.upp = np.array([3, 2, 0, 0]).T

    def trig_spline_filter(self):
        M = np.array([[1, 1, 0, 1], [1, np.sqrt(3 / 4), 0.5, 0.5], [1, 0.5, np.sqrt(3 / 4), -0.5], [1, 0, 1, -1]])
        self.M = np.linalg.inv(M)
        self.u = np.array([1, np.sqrt(1 / 2), np.sqrt(1 / 2), 0]).T
        self.up = np.array([0, -1 * np.sqrt(1 / 2), np.sqrt(1 / 2), -2]).T
        self.upp = np.array([0, -1 * np.sqrt(1 / 2), -1 * np.sqrt(1 / 2), 0]).T

    def b_spline(self):
        self.M = np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 0, 3, 0], [1, 4, 1, 0]]) * 1 / 6
        self.u = np.array([0.125, 0.250, 0.5, 1]).T
        self.up = np.array([0.75, 1, 1, 0]).T
        self.upp = np.array([3, 2, 0, 0]).T

    def bezier_filter(self):
        self.M = np.transpose(np.array([[1, 0, 0, 0], [-3, 3, 0, 0], [3, -6, 3, 0], [-1, 3, -3, 1]]))
        self.u = np.array([0.125, 0.250, 0.5, 1]).T
        self.up = np.array([0.75, 1, 1, 0]).T
        self.upp = np.array([3, 2, 0, 0]).T

    def catmull_rom_filter(self):
        self.M = np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-2, -3, 6, -1], [0, 6, 0, 0]]) * 1 / 6
        self.u = np.array([0.125, 0.25, 0.5, 1]).T
        self.up = np.array([0.75, 1, 1, 0]).T
        self.upp = np.array([3, 2, 0, 0]).T

    def set_filter(self, arg: str):
        switcher = {
            "cubic": self.cubic_spline_filter,
            "bezier": self.bezier_filter,
            "catrom": self.catmull_rom_filter,
            "bspline": self.b_spline,
            "trig": self.trig_spline_filter,
        }
        ## The switcher returns the selected filter function, but doesn't execute it
        fltr_func = switcher.get(arg, lambda: "Invalid filter")
        fltr_func()


def kernel(arg: str):
    print("entered kernel_func")
    fltr = SplineFilter()
    fltr.set_filter(arg)
    k, d, d2 = (fltr.u.T @ fltr.M), (fltr.up.T @ fltr.M), (fltr.upp.T @ fltr.M)
    k_conv_k = scp.signal.convolve(k, k, mode="same")
    d_conv_k = scp.signal.convolve(d, k, mode="same")
    d2_conv_k = scp.signal.convolve(d2, k, mode="same")

    return k_conv_k, d_conv_k, d2_conv_k


def inner_conv(kernel, im_slice):
    print("entered inner_conv")
    ## Graciously borrowed from
    # https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381

    # We implement the "direct" convolution method

    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # We can trust the kernel to be square, thus padding in x and y direction will be equivalent
    padding_hw = kernel.shape[0] - 1

    kernel_hw = kernel.shape[0]

    slice_x = im_slice.shape[0]
    slice_y = im_slice.shape[1]

    strides = 1

    output_x = int(((slice_x - kernel_hw + 2 * padding_hw) / strides) + 1)
    output_y = int(((slice_y - kernel_hw + 2 * padding_hw) / strides) + 1)
    output = np.zeros((output_x, output_y))
    print(output.shape)

    # Apply Equal Padding to All Sides
    if padding_hw != 0:
        padded_slice = np.zeros((im_slice.shape[0] + padding_hw * 2, im_slice.shape[1] + padding_hw * 2))
        padded_slice[int(padding_hw):int(-1 * padding_hw), int(padding_hw):int(-1 * padding_hw)] = im_slice
        print(padded_slice.shape)
    else:
        padded_slice = im_slice

    # Iterate through image
    for y in range(im_slice.shape[1]):
        # Exit Convolution
        if y > im_slice.shape[1] - kernel_hw:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(im_slice.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > im_slice.shape[0] - kernel_hw:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * im_slice[x: x + kernel_hw, y: y + kernel_hw]).sum()
                except:
                    break

    # Normalize cdf to interval [0, 255]
    output = (255 * output / np.max(output))

    return output


def convolution(img, fltr, use_threads):
    ## A more general convolution method for spline filters
    # Supposed to be a shell for the inner_conv method
    # Multithreading has some problems right now that are caused by input image dimensions not divisible by two
    # so we just use the scikit-image convolutions
    # Probably an easy fix

    ## Built-in functions such as skimage convolutions are based on the scipy N-D convolution
    # Thus, they require a great deal of memory allocation,
    # possibly for the purpose of multicore optimization
    # This is not ideal in the Jupyter Notebook-setting, unless you want to change the settings yourself
    ## Because of this, we define our own (simple) convolution method s.t. the filters are easily implemented

    ## By default, the "naivë" computational complexity of the convolution is O(NxM) for an NxM image
    # Thus, it is recommended that you rescale the image before performing the convolution

    ## If the "use threads"-parameter is called with True (optional), a few threads will be used
    # Users are advised to try this if their machines are arbitrarily "powerful enough"
    # Otherwise, simply pass "use_threads=False"

    ## We can assume k_conv_k.shape == d_conv_k.shape == d2_conv_k.shape

    if use_threads:

        from functools import partial
        from multiprocessing import Pool

        sliced_img = [img[0:img.shape[0] // 4, 0:img.shape[1] // 4],
                      img[img.shape[0] // 4:2 * img.shape[0] // 4, img.shape[1] // 4:2 * img.shape[1] // 4],
                      img[2 * img.shape[0] // 4:3 * img.shape[0] // 4, 2 * img.shape[1] // 4:3 * img.shape[1] // 4],
                      img[3 * img.shape[0] // 4:img.shape[0], 3 * img.shape[1] // 4:img.shape[1]]]

        n_slices = len(sliced_img)

        with Pool(processes=n_slices) as pool:

            func = partial(inner_conv, fltr)
            deriv = pool.map(func, sliced_img)
            pool.close()
            pool.join()
            row1 = np.concatenate((deriv[0], deriv[1]), axis=1)
            row2 = np.concatenate((deriv[2], deriv[3]), axis=1)
            new_output = np.concatenate((row1, row2), axis=0)
            return new_output

    else:
        output_img = sk_filt.edges.convolve(img, fltr)
        return output_img


def non_max_supp(filtered_img, wh):
    rows, channels = map(int, filtered_img.shape)

    mask_size = (wh, wh) # dilation mask size tuple
    mx = scp.ndimage.morphology.grey_dilation(input=filtered_img, size=mask_size, structure=np.ones(mask_size))
    # horizontal = rlsa.rlsa(filtered_img, True, False, 5)
    # vertical = rlsa.rlsa(horizontal, False, True, 5)
    show_img(filtered_img, "filtered image")
    show_img(mx, "Grey dilated filtered image")
    # show_img(horizontal)
    # show_img(vertical)

    # write_img("strct_tensor.JPG", filtered_img)
    # write_img("hessian_det.JPG", mx)
    #
    # from skimage.feature.peak import peak_local_max
    # from scipy.ndimage.measurements import center_of_mass
    # from scipy.ndimage import label, maximum_filter
    # from scipy.ndimage.morphology import generate_binary_structure, grey_dilation

    # mx = grey_dilation(input=filtered_img, size=(wh,wh), structure=np.ones((wh,wh)))
    # image_max = maximum_filter(mx, size=(wh,wh), mode='constant')

    # print(values)
    # filtered_img[0:wh, :] = 0
    # filtered_img[:, 0:wh] = 0

    # filtered_img_max = filtered_img * (filtered_img == mx)
    # print(filtered_img_max)
    # filtered_img_max = np.where(filtered_img & mx)
    # coords = np.array(zip(filtered_img_max))
    # vals = [None] * len(coords)
    # or i, coord in enumerate(coords):
    #     vals[i] = coords[coord]

    # vals = -1 * vals
    # print(len(coords), len(vals))
    # sort_idx = np.argsort(vals)
    # sorted_vals = np.sort(vals)

    # print(sorted_vals)




def key_pts(img, fltr, args, use_threads):
    ## A Python rewrite of the keyPoints.m file
    print("entered key_pts")

    if (args != 'get_tensor') and (args != 'get_hessian'):
        print('invalid args')
        return None
    # print("retrieved kernel")

    # k_conv_d = scp.signal.convolve(k, d)
    # d_conv_k = scp.signal.convolve(d, k)
    # d2_conv_k = scp.signal.convolve(d2, k)
    # k_conv_k = scp.signal.convolve(k, k)
    # print("convolved k, d, d2")
    g2D = cv2.getGaussianKernel(ksize=9, sigma=9 / 6)

    ### Avoid too many expensive convolution computations by calling with args
    # print(k_conv_k.shape, d_conv_k.shape)
    # print(np.outer(k_conv_k, d_conv_k).shape)
    # print(img.shape)
    k_conv_k, d_conv_k, d2_conv_k = kernel(fltr)
    if args == 'get_tensor':
        # print("entered get tensor")
        # print("###" * 30)
        # print("Yttre produkten av k konvolverat med k, och k .T")
        # print(np.outer(k_conv_k, d_conv_k.T))
        Ix = convolution(img=img, fltr=np.outer(k_conv_k, d_conv_k.T), use_threads=use_threads)

        # Ix = sk_filt.edges.convolve(img, np.outer(k_conv_k, d_conv_k.T))
        Iy = convolution(img=img, fltr=np.outer(d_conv_k, k_conv_k.T), use_threads=use_threads)
        # print(Ix.shape, Iy.shape)

        # Notes: np.multiply (* is the shorthand) performs element-wise multiplication
        # Not to be confused with np.matmul, or its shorthand @
        Ix2 = Ix * Ix
        Ix2 = convolution(img=Ix2, fltr=g2D, use_threads=use_threads)
        Iy2 = Iy * Iy
        Iy2 = convolution(img=Iy2, fltr=g2D, use_threads=use_threads)

        IxIy = convolution(img=Ix * Iy, fltr=g2D, use_threads=use_threads)


        strct_tensor = np.divide(((Iy2 * Ix2) - (IxIy ** 2)), (Ix2 + Iy2 + np.finfo(np.float).eps))
        show_img(strct_tensor, fltr + " Structural tensor")
        mx = scp.ndimage.morphology.grey_dilation(input=strct_tensor, size=(5, 5), structure=np.ones((5, 5)))
        show_img(mx, fltr + " Struct tensor grey dilation")
        # return non_max_supp(strct_tensor, 5)

    if args == 'get_hessian':
        Ixx = convolution(img=img, fltr=np.outer(k_conv_k, d2_conv_k), use_threads=use_threads)
        Ixx = convolution(img=Ixx, fltr=g2D, use_threads=use_threads)

        Iyy = convolution(img=img, fltr=np.outer(d2_conv_k, k_conv_k), use_threads=use_threads)
        Iyy = convolution(img=Iyy, fltr=g2D, use_threads=use_threads)

        Ixy = convolution(img=img, fltr=np.outer(d_conv_k, d_conv_k), use_threads=use_threads)
        Ixy = convolution(img=Ixy, fltr=g2D, use_threads=use_threads)
        show_img((Ixx * Iyy) - (Ixy ** 2), fltr + " Hessian filtered image")
        mx = scp.ndimage.morphology.grey_dilation(input=(Ixx * Iyy) - (Ixy ** 2), size=(5, 5), structure=np.ones((5,5)))
        show_img(mx, fltr + " Hessian filtered grey dilation")
        # return non_max_supp((Ixx * Iyy) - (Ixy ** 2), 5)


def butterworth(img, cutoff, order):
    # Not sure why this does not work properly. Possibly a problem with the TF

    ## Fourier transform and shift the image
    rows, cols, ft_shift = fourier(img)

    # Define the 2D transfer function

    sample_rate = 255
    normal_cutoff = float(cutoff) / (sample_rate / 2)
    b, a = scp.signal.butter(order, normal_cutoff, btype='lowpass')
    y = scp.signal.filtfilt(b, a, ft_shift)

    ## Multiply the TF and transformed image
    # G = np.matmul(np.transpose(H), ft_shift)
    # Invert the convolution to get a matrix of complex numbers
    inv_y = scp.fft.ifft2(y)
    # convert the complex numbers back to real distances
    real_y = np.sqrt((np.real(inv_y) ** 2) + (np.imag(inv_y) ** 2))
    # normalize G
    return real_y
