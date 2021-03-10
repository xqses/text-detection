import scipy.fft as scp_fourier


def fourier(img):
    # Computes the 2D fourier transform of the image and shifts the result
    ft_img = scp_fourier.fft2(img)
    ft_shift = scp_fourier.fftshift(ft_img)
    rows, cols = ft_shift.shape
    return rows, cols, ft_img
