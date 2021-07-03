from skimage.filters import gabor
from skimage import data, io
from matplotlib import pyplot as plt

image = data.coins()
# detecting edges in a coin image
filt_real, filt_imag = gabor(image, frequency=0.6)
plt.figure()
io.imshow(filt_real)
io.show()

filt_real, filt_imag = gabor(image, frequency=0.1)
plt.figure()
io.imshow(filt_real)
io.show()