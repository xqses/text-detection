from cv2 import ximgproc as xim
from shading_correction import shaCorr

def remove_noise(img):
    return xim.edgePreservingFilter(img, d=15, threshold=20)

def correct_shading(img):
    return shaCorr(img)