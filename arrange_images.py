## This is a script written to open, rescale, correct the shading,
# as well as to write images to new titles,
# these titles should be more easily used by the text detection pipeline
from basics import pyramid, write_img, show_img
from noise_and_shades import correct_shading
import numpy as np
import os
import cv2
from bgrem import ent_thresh

def yield_image(name: str, root: str):
    yield cv2.imdecode(np.fromfile(str(os.path.join(root, name)), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

def process_images():
    e = []
    n_file = 0
    for root, dirs, files in os.walk('image_material', followlinks=True):
        for name in files:
            print("Opening image: ", n_file)
            print(name, root)
            img = next(yield_image(name, root))
            scaled = pyramid(img, direction="down", iterations=2)
            shading_corrected = correct_shading(scaled, edge_preserving=False)
            # Calculate the median image entropy threshold to empirically determine the threshold values of mb, mw
            # e.append(ent_thresh(shading_corrected))
            status = cv2.imwrite("ss\IMG_"+str(n_file+1)+".JPG", shading_corrected)
            # print("Write status", status)
            n_file += 1
            if n_file == 1000:
                print(np.mean(e))
                print(np.median(e))
            if n_file == 2000:
                print(np.mean(e))
                return np.median(e)


print(process_images())