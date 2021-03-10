import cv2
import numpy as np
import scipy as scp

from basics import open_file, preprocessing, pyramid, resize, show_img, write_img
from segmentation import segmentation, find_contours
from sharpening import sharpen
from noise_and_shades import remove_noise, correct_shading
from line_cleaning import line_cleaning


## Image name format should be "IMG_+{# of img}+.JPG"
# For an object with reference to one image, use explicit file name and set multiple=False
# For an object with reference to an array of images, use name="IMG_", set multiple=True
# See function multi() where n_img = desired # of images


#########################################################
##                                                     ##
##                                                     ##
##             Image processing pipeline               ##
##          for the master thesis project of           ##
##      Automatic Handwritten Text Cell Detection      ##
##              in Historical Documents                ##
##                                                     ##
##              Author: Olle Dahlstedt                 ##
##            Last VCS Update: 2021-03-10              ##
##                                                     ##
##                                                     ##
#########################################################


def exec(img):

    scaled_img = pyramid(img, direction="down", iterations=2)
    # show_img(scaled_img, "Downsampled image")

    hsv_img = preprocessing(scaled_img).astype(np.uint8)
    # show_img(hsv_img, "HSV-converted image")

    segmented_img = segmentation(scaled_img, hsv_img).astype(np.uint8)
    # show_img(segmented_img, "Segmented image")

    # noise_removed = remove_noise(segmented_img)
    # show_img(noise_removed, "Noise-removed image")

    shading_corrected = correct_shading(segmented_img).astype(np.uint8)
    # show_img(shading_corrected, "Shading-corrected image")

    gs = cv2.cvtColor(shading_corrected, cv2.COLOR_BGR2GRAY)
    # show_img(gs, "Grayscale image"

    sharpened, bins = sharpen(gs)
    sharpened = sharpened.astype(np.uint8)
    # show_img(sharpened, "Sharpened image")

    lines_cleared = line_cleaning(sharpened, gs)

    for i in range(4):
        shading_corrected = correct_shading(cv2.cvtColor(lines_cleared, cv2.COLOR_GRAY2BGR)).astype(np.uint8)
        gs = cv2.cvtColor(shading_corrected, cv2.COLOR_BGR2GRAY)
        sharpened, bins = sharpen(gs.astype(np.uint8))
        sharpened = sharpened.astype(np.uint8)
        lines_cleared = line_cleaning(sharpened, lines_cleared)

    show_img(lines_cleared, "lines image")
    write_img("cleared_img.jpg", lines_cleared)


def single(os: str):
    os = "win"
    ## EXAMPLE USAGE OF 1 IMAGE
    if os == "linux":
        f_name = 'test_img/IMG_5.JPG'
    if os == "win":
        f_name = 'test_img\IMG_5.JPG'

    f_obj = open_file(name=f_name, multiple=False, n_img=1)

    img = f_obj.img
    return exec(img)

def multi(os, n_img):
    if os == "linux":
        f_name = 'test_img/IMG_'
    if os == "win":
        f_name = 'test_img\IMG_'
    f_obj = open_file(name=f_name, multiple=True, n_img=n_img)
    # We can retrieve less images than we load to the array
    print("####" * 10)
    print("Retrieving image generator. This may take a while, depending on image size and number of images")
    print("####" * 10)
    img_generator = f_obj.get_img_generator(n_img=n_img)

    for i in range(n_img):
        # The image generator is a yield function.
        # Calling it with the next keyword will iterate the function and return one image from the generator loop
        # Saves overhead memory costs (in comparison to keeping a long array of large images in memory)
        img = next(img_generator)
        final = exec(img)
        show_img(final, "Final image")
        write_img("final_image_"+str(i)+".jpg", final)


def main(os: str, multiple_images = False, n_img = 1):
    # For each step of this pipeline, please see the corresponding functions of each script
    # The main script has been cleaned extensively to avoid a huge mess of code
    if multiple_images:
        final_img = multi(os, n_img)
    else:
        final_img = single(os)

main(os="win")
# main(os="win", multiple_images=True, n_img = 5)

