
from matplotlib import pyplot as plt
import cv2
## Help classes ##

class ImageHelper:
    def __init__(self):
        self.id = 0

    def set_img_array(self, name, n_img, img_format):
        self.img_array = [None] * n_img
        for i in range(n_img):
            self.img_array[i] = cv2.imread(name + str(i + 1) + img_format)

    def get_img_generator(self, name, img_format, n_img):
        for i in range(n_img):
            yield cv2.imread(name + str(i + 1) + img_format)

    def get_img_array(self, n_img):
        return self.img_array

    def get_img(self, name):
        return cv2.imread(name)

def write_img(name, img):
    status = cv2.imwrite(name, img)
    return status

def open_file(name: str, multiple: bool, n_img: int):
    if multiple:
        imgs_obj = ImageHelper()
        imgs_obj.set_img_array(name=name, n_img=n_img, img_format=".JPG")
        return imgs_obj
    else:
        img_obj = ImageHelper()
        img_obj.img = img_obj.get_img(name=name)
        return img_obj

def show_img(img, title):
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.title(title)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def preprocessing(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # ret, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    # img_rlsa_horizontal = rlsa.rlsa(thresh, True, False, 5)
    # img_rlsa_vertical = rlsa.rlsa(img_rlsa_horizontal, False, True, 5)
    return hsv_img


def pyramid(img, direction: str, iterations: int):
    ## The pyramid iterates by factors of two in either direction
    # Thus it retains its width to height proportion
    has_color = True
    if len(img.shape) < 3:
        has_color = False
    if direction == "down":
        for i in range(iterations):
            if has_color:
                rows, cols, _channels = map(int, img.shape)
                ## Alternative syntax:
                # rows, cols, _channels = img.shape[0], img.shape[1], img.shape[2]
            else:
                ## If it is a grayscale image, there's no such thing as _channels
                rows, cols = map(int, img.shape)
            img = cv2.pyrDown(img, dstsize=(cols // 2, rows // 2))
    else:
        for i in range(iterations):
            if has_color:
                rows, cols, _channels = map(int, img.shape)
            else:
                rows, cols = map(int, img.shape)
            img = cv2.pyrUp(img, dstsize=(cols * 2, rows * 2))
    return img

def resize(img, dim):
    return cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

class ContourHelper:
    def __init__(self, arr):
        self.array = arr

    def array_gen(self):
        for i in range(len(self.array)):
            yield self.array[i]
