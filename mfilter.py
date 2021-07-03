import numpy as np
import sys


def process(image, size, window=1, threshold=0., spam=False):

    ## set filter window and image dimensions
    filter_window = 2*window + 1
    xlength, ylength = size
    vlength = filter_window*filter_window
    if spam:
        print('Image length in X direction: {}'.format(xlength))
        print('Image length in Y direction: {}'.format(ylength))
        print('Filter window size: {0} x {0}'.format(filter_window))

    ## create 2-D image array and initialize window
    image_array = np.reshape(np.array(image, dtype=np.uint8), (ylength, xlength))
    filter_window = np.array(np.zeros((filter_window, filter_window)))
    target_vector = np.array(np.zeros(vlength))
    pixel_count = 0

    try:
        ## loop over image with specified window filter_window
        for y in range(window, ylength-(window+1)):
            for x in range(window, xlength-(window+1)):
            ## populate window, sort, find median
                filter_window = image_array[y-window:y+window+1, x-window:x+window+1]
                target_vector = np.reshape(filter_window, ((vlength),))
                ## numpy sort
                median = median_demo(target_vector, vlength)
                ## C median library
                # median = medians_1D.quick_select(target_vector, vlength)
            ## check for threshold
                if not threshold > 0:
                    image_array[y, x] = median
                    pixel_count += 1
                else:
                    scale = np.zeros(vlength)
                    for n in range(vlength):
                        scale[n] = abs(int(target_vector[n]) - int(median))
                    scale = np.sort(scale)
                    Sk = 1.4826 * (scale[vlength//2])
                    if abs(int(image_array[y, x]) - int(median)) > (threshold * Sk):
                        image_array[y, x] = median
                        pixel_count += 1

    except TypeError as err:
        print('Error in processing function:'.format(err))
        sys.exit(2)
        ## ,NameError,ArithmeticError,LookupError

    print('{} pixel(s) filtered out of {}'.format(pixel_count, xlength*ylength))
    ## convert array back to sequence and return
    return np.reshape(image_array, (xlength*ylength)).tolist()

def median_demo(target_array, array_length):
    sorted_array = np.sort(target_array)
    median = sorted_array[array_length//2]
    return median