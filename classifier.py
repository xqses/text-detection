import random
import re
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from PST import PST
from basics import open_file
from bgrem import bgremoval
from line_cleaning import line_cleaning
from morphology import get_morph
from sharpening import sharpen

def get_data():
    kp_data = pd.read_csv("kp_data.csv")
    return kp_data

def normalization(img, k):
    try:
        assert len(img.shape) < 3
    except:
        print("Normalization cannot be done in color")
        print("Converting color image to grayscale")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean, STD = cv2.meanStdDev(img.astype(np.uint8))
    # Clip frame to lower and upper STD
    clipped = np.clip(img, mean - k * STD, mean + k * STD).astype(np.uint8)
    # Normalize to range
    img = cv2.normalize(clipped, clipped, 0, 255, norm_type=cv2.NORM_MINMAX)
    # show_img(img, "normalized")

    return img

def load_haar_features(img, feature_type, feature_coord=None):
    from skimage.feature import haar_like_feature
    from skimage.transform import integral_image
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    print("Passed integral image")
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)

def train_haar(labels):
    from dask import delayed
    from sklearn.model_selection import train_test_split
    from skimage.util import img_as_float

    f_name = 'labeled_subimages\sub_img_'
    n_img = len(labels)

    feature_types = ['type-2-x', 'type-2-y',
                     'type-3-x', 'type-3-y',
                     'type-4']

    features = []
    print("####" * 10)
    print("Retrieving image generator. This may take a while, depending on image size and number of images")
    print("####" * 10)
    img_generator = open_file(name=f_name, multiple=True, n_img=n_img, sub_img=True)

    for i, img in enumerate(img_generator):
        print(i)
        features.append(load_haar_features(img, feature_types))
    X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=150,
                                                        random_state=0,
                                                        stratify=labels)
    print(X_train)

def load_descriptors(labels):
    f_name = 'labeled_subimages\sub_img_'

    n_img = len(labels)

    sift = cv2.SIFT_create(nfeatures = 0,nOctaveLayers = 3,contrastThreshold = 0.04,edgeThreshold = 15,sigma = 0.8 )
    boost = cv2.HOGDescriptor()

    # init BoW cluster
    # BOW = cv2.BOWKMeansTrainer(4)
    # We can retrieve less images than we load to the array
    print("####" * 10)
    print("Retrieving image generator. This may take a while, depending on image size and number of images")
    print("####" * 10)
    img_generator = open_file(name=f_name, multiple=True, n_img=n_img, sub_img=True)
    descriptors = []
    true_undefined = 0
    false_undefined = 0


    for i in range(n_img):
        img = next(img_generator)
        gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # kp_boost = boost.compute(gs)
        # print(gs.shape)
        # _, des = boost.compute(gs, kp_boost)
        # print(des)

        ### Some selected preprocessing
        s1 = sharpen(gs, type="nct").astype(np.uint8)
       # norm = normalization(s1, k=0.8)
        # compute the descriptors with BRIEF
        kp_sift = sift.detect(s1, None)
        _, des = sift.compute(s1, kp_sift)
        # print(des.shape)
        if des is not None and len(des) != 0:
            descriptors.append(des)
        else:
            if labels[i] == 1.0:
                true_undefined += 1
            else:
                false_undefined += 1
            labels = labels.drop([i])

    print(true_undefined / (true_undefined + false_undefined))

    # Shuffle descriptors

    ## Since we have now dropped many labels we will need to concatenate the new list of indices
    ## (we cannot use enumerate because some indices do not exist in the list anymore)
    ## hence, zip the new indices with the descriptors and shuffle the zipped list
    x = list(zip(list(labels.index.values), descriptors))
    split_index = len(descriptors) - (len(descriptors) // 5)
    random.shuffle(x)
    indices, shuffled_descriptors = zip(*x)

    new_labels = labels.reindex(indices)

    train_descriptors = shuffled_descriptors[:split_index]
    test_descriptors = shuffled_descriptors[split_index:]
    train_labels = new_labels[:split_index]
    test_labels = new_labels[split_index:]

    print("-" * 20)
    print("In train:")
    print("Number of label 1:", list(train_labels).count(1))
    print("Number of label 2:", list(train_labels).count(2))
    print("Number of label 3:", list(train_labels).count(3))
    print("Number of label 4:", list(train_labels).count(4))

    print("-" * 20)
    print("In test:")
    print("Number of label 1:", list(test_labels).count(1))
    print("Number of label 2:", list(test_labels).count(2))
    print("Number of label 3:", list(test_labels).count(3))
    print("Number of label 4:", list(test_labels).count(4))
    print("-" * 20)

    return train_descriptors, test_descriptors, train_labels, test_labels


# descriptors = load_descriptors()


def generate_labels(img):
    try:
        kpd = pd.read_csv("kp_data.csv")

    except:
        print("Dataframe does not exist, creating new")
        kpd = pd.DataFrame(columns=["label"])

    iteration = len(kpd) + 1

    copy3c = deepcopy(img)
    fresh3c = deepcopy(copy3c)
    gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ### Some selected preprocessing
    s1 = sharpen(gs, type="nct").astype(np.uint8)
    s_copy = deepcopy(s1)
    mask1 = np.zeros_like(gs).astype(np.uint8)
    mask1[PST(s1)[0]] = 255
    rem_bg = (bgremoval(s1, f=3, g=3, th=2.0, ct=1) * 255).astype(np.uint8)
    _, thresh = cv2.threshold(rem_bg, 0, 255, cv2.THRESH_OTSU)
    b_img = line_cleaning(mask1, thresh,
                          use_Canny=False, is_binary=True).astype(np.uint8)
    grad = get_morph(b_img, morph="gradient", sz=(3, 3), threshold=True)
    # show_img(grad)

    ### Go for kps
    # sift = cv2.SIFT_create()
    out = cv2.connectedComponentsWithStats(grad, connectivity=8)
    # full_kps = sift.detect(s1, None)
    # cv2.drawKeypoints(fresh3c, full_kps, fresh3c,
    #                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS).astype(np.uint8)
    # show_img(fresh3c)
    n_labels, labels, stats, centroids = out
    # print("n labels:", n_labels)
    # show_img(thresh_img, "grad")

    # Require both marked and unmarked images
    # Marked images are for labeling purposes only
    # Unmarked images are used to save subimages for later keypoint extraction
    marked = []
    unmarked = []

    for i in range(1, n_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        # print(area)
        # (cX, cY) = centroids[i]
        if 50 < area < 40000:
            copy_clean = deepcopy(fresh3c)
            x_start, y_start = (x // 2), (y // 2)
            unmarked.append(fresh3c[y:y + h, x:x + w])
            cv2.rectangle(copy_clean, (x, y), (x + w, y + h), (255, 0, 255), 1)
            marked.append(copy_clean[y_start:y_start + y + h + 200, x_start:x_start + x + w + 400])

    print("-----------------")
    print("CREATING SUB-IMAGE GENERATOR")

    n_subs = len(marked)

    marked_iter = iter(marked)
    unmarked_iter = iter(unmarked)

    del marked
    del unmarked
    i = 1

    ## Make sure not to run imshow more than once
    ## Around ~100 images in matplotlib will be very very slow
    ## Create one instance of an artist instead and refresh the data using set_data()

    ax = plt.axes()
    artist = ax.imshow(next(marked_iter))

    for img in marked_iter:
        print("-----------------")
        print("LABEL SUBIMAGE", i, "OF", n_subs)
        i += 1

        artist.set_data(img)
        plt.pause(0.05)

        ## Categories:
        # 1: undefined
        # 2: machine-typed text
        # 3: handwritten text
        # 4: handwritten digits
        s = input("Define category: ")
        df_entry = {"label": s}
        kpd = kpd.append(df_entry, ignore_index=True)
        kpd.to_csv('kp_data.csv', index=False)

    for img in unmarked_iter:
        _ = cv2.imwrite("labeled_subimages\sub_img_" + str(iteration) + ".png", img)
        iteration += 1


def clean_data():
    data = get_data()
    print(len(data))
    for i, row in data.iterrows():
        try:
            if int(row["label"]) == 1:
                continue
            if int(row["label"]) == 2:
                continue
            if int(row["label"]) == 3:
                continue
            if int(row["label"]) == 4:
                continue
            if len(row["label"]) > 1:
                row["label"] = row["label"][0]
            print(row["label"])
        except:
            try:
                if re.search("[0-9]+", row["label"]) != None:
                    print("Regex matched")
                    row["label"] = re.search("[0-9]+", row["label"]).group()
            except:
                print("Regex didnt match, no number exists")
                data = data.drop(i)

    print(len(data))
    data.to_csv('kp_data.csv', index=False)
    return data


from scipy.cluster.vq import kmeans, vq


def bovw(train_labels, test_labels, train_descriptors, test_descriptors):
    from sklearn.metrics import plot_confusion_matrix, accuracy_score
    # print(train_descriptors.shape, test_descriptors.shape)
    # print("in bovw")
    try:
        assert len(train_descriptors) + len(test_descriptors) == len(train_labels) + len(test_labels)
    except AssertionError:
        print("Labels and descriptors length mismatch. Clean data before continuing!")

    train_descriptor_stack = train_descriptors[0]
    test_descriptor_stack = test_descriptors[0]
    # print(train_descriptor_stack.shape)
    # print(test_descriptor_stack.shape)

    for i, descriptor in enumerate(train_descriptors[1:]):
        if descriptor is not None:
            # print(descriptor)
            train_descriptor_stack = np.vstack((train_descriptor_stack, descriptor))
        else:
            train_labels = train_labels.drop(i)

    for i, descriptor in enumerate(test_descriptors[1:]):
        if descriptor is not None:
            test_descriptor_stack = np.vstack((test_descriptor_stack, descriptor))
        else:
            test_labels = test_labels.drop(i)

    # except:
    # print("Ran into error with some descriptor, continue from last legal operation")
    # labels.drop()

    # data = data.drop_duplicates()
    # train = data.iloc[:600]
    # test = data.iloc[600:]

    print(len(train_descriptors), len(train_labels))
    print(len(test_descriptors), len(test_labels))

    k = 20

    # print(train_descriptors.shape)
    voc, variance = kmeans(train_descriptor_stack, k, 1)
    # print(voc)
    # print(variance)
    im_features = np.zeros((len(train_labels), k), "float32")
    for i in range(len(train_labels)):
        words, distance = vq(train_descriptors[i], voc)
        for w in words:
            im_features[i][w] += 1

    from sklearn.preprocessing import StandardScaler
    stdslr = StandardScaler().fit(im_features)
    im_features = stdslr.transform(im_features)

    from sklearn.svm import LinearSVC
    clf = LinearSVC(max_iter=150000)
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    clf.fit(im_features, np.array(train_labels))
    gnb.fit(im_features, np.array(train_labels))

    test_features = np.zeros((len(test_labels), k), "float32")
    for i in range(len(test_labels)):
        words, distance = vq(test_descriptors[i], voc)
        for w in words:
            test_features[i][w] += 1

    test_features = stdslr.transform(test_features)

    # print(list(test_labels))
    # print(clf.predict(test_features))
    svm_accuracy = accuracy_score(list(test_labels), clf.predict(test_features))
    gnb_accuracy = accuracy_score(list(test_labels), gnb.predict(test_features))
    print("Support Vector Machine accuracy:", svm_accuracy)
    print("Gaussian Naive Bayes accuracy:", gnb_accuracy)
    plot_confusion_matrix(clf, test_features, list(test_labels))
    plt.show()
    # print(confusion_matrix(list(test_labels), clf.predict(test_features)))
    # print(confusion_matrix(list(test_labels), gnb.predict(test_features)))


# data = clean_data()
# bovw(data, descriptors)


def train_classifier(img):
    ## Update this key manually for now
    ## Can probably solve checking length of file list or length of labels in dataframe if we want to but cba

    # iteration = generate_labels(img)
    data = clean_data()
    labels = data["label"]
    # train_haar(labels)
    # print(labels.iloc[2])
    train_descriptors, test_descriptors, train_labels, test_labels = load_descriptors(labels)
    bovw(train_labels=train_labels, test_labels=test_labels, train_descriptors=train_descriptors, test_descriptors=test_descriptors)

    return iteration
