if edge_preserving:
    ## Added by OD 2021-03-05

    # Division of subregions
    ## Resizing the image to roughly kern shape x 2
    ## Let subregion around each pixel be ceil(kern shape / 10)
    # I.e kern shape = 101, 143 -> subregion = ceil((202, 146) / 10) = (21, 15)
    # We would create our subwindow around each pixel thus from pixel at coord (21, 15)
    # Importantly, this means we will not blur the border of the image, which is fine in our material
    resized = resize(img, (krn.shape[1], krn.shape[0]))
    print(resized.shape)
    d = 30
    window_y, window_x = int(np.ceil(resized.shape[0] // d)), int(np.ceil(resized.shape[1] // d))
    # We cannot guarantee that either y or x region will be odd, so add if clause
    window_y += 1 if window_y % 2 == 0 else 0
    window_x += 1 if window_x % 2 == 0 else 0
    kern = np.ones((window_y * 2, window_x * 2, 3))

    ### Determine the number of subwindows ###
    # Since we will not blur the borders,
    # we will have one subregion around each pixel except the border pixels

    _, sigma, _ = mask(np.max([window_y, window_x]), 0)
    div = cv2.GaussianBlur(kern, (window_y, window_x), sigma)

    inits = {}
    # Using a nested dictionary for faster lookup
    # This makes things slightly complicated-looking (and coming up with)
    # This also means most operations are done within the loop
    # But it saves runtime, so that's nice

    # As we need a window around each pixel, this cannot be run outside two loops
    for y in range(window_y, resized.shape[0] - window_y):
        inits[y] = {}
        # print("y =", y, "window y =", window_y, "y - window_y = ", (y - window_y))
        for x in range(window_x, resized.shape[1] - window_x):
            inits[y][x] = {}
            # print("x =", x, "window x =", window_x, "x - window_x = ", (x - window_x))
            subwindow = resized[y - window_y:y + window_y, x - window_x:x + window_x]

            ### Blur each window ###
            # Note: in order to avoid too strong blurring (erasing all color information),
            # Blur with kernel size half of each window

            p = cv2.GaussianBlur(subwindow, (window_y, window_x), sigma) / div
            # Color mean within each (now blurred) window
            color_means = np.array([np.mean(p[:, :, 0]), np.mean(p[:, :, 1]), np.mean(p[:, :, 2])])
            inits[y][x]["cmeans"] = color_means

            # pixelwise distances (5)
            pd = np.array([np.linalg.norm(np.array([resized[y, x][0], color_means[0]]), ord=2),
                           np.linalg.norm(np.array([resized[y, x][1], color_means[1]]), ord=2),
                           np.linalg.norm(np.array([resized[y, x][2], color_means[2]]), ord=2)])

            inits[y][x]["pixelwise_distance"] = pd

            # Mean pixelwise distance (6)
            mpd = np.mean(pd)
            inits[y][x]["mean_pixelwise_distamce"] = mpd

    t = 20
    im = np.zeros(resized.shape)
    wts = np.ones(resized.shape)
    for y in range(window_y, resized.shape[0] - window_y):
        for x in range(window_x, resized.shape[1] - window_x):
            im[y, x] = im[y, x] + (
                        resized[y, x] * ((t - inits[y][x]["pixelwise_distance"]) ** 2) * inits[y][x]["cmeans"])
            wts[y, x] = wts[y, x] + (resized[y, x] * (t - inits[y][x]["pixelwise_distance"]) ** 2)

    p = im / wts
    p = p[:, :, 2]
