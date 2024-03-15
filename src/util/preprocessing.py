import cv2
import numpy as np


class Preprocess:
    """
    Class for removing hair (artifacts) from an image and morphologic close transformation
    """

    @staticmethod
    def remove_artifacts_sevenpoint(img, target_size=(512, 384)):
        img_pre = remove_hair_v11(img)
        img_pre = remove_ink_patches(img_pre)

        h_raw, w_raw, _ = img.shape
        h_pre, w_pre, _ = img_pre.shape

        if (h_raw == h_pre) & (w_raw == w_pre):
            return img_pre
        else:
            return cv2.resize(img_pre, target_size)


    @staticmethod
    def remove_artifact(img):
        # perform closing to remove hair
        kernel = np.ones((15, 15), np.uint8)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
        return closing

    @staticmethod
    def equalize_histy_channel(img):
        """
        ASLM Noise Removal. It equalizes the Y channel of an YUV image, Y contains the intensity information
        https://www.opencv-srf.com/2018/02/histogram-equalization.html
        http://users.diag.uniroma1.it/bloisi/papers/bloisi-CMIG-2016-draft.pdf
        :param img:
        :return:
        """
        # convert image to YUV
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        # convert image to RGB
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img

    @staticmethod
    def remove_artifact_rgb(img):
        """
        DR noise removal. It applies morphologic close transformation on each channel of RGB image
        kernel of (11,11) based on hair size
        :param img:
        :return:
        """
        # median filter
        img_median = cv2.medianBlur(img, 5, 5)
        # split RGB channels
        img_b, img_g, img_r = cv2.split(img_median)
        # kernel of 11 * 11
        kernel = np.ones((11, 11), np.uint8)
        # perform morphologic closing on each RGB channel
        imgClosingB = cv2.morphologyEx(img_b, cv2.MORPH_CLOSE, kernel)
        imgClosingG = cv2.morphologyEx(img_g, cv2.MORPH_CLOSE, kernel)
        imgClosingR = cv2.morphologyEx(img_r, cv2.MORPH_CLOSE, kernel)
        # merge the 3 channels
        img_result = cv2.merge((imgClosingB, imgClosingG, imgClosingR))
        return img_result

    @staticmethod
    def remove_artifact_yuv(img):
        """
        DR noise removal
        (1) convert BGR to YUV;
        (2) process;
        (3) convert YUV to RGB for OTSU apply morphologic close transformation on each channel of YUV image
        kernel of (11,11) based on hair size
        :param img:
        :return:
        """
        # median filter
        img_median = cv2.medianBlur(img, 5, 5)
        # split YUV channels
        img_yuv = cv2.cvtColor(img_median, cv2.COLOR_RGB2YUV)
        img_v, img_u, img_y = cv2.split(img_yuv)
        # kernel of 11 * 11
        kernel = np.ones((11, 11), np.uint8)
        # perform morphologic closing on each RGB channel
        img_closing_v = cv2.morphologyEx(img_v, cv2.MORPH_CLOSE, kernel)
        img_closing_u = cv2.morphologyEx(img_u, cv2.MORPH_CLOSE, kernel)
        img_closing_y = cv2.morphologyEx(img_y, cv2.MORPH_CLOSE, kernel)
        # merge the 3 channels
        img_result = cv2.merge((img_closing_v, img_closing_u, img_closing_y))
        # back to RGB
        img_result = cv2.cvtColor(img_result, cv2.COLOR_YUV2RGB)
        return img_result



    @staticmethod
    def SLIC(img):
        """
        Use the SLIC clustering to extract borders
        https://jayrambhia.com/blog/superpixels-slic
        returns the result models
        :param img:
        :return:
        """
        slic = cv2.ximgproc.createSuperpixelSLIC(img, algorithm=cv2.ximgproc.MSLIC, region_size=300, ruler=0.075)
        color_img = np.zeros(img.shape, np.uint8)
        color_img[:] = (0, 0, 0)
        for n in range(2):
            slic.iterate(2)
        slic.enforceLabelConnectivity()
        mask = slic.getLabelContourMask(False)
        # stitch foreground & background together
        mask_inv = cv2.bitwise_not(mask)
        result_bg = cv2.bitwise_and(img, img, mask=mask_inv)
        result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
        result = cv2.add(result_bg, result_fg)
        # cv2.imshow('SLIC',mask_inv)
        return result

    @staticmethod
    def perform_kmeans(img, k_clusters=5):
        """
        It uses KMEANS clustering
        :param img:
        :param k_clusters:
        :return:
        """
        # apply KMEANS
        Z = img.reshape((-1, 3))
        # convert to np.float32
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(Z, k_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result = res.reshape((img.shape))
        return result, center

    @staticmethod
    def remove_tint(img):
        """
        Return a mask that removes tint effect from corners of the models
        https://stackoverflow.com/questions/42594993/gradient-mask-blending-in-opencv-python
        :param img:
        :return:
        """
        H, W = img.shape[:2]
        mask = np.zeros((H, W), np.uint8)
        cv2.circle(mask, (W // 2, H // 2), W // 2 + W // 50, (150, 150, 150), -1, cv2.LINE_AA)
        mask = cv2.blur(mask, (321, 321))
        mask = 160 - mask
        return mask

    @staticmethod
    def has_tint(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        H, W = img_gray.shape[:2]
        mask = np.zeros((H, W), np.uint8)
        cv2.circle(mask, (W // 2, H // 2), W // 2 + W // 50, (255, 255, 255), -1, cv2.LINE_AA)
        mask = cv2.blur(mask, (21, 21))
        # mask = 255 - mask
        mask = cv2.subtract(img_gray, mask)
        seuil = 140
        return mask[10, 10] < seuil


def remove_lens(img):

    copy = img

    kernel = np.ones((9, 9), np.uint8)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 90])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_black, upper_black)
    mask = 255 - mask
    img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, channels = img.shape

    border_threshold_R = 4
    border_threshold_G = 4
    border_threshold_B = 4

    borderup = 0
    borderdown = 0
    borderleft = 0
    borderright = 0

    upone = 0
    downone = 0
    leftone = 0
    rightone = 0

    upthree = 0
    downthree = 0
    leftthree = 0
    rightthree = 0

    # Checking whether the top border exists in the middle of the image
    for i in range(int(height / 2)):
        mid_pixel_top_half = img[i][int(width / 2)]
        R, G, B = mid_pixel_top_half[2], mid_pixel_top_half[1], mid_pixel_top_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            borderup += 1
        else:
            break

    # Checking whether the bottom border exists in the middle of the image
    for i in range(height - 1, int(height / 2) - 1, -1):
        mid_pixel_bottom_half = img[i][int(width / 2)]
        R, G, B = mid_pixel_bottom_half[2], mid_pixel_bottom_half[1], mid_pixel_bottom_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            borderdown += 1
        else:
            break

    # Checking whether the left border exists in the middle of the image
    for i in range(int(width / 2)):
        mid_pixel_top_half = img[int(height / 2)][i]
        R, G, B = mid_pixel_top_half[2], mid_pixel_top_half[1], mid_pixel_top_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            borderleft += 1
        else:
            break

    # Checking whether the right border exists in the middle of the image
    for i in range(width - 1, int(width / 2) - 1, -1):
        mid_pixel_bottom_half = img[int(height / 2)][i]
        R, G, B = mid_pixel_bottom_half[2], mid_pixel_bottom_half[1], mid_pixel_bottom_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            borderright += 1
        else:
            break

        #############################

    # Checking whether the top border exists in the first quater of the image
    for i in range(int(height / 2)):
        mid_pixel_top_half = img[i][int(width / 4)]
        R, G, B = mid_pixel_top_half[2], mid_pixel_top_half[1], mid_pixel_top_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            upone += 1
        else:
            break

    # Checking whether the bottom border exists in the first quater of the image
    for i in range(height - 1, int(height / 2) - 1, -1):
        mid_pixel_bottom_half = img[i][int(width / 4)]
        R, G, B = mid_pixel_bottom_half[2], mid_pixel_bottom_half[1], mid_pixel_bottom_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            downone += 1
        else:
            break

    # Checking whether the left border exists in the first quater of the image
    for i in range(int(width / 2)):
        mid_pixel_top_half = img[int(height / 4)][i]
        R, G, B = mid_pixel_top_half[2], mid_pixel_top_half[1], mid_pixel_top_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            leftone += 1
        else:
            break

    # Checking whether the right border exists in the first quater of the image
    for i in range(width - 1, int(width / 2) - 1, -1):
        mid_pixel_bottom_half = img[int(height / 4)][i]
        R, G, B = mid_pixel_bottom_half[2], mid_pixel_bottom_half[1], mid_pixel_bottom_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            rightone += 1
        else:
            break

        #############################

    # Checking whether the top border exists in the last quater of the image
    for i in range(int(height / 2)):
        mid_pixel_top_half = img[i][int(3 * width / 4) - 1]
        R, G, B = mid_pixel_top_half[2], mid_pixel_top_half[1], mid_pixel_top_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            upthree += 1
        else:
            break

    # Checking whether the bottom border exists in the last quater of the image
    for i in range(height - 1, int(height / 2) - 1, -1):
        mid_pixel_bottom_half = img[i][int(3 * width / 4) - 1]
        R, G, B = mid_pixel_bottom_half[2], mid_pixel_bottom_half[1], mid_pixel_bottom_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            downthree += 1
        else:
            break

    # Checking whether the left border exists in the last quater of the image
    for i in range(int(width / 2)):
        mid_pixel_top_half = img[int(3 * height / 4) - 1][i]
        R, G, B = mid_pixel_top_half[2], mid_pixel_top_half[1], mid_pixel_top_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            leftthree += 1
        else:
            break

    # Checking whether the right border exists in the last quater of the image
    for i in range(width - 1, int(width / 2) - 1, -1):
        mid_pixel_bottom_half = img[int(3 * height / 4) - 1][i]
        R, G, B = mid_pixel_bottom_half[2], mid_pixel_bottom_half[1], mid_pixel_bottom_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            rightthree += 1
        else:
            break

    count = 0
    finalup = 0
    finaldown = 0
    finalleft = 0
    finalright = 0

    if upone > 0 and upthree > 0 and borderup > 0:
        count += 1
        listVal = [borderup, upone, upthree]
        finalup = max(listVal)

    if downone > 0 and downthree > 0 and borderdown > 0:
        count += 1
        listVal = [borderdown, downone, downthree]
        finaldown = max(listVal)

    if (rightone > 0 and rightthree > 0 and borderright > 0):
        count += 1
        listVal = [borderright, rightone, rightthree]
        finalright = max(listVal)

    if (leftone > 0 and leftthree > 0 and borderleft > 0):
        count += 1
        listVal = [borderleft, leftone, leftthree]
        finalleft = max(listVal)

    if (count > 0):
        subimage = copy[finalup + 10: height - finaldown - 10, finalleft + 10: width - finalright - 10]
        #         final = cv2.resize(subimage, (width, height))
        return subimage
    else:
        return copy


def remove_hair_inpaint(img_src):
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(1, (17, 17))
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    img_dst = cv2.inpaint(img_src, thresh2, 1, cv2.INPAINT_TELEA)
    return img_dst


def remove_hair(img):
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Changing the shape and the size of the kernal
    # Kernal shows how any pixel in an image combines with different amounts of neigbouring pixels
    kernel = cv2.getStructuringElement(1, (5, 5))

    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    ret, thresh2 = cv2.threshold(blackhat, 100, 255, cv2.THRESH_BINARY)

    # Filling the gaps that were taken by the pixels that showed hair by mixing up with the nearby pixel colours
    dst = cv2.inpaint(img, thresh2, 15, cv2.INPAINT_TELEA)

    return dst


def remove_patches_v2(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([150, 255, 255])
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.dilate(mask, kernel, iterations=4)
    dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return dst


def remove_ink_patches(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_values1 = np.array([110, 50, 70])
    lower_values2 = np.array([110, 100, 120])
    upper_values = np.array([138, 255, 255])

    bluepen_mask1 = cv2.inRange(hsv_image, lower_values1, upper_values)
    bluepen_mask2 = cv2.inRange(hsv_image, lower_values2, upper_values)

    non_black1 = np.sum(bluepen_mask1 != 0)
    non_black2 = np.sum(bluepen_mask2 != 0)
    percentage = 0

    if non_black1 != 0:
        percentage = non_black2 / non_black1 * 100

    flags = cv2.INPAINT_TELEA

    if percentage > 50:
        output = cv2.inpaint(img, bluepen_mask1+bluepen_mask2, 100, flags=flags)
    else:
        output = img

    return output


def remove_patches_v11(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100,50,50])
    upper_blue = np.array([150,255,255])
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.dilate(mask,kernel,iterations = 4)
    dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return dst


def remove_hair_v11(src):

    # Convert the original image to grayscale
    grayScale = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1, (17, 17))

    # Perform the blackHat filtering on the grayscale image to find the (hair countours)
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair countours in preparation for the inpainting
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # inpaint the original image depending on the mask
    dst = cv2.inpaint(src, thresh2, 1, cv2.INPAINT_TELEA)

    return dst


def remove_null_variables(df, threshold=0.6):
    # Calculate the proportion of null values per variable
    null_proportions = df.isnull().mean()

    # Select variables with more than the threshold of null values
    null_variables = null_proportions[null_proportions > threshold].index

    # Remove the null variables from the DataFrame
    df_without_nulls = df.drop(null_variables, axis=1)

    return df_without_nulls

