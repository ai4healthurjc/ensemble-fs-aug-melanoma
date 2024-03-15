import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from pathlib import Path
from scipy.spatial import distance
import math
# from artifacts.artifacts import remove_hair, remove_ink_patches, remove_lens
from util.prepro import scale_contour
import os
from queue import Queue
from threading import Thread
from time import time
import matplotlib.pyplot as plt
import logging
import coloredlogs
# from utils.preprocess import Preprocess
from skimage.segmentation import chan_vese
# from utils.visualization import plot_img_raw_and_masked

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


class ExtractWorker(Thread):

    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            directory, link = self.queue.get()
            try:
                print('extract_features')
                # extract_features_batch(dict_batch_imgs['batch_path_filenames'], dict_batch_imgs['batch_meta'])
            finally:
                self.queue.task_done()


def test_segmenter(img_pre):
    img_pre_copy = img_pre.copy()
    img_gray_aux = cv2.cvtColor(img_pre_copy, cv2.COLOR_BGR2GRAY)
    cv = chan_vese(img_gray_aux, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
                   max_num_iter=200, dt=0.5, init_level_set="checkerboard",
                   extended_output=True)
    result = cv2.bitwise_and(img_pre_copy, img_pre_copy, mask=cv[0])
    return result


def detect_milky_red_areas(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([170, 140, 140])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    output_img = img.copy()
    output_img[np.where(mask1 == 0)] = 0

    # Check whether the red area is very small in the code
    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask1 == 0)] = 0

    gray = cv2.cvtColor(output_hsv, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    output_red = img.copy()
    result = cv2.bitwise_and(output_red, output_red, mask=thresh)

    all_area = np.sum(output_img != 0)
    selected_area = np.sum(result != 0)

    if all_area == 0:
        return 0
    else:
        percentage = selected_area / all_area * 100
        if percentage > 15:
            return 1
        else:
            return 0


def detect_rosettes(img, cnt, state):
    hh, ww = img.shape[:2]
    if state == 1:
        c = scale_contour(cnt, 0.7)
        mask3 = np.zeros((hh, ww), dtype=np.uint8)
        mask3 = cv2.drawContours(mask3, [c], 0, (255, 255, 255), cv2.FILLED)
        result = cv2.bitwise_and(img, img, mask=mask3)
    else:
        result = img

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result = cv2.equalizeHist(gray)

    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 100

    params.filterByCircularity = True
    params.minCircularity = 0.6

    params.filterByConvexity = True
    params.minConvexity = 0.8

    params.filterByInertia = True
    params.minInertiaRatio = 0.3

    params.filterByColor = True
    params.blobColor = 255

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(result)

    blank = np.zeros((1, 1))

    if (len(keypoints) > 4):
        return 1
    else:
        return 0


def detect_regression_structure(img, cnt, state):
    hh, ww = img.shape[:2]
    if(state == 1):
        c = scale_contour(cnt, 0.6)

        mask3 = np.zeros((hh,ww), dtype=np.uint8)
        mask3 = cv2.drawContours(mask3, [c], 0, (255,255,255), cv2.FILLED)

        result = cv2.bitwise_and(img, img, mask=mask3)

    else:
        result = img

    lower_white = np.array([0,0,145])
    upper_white = np.array([175,123,255])

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

    mask2 = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(result, result, mask=mask2)

    allArea = np.sum(result != 0)
    selectedArea = np.sum(res != 0)

    if allArea == 0:
        return 0
    else:
        percentage = selectedArea/allArea*100
        if percentage > 60:
            return 1
        else:
            return 0


def detect_blue_white_veil(img, cnt, state):

    hh, ww = img.shape[:2]

    if state == 1:
        c = scale_contour(cnt, 0.85)
        mask3 = np.zeros((hh,ww), dtype=np.uint8)
        mask3 = cv2.drawContours(mask3, [c], 0, (255,255,255), cv2.FILLED)
        result = cv2.bitwise_and(img, img, mask=mask3)
    else:
        result = img

    # Defining the colour range that we are looking for. in the lesion
    lower_bluegray = np.array([94, 80, 2])
    upper_bluegray = np.array([126, 255, 255])

    # Converting the image to the HSV format
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

    lower_gray = np.array([94, 70, 80], np.uint8)
    upper_gray = np.array([160, 100, 255], np.uint8)

    # Creating a mask from the area that the given colour appears in the lesion
    mask2 = cv2.inRange(hsv, lower_bluegray, upper_bluegray)
    # Cropping the image using the mask
    res = cv2.bitwise_and(result,result, mask= mask2)

    # Get the area of the contour and the area taken by the considered colour
    allArea = np.sum(result != 0)
    selectedArea = np.sum(res != 0)

    if(allArea == 0):
        return(0)
    else:
        percentage = selectedArea/allArea*100
        if(percentage > 15):
            return(1)
        else:
            return(0)


def extract_abcd_features(img_raw, img_pre, img_segmented, img_name):

    copy = img_pre
    image, con, percentage, hull, hull_image = get_contour(img_pre)

    try:
        if percentage == 100:
            state = 0
            asymmetry_value = 0
            border_value = 0
            brown, red, blue, white, black, colour_value, colour_real_value = detect_colour(copy, con, state)
            diameter_value = 0

            width_value = 0
            height_value = 0
            diameter_real_value = 0
            globules_value = detect_globules(copy)
            blotches_value = detect_blotches(copy)
            milky_red_value = detect_milky_red_areas(copy)
            rosettes_value = detect_rosettes(copy, con, state)
            regression_structure_value = detect_regression_structure(copy, con, state)
            blue_white_veil_value = detect_blue_white_veil(copy, con, state)
            atypical_value = detect_atypical_network(copy, con, state)
            streaks_value = 0

        else:
            state = 1
            asymmetry_value, asymmetry_real_value, percentage_value = detect_asymmetry(hull_image, hull)
            border_value, border_real_value, border_new_value = detect_border_irregularity(con, hull)
            brown, red, blue, white, black, colour_value, colour_real_value = detect_colour(hull_image, hull, state)
            diameter_value, diameter_real_value, width_value, height_value = detect_diameter(hull_image, hull)

            globules_value = detect_globules(hull_image)
            blotches_value = detect_blotches(copy)
            milky_red_value = detect_milky_red_areas(copy)
            rosettes_value = detect_rosettes(hull_image, hull, state)
            regression_structure_value = detect_regression_structure(hull_image, hull, state)
            blue_white_veil_value = detect_blue_white_veil(hull_image, hull, state)
            atypical_value = detect_atypical_network(hull_image, hull, state)
            streaks_value = detect_streaks(hull_image, hull)

        dict_abcd = {
            'asymmetry': asymmetry_value,
            'border': border_value,
            'color': colour_value,
            'diameter': diameter_value
        }

        dict_seven = {
            'globules': globules_value,
            'blotches': blotches_value,
            'read_areas': milky_red_value,
            'rosettes': rosettes_value,
            'regression_structure': regression_structure_value,
            'blue_white': blue_white_veil_value,
            'atypical_network': atypical_value,
            'streaks': streaks_value
        }

    except:

        dict_abcd = {
            'asymmetry': 0,
            'border': 0,
            'color': 0,
            'diameter': 0
        }

        dict_seven = {
            'globules': 0,
            'blotches': 0,
            'read_areas': 0,
            'rosettes': 0,
            'regression_structure': 0,
            'blue_white': 0,
            'atypical_network': 0,
            'streaks': 0
        }

    return dict_abcd, dict_seven


def detect_globules(img):

    h, w = img.shape[:2]

    params = assign_blobdetector_params()
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)

    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(img, keypoints, blank, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    n_blobs = len(keypoints)

    if n_blobs > 15:
        return 1
    else:
        return 0


def detect_blotches(img):
    # Obtain a new image by applying GaussianBlur to smooth the contour edges
    img = cv2.GaussianBlur(img, (5, 5), 0)
    src3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kernel = np.ones((7, 7), np.uint8)
    # Filling the points inside the foreground objects
    src2 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    src1 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
    # retval2,threshold2 = cv2.threshold(src1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # The thresholding values are being set to detect the lesion clearly
    # If the pixel value is lesser than the threshold the the pixel will be set to 0. Can only work grayscale images
    # Creating a mask to crop the lesion in the image
    # Obtaing the lesion from the image using created mask
    ret, thresh2 = cv2.threshold(src1, 65, 255, cv2.THRESH_BINARY_INV)
    mask3 = cv2.cvtColor(thresh2, cv2.COLOR_BGR2RGB)
    im_thresh_color = cv2.bitwise_and(src3, mask3)

    # The thresholding values are being set to detect the lesion clearly
    # If the pixel value is lesser than the threhold the the pixel will be set to 0. Can only work grayscale images
    # Creating a mask to crop the lesion in the image
    # Obtaing the lesion from the image using created mask
    retOther, threshOther = cv2.threshold(src1, 125, 255, cv2.THRESH_BINARY_INV)
    maskOther = cv2.cvtColor(threshOther, cv2.COLOR_BGR2RGB)
    img_threshold_color_other = cv2.bitwise_and(src3, maskOther)

    # Obtaining the larger area from the contour
    img_gray = cv2.cvtColor(img_threshold_color_other, cv2.COLOR_BGR2GRAY)
    # Get the external contours not the entire hierarchy of contours (unlike TREE)
    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_area = 0

    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        c = scale_contour(c, 0.8)
        all_area = cv2.contourArea(c)

        # Obtaining the larger area from the contour
    img_gray = cv2.cvtColor(im_thresh_color, cv2.COLOR_BGR2GRAY)
    # Get the external contours not the entire hierarchy of contours (unlike TREE)
    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blotch_area = 0

    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        blotch_area = cv2.contourArea(c)
    percentage = 0

    if all_area != 0:
        percentage = blotch_area / all_area * 100

    if 30 < percentage < 90:
        return 1
    else:
        return 0


def detect_streaks(img, cnt):

    hh, ww = img.shape[:2]
    c1 = scale_contour(cnt, 1.1)
    c = scale_contour(cnt, 0.75)

    mask_small = np.zeros((hh, ww), dtype=np.uint8)
    mask_small = cv2.drawContours(mask_small, [c], 0, (255, 255, 255), cv2.FILLED)
    result = cv2.bitwise_and(img, img, mask=mask_small)

    mask_large = np.zeros((hh, ww), dtype=np.uint8)
    mask_large = cv2.drawContours(mask_large, [c1], 0, (255, 255, 255), cv2.FILLED)
    result2 = cv2.bitwise_and(img, img, mask=mask_large - mask_small)

    hsv = cv2.cvtColor(result2, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 90])
    masknew = cv2.inRange(hsv, lower_black, upper_black)
    res = cv2.bitwise_and(result2, result2, mask=masknew)

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_area = 0

    for c in contours:
        all_area += cv2.contourArea(c)
        cv2.drawContours(result2, c, -1, (0, 0, 255), 2)

    if all_area > 0:
        gray_scale = cv2.cvtColor( result2, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

        blackhat = cv2.morphologyEx(gray_scale, cv2.MORPH_BLACKHAT, kernel)
        contours,hierarchy = cv2.findContours(blackhat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        invert = cv2.bitwise_not(cv2.cvtColor(thresh2, cv2.COLOR_BGR2RGB))
        dst = cv2.inpaint(result2, thresh2, 1, cv2.INPAINT_TELEA)
        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 200

        params.filterByCircularity = True
        params.minCircularity = 0.01

        params.filterByConvexity = True
        params.minConvexity = 0.4

        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(invert)
        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(result2, keypoints, blank, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        n_white_pix = np.sum(invert == 255)
        n_black_pix = np.sum(invert == 0)
        test1 = np.sum(result2 != 0)

        keys = 0
        for k in keypoints:
            keys += (k.size*k.size*3.14*0.25)
        non_black = np.sum(result2 != 0)
        test2 = np.sum(result2 == 0)

        key_percentage = keys/n_black_pix*100
        contour_percentage = all_area/non_black*100
        total = key_percentage + contour_percentage

        if total > 25 or (contour_percentage > 0.4 and key_percentage > 6 and total > 12) or (total > 12 and len(keypoints) > 10):
            return 1
        else:
            return 0
    else:
        return 0


def detect_atypical_network(img, cnt, state):

    hh, ww = img.shape[:2]

    if state == 1:
        c = scale_contour(cnt, 0.85)
        # Creating a mask to crop the lesion in the image
        mask3 = np.zeros((hh, ww), dtype=np.uint8)
        mask3 = cv2.drawContours(mask3, [c], 0, (255, 255, 255), cv2.FILLED)
        # Obtain the lesion from the image using created mask
        result = cv2.bitwise_and(img, img, mask=mask3)
    else:
        result = img

    # Obtaining the area of the lesion
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea) # DCM
    all_area = cv2.contourArea(c)

    # Obtaining the area where the atypical network is covering
    grayScale = cv2.cvtColor( result, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
    black_hat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    ret, thresh2 = cv2.threshold(black_hat, 10, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detect the blobs in the based on the given criteria
    blobdetector_params = cv2.SimpleBlobDetector_Params()

    blobdetector_params.filterByArea = True
    blobdetector_params.minArea = 20
    blobdetector_params.maxArea = 50

    blobdetector_params.filterByCircularity = True
    blobdetector_params.minCircularity = 0.5

    blobdetector_params.filterByConvexity = True
    blobdetector_params.minConvexity = 0.01

    blobdetector_params.filterByInertia = False
    blobdetector_params.minInertiaRatio = 0.1

    detector = cv2.SimpleBlobDetector_create(blobdetector_params)
    keypoints = detector.detect(thresh2)

    blank = np.zeros((1, 1))
    # Marking the detected blobs
    blobs = cv2.drawKeypoints(thresh2, keypoints, blank, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    blobs = cv2.drawKeypoints(result, keypoints, blank, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Obtaining the pixel areas of black and white
    n_white_pix = np.sum(thresh2 == 255)
    n_black_pix = np.sum(thresh2 == 0)
    percentage = n_white_pix/all_area*100

    if len(keypoints) > 15 and percentage > 30:
        return 1
    else:
        return 0



def detect_colour(img, cnt, state):
    hh, ww = img.shape[:2]
    count = 0

    if state == 1:
        c = scale_contour(cnt, 0.85)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        mask3 = np.zeros((hh, ww), dtype=np.uint8)
        mask3 = cv2.drawContours(mask3, [c], 0, (255, 255, 255), cv2.FILLED)
        result = cv2.bitwise_and(img, img, mask=mask3)
    else:
        result = img

    lower_brown1 = np.array([0, 120, 0])
    upper_brown1 = np.array([90, 150, 190])
    lower_brown2 = np.array([135, 120, 0])
    upper_brown2 = np.array([180, 150, 100])

    lower_red = np.array([160, 130, 150])
    upper_red = np.array([210, 255, 255])

    lower_blue = np.array([94, 70, 80])
    upper_blue = np.array([130, 255, 255])

    lower_white = np.array([0, 0, 100])
    upper_white = np.array([225, 90, 255])

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 90])

    allArea = 0
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        allArea += cv2.contourArea(c)

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv, lower_black, upper_black)
    res = cv2.bitwise_and(result,result, mask=mask2)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selectedArea = 0

    for c in contours:
        selectedArea += cv2.contourArea(c)

    ####### TODO Remove
    if allArea == 0:
        allArea = 0.000001
        percentage = selectedArea / allArea * 100
    else:
        percentage = selectedArea / allArea * 100
    #######

    if percentage > 5:
        count += 1
        black = 1
    else:
        black = 0

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(result,result, mask= mask2)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    contours,hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selectedArea = 0
    for c in contours:
        selectedArea += cv2.contourArea(c)
    percentage = selectedArea/allArea*100
    if(percentage > 5):
        count += 1
        white = 1
    else:
        white = 0

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(result,result, mask= mask2)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selectedArea = 0

    for c in contours:
        selectedArea += cv2.contourArea(c)
    percentage = selectedArea / allArea*100

    if percentage > 5:
        count += 1
        blue = 1
    else:
        blue = 0

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(result,result, mask= mask2)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selectedArea = 0
    for c in contours:
        selectedArea += cv2.contourArea(c)
    percentage = selectedArea/allArea*100
    if percentage > 5:
        count += 1
        red = 1
    else:
        red = 0

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
    mask2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
    mask = mask1 + mask2
    res = cv2.bitwise_and(result, result, mask=mask)

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selectedArea = 0

    for c in contours:
        selectedArea += cv2.contourArea(c)
    percentage = selectedArea/allArea*100

    if percentage > 5:
        count += 1
        brown = 1
    else:
        brown = 0

    if count > 2:
        return brown, red, blue, white, black, 1, count
    else:
        return brown, red, blue, white, black, 0, count


def detect_asymmetry(img, cnt):

    distance_ratio = 0
    c = cnt

    # area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)

    # Obtaining the top, bottem, left and right points of the lesion
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBottom = tuple(c[c[:, :, 1].argmax()][0])

    # Drawing the contour and the top, bottem, left and right points of the lesion
    cv2.drawContours(img, c, -1, (0, 255, 255), 2)
    cv2.circle(img, extLeft, 8, (0, 0, 255), -1)
    cv2.circle(img, extRight, 8, (0, 255, 0), -1)
    cv2.circle(img, extTop, 8, (255, 0, 0), -1)
    cv2.circle(img, extBottom, 8, (255, 255, 0), -1)

    # Obtaining the distance between given two points in the XY cordinate plane.
    # Proven that it is equal to the pythogorous theorem
    dist1 = math.sqrt(((extTop[0] - extBottom[0])**2) + ((extTop[1] - extBottom[1])**2))
    dist2 = math.sqrt(((extLeft[0] - extRight[0])**2) + ((extLeft[1] - extRight[1])**2))

    distance_diff = abs(dist1 - dist2)
    if dist1 == 0 and dist2 == 0:
        distance_ratio = 0
    else:
        distance_ratio = (distance_diff/(dist1+dist2)) * 100

    # Obtaining the center point of the contour
    M = cv2.moments(c)

    ##### TODO Remove
    if M["m00"] == 0:
        M["m00"] = 0.00001
    else:
        M["m00"] = M["m00"]
    #####

    center_X = int(M["m10"] / M["m00"])
    center_Y = int(M["m01"] / M["m00"])
    contour_center = (center_X, center_Y)

    # Obtaining the distance from the center of the lesion to the top, bottem, left and right points of the lesion
    x = abs(distance.euclidean(extRight, contour_center)-distance.euclidean(extLeft, contour_center))
    y = abs(distance.euclidean(extTop, contour_center)-distance.euclidean(extBottom, contour_center))

    if distance_ratio > 5 or x > 25 or y > 25:
        if x <= 10 or y <= 10:
            return 0, round(distance_ratio, 4), perimeter
        else:
            return 1, round(distance_ratio, 4), perimeter
    else:
        return 0, round(distance_ratio, 4), perimeter


def detect_border_irregularity(cnt, hull):
    comp = 0
    comp_new = 0

    if len(cnt) > 4:
        ellipse = cv2.fitEllipse(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        ellipse_cnt = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])),
                                       (int(ellipse[1][0]), int(ellipse[1][1])),
                                       int(ellipse[2]), 0, 360, 1)

        # Check the difference between given two contours to check for the irregularity
        comp_ori = cv2.matchShapes(cnt, ellipse_cnt, 1, 0.0)
        comp_new = cv2.matchShapes(cnt, hull, 1, 0.0)

    if comp_new > 0.3:
        return 1, round(comp_ori, 4), round(comp_new, 4)
    else:
        return 0, round(comp_ori, 4), round(comp_new, 4)


def detect_diameter(img, cnt):

    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)

    diameter = 2 * radius / 100 * 2
    # Obtaining the rectangle that could cover the entire contour
    x, y, w, h = cv2.boundingRect(cnt)

    if h >= w:
        d = h
    else:
        d = w

    d = d * 25.4 / 1280

    if d > 6:
        return 1, round(d, 4), w, h
    else:
        return 0, round(d, 4), w, h


def get_contour(img):

    hh, ww = img.shape[:2]
    border = False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (17, 17), 32)
    ret, thresh_blurred = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours_blurred, hierarchy = cv2.findContours(thresh_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ret, thresh_normal = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours_normal, hierarchy = cv2.findContours(thresh_normal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sorting the contours based on their sizes in the ascending order
    cnts_sorted = sorted(contours_blurred, key=lambda x: cv2.contourArea(x))

    i = len(cnts_sorted) - 1

    if len(cnts_sorted) > 0:
        status = True
        valid = True
        while status:
            if i < 0:
                status = False
                valid = False
                break
            ca = cnts_sorted[i]
            count = 0
            # Filters the contours that are touching the border of the image
            for a in range(len(ca)):
                if (ww - 5) < ca[a][0][0]:
                    count += 1
            for a in range(len(ca)):
                if (hh - 5) < ca[a][0][1]:
                    count += 1
            if (0 in ca) or (count != 0):
                i -= 1
            else:
                status = False

        if valid:
            maskBlurred = np.zeros((hh, ww), dtype=np.uint8)
            maskBlurred = cv2.drawContours(maskBlurred, [ca], 0, (255, 255, 255), cv2.FILLED)
            blurredArea = np.sum(maskBlurred != 0)
            blurredCon = ca
        else:
            maskBlurred = np.zeros((hh, ww), dtype=np.uint8)
            blurredArea = 0
            blurredCon = cnts_sorted[len(cnts_sorted)-1]

    # Obtains the largest contour detected even though it may touch the border of the image

        maskBlurredLarge = np.zeros((hh, ww), dtype=np.uint8)
        maskBlurredLarge = cv2.drawContours(maskBlurredLarge, [cnts_sorted[len(cnts_sorted)-1]], 0, (255, 255, 255), cv2.FILLED)
        blurredLargeArea = np.sum(maskBlurredLarge != 0)
        blurredLargeCon = cnts_sorted[len(cnts_sorted)-1]
        percentage = blurredLargeArea / (hh*ww) * 100
        x, y, w, h = cv2.boundingRect(cnts_sorted[len(cnts_sorted)-1])
        coord = []
        coord.append((0, 0))
        coord.append((0, hh))
        coord.append((ww, 0))
        coord.append((ww, hh))
        countCor = 0

        for cor in coord:
            if cor == (x, y) or cor == (x + w, y) or cor == (x, y + h) or cor == (x + w, y + h):
                countCor += 1

        if(percentage > 25 and countCor == 0) or (percentage > 60):
            selected = maskBlurredLarge
            selected_area = blurredLargeArea
            selectedCon = blurredLargeCon
            border = True
        else:
            selected = maskBlurred
            selected_area = blurredArea
            selectedCon = blurredCon
    else:
        selected = np.zeros((hh, ww), dtype=np.uint8)
        selected_area = 0

    cnts_sorted = sorted(contours_normal, key=lambda x: cv2.contourArea(x))
    i = len(cnts_sorted) - 1
    if len(cnts_sorted) > 0:
        status = True
        valid = True
        while status:
            if i < 0:
                status = False
                valid = False
                break
            ca = cnts_sorted[i]
            count = 0
            # Filters the contours that are touching the border of the image
            for a in range(len(ca)):
                if((ww-5) < ca[a][0][0]):
                    count += 1
            for a in range(len(ca)):
                if((hh-5) < ca[a][0][1]):
                    count += 1
            if (0 in ca) or (count != 0):
                i -= 1
            else:
                status = False
        if valid:
            maskNormal = np.zeros((hh, ww), dtype=np.uint8)
            maskNormal = cv2.drawContours(maskNormal, [ca], 0, (255, 255, 255), cv2.FILLED)
            normalArea = np.sum(maskNormal != 0)
            normalCon = ca
        else:
            normalArea = 0
            normalCon = cnts_sorted[len(cnts_sorted)-1]
    else:
        normalArea = 0

    if (selected_area < normalArea) or (border and (selected_area / 2) < normalArea):
        selected = maskNormal
        selectedCon = normalCon
        selected_area = normalArea

    src = img
    src = cv2.GaussianBlur(src,(5,5),0)
    src3 = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    kernal = np.ones((7, 7), np.uint8)
    src2 = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernal)
    src1 = cv2.cvtColor(src2, cv2.COLOR_BGR2RGB)
    retOther, threshOther = cv2.threshold(src1, 125, 255, cv2.THRESH_BINARY_INV)
    maskOther= cv2.cvtColor(threshOther, cv2.COLOR_BGR2RGB)
    im_thresh_color_other = cv2.bitwise_and(src3, maskOther)
    gray = cv2.cvtColor(im_thresh_color_other, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_sorted = sorted(contours, key=lambda x: cv2.contourArea(x))

    i = len(cnts_sorted) - 1

    if len(cnts_sorted) > 0:
        status = True
        valid = True
        while status:
            if i < 0:
                status = False
                valid = False
                break
            ca = cnts_sorted[i]
            count = 0
            # Filters the contours that are touching the border of the image
            for a in range(len(ca)):
                if (ww - 5) < ca[a][0][0]:
                    count += 1
            for a in range(len(ca)):
                if (hh - 5) < ca[a][0][1]:
                    count += 1
            if (0 in ca) or (count != 0):
                i -= 1
            else:
                status = False

    else:
        blotchArea = 0

    maskBlotch = np.zeros((hh, ww), dtype=np.uint8)
    maskBlotch = cv2.drawContours(maskBlotch, [ca], 0, (255, 255, 255), cv2.FILLED)
    blotchArea = np.sum(maskBlotch != 0)
    blotchCon = ca

    if (selected_area < blotchArea) or (border and (selected_area / 2) < blotchArea):
        selected = maskBlotch
        selectedCon = blotchCon
        selected_area = blotchArea

    i = img
    result = cv2.bitwise_and(i, i, mask=selected)
    percentage = selected_area / (hh*ww) * 100

    maskBlotch = np.zeros((hh, ww), dtype=np.uint8)
    hull = cv2.convexHull(selectedCon)
    maskBlotch = cv2.drawContours(maskBlotch, [hull], 0, (255, 255, 255), cv2.FILLED)
    hullResult = cv2.bitwise_and(i, i, mask=maskBlotch)

    # Obtaining the center of the image
    image_center = np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).shape) / 2
    value = image_center[1]
    image_center[1] = image_center[0]
    image_center[0] = value
    image_center = tuple(image_center.astype('int32'))
    # Obtaining the center of the contour
    M = cv2.moments(selectedCon)
    center_X = int(M["m10"] / M["m00"])
    center_Y = int(M["m01"] / M["m00"])
    contour_center = (center_X, center_Y)
    # Below calculated value is the distance between the center of the contour and the center of the image
    distance_to_center = (distance.euclidean(image_center, contour_center))

    # This is used to check whether the contour touches the corners of the image
    corners = 0
    for a in range(len(selectedCon)):
        if 0 == selectedCon[a][0][0] and 0 == selectedCon[a][0][1]:
            corners += 1
        if 0 == selectedCon[a][0][0] and (hh - 1) == selectedCon[a][0][1]:
            corners += 1
        if (ww - 1) == selectedCon[a][0][0] and (hh - 1) == selectedCon[a][0][1]:
            corners += 1
        if (ww - 1) == selectedCon[a][0][0] and 0 == selectedCon[a][0][1]:
            corners += 1

    # This will check for different threshold values and obntain different parts of the lesion as contours
    # to obtain the best extracted lesion
    if percentage < 1 or (percentage < 3 and distance_to_center > 15) or percentage > 90 or (percentage < 10 and distance_to_center > 100) or corners > 0:
        value = 165
        existingPercentage = 0
        existingStatus = False
        common = selectedCon
        while value > 50:
            con, per = variableContour(img, value)
            value -= 3
            if per > 0:
                M = cv2.moments(con)
                try:
                    center_X = int(M["m10"] / M["m00"])
                    center_Y = int(M["m01"] / M["m00"])
                    contour_center = (center_X, center_Y)
                    distance_to_center = (distance.euclidean(image_center, contour_center))
                except:
                    distance_to_center = 200
                if existingPercentage < per < 75 and (distance_to_center < 100 or per > 2):
                    existingStatus = True
                    existingPercentage = per
                    existingCon = con

        if not(existingStatus):
            existingCon = common

        maskBlotch = np.zeros((hh, ww), dtype=np.uint8)
        maskBlotch = cv2.drawContours(maskBlotch, [existingCon], 0, (255, 255, 255), cv2.FILLED)
        blotchArea = np.sum(maskBlotch != 0)
        blotchCon = existingCon
        percentage = blotchArea / (hh*ww) * 100

        M = cv2.moments(existingCon)
        center_X = int(M["m10"] / M["m00"])
        center_Y = int(M["m01"] / M["m00"])
        contour_center = (center_X, center_Y)
        distance_to_center = (distance.euclidean(image_center, contour_center))
        if (percentage < 5 and distance_to_center > 100) or percentage > 75:
            return img, blotchCon, 100, blotchCon, img
        else:
            i = img
            result = cv2.bitwise_and(i, i, mask=maskBlotch)
            maskBlotch = np.zeros((hh, ww), dtype=np.uint8)
            hull = cv2.convexHull(existingCon)
            maskBlotch = cv2.drawContours(maskBlotch, [hull], 0, (255, 255, 255), cv2.FILLED)
            hullResult = cv2.bitwise_and(i, i, mask=maskBlotch)
            return result, blotchCon, percentage, hull, hullResult

    return result, selectedCon, percentage, hull, hullResult


def variableContour(img, value):
    src = img
    # Getting the hight and the width of the image
    hh, ww = img.shape[:2]
    src = cv2.GaussianBlur(src, (5, 5), 0)
    src3 = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    kernal = np.ones((7, 7), np.uint8)
    src2 = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernal)
    src1 = cv2.cvtColor(src2, cv2.COLOR_BGR2RGB)
    retOther, threshOther = cv2.threshold(src1, value, 255, cv2.THRESH_BINARY_INV)
    maskOther = cv2.cvtColor(threshOther, cv2.COLOR_BGR2RGB)
    im_thresh_color_other = cv2.bitwise_and(src3, maskOther)
    gray = cv2.cvtColor(im_thresh_color_other, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    i = len(cntsSorted) - 1
    if (len(cntsSorted) > 0):
        status = True
        valid = True
        while status:
            if (i < 0):
                status = False
                valid = False
                break
            ca = cntsSorted[i]
            count = 0
            for a in range(len(ca)):
                if ((ww - 5) < ca[a][0][0]):
                    count += 1
            for a in range(len(ca)):
                if ((hh - 5) < ca[a][0][1]):
                    count += 1
            if (0 in ca) or (count != 0):
                i -= 1
            else:
                status = False

        if (valid):
            maskBlotch = np.zeros((hh, ww), dtype=np.uint8)
            maskBlotch = cv2.drawContours(maskBlotch, [ca], 0, (255, 255, 255), cv2.FILLED)
            blotchArea = np.sum(maskBlotch != 0)
            blotchCon = ca
        else:
            blotchArea = 0
            blotchCon = None

    else:
        blotchArea = 0
        blotchCon = None

    percentage = blotchArea / (hh * ww) * 100

    return blotchCon, percentage