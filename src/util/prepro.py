import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image
import copy


def scale_contour(cnt, scale):
    M = cv2.moments(cnt)

    #### TODO Remove
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        # set values as what you need in the situation
        cx, cy = 0, 0
    ####

    # cx = int(M['m10']/M['m00'])
    # cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


def resize_img(img, new_shape=(256, 256)):
    img_resized = cv2.resize(img, new_shape)
    return img_resized


def convert_to_gray_img(img_resized):
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    return img_gray.astype(np.uint8)


def apply_segmentation_using_mask(img, mask, plot_img=False):
    white = np.zeros(img.shape, np.uint8) * 255
    white[:, :, 0] = mask
    white[:, :, 1] = mask
    white[:, :, 2] = mask
    img_segmented = cv2.bitwise_and(img, white)

    if plot_img:
        plt.imshow(img_segmented)
        plt.show()

    return img_segmented


def apply_mask_cv(img, mask):
    img_masked = cv2.bitwise_and(img, mask)
    return img_masked


def euclidean2d(v1, v2):
    dis = math.pow(v1, 2) + math.pow(v2, 2)
    return dis


def euclidean3d(v1, v2):
    x = math.pow(int(v1[0]) - int(v2[0]), 2)
    y = math.pow(int(v1[1]) - int(v2[1]), 2)
    z = math.pow(int(v1[2]) - int(v2[2]), 2)
    dis = math.sqrt(x + y + z)
    return dis


def bgr2lab_cv(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return lab


def bgr2gray_cv(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def rotate_bound(image, angle):
    #Determine centerpoint of image
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    #Find nerw boundry for image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    #Adjust rotation to account for translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    #Perform rotation
    return cv2.warpAffine(image, M, (nW, nH))


def get_array_gray_levels_with_frequency(img_gray, lvldtype=np.uint8):

    aryoflst = np.empty(0, np.dtype([('glvl', lvldtype), ('freq', np.uint32, (1,))]), 'C')

    for x in range(0, img_gray.shape[0], 1):
        for y in range(0, img_gray.shape[1], 1):
            aryoflst = __ins(aryoflst, img_gray[x, y], index=aryoflst.size)

    return aryoflst


def get_array_gray_levels_with_frequencyv2():
    print('holi2')

def __ins(arr, ins_val, index):
    if arr.size == 0:
        arr = np.insert(arr, index, (ins_val, np.array([ 1 ], np.uint32)), 0)
        return arr
    else:
        fnd_idx = search(arr, ins_val, 0, arr.size-1)
        if fnd_idx >= 0:
            ((arr[fnd_idx])[1])[0] = np.uint32(((arr[fnd_idx])[1])[0]) + np.uint32(1)
            return arr
        else:
            while index >= 0:
                    if ins_val > (arr[index - 1])[0]:
                        arr = np.insert(arr, index, (ins_val, np.array([1], np.uint32)), 0)
                        break
                    if ins_val < (arr[index - 1])[0]:
                        if (index == 0):
                            arr = np.insert(arr, index, (ins_val, np.array([1], np.uint32)), 0)
                        index = index - 1
                        continue
                    else:
                        ((arr[index - 1])[1])[0] = np.uint32(((arr[index - 1])[1])[0]) + np.uint32(1)
                        break
            return arr


def search(arr, ins_val, low, high):
    fnd_idx = -1
    if arr.size == 0:
        pass
    else:
        while low <= high:
            mid = int(low + ((high - low) / 2))
            if ins_val > (arr[mid])[0]:
                low = mid + 1
                continue
            if ins_val < (arr[mid])[0]:
                high = mid - 1
                continue
            if ins_val == (arr[mid])[0]:
                fnd_idx = mid
                break
    return fnd_idx




def strings2numeric(strings, names, numeric_vals, sentinel=-1):
    """Convert strings to numeric values.
    Args:
        strings: a list of strings to convert to numeric tags.
        names: a list of unique strings that correspond to the numeric values.
        numeric_vals: a list of integers that correspond to the ordering in names.
        sentinel: a value that is not in the numeric_vals.
    Returns:
        a numpy array of the numeric values to use (instead of the strings)
    """

    if sentinel in numeric_vals:
        raise ValueError("`sentinel` should not occur in `numeric_val`.")

    strings = np.asarray(strings)

    numeric = np.ones(shape=len(strings), dtype=int) * sentinel

    for class_idx, label in zip(numeric_vals, names):
        # If this is a list, then group all these (sublist) items with the same numeric label.
        if isinstance(label, list):
            for l in label:
                numeric[np.asarray(strings == l)] = class_idx

        else:  # This is a single sub-label already.
            numeric[np.asarray(strings == label)] = class_idx

    if np.any(numeric == sentinel):
        missing_indexes = np.where(numeric == sentinel)
        first_missing_str = strings[missing_indexes[0]]
        raise ValueError(
            "The value `%s` in `strings` do not exist in `names`. Did you spell something wrong?" % first_missing_str)

    return numeric


def crop_resize_img(img_path, target_size, crop_amount):
    """Load, crop, and resize an image."""
    img = image.load_img(img_path)
    x = image.img_to_array(img)
    x = x[crop_amount:-crop_amount, crop_amount:-crop_amount, :]
    ximg = Image.fromarray(np.uint8(x))
    ximg_resize = ximg.resize((target_size[0], target_size[1]))
    x = image.img_to_array(ximg_resize)

    return x


def quick_sort(arr, low, high):
    if low < high:
        pi = __partition(arr, low, high)
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)


def __partition(arr, low, high):
    i = (low - 1)
    pivot = (arr[high])[0]
    for j in range(low, high,1):
        if (arr[j])[0] <= pivot:
            i = i + 1
            temp = copy.deepcopy(arr[i])
            arr[i] = copy.deepcopy(arr[j])
            arr[j] = copy.deepcopy(temp)
    temp2 = copy.deepcopy(arr[i+1])
    arr[i+1] = copy.deepcopy(arr[high])
    arr[high] = copy.deepcopy(temp2)
    return (i+1)
