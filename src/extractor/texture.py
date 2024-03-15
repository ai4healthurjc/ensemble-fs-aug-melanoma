import cv2
import numpy as np
import math
import skimage
from skimage.feature import graycomatrix, graycoprops


def get_array_gray_levels_with_frequency(img_gray, lvldtype=np.uint8):

    aryoflst = np.empty(0, np.dtype([('glvl', lvldtype), ('freq', np.uint32, (1,))]), 'C')

    for x in range(0, img_gray.shape[0], 1):
        for y in range(0, img_gray.shape[1], 1):
            aryoflst = __ins(aryoflst, img_gray[x, y], index=aryoflst.size)

    return aryoflst


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


def __showGLCM(feobj):
    print(feobj.getGLCM())


def get_haralick_features(feobj):

    dict_haralick = {
        "haralick_asm": feobj.getAngularSecondMomentASM(),
        "haralick_energy": feobj.getEnergy(),
        "haralick_entropy": feobj.getEntropy(),
        "haralick_contrast": feobj.getContrast(),
        "haralick_homogeneity": feobj.getHomogeneity(),
        "haralick_directional_moment": feobj.getDm(),
        "haralick_correlation": feobj.getCorrelation(),
        "haralick_corr": feobj.getHarCorrelation(),
        "haralick_cluster_shade": feobj.getClusterShade(),
        "haralick_cluster_prominence": feobj.getClusterProminence(),
        "haralick_moment1": feobj.getMoment1(),
        "haralick_moment2": feobj.getMoment2(),
        "haralick_moment3": feobj.getMoment3(),
        "haralick_moment4": feobj.getMoment4(),
        "haralick_dasm": feobj.getDasm(),
        "haralick_differential_mean": feobj.getDmean(),
        "haralick_differential_entropy": feobj.getDentropy()
    }

    return dict_haralick


def get_kings_features(feobj4):

    dict_king = {
        'king_coarseness': feobj4.getKingsCoarseness(),
        'king_contrast': feobj4.getKingsContrast(),
        'king_busyness': feobj4.getKingsBusyness(),
        'king_complexity': feobj4.getKingsComplexity(),
        'king_strength': feobj4.getKingsStrength()
    }

    return dict_king


def extract_haralick_king_features(img, mask):

    img_gray = cv2.bitwise_and(img, img, mask=mask)
    # img_gray = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2GRAY)

    feobj = TextureHaralick(img_gray)
    dict_haralick = get_haralick_features(feobj)

    feobj3 = TextureKing(img_gray)
    dict_king = get_kings_features(feobj3)

    dict_texture = {}
    dict_texture.update(dict_haralick)
    dict_texture.update(dict_king)

    return dict_texture


def texturef(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = skimage.img_as_ubyte(image)
    # Calcule de matrice GLCM
    glcm = graycomatrix(image, [1], [0], 256, symmetric=True, normed=True)
    # extraction des moments statistiques [Homogeneity, Correlation, Contrast, Energie.]
    feature = graycoprops(glcm, 'dissimilarity')[0]
    feature = np.concatenate([feature, graycoprops(glcm, 'correlation')[0]])
    feature = np.concatenate([feature, graycoprops(glcm, 'contrast')[0]])
    feature = np.concatenate([feature, graycoprops(glcm, 'energy')[0]])
    return feature


class TextureHaralick(object):
    """
    Haralick introduced Gray Level Co-occurrence Matrix (GLCM), using various statistical and differential
    textural-features of an image are extracted. These features, are primarily based on the 'thermodynamical'
    aspects of a textured image. GLCM shows how often each gray level occurs at a pixel located at a fixed
    geometric position relative to another pixel, as a function of the gray level.
    Listed below are the set of Haralick's texture features,
    (1) Angular Second Moment(ASM)
    (2) Energy
    (3) Entropy
    (4) Contrast
    (5) Homogeneity
    (6) Directional-Moment(DM)
    (7) Correlation
    (8) Haralick-Correlation
    (9) Cluster-Shade
    (10) Cluster-Prominence
    (11) Moment-1
    (12) Moment-2
    (13) Moment-3
    (14) Moment-4
    (15) Differential-ASM(DASM)
    (16) Differential-Mean(DMEAN)
    (17) Differential-Entropy(DENTROPY)
    """

    def __init__(self, img, offset=()):
        glvl_with_freq = get_array_gray_levels_with_frequency(img)
        if len(offset) == 0:
            self.__glcm = self._generate_glcm(img, glvl_with_freq)
        else:
            self.__glcm = self._generate_glcm(img, glvl_with_freq, offset)

        (self.__asm, self.__energy, self.__entropy, self.__contrast, self.__idm_homogeneity, self.__dm,
         self.__correlation, self.__har_correlation, self.__cluster_shade, self.__cluster_prominence, self.__moment1,
         self.__moment2, self.__moment3, self.__moment4, self.__dasm, self.__dmean,
         self.__dentropy) = self._generate_haralick_features(self.__glcm, glvl_with_freq)

    def _generate_glcm(self, img, glvlwthfreq, offset=(0, 1)):
        coocurmat = np.zeros((glvlwthfreq.size, glvlwthfreq.size), np.uint32, 'C')
        for i in range(0, (img.shape)[0], 1):
            for j in range(0, (img.shape)[1], 1):
                if ((((i + offset[0]) < 0) | ((i + offset[0]) >= img.shape[0])) | (
                        ((j + offset[1]) < 0) | ((j + offset[1]) >= img.shape[1]))):
                    continue
                else:
                    first = search(glvlwthfreq, img[i, j], 0, glvlwthfreq.size - 1)
                    second = search(glvlwthfreq, img[(i + offset[0]), (j + offset[1])], 0, glvlwthfreq.size - 1)
                    coocurmat[first, second] = np.uint32(coocurmat[first, second]) + np.uint32(1)
        return coocurmat

    def _generate_haralick_features(self, glcm, glvlwthfreq):
        sumofglcm = glcm.sum(axis=None, dtype=float)
        asm = 0.0
        correlation = 0.0
        har_correlation = 0.0
        entropy = 0.0
        contrast = 0.0
        idm_homogeneity = 0.0
        cluster_shade = 0.0
        cluster_prominence = 0.0
        m1 = 0.0
        m3 = 0.0
        m4 = 0.0
        dm = 0.0
        ux = 0.0
        uy = 0.0
        vx = 0.0
        vy = 0.0

        (energy, m2, asm, entropy, contrast, idm_homogeneity, dm, ux, uy, m1, m3, m4) = self._genHarFeatPt1(glcm,
                                                                                                            glvlwthfreq,
                                                                                                            asm,
                                                                                                            entropy,
                                                                                                            contrast,
                                                                                                            idm_homogeneity,
                                                                                                            dm, ux, uy,
                                                                                                            m1, m3, m4,
                                                                                                            sumofglcm)

        (cluster_shade, cluster_prominence, correlation, har_correlation) = self._genHarFeatPt2(glcm, glvlwthfreq,
                                                                                                ux, uy, vx, vy,
                                                                                                correlation,
                                                                                                cluster_shade,
                                                                                                cluster_prominence,
                                                                                                har_correlation,
                                                                                                sumofglcm)
        dasm = 0.0
        dmean = 0.0
        dentropy = 0.0
        for k in range(0, glvlwthfreq.size, 1):
            psum = 0.0
            for i in range(0, (glcm.shape)[0], 1):
                for j in range(0, (glcm.shape)[1], 1):
                    if (math.fabs(i - j) == k):
                        psum = psum + (float(glcm[i, j]) / sumofglcm)
                    else:
                        continue
            (dasm, dmean) = ((dasm + math.pow(psum, 2)), (dmean + (k * psum)))
            if psum <= 0.0:
                dentropy = dentropy + 0.0
                continue
            else:
                dentropy = dentropy + (psum * (- math.log(psum)))
        return (asm, energy, entropy, contrast, idm_homogeneity, dm, correlation, har_correlation, cluster_shade,
                cluster_prominence, m1, m2, m3, m4, dasm, dmean, dentropy)

    def _genHarFeatPt1(self, glcm, glvlwthfreq, asm, entropy, contrast, idm_homogeneity, dm, ux, uy, m1, m3, m4,
                       sumofglcm):
        i = 0
        for x in glcm:
            j = 0
            for item in x:
                y = float(item) / sumofglcm
                if y == 0.0:
                    pass
                else:
                    asm = asm + math.pow(y, 2)
                    entropy = entropy + (y * (- math.log(y)))
                    contrast = contrast + (math.pow((float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])), 2) * y)
                    idm_homogeneity = idm_homogeneity + (
                                (1 / (1 + math.pow((float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])), 2))) * y)
                    dm = dm + (math.fabs(float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])) * y)
                    ux = ux + (float((glvlwthfreq[i])[0]) * y)
                    uy = uy + (float((glvlwthfreq[j])[0]) * y)
                    m1 = m1 + ((float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])) * y)
                    m3 = m3 + (math.pow((float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])), 3) * y)
                    m4 = m4 + (math.pow((float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])), 4) * y)
                j = j + 1
            i = i + 1
        return (math.sqrt(asm), contrast, asm, entropy, contrast, idm_homogeneity, dm, ux, uy, m1, m3, m4)

    def _genHarFeatPt2(self, glcm, glvlwthfreq, ux, uy, vx, vy, correlation, cluster_shade, cluster_prominence,
                       har_correlation, sumofglcm):
        i = 0
        for x in glcm:
            j = 0
            for item in x:
                y = float(item) / sumofglcm
                if (y == 0.0):
                    pass
                else:
                    vx = vx + (math.pow((float((glvlwthfreq[i])[0]) - ux), 2) * y)
                    vy = vy + (math.pow((float((glvlwthfreq[j])[0]) - uy), 2) * y)
                    correlation = correlation + (
                                (float((glvlwthfreq[i])[0]) - ux) * (float((glvlwthfreq[j])[0]) - uy) * y)
                    cluster_shade = cluster_shade + (
                                math.pow(((float((glvlwthfreq[i])[0]) - ux) + (float((glvlwthfreq[j])[0]) - uy)),
                                         3) * y)
                    cluster_prominence = cluster_prominence + (
                                math.pow(((float((glvlwthfreq[i])[0]) - ux) + (float((glvlwthfreq[j])[0]) - uy)),
                                         4) * y)
                    har_correlation = har_correlation + (
                                (float((glvlwthfreq[i])[0]) * float((glvlwthfreq[j])[0]) * y) - math.pow(
                            ((ux + uy) / 2), 2))
                j = j + 1
            i = i + 1
        (vx, vy) = (math.sqrt(vx), math.sqrt(vy))
        (correlation, har_correlation) = ((correlation / (vx * vy)), (har_correlation / math.pow(((vx + vy) / 2), 2)))
        return (cluster_shade, cluster_prominence, correlation, har_correlation)

    def getGLCM(self):
        return self.__glcm

    def getAngularSecondMomentASM(self):
        return self.__asm

    def getEnergy(self):
        return self.__energy

    def getEntropy(self):
        return self.__entropy

    def getContrast(self):
        return self.__contrast

    def getHomogeneity(self):
        return self.__idm_homogeneity

    def getDm(self):
        return self.__dm

    def getCorrelation(self):
        return self.__correlation

    def getHarCorrelation(self):
        return self.__har_correlation

    def getClusterShade(self):
        return self.__cluster_shade

    def getClusterProminence(self):
        return self.__cluster_prominence

    def getMoment1(self):
        return self.__moment1

    def getMoment2(self):
        return self.__moment2

    def getMoment3(self):
        return self.__moment3

    def getMoment4(self):
        return self.__moment4

    def getDasm(self):
        return self.__dasm

    def getDmean(self):
        return self.__dmean

    def getDentropy(self):
        return self.__dentropy



class TextureKing(object):
    """
    King's texture features are also based on the human visual perception of images. It is computationally more
    optimal as well as extracts more features. King's method introduces the notion of NGTDM
    (Neighborhood Gray-Tone Difference Matrix).
    Listed below are the set of King's texture features:
    (1) King's-Coarseness
    (2) King's-Contrast
    (3) King's-Busyness
    (4) King's-Complexity
    (5) King's-Strength
    """

    def __init__(self, img, d=2, e=0.3):
        glvlwthfreq = get_array_gray_levels_with_frequency(img)
        self._ngtdm = self._generate_ngtdm(img, glvlwthfreq, d)
        (self.__coarseness, factor) = self._generate_kings_coarseness(glvlwthfreq, img.size, e)
        self.__contrast = self._generate_kings_contrast(glvlwthfreq, img.size)
        self.__busyness = self.__generateBusyness(glvlwthfreq, img.size, factor)
        self.__complexity = self.__generateComplexity(glvlwthfreq, img.size)
        self.__strength = self.__generateStrength(glvlwthfreq, img.size, e)

    def _generate_ngtdm(self, img, glvlwthfreq, d):
        ngtdm = np.zeros(glvlwthfreq.shape, float, 'C')
        for i in range(0, img.shape[0], 1):
            for j in range(0, img.shape[1], 1):
                if img[i, j] == 0:
                    continue
                else:
                    index = search(glvlwthfreq, img[i, j], 0, glvlwthfreq.size - 1)
                    ngtdm[index] = ngtdm[index] + np.fabs(img[i, j] - (self.__calculateSubSum(img, i, j, d) / (np.power(((2 * d) + 1), 2) - 1)))
        return ngtdm

    def __calculateSubSum(self, img, i, j, d):
        sum = 0.0
        m = -d
        while(m < d):
            n = -d
            while(n < d):
                (x, y) = self.__checkLimits((i + m), (j + n), img.shape)
                sum = sum + img[x, y]
                n = n + 1
            m = m + 1
        return sum

    def __checkLimits(self, x, y, shape):
        if (x < 0):
            x = 0
        if (x >= shape[0]):
            x = shape[0] - 1
        if (y < 0):
            y = 0
        if (y >= shape[1]):
            y = shape[1] - 1
        return (x, y)

    def _generate_kings_coarseness(self, glvlwthfreq, totpix, e):
        sum = 0.0
        for i in range(0, glvlwthfreq.size, 1):
            sum = sum + ((float((glvlwthfreq[i])[1]) / float(totpix)) * self._ngtdm[i])
        return ((1 /(e + sum)), sum)

    def _generate_kings_contrast(self, glvlwthfreq, totpix):
        sum = 0.0
        for i in range(0, glvlwthfreq.size, 1):
            for j in range(0, glvlwthfreq.size, 1):
              if((glvlwthfreq[i])[0] == (glvlwthfreq[j])[0]):
                  continue
              else:
                  sum = sum + (((float((glvlwthfreq[i])[1])) / float(totpix)) * ((float((glvlwthfreq[j])[1])) / float(totpix)) * np.power((float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])), 2))
        sum = sum * (1.0 / float(glvlwthfreq.size * (glvlwthfreq.size - 1))) * ((1.0 / np.power(float(totpix), 2)) * (self._ngtdm).sum(axis=None, dtype=float))
        return sum

    def __generateBusyness(self, glvlwthfreq, totpix, factor):
        sum = 0.0
        for i in range(0, glvlwthfreq.size, 1):
            for j in range(0, glvlwthfreq.size, 1):
                    sum = sum + ((float((glvlwthfreq[i])[0]) * ((float((glvlwthfreq[i])[1])) / float(totpix))) - (float((glvlwthfreq[j])[0]) * ((float((glvlwthfreq[j])[1])) / float(totpix))))
        sum = factor / sum
        return sum

    def __generateComplexity(self, glvlwthfreq, totpix):
        sum = 0.0
        for i in range(0, glvlwthfreq.size, 1):
            for j in range(0, glvlwthfreq.size, 1):
                sum = sum + ((np.fabs(float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])) / (np.power(float(totpix), 2) * (((float((glvlwthfreq[i])[1])) / float(totpix)) + ((float((glvlwthfreq[j])[1])) / float(totpix))))) * ((((float((glvlwthfreq[i])[1])) / float(totpix)) * self._ngtdm[i]) + (((float((glvlwthfreq[j])[1])) / float(totpix)) * self._ngtdm[j])))
        return sum

    def __generateStrength(self, glvlwthfreq, totpix, e):
        sum = 0.0
        for i in range(0, glvlwthfreq.size, 1):
            for j in range(0, glvlwthfreq.size, 1):
                sum = sum + ((((float((glvlwthfreq[i])[1])) / float(totpix)) + ((float((glvlwthfreq[j])[1])) / float(totpix))) * np.power((float((glvlwthfreq[i])[0]) - float((glvlwthfreq[j])[0])), 2))
        sum = sum / (e + (self._ngtdm).sum(axis=None, dtype=float))
        return sum

    def getNGTDM(self):
        return self._ngtdm

    def getKingsCoarseness(self):
        return self.__coarseness

    def getKingsContrast(self):
        return self.__contrast

    def getKingsBusyness(self):
        return self.__busyness

    def getKingsComplexity(self):
        return self.__complexity

    def getKingsStrength(self):
        return self.__strength