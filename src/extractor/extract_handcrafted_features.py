import pandas as pd
import cv2
import logging
import coloredlogs
from pyfeats import *
from pathlib import Path
import itertools
from operator import itemgetter
from multiprocessing import Pool
import traceback
from extractor.texture import extract_haralick_king_features
from extractor.color import Color
from extractor.abcd import extract_abcd_features

from util.loader import read_img, load_images_by_dataset_name
from util.preprocessing import Preprocess
import util.consts as consts

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def get_paths_dataset(dataset):
    if dataset == 'ph2':
        return Path.joinpath(consts.PATH_PROJECT_DATA_PH2_DIR, consts.PH2_DATASET_NAME_STATISTICS)
    elif dataset == 'sevenpoint':
        return Path.joinpath(consts.PATH_PROJECT_DATA_SEVEN_POINT, consts.SEVEN_DATASET_NAME_STATISTICS)
    else:
        return Path.joinpath(consts.PATH_PROJECT_DATA_PH2_DIR, consts.PH2_DATASET_NAME_STATISTICS)


def extract_features_batch(list_path_imgs: list) -> list:

    list_img_features = []
    type_img = 'pre'

    for path_img, path_mask in list_path_imgs:

        img_raw, img_raw_name = read_img(path_img, type_images='images', flag_resize=True)
        img_mask, img_mask_name = read_img(path_mask, type_images='masks', flag_resize=False)

        try:
            img_pre = Preprocess.remove_artifacts_sevenpoint(img_raw)
            img_segmented = cv2.bitwise_and(img_pre, img_pre, mask=img_mask)
            img_gray_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

            if type_img == 'raw':
                img_gray_to_extract = img_gray
            else:
                img_gray_to_extract = img_gray_pre

            logger.info('img-name: {}, mask: {}'.format(img_raw_name, img_mask_name))

            if img_raw_name != img_mask_name:
                logger.error('Image and mask names are not matching!!!!')

            dict_color = extract_color_features(img_raw, img_mask, img_raw_name)
            dict_abcd, dict_seven = extract_abcd_features(img_raw, img_pre, img_segmented, img_raw_name)
            dict_multiscale = extract_multiscale_features(img_gray_to_extract, img_mask, img_raw_name)
            dict_texture = extract_texture_features(img_gray_to_extract, img_mask, img_raw_name)

            dict_features = {
                **dict_abcd,
                **dict_seven,
                **dict_texture,
                **dict_color,
                **dict_multiscale,
                'img_name': img_raw_name
            }

            list_img_features.append(dict_features)
        except Exception:
            traceback.print_exc()
            logger.error('Error extracting features')

    return list_img_features


def convert_lists_to_dict(list_names, list_values):
    dict_feat = dict(zip(list_names, list_values))
    return dict_feat


def extract_color_features(image_color, mask, image_name):
    color_extractor = Color()
    dict_color = color_extractor.extract_color_features(image_color, mask)
    return dict_color


def extract_multiscale_features(image, mask, image_name):

    dict_multiscale_features = {}

    dwt_feat, dwt_labels = dwt_features(image, mask, wavelet='bior3.3', levels=3)
    dict_multiscale_features.update(convert_lists_to_dict(dwt_labels, dwt_feat))
    logger.info(
        'Multiscale Features - Discrete Wavelet Transform (DWT) {} for image {}'.format(len(dwt_labels), image_name))

    wp_feat, wp_labels = wp_features(image, mask, wavelet='coif1', maxlevel=3)
    dict_multiscale_features.update(convert_lists_to_dict(wp_labels, wp_feat))
    logger.info('Multiscale Features - Wavelet Packets (WP) {} for image {}'.format(len(wp_labels), image_name))

    return dict_multiscale_features


def extract_texture_features(image, mask, image_name):
    dict_texture_features = {}

    fdta_feat, fdta_labels = fdta(image, mask, s=3)
    dict_fdta = convert_lists_to_dict(fdta_labels, fdta_feat)
    dict_texture_features.update(dict_fdta)
    logger.info('Texture Features - Fractal Dimension Texture Analysis (FDTA) {} for image {}'.format(len(fdta_labels),
                                                                                                      image_name))
    fos_feat, fos_labels = fos(image, mask)
    dict_fos = convert_lists_to_dict(fos_labels, fos_feat)
    dict_texture_features.update(dict_fos)
    logger.info('Texture Features - First order statistics (FOS) {} for image {}'.format(len(fos_labels), image_name))

    glcm_feat_mean, glcm_features_range, glcm_labels_mean, glcm_labels_range = glcm_features(image, ignore_zeros=True)
    dict_glcm = convert_lists_to_dict(glcm_labels_mean, glcm_feat_mean)
    dict_texture_features.update(dict_glcm)
    logger.info('Texture Features - Gray Level Co-occurence Matrix (GLCM) for image {}'.format(image_name))

    glds_feat, glds_labels = glds_features(image, mask, Dx=[0, 1, 1, 1], Dy=[1, 1, 0, -1])
    dict_glds = convert_lists_to_dict(glds_labels, glds_feat)
    dict_texture_features.update(dict_glds)
    logger.info('Texture Features - Gray Level Difference Statistics (GLDS) {} for image {}'.format(len(glds_labels), image_name))

    glrlm_feat, glrlm_labels = glrlm_features(image, mask, Ng=256)
    dict_glrlm = convert_lists_to_dict(glrlm_labels, glrlm_feat)
    dict_texture_features.update(dict_glrlm)
    logger.info('Texture Features - Gray Level Run Length Matrix (GLRLM) {} for image {}'.format(len(glrlm_labels), image_name))

    glszm_feat, glszm_labels = glszm_features(image, mask)
    dict_glszm = convert_lists_to_dict(glszm_labels, glszm_feat)
    dict_texture_features.update(dict_glszm)
    logger.info('Texture Features - Gray Level Size Zone Matrix (GLSZM) {} for image {}'.format(len(glszm_labels), image_name))

    hos_feat, hos_labels = hos_features(image, th=[135, 140])
    dict_hos = convert_lists_to_dict(hos_labels, hos_feat)
    dict_texture_features.update(dict_hos)
    logger.info('Texture Features - Higher Order Spectra (HOS) {} for image {}'.format(len(hos_labels), image_name))

    dict_haralick_king_features = extract_haralick_king_features(image, mask)
    dict_texture_features.update(dict_haralick_king_features)

    lbp_feat, lbp_labels = lbp_features(image, mask, P=[8, 16, 24], R=[1, 2, 3])
    dict_lbp = convert_lists_to_dict(lbp_labels, lbp_feat)
    dict_texture_features.update(dict_lbp)
    logger.info('Texture Features - Local Binary Pattern (LPB) {} for image {}'.format(len(lbp_labels), image_name))

    sfm_feat, sfm_labels = sfm_features(image, mask, Lr=4, Lc=4)
    dict_sfm = convert_lists_to_dict(sfm_labels, sfm_feat)
    dict_texture_features.update(dict_sfm)
    logger.info(
        'Texture Features - Statistical Feature Matrix (SFM) {} for image {}'.format(len(sfm_labels), image_name))

    return dict_texture_features


def extract_statistics(dataset: str, batch_size=10, n_jobs=1, save_df=False):

    list_imgs = load_images_by_dataset_name(dataset_name=dataset, batch_size=batch_size)
    list_num_imgs = [len(dict_imgs['batch_path_filenames']) for dict_imgs in list_imgs]
    logger.info('Dataset: {}, images loaded: {}'.format(dataset, sum(list_num_imgs)))

    list_batch_imgs = list(map(itemgetter('batch_path_filenames'), list_imgs))

    pool = Pool(processes=n_jobs)
    result_list_pool = pool.map(extract_features_batch, list_batch_imgs)
    list_dict_features = list(itertools.chain.from_iterable(result_list_pool))
    df_features = pd.DataFrame(list_dict_features)

    if save_df:
        path_statistics = get_paths_dataset(dataset)
        df_features.to_csv(str(path_statistics), index=False)

    return df_features
