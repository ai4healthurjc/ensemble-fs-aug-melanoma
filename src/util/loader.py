import numpy as np
import pandas as pd
import math
import cv2
from sklearn.utils import resample
from pathlib import Path
from itertools import islice
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import util.consts as consts
import logging
import coloredlogs
from sklearn.model_selection import train_test_split
import glob
import re
import joblib

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def read_img(path_img: Path, type_images='images', target_size=(512, 384), flag_resize=False):
    if type_images == 'images':
        img = cv2.imread(str(path_img), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(path_img), 0)

    img_name = str(path_img.stem)

    if flag_resize:
        img_resized = cv2.resize(img, target_size)
    else:
        img_resized = img

    return img_resized, img_name


def get_imgs_ordered(path_dir: Path, pattern: str, sort: bool = True, recursive: bool = True):

    if not path_dir.is_dir():
        raise ValueError(f"'{path_dir}' is not a directory.")

    frec = lambda p, g: p.rglob(g) if recursive else p.glob(g)
    fsort = lambda x: sorted(x) if sort else x

    return fsort(frec(path_dir, pattern))


def load_meta(dataset_name):

    if dataset_name == 'sevenpoint':
        path_metadata = str(Path.joinpath(consts.PATH_PROJECT_DATA_SEVEN_POINT_META))
        df_meta = pd.read_csv(path_metadata)
        return df_meta


def load_batch_images_ph2(dataset_name, batch_size):
    path_imgs = Path(consts.PATH_PROJECT_DATA_PH2_DIR_IMAGES)
    path_masks = Path(consts.PATH_PROJECT_DATA_PH2_DIR_MASKS_DOBLEUNET)

    return load_batch_images(path_imgs, path_masks, batch_size, dataset_name)


def load_batch_images_sevenpoint(dataset_name, batch_size):
    path_imgs = Path(consts.PATH_PROJECT_DATA_SEVEN_POINT_IMAGES)
    path_masks = Path(consts.PATH_PROJECT_DATA_SEVEN_POINT_MASKS_DOBLEUNET)
    return load_batch_images(path_imgs, path_masks, batch_size, dataset_name)


def load_images(dataset_name):

    if dataset_name == 'ph2':
        images = []
        filepaths = []
        path_stats = Path.joinpath(consts.PATH_PROJECT_DIR, 'data', 'raw', 'ph2', 'images')
        for filepath in glob.iglob(str(path_stats)+"/**/*.*", recursive=True):
            if not filepath.endswith((".png", ".jpg", ".JPEG", ".bmp", ".BMP", ".JPG")):
                continue
            print(filepath)
            images.append(filepath)

        def extract_number(file_path):
            match = re.search(r'IMD(\d+).jpg', file_path)
            return int(match.group(1)) if match else -1
        
        sorted_file_paths = sorted(images, key=extract_number)

        imgs_resize = []
        for i in range(len(sorted_file_paths)):
            imgs_resize.append(cv2.resize(cv2.imread(sorted_file_paths[i]), (224, 224)))

    if dataset_name == 'derm7pt':
        images = []
        filepaths = []
        path_stats = Path.joinpath(consts.PATH_PROJECT_DIR, 'data', 'raw','derm7pt', 'images')
        for filepath in glob.iglob(str(path_stats)+"/**/*.*", recursive=True):
            if not filepath.endswith((".png", ".jpg", ".JPEG", ".bmp", ".BMP", ".JPG")):
                continue
            print(filepath)
            images.append(filepath)
        
        imgs_resize = []
        for i in range(len(images)):
            imgs_resize.append(cv2.resize(cv2.imread(images[i]), (224, 224)))

    return imgs_resize
     

def load_stats_by_dataset_name_partition(dataset_name, modality, idx):

    if dataset_name == 'ph2' and modality == 'statistics':
        path_stats = Path.joinpath(consts.PATH_PROJECT_DIR, 'data', 'fs', 'ph2', f'statistics_FS_{idx}_PH2.csv')
        print(path_stats)
        df_stats = pd.read_csv(str(path_stats))

        df_gt = load_ground_truth_dataset(dataset_name=dataset_name)
        df_merge = df_stats.merge(df_gt, how='inner', on='img_name')
        df_merge = df_merge.drop(columns=['label', 'img_name'])
        df_features = df_merge.iloc[:, :-1]
        y_label_encoded = df_merge['label_encoded'].values
    
    if dataset_name == 'derm7pt' and modality == 'statistics':
        path_stats = Path.joinpath(consts.PATH_PROJECT_DIR, 'data', 'fs', 'derm7pt', f'statistics_FS_{idx}_7point.csv')
        print(path_stats)
        df_stats = pd.read_csv(str(path_stats))

        df_gt = load_ground_truth_dataset(dataset_name=dataset_name)
        df_merge = df_stats.merge(df_gt, how='inner', on='img_name')
        df_merge = df_merge.drop(columns=['label', 'img_name'])
        df_features = df_merge.iloc[:, :-1]
        y_label_encoded = df_merge['label_encoded'].values

    if dataset_name == 'ph2' and modality == 'embeddings':
        path_stats = Path.joinpath(consts.PATH_PROJECT_DIR, 'data', 'fs', 'ph2', f'embeddings_FS_{idx}_PH2.csv')
        df_stats = pd.read_csv(str(path_stats))
        df_gt = load_ground_truth_dataset(dataset_name=dataset_name)
        df_merge = df_stats.merge(df_gt, how='inner', on='img_name')
        df_merge = df_merge.drop(columns=['label', 'img_name'])
        df_features = df_merge.iloc[:, :-1]
        y_label_encoded = df_merge['label_encoded'].values
    
    if dataset_name == 'derm7pt' and modality=='embeddings':
        path_stats = Path.joinpath(consts.PATH_PROJECT_DIR, 'data', 'fs','derm7pt', f'embeddings_FS_{idx}_7point.csv')
        print(path_stats)
        df_stats = pd.read_csv(str(path_stats))

        df_gt = load_ground_truth_dataset(dataset_name=dataset_name)
        df_merge = df_stats.merge(df_gt, how='inner', on='img_name')
        df_merge = df_merge.drop(columns=['label', 'img_name'])
        df_features = df_merge.iloc[:, :-1]
        y_label_encoded = df_merge['label_encoded'].values

    return df_features, y_label_encoded


def load_stats_by_dataset_name(dataset_name):

    if dataset_name == 'ph2':
        path_stats = Path.joinpath(consts.PATH_PROJECT_DIR, 'data', 'raw', 'ph2', 'df_statistics_ph2.csv')
        df_stats = pd.read_csv(str(path_stats))
        print(df_stats)

        df_gt = load_ground_truth_dataset(dataset_name=dataset_name)
        df_merge = df_stats.merge(df_gt, how='inner', on='img_name')
        v_img_name = df_merge['img_name'].values
        df_merge = df_merge.drop(columns=['label', 'img_name'])
        df_features = df_merge.iloc[:, :-1]
        print(df_features)
        y_label_encoded = df_merge['label_encoded'].values

    if dataset_name == 'derm7pt':
        path_stats = Path.joinpath(consts.PATH_PROJECT_DIR, 'data', 'raw', 'derm7pt', f'df_statistics_seven.csv')
        df_stats = pd.read_csv(str(path_stats))
        df_gt = load_ground_truth_dataset(dataset_name=dataset_name)
        df_merge = df_stats.merge(df_gt, how='inner', on='img_name')
        v_img_name = df_merge['img_name'].values
        missing_names_stats = set(df_stats['img_name']) - set(df_gt['img_name'])
        missing_names_gt = set(df_gt['img_name']) - set(df_stats['img_name'])

        df_merge = df_merge.drop(columns=['label', 'img_name'])
        df_features = df_merge.iloc[:, :-1]
        y_label_encoded = df_merge['label_encoded'].values

    return df_features, y_label_encoded, v_img_name


def load_embeddings_by_dataset_name(dataset_name, idx):

    if dataset_name == 'ph2':
        path_emb = Path.joinpath(consts.PATH_PROJECT_DIR, 'reports', 'embeddings', f'embeddings_{idx}_ph2.csv')
        print(path_emb)
        df_emb = pd.read_csv(str(path_emb))
        print(df_emb)
    
    if dataset_name == 'derm7pt':
        path_emb = Path.joinpath(consts.PATH_PROJECT_DIR, 'reports', 'embeddings', f'embeddings_{idx}_derm7pt.csv')
        print(path_emb)
        df_emb = pd.read_csv(str(path_emb))
        print(df_emb)

    return df_emb


def load_synthetic_stats_by_dataset_name(dataset_name, type_encoding, type_sampling, oversampler, classifier, ir, n_epochs, batch_size, seed):

    path_stats = Path.joinpath(consts.PATH_PROJECT_SYNTHETIC_DATA, 'ctgan', 'statistics', f'df_x_train_resampled_scaled_{dataset_name}_{type_encoding}_{type_sampling}_{oversampler}_{classifier}_ir_{ir}_epoch_{n_epochs}_batch_{batch_size}_seed_{seed}.csv')
    df_stats = pd.read_csv(str(path_stats))
    names = df_stats.columns

    df_label = Path.joinpath(consts.PATH_PROJECT_SYNTHETIC_DATA, 'ctgan', 'statistics', f'df_y_train_resampled_scaled_{dataset_name}_standard_{type_sampling}_{oversampler}_{classifier}_ir_{ir}_epoch_{n_epochs}_batch_{batch_size}_seed_{seed}.csv')
    df_label = pd.read_csv(str(df_label))

    df_stats['label'] = df_label['label']

    return df_stats, names


def load_model_stats(dataset_name, type_encoding, type_sampling, oversampler, classifier, ir, n_epochs, batch_size, seed):
    path_model = Path.joinpath(consts.PATH_PROJECT_MODELS, 'statistics', f'model_clf_{dataset_name}_{type_encoding}_{type_sampling}_{oversampler}_{classifier}_ir_{ir}_epoch_{n_epochs}_batch_{batch_size}_seed_{seed}.sav')
    model = joblib.load(path_model)
    return model


def split_train_test(x_features, y_label, test_size=0.2, seed_value=1234):

    x_train, x_test, y_train, y_test = train_test_split(x_features,
                                                        y_label,
                                                        stratify=y_label,
                                                        test_size=test_size,
                                                        random_state=seed_value
                                                        )

    return x_train, y_train, x_test, y_test


def get_prefix_images_dataset(dataset_name):
    if dataset_name == 'isic':
        return "ISIC_*.jpg", "ISIC_*.jpg", "image_name", "target"
    elif dataset_name == 'isic2020':
        return "ISIC_*.jpg", "ISIC_*.jpg", "image_name", "target"
    elif dataset_name == 'ph2':
        return "IMD*.jpg", "IMD*.png", "img_id", "diagnostic"
    elif dataset_name == 'sevenpoint':
        return "*.jpg", "*.png", "img_id", "diagnostic"


def load_batch_images(path_imgs_dir: Path,
                      path_masks_dir: Path,
                      batch_size: int,
                      dataset_name: str
                      ):

    prefix_imgs, prefix_masks, img_colum_name, target_id = get_prefix_images_dataset(dataset_name)
    imgs_filenames = get_imgs_ordered(path_imgs_dir, prefix_imgs)
    masks_filenames = get_imgs_ordered(path_masks_dir, prefix_masks)
    files_filenames = zip(imgs_filenames, masks_filenames)

    list_dict_batches = []
    batch_id = 0

    while True:

        list_batch_img_path_filenames = list(islice(files_filenames, batch_size))

        if not list_batch_img_path_filenames:
            break

        logger.info(
            'Dataset: {}, batch {} with {} images'.format(dataset_name, batch_id, len(list_batch_img_path_filenames))
        )

        dict_batch_imgs = {
            'batch_id': batch_id,
            'batch_path_filenames': list_batch_img_path_filenames,
            'img_name_id': img_colum_name,
            'target_id': target_id
        }

        list_dict_batches.append(dict_batch_imgs)
        batch_id += 1

    return list_dict_batches


def get_meta_dataset(dataset_name):
    if dataset_name == 'ph2':
        return consts.PATH_PROJECT_DATA_PH2_GROUND_TRUTH, 'Name', 'Clinical Diagnosis'
    elif dataset_name == 'derm7pt':
        return consts.PATH_PROJECT_DATA_SEVEN_POINT_GROUND_TRUTH, 'derm', 'diagnosis'


def load_ground_truth_dataset(dataset_name):

    path_meta, img_name_col, label_col = get_meta_dataset(dataset_name)
    df_gt = pd.read_csv(str(path_meta))

    y_label = df_gt.loc[:, label_col].values
    y_label_encoded = LabelEncoder().fit_transform(y_label)

    logger.info('y-label-raw: {}'.format(np.unique(y_label)))
    logger.info('y-label-encoded: {}'.format(np.unique(y_label_encoded)))

    df_gt['label_encoded'] = y_label_encoded
    df_gt = df_gt.rename(columns={img_name_col: 'img_id', label_col: 'label'})
    df_gt['img_name'] = df_gt['img_id'].apply(lambda x: x.split('.')[0])
    df_gt_filtered = df_gt.loc[:, ['img_name', 'label', 'label_encoded']]

    print(df_gt_filtered)

    return df_gt_filtered


def remove_features_with_nan(df, threshold=0.6):

    # Calculate the proportion of null values per variable
    null_proportions = df.isnull().mean()

    # Select variables with more than the threshold of null values
    null_variables = null_proportions[null_proportions > threshold].index

    # Remove the null variables from the DataFrame
    df_without_nulls = df.drop(null_variables, axis=1)

    return df_without_nulls


def normalize_dataframe(df_features, type_scaler='standard'):

    # Select scaler
    if type_scaler == 'standard':
        myscaler = StandardScaler()
    else:
        myscaler = MinMaxScaler()

    # Remove features with all nans
    df_features = df_features.loc[:, ~df_features.isnull().all()]

    # Remove features with 60% of nans
    df_features = remove_features_with_nan(df_features, threshold=0.6)

    list_vars_categorical, list_vars_numerical = get_categorical_numerical_var_names(df_features)
    # list_vars_categorical.remove("img_name")

    logger.info('# vars-cat: {}'.format(len(list_vars_categorical)))
    logger.info('# vars-num: {}'.format(len(list_vars_numerical)))

    if len(list_vars_categorical) != 0:
        df_features[list_vars_categorical] = MinMaxScaler().fit_transform(df_features[list_vars_categorical].values)

    if len(list_vars_numerical) != 0:
        df_features[list_vars_numerical] = myscaler.fit_transform(df_features[list_vars_numerical].values)

    for var_cat in list_vars_categorical:
        df_features[var_cat] = df_features[var_cat].fillna(df_features[var_cat].mode()[0])

    for var_num in list_vars_numerical:
        df_features[var_num] = df_features[var_num].fillna(df_features[var_num].mean())

    return df_features


def get_categorical_numerical_var_names(df_data: pd.DataFrame) -> (list, list):

    df_info = identify_type_features(df_data)

    list_numerical_vars = list(df_info[df_info['type'] == consts.TYPE_FEATURE_CONTINUOUS].index)
    list_categorical_vars = list(df_info[df_info['type'] == consts.TYPE_FEATURE_DISCRETE].index)

    return list_categorical_vars, list_numerical_vars


def identify_type_features(df, discrete_threshold=10, debug=False):
    """
    Categorize every feature/column of df as discrete or continuous according to whether or not the unique responses
    are numeric and, if they are numeric, whether or not there are fewer unique renponses than discrete_threshold.
    Return a dataframe with a row for each feature and columns for the type, the count of unique responses, and the
    count of string, number or null/nan responses.
    """
    counts = []
    string_counts = []
    float_counts = []
    null_counts = []
    types = []
    for col in df.columns:
        responses = df[col].unique()
        counts.append(len(responses))
        string_count, float_count, null_count = 0, 0, 0
        for value in responses:
            try:
                val = float(value)
                if not math.isnan(val):
                    float_count += 1
                else:
                    null_count += 1
            except:
                try:
                    val = str(value)
                    string_count += 1
                except:
                    print('Error: Unexpected value', value, 'for feature', col)

        string_counts.append(string_count)
        float_counts.append(float_count)
        null_counts.append(null_count)
        types.append('d' if len(responses) < discrete_threshold or string_count > 0 else 'c')

    df_feature_info = pd.DataFrame(
        {'count': counts,
         'string_count': string_counts,
         'float_count': float_counts,
         'null_count': null_counts,
         'type': types},
        index=df.columns
    )

    if debug:
        print(f'Counted {sum(df_feature_info["type"] == "d")} discrete features and {sum(df_feature_info["type"] == "c")} continuous features')

    return df_feature_info


def get_balanced_dataset(m_min: np.matrix, m_maj: np.matrix, seed_value: int) -> np.matrix:
    n_min_samples = m_min.shape[0]
    matrix_maj_downsampled = resample(m_maj, replace=False, n_samples=n_min_samples, random_state=seed_value)
    return matrix_maj_downsampled


def balance_partition(df_x, y, seed_value):

    rus = RandomUnderSampler(random_state=seed_value)
    x_resampled, y_resampled = rus.fit_resample(df_x, y)

    logger.info('# samples per class: {}'.format(sorted(Counter(y_resampled).items())))

    return x_resampled, y_resampled


def load_images_by_dataset_name(dataset_name, batch_size):
    if dataset_name == 'ph2':
        return load_batch_images_ph2(dataset_name, batch_size=batch_size)
    elif dataset_name == 'sevenpoint':
        return load_batch_images_sevenpoint(dataset_name, batch_size=batch_size)




