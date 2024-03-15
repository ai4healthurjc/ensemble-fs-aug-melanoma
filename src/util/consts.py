from pathlib import Path

PATH_PROJECT_DIR = Path(__file__).resolve().parents[2]
PATH_PROJECT_DATA = Path.joinpath(PATH_PROJECT_DIR, 'data')
PATH_PROJECT_DATA_RAW = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw')
PATH_PROJECT_DATA_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'preprocessed')

PATH_PROJECT_FS_SCORES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'fs', 'scores')
PATH_PROJECT_FS_FIGURES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'fs', 'figures')

PATH_PROJECT_MODELS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'models')
PATH_PROJECT_EMBEDDINGS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'embeddings')

PATH_PROJECT_FIGURES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'figures')
PATH_PROJECT_BOOTSTRAP = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'bootstrap')
PATH_PROJECT_SHAP = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'shap')

PATH_PROJECT_COEFS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'coefs')
PATH_PROJECT_METRICS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'metrics')
PATH_PROJECT_SYNTHETIC_DATA = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'synthetic_data')
PATH_PROJECT_FS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'fs')
PATH_PROJECT_MODEL_SHAP = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'model_shap', 'knn')
PATH_PROJECT_PARAMS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'params')

PATH_PROJECT_REPORTS_MODELS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'models')

PATH_PROJECT_DATA_SEVEN_POINT = Path.joinpath(PATH_PROJECT_DIR, 'data', 'derm7pt')
PATH_PROJECT_DATA_SEVEN_POINT_META = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'derm7pt', 'meta', 'meta.csv')
PATH_PROJECT_DATA_SEVEN_POINT_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'derm7pt', 'images')
PATH_PROJECT_DATA_SEVEN_POINT_MASKS_DOBLEUNET = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'derm7pt', 'masks_dobleunet')
PATH_PROJECT_DATA_SEVEN_POINT_TRAIN = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'derm7pt', 'meta', 'train_indexes.csv')
PATH_PROJECT_DATA_SEVEN_POINT_TEST = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'derm7pt', 'meta', 'test_indexes.csv')
PATH_PROJECT_DATA_SEVEN_POINT_GROUND_TRUTH = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'derm7pt', 'metadatos_procesados_7_point_dataset_ordenado.csv')
SEVEN_DATASET_NAME_STATISTICS = 'df_statistics_seven.csv'

PATH_PROJECT_DATA_PH2_DIR = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'ph2')
PATH_PROJECT_DATA_PH2_DIR_IMAGES = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'ph2', 'images')
PATH_PROJECT_DATA_PH2_DIR_MASKS_DOBLEUNET = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'ph2', 'masks_dobleunet')
PATH_PROJECT_DATA_PH2_DIR_MASKS_GROUND_TRUTH = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'ph2', 'masks_ground_truth')
PATH_PROJECT_DATA_PH2_GROUND_TRUTH = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'ph2', 'metadatos_procesados_PH2_v2.csv')
PATH_PROJECT_DATA_PH2_STATISTICS = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw', 'ph2', 'statistics_FS_0_PH2.csv')
PH2_DATASET_NAME_STATISTICS = 'df_statistics_ph2.csv'

COL_NAMES_ABCD = ['asymmetry', 'border', 'color', 'diameter']

COL_NAMES_SEVEN = ['globules', 'blotches', 'read_areas', 'rosettes', 'regression_structure', 'blue_white',
                   'atypical_network', 'streaks']

COL_NAMES_PAD = []

TYPE_FEATURE_CONTINUOUS = 'c'
TYPE_FEATURE_DISCRETE = 'd'

LABEL_CLASS_MIN_REAL = 1000
LABEL_CLASS_MIN_SYNTHETIC = 2000
LABEL_CLASS_MAJ_REAL = 3000

SUFFIX_FS = '_fs'
