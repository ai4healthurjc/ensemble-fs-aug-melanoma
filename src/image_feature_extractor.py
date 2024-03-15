import pandas as pd
from pathlib import Path
from multiprocessing import cpu_count
import argparse
import logging
import coloredlogs

import util.consts as consts
from util.loader import normalize_dataframe, load_ground_truth_dataset, balance_partition
from clfs.tabular import train_fs_clf_with_different_k_features
from extractor.extract_handcrafted_features import extract_statistics

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def parse_arguments(parser):
    parser.add_argument('--dataset', default='ph2', type=str)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--k_features', default=10, type=int)
    parser.add_argument('--estimator', default='dt', type=str)
    parser.add_argument('--score_metric', default='roc_auc', type=str)
    parser.add_argument('--fs', default='relief', type=str)
    parser.add_argument('--train_models', default=False, type=bool)
    parser.add_argument('--extract_features', default=False, type=bool)
    parser.add_argument('--n_jobs', default=1, type=int)
    parser.add_argument('--seed_value', default=3232, type=int)
    parser.add_argument('--features', default='texture', type=str)

    return parser.parse_args()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract image features')
    args = parse_arguments(parser)

    n_procs = cpu_count()
    n_jobs = n_procs if args.n_jobs == -1 or args.n_jobs > n_procs else args.n_jobs
    logger.info('n_jobs selected: {}'.format(n_jobs))

    if args.extract_features:
        df_extracted_features = extract_statistics(dataset=args.dataset, batch_size=args.batch_size, n_jobs=args.n_jobs, save_df=True)
    else:
        path_statistics = Path.joinpath(consts.PATH_PROJECT_DATA_PH2_DIR, consts.PATH_PROJECT_DATA_PH2_STATISTICS)
        df_extracted_features = pd.read_csv(str(path_statistics))

    if args.train_models:
        df_extracted_features_scaled = normalize_dataframe(df_extracted_features)
        df_gt = load_ground_truth_dataset(dataset_name=args.dataset)
        df_merge = df_extracted_features_scaled.merge(df_gt, how='inner', on='img_name')

        df_merge = df_merge.drop(columns=['label', 'img_name'])
        df_features = df_merge.iloc[:, :-1]
        y_label_encoded = df_merge['label_encoded'].values

        df_x_features_resampled, y_label_resampled = balance_partition(df_features, y_label_encoded, seed_value=args.seed_value)

        df_metrics, df_scores = train_fs_clf_with_different_k_features(df_x_features_resampled,
                                                                       y_label_resampled,
                                                                       args.fs,
                                                                       args.estimator,
                                                                       k_features=args.k_features,
                                                                       as_frame=True,
                                                                       scoring_estimator=args.score_metric
                                                                       )

        print(df_metrics)
