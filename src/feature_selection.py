import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from multiprocessing import cpu_count
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import coloredlogs
import logging

import fs.ensemble_fs as fs
from util.loader import load_stats_by_dataset_name, load_embeddings_by_dataset_name
import util.consts as consts
from util.preprocessing import remove_null_variables
from util.plotter import generate_feature_importance_plot
from clfs.tabular import train_clf_with_selected_features

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def select_optimal_features_ensemblefs(X,
                                       y,
                                       df_voting_sorted,
                                       v_img_name,
                                       estimator_name,
                                       scoring_estimator,
                                       seed_value
                                       ):

    list_total_metrics = []

    for per_features in range(10, 110, 10):
        n_selected_features = round(df_voting_sorted.shape[0] * (per_features / 100))
        v_selected_features = df_voting_sorted.iloc[:n_selected_features, :]['var_name']
        df_selected_features = X.loc[:, v_selected_features]

        list_df_metrics = train_clf_with_selected_features(df_selected_features,
                                                           y,
                                                           bbdd_name='',
                                                           estimator_name=estimator_name,
                                                           as_frame=False,
                                                           scoring_estimator=scoring_estimator,
                                                           )

        list_total_metrics.extend(list_df_metrics)
    df_metrics = pd.DataFrame(list_total_metrics)
    print(df_metrics)
    df_scoring_clf = df_metrics[(df_metrics['metric'] == scoring_estimator)]
    best_k_features = df_scoring_clf.loc[df_scoring_clf['mean'].idxmax()]['k_features']
    v_best_features = df_voting_sorted.iloc[:best_k_features, :]['var_name']
    df_best_features = X.loc[:, v_best_features]
    df_best_features['img_name'] = v_img_name
    path_stats = Path.joinpath(consts.PATH_PROJECT_DIR, 'data', 'fs', 'ph2', f'statistics_FS_{seed_value}_PH2.csv')
    df_best_features.to_csv(str(path_stats), index=False)


def parse_arguments(parser):
    parser.add_argument('--dataset', default='ph2', type=str)
    parser.add_argument('--modality', default='statistics', type=str)
    parser.add_argument('--n_boots', default=10, type=int)
    parser.add_argument('--n_jobs', default=4, type=int)
    parser.add_argument('--fs', default='relief', type=str)
    parser.add_argument('--estimator', default='dt', type=str)
    parser.add_argument('--scoring_estimator', default='roc_auc', type=str)
    parser.add_argument('--agg_func', default='mean', type=str)
    parser.add_argument('--n_seeds', default=5, type=int)

    return parser.parse_args()


parser = argparse.ArgumentParser(description='ensemble feature selection experiments')
args = parse_arguments(parser)

n_procs = cpu_count()
n_jobs = n_procs if args.n_jobs == -1 or args.n_jobs > n_procs else args.n_jobs
logger.info('n_jobs selected: {}'.format(n_jobs))

for idx in range(args.n_seeds):

    if args.modality == 'statistics':
        features, y_label, v_img_name = load_stats_by_dataset_name(args.dataset)
        features_pre = remove_null_variables(features, threshold=0.6)
        features_pre['img_name'] = v_img_name
        X_train, X_test, Y_train, Y_test = train_test_split(features_pre, y_label, stratify=y_label, test_size=0.2, random_state=idx)
        
    if args.modality == 'embeddings':
        features, y_label, v_img_name = load_stats_by_dataset_name(args.dataset)
        embeddings = load_embeddings_by_dataset_name(args.dataset, idx)
        embeddings['img_name'] = v_img_name
        X_train, X_test, Y_train, Y_test = train_test_split(embeddings, y_label, stratify=y_label, test_size=0.2, random_state=idx)

    v_img_name_train = X_train['img_name'].values
    X_train = X_train.drop(columns=['img_name'])

    binary_vars = [col for col in X_train.columns if set(X_train[col].dropna().unique()) == {0, 1}]

    if binary_vars and not X_train[binary_vars].empty:
        mode_values = X_train[binary_vars].mode().iloc[0]
        X_train[binary_vars] = X_train[binary_vars].fillna(mode_values)

    numeric_vars = [col for col in X_train.columns if col not in binary_vars]
    X_train[numeric_vars] = X_train[numeric_vars].fillna(X_train[numeric_vars].mean())
    print("binary_vars:", binary_vars)
    print("Moda de X_train[binary_vars]:", X_train[binary_vars].mode())

    rus = RandomUnderSampler(sampling_strategy='all', random_state=idx)
    X_res, y_res = rus.fit_resample(X_train, Y_train)

    scaler = StandardScaler()
    X_train[numeric_vars] = scaler.fit_transform(X_train[numeric_vars])
    # X_test[numeric_vars] = scaler.transform(X_test[numeric_vars])

    y_res = np.array(y_res)
    v_feature_names = X_train.columns

    list_vars_categorical, list_vars_numerical = fs.get_categorical_numerical_names(X_res)
    Z_selected, Z_scores = fs.compute_zmatrix_bootstrap(X_res,
                                                        y_res,
                                                        args.fs,
                                                        v_feature_names,
                                                        list_vars_categorical,
                                                        list_vars_numerical,
                                                        M=args.n_boots,
                                                        n_jobs=n_jobs)

    df_ensemble_voting_sorted, df_ensemble_mean_sorted = fs.run_ensemble_agg(X_res,
                                                                             Z_selected,
                                                                             Z_scores,
                                                                             args.agg_func)

    print(df_ensemble_voting_sorted)
    csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_FS_SCORES, args.modality, f'score_{args.dataset}_{args.fs}_{idx}.csv'))

    df_ensemble_voting_sorted.to_csv(csv_file_path, index=False)
    
    # file_path = str(Path.joinpath(consts.PATH_PROJECT_FS_FIGURES, args.modality, f'graphic_fs_{args.dataset}_{args.fs}_{idx}.pdf'))
    # generate_feature_importance_plot(df_ensemble_voting_sorted, file_path)

    select_optimal_features_ensemblefs(X_train,
                                       Y_train,
                                       df_ensemble_voting_sorted,
                                       v_img_name_train,
                                       args.estimator,
                                       args.scoring_estimator,
                                       idx
                                       )



