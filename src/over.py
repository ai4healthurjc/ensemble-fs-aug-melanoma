import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from ctgan import CTGAN
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import logging
import coloredlogs
from locale import atof, setlocale, LC_NUMERIC
from datetime import datetime
from collections import Counter
from operator import itemgetter

from util.loader import load_stats_by_dataset_name_partition, split_train_test, normalize_dataframe
from util.plotter import generate_graphic_metrics
from util.evaluation_metrics import compute_classification_prestations
import util.consts as consts

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

list_conventional_oversamplers = ['rus', 'smoten']
list_interpretable_models = ['dt', 'lasso', 'rf', 'svm']

setlocale(LC_NUMERIC, '')


def save_model_artifacts(oversampler_model,
                         oversampler_name,
                         generic_name_partition,
                         df_x_train_class_min_real,
                         df_x_train_class_min_syn
                         ):

    pickle.dump(oversampler_model, open(str(Path.joinpath(consts.PATH_PROJECT_MODELS, args.modality, 
                                                'model_oversampler_{}.sav'.format(generic_name_partition))), 'wb'))
    # oversampler_model.save(
    #     str(Path.joinpath(consts.PATH_PROJECT_MODELS, 'model_oversampler_{}.h5'.format(generic_name_partition)))
    # )

    df_x_train_class_min_syn.to_csv(
        str(Path.joinpath(consts.PATH_PROJECT_SYNTHETIC_DATA, oversampler_name, args.modality,
                          'df_synthetic_class_min_{}.csv'.format(generic_name_partition))),
        index=False
    )

    df_x_train_class_min_real.to_csv(
        str(Path.joinpath(consts.PATH_PROJECT_SYNTHETIC_DATA,
                          oversampler_name, args.modality,
                          'df_real_class_min_{}.csv'.format(generic_name_partition))),
        index=False
    )


def join_real_synthetic_data(x_real, x_synthetic, y_train, num_samples_class_maj, num_samples_class_min_syn, id_label_min):

    x_train_resampled = np.concatenate((x_real, x_synthetic), axis=0)
    y_train_resampled = np.concatenate((y_train, np.reshape(np.full((num_samples_class_min_syn, 1), id_label_min), -1)))

    num_samples_class_min_real = y_train.shape[0]

    y_meta_class_min_real = np.reshape(np.full((num_samples_class_min_real, 1), consts.LABEL_CLASS_MIN_REAL), -1)
    y_meta_class_min_syn = np.reshape(np.full((num_samples_class_min_syn, 1), consts.LABEL_CLASS_MIN_SYNTHETIC), -1)
    y_meta_class_min_real_syn = np.concatenate((y_meta_class_min_real, y_meta_class_min_syn))
    y_meta_class_maj_real = np.reshape(np.full((num_samples_class_maj, 1), consts.LABEL_CLASS_MAJ_REAL), -1)

    y_meta_class_maj_min = np.concatenate((y_meta_class_maj_real, y_meta_class_min_real_syn))

    return x_train_resampled, y_train_resampled, y_meta_class_maj_min


def get_x_train_classes_and_ids(x_train, y_train, v_column_names, imbalance_ratio):

    c_items = Counter(y_train)
    id_label_min, num_samples_min = min(c_items.items(), key=itemgetter(1))
    id_label_maj, num_samples_maj = max(c_items.items(), key=itemgetter(1))

    df_x_train_with_label = pd.DataFrame(x_train, columns=v_column_names)
    df_x_train_with_label['label'] = y_train

    df_x_train_class_min_with_label = df_x_train_with_label[df_x_train_with_label.loc[:, 'label'] == id_label_min]
    df_x_train_class_maj_with_label = df_x_train_with_label[df_x_train_with_label.loc[:, 'label'] == id_label_maj]

    df_x_train_class_min = df_x_train_class_min_with_label.iloc[:, :-1]
    df_x_train_class_maj = df_x_train_class_maj_with_label.iloc[:, :-1]

    y_train_class_min = df_x_train_class_min_with_label.loc[:, 'label']
    y_train_class_maj = df_x_train_class_maj_with_label.loc[:, 'label']

    num_synthetic_samples_required = int(imbalance_ratio * num_samples_maj)
    num_synthetic_samples = num_synthetic_samples_required - num_samples_min

    return df_x_train_class_min, y_train_class_min, df_x_train_class_maj, y_train_class_maj, num_synthetic_samples, id_label_min, id_label_maj


def parse_arguments(parser):
    parser.add_argument('--dataset', default='ph2', type=str)
    parser.add_argument('--modality', default='statistics', type=str)
    parser.add_argument('--classifier', default='dt', type=str)
    parser.add_argument('--type_sampling', default='over', type=str)
    parser.add_argument('--oversampler', default='ctgan', type=str)
    parser.add_argument('--type_encoding', default='standard', type=str)
    parser.add_argument('--graphics', default=False, type=bool)
    parser.add_argument('--email', default='test@test.com', type=str)
    parser.add_argument('--n_seeds', default=5, type=int)
    parser.add_argument('--n_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--imbalance_ratio', default='0.5', type=str)
    parser.add_argument('--weight', default='0.0', type=str)
    parser.add_argument('--verbose', default=False, type=bool)
    parser.add_argument('--flag_fs', default=True, type=bool)
    parser.add_argument('--cuda', default=False, type=bool)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='Oversampling experiments')
args = parse_arguments(parser)

imbalance_ratio = atof(args.imbalance_ratio)

list_acc_values = []
list_specificity_values = []
list_recall_values = []
list_auc_values = []
list_feature_importance = []

generic_name = '{}_{}_{}_{}_{}_ir_{}_epoch_{}_batch_{}'.format(args.dataset,
                                                               args.type_encoding,
                                                               args.type_sampling,
                                                               args.oversampler,
                                                               args.classifier,
                                                               imbalance_ratio,
                                                               args.n_epochs,
                                                               args.batch_size
                                                               )

for idx in range(args.n_seeds):

    generic_name_partition = '{}_seed_{}'.format(generic_name, idx)
    logger.info('Experiment with: {}'.format(generic_name_partition))

    df_x_features, y_label = load_stats_by_dataset_name_partition(args.dataset, args.modality, idx)
    df_x_features_scaled = normalize_dataframe(df_x_features, args.type_encoding)

    v_column_names = df_x_features_scaled.columns.values
    x_train, y_train, x_test, y_test = split_train_test(df_x_features_scaled.copy().values,
                                                        y_label,
                                                        test_size=0.2,
                                                        seed_value=idx
                                                        )

    df_x_train_class_min_real, y_train_class_min_real, df_x_train_class_maj, y_train_class_maj, n_samples_syn, id_label_min, id_label_maj = get_x_train_classes_and_ids(
        x_train, y_train, v_column_names, imbalance_ratio)

    num_samples_maj = df_x_train_class_maj.shape[0]
    num_samples_min = df_x_train_class_min_real.shape[0]

    # seed_value = idx
    # torch.manual_seed(seed_value)
    # np.random.seed(seed_value)

    # for undersampling_idx in range(5):
    #     # Semilla para el submuestreo (undersampling)
    #     seed_value_undersampling = undersampling_idx  # Cambia la semilla en cada iteración
    #     torch.manual_seed(seed_value_undersampling)
    #     np.random.seed(seed_value_undersampling)

    if args.oversampler == 'ctgan':
        logger.info('Resampling with CTGAN - oversampling')
        oversampler_model = CTGAN(epochs=args.n_epochs,
                                  batch_size=args.batch_size,
                                  cuda=args.cuda,
                                  verbose=True
                                  )
        oversampler_model.fit(df_x_train_class_min_real, v_column_names)
    else:
        logger.info('Resampling with RUS - undersampling')
        oversampler_model = RandomUnderSampler(random_state=idx, sampling_strategy=imbalance_ratio)

    if args.oversampler in list_conventional_oversamplers:
        x_train_maj_resampled_min, y_train_maj_resampled_min = oversampler_model.fit_resample(x_train, y_train)
        y_meta_train_with_both_min_maj = y_train_maj_resampled_min
    else:
        df_x_train_class_min_syn = oversampler_model.sample(n_samples_syn)

        save_model_artifacts(oversampler_model, args.oversampler, generic_name_partition, df_x_train_class_min_real, df_x_train_class_min_syn)

        x_train_resampled_class_min, y_train_resampled_class_min, y_meta_train_with_both_min_maj = join_real_synthetic_data(
            df_x_train_class_min_real.values,
            df_x_train_class_min_syn.values,
            y_train_class_min_real,
            num_samples_maj,
            n_samples_syn,
            id_label_min
        )

        x_train_maj_resampled_min = np.concatenate((df_x_train_class_maj.values, x_train_resampled_class_min), axis=0)
        y_train_maj_resampled_min = np.concatenate((y_train_class_maj, y_train_resampled_class_min))

    if args.type_sampling == 'hybrid':
        logger.info('Training with hybrid approach')

        rus = RandomUnderSampler(random_state=idx)
        x_train_resampled, y_train_resampled = rus.fit_resample(x_train_maj_resampled_min, y_train_maj_resampled_min)
        v_indices_resampled = rus.sample_indices_
        y_label_real_syn_total_resampled = y_meta_train_with_both_min_maj[v_indices_resampled]

    elif args.type_sampling == 'over':
        logger.info('Training with oversampling approach')

        x_train_resampled = x_train_maj_resampled_min
        y_train_resampled = y_train_maj_resampled_min
        y_label_real_syn_total_resampled = y_meta_train_with_both_min_maj

    else:
        logger.info('Training with rus/smoten')
        x_train_resampled, y_train_resampled = oversampler_model.fit_resample(x_train, y_train)
        y_label_real_syn_total_resampled = [consts.LABEL_CLASS_MAJ_REAL if label == 0 else consts.LABEL_CLASS_MIN_REAL for label in y_train_resampled]

    logger.info('Resampled x_train dataset shape {}'.format(x_train_resampled.shape))
    logger.info('Resampled y_train dataset shape {}'.format(Counter(y_train_resampled)))

    if args.type_encoding == 'target':
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_resampled_scaled = scaler.transform(x_train_resampled)
        x_test_scaled = scaler.transform(x_test)
    else:
        x_train_resampled_scaled = x_train_resampled
        x_test_scaled = x_test

    df_x_train_resampled_scaled = pd.DataFrame(x_train_resampled_scaled, columns=v_column_names)
    df_x_train_resampled_scaled.to_csv(Path.joinpath(consts.PATH_PROJECT_SYNTHETIC_DATA, args.oversampler,
                                                     args.modality,
                                                     'df_x_train_resampled_scaled_{}.csv'.format(generic_name_partition)), index=False)

    y_labels_maj_min = np.c_[y_train_resampled, y_label_real_syn_total_resampled]
    df_y_labels_maj_min = pd.DataFrame(y_labels_maj_min, columns=['label', 'label_meta'])

    df_y_labels_maj_min.to_csv(Path.joinpath(consts.PATH_PROJECT_SYNTHETIC_DATA, args.oversampler, args.modality,
                                            'df_y_train_resampled_scaled_{}.csv'.format(generic_name_partition)),
                            index=False)

    if args.classifier == 'svm':
        model_clf_generic = LinearSVC(max_iter=100000, random_state=idx)

        hyperparameter_space = {
            'C': np.logspace(-0.9, 0.9, 10)
            # 'C': [0.1, 1, 10, 100, 1000],
        }

        grid_cv = GridSearchCV(estimator=model_clf_generic, param_grid=hyperparameter_space, scoring='roc_auc', cv=5)
        grid_cv.fit(x_train_resampled_scaled, y_train_resampled)
        clf_model = grid_cv.best_estimator_
        clf_model.fit(x_train_resampled_scaled, y_train_resampled)
        y_pred = clf_model.predict(x_test_scaled)

        logger.info('Training with {}, params: {}, best_params: {}'.format(args.classifier, clf_model.get_params(),
                                                                        grid_cv.best_params_))

        list_feature_importance.append(clf_model.coef_.reshape((-1)))

    elif args.classifier == 'dt':
        lenght_train = len(x_train_resampled_scaled)
        lenght_15_percent_val = int(0.15 * lenght_train)
        lenght_20_percent_val = int(0.20 * lenght_train)

        tuned_parameters = {
            'max_depth': range(2, 8, 2),
            'min_samples_split': range(lenght_15_percent_val, lenght_20_percent_val),
        }

        model_tree = DecisionTreeClassifier(random_state=idx)

        grid_cv = GridSearchCV(estimator=model_tree, param_grid=tuned_parameters, scoring='roc_auc', cv=5)
        grid_cv.fit(x_train_resampled_scaled, y_train_resampled)
        clf_model = grid_cv.best_estimator_

        clf_model.fit(x_train_resampled_scaled, y_train_resampled)
        y_pred = clf_model.predict(x_test_scaled)
        logger.info('Training with dt {}'.format(grid_cv.best_params_))

        list_feature_importance.append(clf_model.feature_importances_)

    elif args.classifier == 'knn':

        hyperparams_space = {
            'n_neighbors': range(1, 11)
        }        

        grid_cv = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=hyperparams_space, scoring='roc_auc', cv=5)
        grid_cv.fit(x_train_resampled_scaled, y_train_resampled)
        clf_model = grid_cv.best_estimator_
        clf_model.fit(x_train_resampled_scaled, y_train_resampled)

        y_pred = clf_model.predict(x_test_scaled)

        logger.info('Training with knn {}'.format(grid_cv.best_params_))

        list_feature_importance.append(0.0)

    elif args.classifier == 'lasso':
        hyperparams_space = {
            "C": np.logspace(-1.5, 0.4, 10),
            "penalty":["l1"]
        }

        #model_clf = Lasso(max_iter=1000, random_state=seed_value)
        model_clf = LogisticRegression(solver='liblinear',max_iter=1000, random_state=idx)

        grid_cv = GridSearchCV(estimator=model_clf, param_grid=hyperparams_space, scoring='roc_auc', cv=5)
        grid_cv.fit(x_train_resampled_scaled, y_train_resampled)

        clf_model = grid_cv.best_estimator_
        clf_model.fit(x_train_resampled_scaled, y_train_resampled)
        y_pred = clf_model.predict(x_test_scaled)

        for i in range(len(y_pred)):
            if y_pred[i] >= 0.5:
                y_pred[i] = 1.0
            else:
                y_pred[i] = 0.0

        list_feature_importance.append(clf_model.coef_)

        logger.info('Training with lasso {}'.format(clf_model.get_params()))

    # save_parameters(generic_name_partition, clf_model.get_params())
    # save_samples_wrong_classified(x_test_scaled, y_test, y_pred, args.oversampler, generic_name_partition)

    acc_val, specificity_val, recall_val, roc_auc_val = compute_classification_prestations(y_test, y_pred)

    list_acc_values.append(acc_val)
    list_specificity_values.append(specificity_val)
    list_recall_values.append(recall_val)
    list_auc_values.append(roc_auc_val)

    pickle.dump(clf_model, open(str(Path.joinpath(consts.PATH_PROJECT_MODELS, args.modality,
                                                'model_clf_{}.sav'.format(generic_name_partition))), 'wb'))

mean_std_specificity = np.mean(list_specificity_values), np.std(list_specificity_values)
mean_std_accuracy = np.mean(list_acc_values), np.std(list_acc_values)
mean_std_recall = np.mean(list_recall_values), np.std(list_recall_values)
mean_std_auc = np.mean(list_auc_values), np.std(list_auc_values)

print('accuracy:', mean_std_accuracy)
print('specificity:', mean_std_specificity)
print('recall:', mean_std_recall)
print('AUC:', mean_std_auc)

# metrics = mean_std_accuracy, mean_std_recall, mean_std_specificity, mean_std_auc
current_time = datetime.now()
str_date_time = current_time.strftime("%m/%d/%Y")
# df_metrics = pd.DataFrame(metrics, columns=['mean', 'std'], index=['accuracy', 'recall', 'specificity', 'auc'])
# df_metrics.to_csv(str(Path.joinpath(consts.PATH_PROJECT_METRICS, args.oversampler, args.modality,
#                                     'metrics_{}.csv'.format(generic_name))))

exp_name = '{}+{}+{}'.format(args.oversampler, args.classifier, imbalance_ratio)
exp_name = '{}+{}'.format(exp_name, 'fs') if args.flag_fs else exp_name

new_row_auc = {'run_date': str_date_time,
               'model': exp_name,
               'eval_metric': 'auc',
               'type_encoding': args.type_encoding,
               'type_sampling': args.type_sampling,
               'mean': mean_std_auc[0],
               'std': mean_std_auc[1]}

new_row_sensitivity = {'run_date': str_date_time,
                       'model': exp_name,
                       'eval_metric': 'sensitivity',
                       'type_encoding': args.type_encoding,
                       'type_sampling': args.type_sampling,
                       'mean': mean_std_recall[0],
                       'std': mean_std_recall[1]}
new_row_specificity = {'run_date': str_date_time,
                       'model': exp_name,
                       'eval_metric': 'specificity',
                       'type_encoding': args.type_encoding,
                       'type_sampling': args.type_sampling,
                       'mean': mean_std_specificity[0],
                       'std': mean_std_specificity[1]}

new_row_accuracy = {'run_date': str_date_time,
                       'model': exp_name,
                       'eval_metric': 'accuracy',
                       'type_encoding': args.type_encoding,
                       'type_sampling': args.type_sampling,
                       'mean': mean_std_accuracy[0],
                       'std': mean_std_accuracy[1]}

csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_METRICS, args.oversampler, args.modality,
                                    f'metrics_{args.dataset}_{args.modality}_{args.type_encoding}_{args.type_sampling}_{args.oversampler}_epochs_{args.n_epochs}.csv'))

if os.path.exists(csv_file_path):
    try:
        df_metrics_classification = pd.read_csv(csv_file_path)
    except pd.errors.EmptyDataError:
        # El archivo está vacío, puedes manejarlo como desees
        df_metrics_classification = pd.DataFrame()
else:
    df_metrics_classification = pd.DataFrame()

df_metrics_classification = pd.concat([df_metrics_classification, pd.DataFrame([new_row_auc])], ignore_index=True)
df_metrics_classification = pd.concat([df_metrics_classification, pd.DataFrame([new_row_sensitivity])], ignore_index=True)
df_metrics_classification = pd.concat([df_metrics_classification, pd.DataFrame([new_row_specificity])], ignore_index=True)
df_metrics_classification = pd.concat([df_metrics_classification, pd.DataFrame([new_row_accuracy])], ignore_index=True)

# df_metrics_classification = df_metrics_classification.append(new_row_auc, ignore_index=True)
# df_metrics_classification = df_metrics_classification.append(new_row_sensitivity, ignore_index=True)
df_metrics_classification.to_csv(csv_file_path, index=False)

print(df_metrics_classification)

if args.graphics:
    graphic_filepath = str(Path.joinpath(consts.PATH_PROJECT_METRICS,
                                         args.oversampler, args.modality, 'figures',
                                         f'graphic_{args.oversampler}_{args.modality}_{args.dataset}.pdf')
                           )
    print(generate_graphic_metrics(df_metrics_classification, graphic_filepath, args.type_sampling))
    
