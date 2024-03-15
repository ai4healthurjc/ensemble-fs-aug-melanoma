import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#import pyreadstat
import shap

import argparse
from pathlib import Path
from multiprocessing import cpu_count
import coloredlogs
import logging

from util.loader import load_synthetic_stats_by_dataset_name, load_model_stats, load_stats_by_dataset_name_partition
import util.consts as consts


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def parse_arguments(parser):
    parser.add_argument('--dataset', default='ph2', type=str)
    # parser.add_argument('--modality', default='statistics', type=str)
    parser.add_argument('--type_sampling', default='over', type=str)
    parser.add_argument('--oversampler', default='ctgan', type=str)
    parser.add_argument('--type_encoding', default='standard', type=str)
    parser.add_argument('--n_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--classifier', default='dt', type=str)
    parser.add_argument('--ir', default=0.7, type=float)
    parser.add_argument('--seed', default=0, type=int)

    return parser.parse_args()


parser = argparse.ArgumentParser(description='ensemble feature selection experiments')
args = parse_arguments(parser)

n_procs = cpu_count()


model=load_model_stats (args.dataset,  args.type_encoding, args.type_sampling, args.oversampler, args.classifier, args.ir, args.n_epochs, args.batch_size, args.seed)

features, names_features = load_synthetic_stats_by_dataset_name(args.dataset,  args.type_encoding, args.type_sampling, args.oversampler, args.classifier, args.ir, args.n_epochs, args.batch_size, args.seed)


y_label_train=features['label']
features_train = features.drop(columns=['label'])

statistics, y_label=load_stats_by_dataset_name_partition(args.dataset, 'statistics', args.seed)

X_train, X_test, y_train, y_test = train_test_split(statistics, y_label, test_size=0.2,  stratify=y_label, random_state=args.seed)

binary_vars = [col for col in X_train.columns if set(X_train[col].dropna().unique()) == {0, 1}]

# # Reemplazar los valores nulos de las variables numéricas por la media
numeric_vars = [col for col in X_train.columns if col not in binary_vars]


# Normalizar las variables numéricas utilizando StandardScaler
scaler = StandardScaler()
X_train[numeric_vars]= scaler.fit_transform(X_train[numeric_vars])
X_test[numeric_vars] = scaler.transform(X_test[numeric_vars])


explainer=shap.Explainer(model, features_train)
shap_values=explainer.shap_values(X_test)
print(shap_values)

nombres=X_test.columns.tolist()

fig=plt.figure()

shap.summary_plot(shap_values, X_test, feature_names=nombres)
plt.ylim(bottom=0)  # Establecer el límite inferior a 0 o al valor deseado

# plt.tight_layout()
graphic_filepath = str(Path.joinpath(consts.PATH_PROJECT_SHAP, f'shap_{args.dataset}_{args.classifier}_{args.ir}_statistics_seed_{args.seed}.pdf'))

fig.savefig(graphic_filepath, format='pdf')

plt.close(fig)
