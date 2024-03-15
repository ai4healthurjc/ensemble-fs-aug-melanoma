import argparse
from pathlib import Path
from multiprocessing import cpu_count
import coloredlogs
import logging
import coloredlogs

from util.loader import load_synthetic_stats_by_dataset_name
import util.consts as consts
from util.bootstrap_utils import compute_boostrap
from util.plotter import bootstrap_graphic

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
    parser.add_argument('--bootstrap_graphic', default='yes', type=str)


    return parser.parse_args()


parser = argparse.ArgumentParser(description='ensemble feature selection experiments')
args = parse_arguments(parser)

# n_procs = cpu_count()
# n_jobs = n_procs if args.n_jobs == -1 or args.n_jobs > n_procs else args.n_jobs
# logger.info('n_jobs selected: {}'.format(n_jobs))

features, names_features = load_synthetic_stats_by_dataset_name(args.dataset, args.type_encoding, args.type_sampling, args.oversampler, args.classifier, args.ir, args.n_epochs, args.batch_size, args.seed)

file_path = str(Path.joinpath(consts.PATH_PROJECT_BOOTSTRAP, f'df_var_importance_bootstrap_seed_{args.seed}_statistics_{args.dataset}_{args.classifier}_{args.ir}.csv'))
bootstrap_df = compute_boostrap(features, names_features, 'label', file_path, flag_save_importance=True)

graphic_filepath = str(Path.joinpath(consts.PATH_PROJECT_BOOTSTRAP, f'graphic_df_var_importance_bootstrap_seed_{args.seed}_statistics_{args.dataset}_{args.classifier}_{args.ir}.pdf'))

if args.bootstrap_graphic=='yes':
    bootstrap_graphic(bootstrap_df, graphic_filepath)



