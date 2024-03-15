import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from imblearn.under_sampling import RandomUnderSampler
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from keras.applications import ResNet50
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from tensorflow.keras import Model, layers
from keras import layers
from keras.optimizers import Adam
from util.evaluation_metrics import compute_classification_prestations_cnn

import logging
import coloredlogs

from util.loader import load_images, load_stats_by_dataset_name
import util.consts as consts

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def build_model(backbone, lr=5e-4):

    for layer in backbone.layers:
        layer.trainable = False

    specific_layer = backbone.get_layer('conv5_block3_3_conv')

    specific_layer.trainable = True

    output = specific_layer.output

    x = layers.GlobalAveragePooling2D()(output)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    output = layers.Dense(2, activation='softmax')(x)

    model = Model(inputs=backbone.input, outputs=output)

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=lr),
        metrics=['accuracy']
    )

    return model


def parse_arguments(parser):
    parser.add_argument('--dataset', default='ph2', type=str)
    parser.add_argument('--extract_embeddings', default=False, type=bool)
    parser.add_argument('--n_seeds', default=5, type=int)
    parser.add_argument('--n_jobs', default=1, type=int)
    parser.add_argument('--n_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--n_epochs_early_stop', default=15, type=int)
    parser.add_argument('--patience_lr', default=5, type=int)
    parser.add_argument('--initial_lr', default=1e-4, type=float)

    return parser.parse_args()


parser = argparse.ArgumentParser(description='ensemble feature selection experiments')
args = parse_arguments(parser)

n_procs = cpu_count()
n_jobs = n_procs if args.n_jobs == -1 or args.n_jobs > n_procs else args.n_jobs
logger.info('n_jobs selected: {}'.format(n_jobs))

images = load_images(args.dataset)
features, y_label, _ = load_stats_by_dataset_name(args.dataset)

BATCH_SIZE = args.batch_size

train_generator = ImageDataGenerator(
        zoom_range=2, 
        rotation_range=90,
        horizontal_flip=True, 
        vertical_flip=True, 
    )

resnet = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

model = build_model(resnet, lr=args.initial_lr)
model.summary()

for layer in model.layers:
    print(layer.name, layer.trainable)
    
learn_control = ReduceLROnPlateau(monitor='val_accuracy', patience=args.patience_lr,
                                  verbose=1, factor=0.2, min_lr=1e-7)

early = EarlyStopping(monitor="val_accuracy", 
                      mode="max", 
                      patience=args.n_epochs_early_stop) 

list_acc_values = []
list_specificity_values = []
list_recall_values = []
list_auc_values = []

for idx in range(args.n_seeds):

    X_train, X_test, y_train, y_test = train_test_split(images, y_label, test_size=0.2,  stratify=y_label, random_state=idx)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.15, random_state=idx)
    rus = RandomUnderSampler(sampling_strategy='all', random_state=idx)

    X_train = np.array(X_train)
    X_val = np.array(X_val)

    X_train_flattened = X_train.reshape(X_train.shape[0], -1)
    X_val_flattened = X_val.reshape(X_val.shape[0], -1)

    X_train_under, y_train_under = rus.fit_resample(X_train_flattened, y_train)
    X_val_under, y_val_under = rus.fit_resample(X_val_flattened, y_val)

    unique_values_train, counts_train = np.unique(y_train_under, return_counts=True)
    unique_values_test, counts_test = np.unique(y_test, return_counts=True)

    print("Train:", dict(zip(unique_values_train, counts_train)))
    print("Test:", dict(zip(unique_values_test, counts_test)))

    X_train_under = X_train_under.reshape(-1, 224, 224, 3)
    X_val_under = X_val_under.reshape(-1, 224, 224, 3)

    X_train = np.array(X_train_under)
    X_val = np.array(X_val_under)
    X_test = np.array(X_test)
    
    Y_train = to_categorical(y_train_under, num_classes= 2)
    Y_val = to_categorical(y_val_under, num_classes= 2)
    Y_test = to_categorical(y_test, num_classes= 2)

    model_file_path = str(
        Path.joinpath(consts.PATH_PROJECT_MODELS, 'resnet', f'weights_resnet_cnn_{args.dataset}_idx{idx}.hdf5')
    )

    checkpoint = ModelCheckpoint(model_file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    model = build_model(resnet, lr=args.initial_lr)
    model.summary()

    for layer in model.layers:
        print(layer.name, layer.trainable)

    history = model.fit_generator(
        train_generator.flow(X_train, Y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=X_train.shape[0] / BATCH_SIZE,
        epochs=args.n_epochs,
        validation_data=(X_val, Y_val),
        callbacks=[learn_control, early, checkpoint]
    )

    hist_df = pd.DataFrame(history.history) 
    hist_file_path = str(Path.joinpath(consts.PATH_PROJECT_MODELS, 'resnet',
                                       f'history_model_resnet_cnn_{args.dataset}_idx{idx}.csv')
                         )
    
    with open(hist_file_path, mode='w') as f:
        hist_df.to_csv(f)

    model.load_weights(model_file_path)
    model.save(str(Path.joinpath(consts.PATH_PROJECT_MODELS, 'resnet', f"model_resnet_cnn_{args.dataset}_idx{idx}.h5")))

    if args.extract_embeddings:
        emb = Model(inputs=model.input, outputs=model.get_layer(f'global_average_pooling2d_{idx+1}').output)
        images = np.array(images)
        X_emb = emb.predict(images)
        
        column_names = [str(i) for i in range(2048)]
        df_emb = pd.DataFrame(X_emb, columns=column_names)
        emb_file_path = str(Path.joinpath(consts.PATH_PROJECT_EMBEDDINGS, f'embeddings_{idx}_{args.dataset}.csv'))
        df_emb.to_csv(emb_file_path, index=False)
    else:
        continue

    Y_pred = model.predict(X_test)
    acc_val, specificity_val, recall_val, roc_auc_val = compute_classification_prestations_cnn(Y_test, Y_pred)

    print(acc_val, recall_val, specificity_val, roc_auc_val)

    list_acc_values.append(acc_val)
    list_specificity_values.append(specificity_val)
    list_recall_values.append(recall_val)
    list_auc_values.append(roc_auc_val)

mean_std_specificity = np.mean(list_specificity_values), np.std(list_specificity_values)
mean_std_accuracy = np.mean(list_acc_values), np.std(list_acc_values)
mean_std_recall = np.mean(list_recall_values), np.std(list_recall_values)
mean_std_auc = np.mean(list_auc_values), np.std(list_auc_values)

print('accuracy:', mean_std_accuracy)
print('specificity:', mean_std_specificity)
print('recall:', mean_std_recall)
print('AUC:', mean_std_auc)

exp_name = '{}+{}+{}'.format(args.dataset, 'cnn_resnet', args.n_seeds)

new_row_auc = {
    'model': exp_name,
    'eval_metric': 'auc',
    'mean': mean_std_auc[0],
    'std': mean_std_auc[1]
}

new_row_sensitivity = {
    'model': exp_name,
    'eval_metric': 'sensitivity',
    'mean': mean_std_recall[0],
    'std': mean_std_recall[1]
}

new_row_specificity = {
    'model': exp_name,
    'eval_metric': 'specificity',
    'mean': mean_std_specificity[0],
    'std': mean_std_specificity[1]
}

new_row_accuracy = {
    'model': exp_name,
    'eval_metric': 'accuracy',
    'mean': mean_std_accuracy[0],
    'std': mean_std_accuracy[1]
}

csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_METRICS, 'resnet', 'metrics_classification_resnet_{}.csv'.format(args.dataset)))

if os.path.exists(csv_file_path):
    try:
        df_metrics_classification = pd.read_csv(csv_file_path)
    except pd.errors.EmptyDataError:
        df_metrics_classification = pd.DataFrame()
else:
    df_metrics_classification = pd.DataFrame()

df_metrics_classification = pd.concat([df_metrics_classification, pd.DataFrame([new_row_auc])], ignore_index=True)
df_metrics_classification = pd.concat([df_metrics_classification, pd.DataFrame([new_row_sensitivity])], ignore_index=True)
df_metrics_classification = pd.concat([df_metrics_classification, pd.DataFrame([new_row_specificity])], ignore_index=True)
df_metrics_classification = pd.concat([df_metrics_classification, pd.DataFrame([new_row_accuracy])], ignore_index=True)
df_metrics_classification.to_csv(csv_file_path, index=False)




   

