Ensemble-fs-tabular-augmentation-melanoma
====

Repository for reproducibility results for the paper "Ensemble feature selection and tabular data augmentation with generative adversarial networks to enhance cutaneous melanoma identification and interpretability", which is focused on combining ensemble feature selection and tabular generative adversarial network (CTGAN) to classify skin lesions in imbalanced datasets.

## Installation and setup

To download the source code, you can clone it from the Github repository.
```console
git clone git@github.com:ai4healthurjc/ensemble-fs-aug-melanoma.git
```

Before installing libraries, ensuring that a Python virtual environment is activated (using conda o virtualenv). To install Python libraries run: 

```console
pip install -r requirements_venv.txt 
```

If you have any issue with skrebate, please install the following modified version:
```console
pip install git+https://github.com/cdchushig/scikit-rebate.git@1efbe530a46835c86f2e50f17342541a3085be9c
```

## Download dermoscopy images from public datasets

Datasets with dermoscopy images are publicly available in the following websites:

1. [Link to PH2 dataset](https://www.fc.up.pt/addi/ph2%20database.html)
2. [Link to Derm7pt dataset](https://derm.cs.sfu.ca/Welcome.html)

To replicate the results, download images from PH2 and Derm7pt datasets. After downloading data, you have to put folders (PH2, Derm7pt) in **data** folder. Specifically: 
- data/raw/ph2/images **(images)**, data/raw/ph2/masks_dobleunet **(masks)**;
- data/raw/derm7pt/images **(images)**, data/raw/derm7pt/masks_dobleunet **(masks)**.  

## To obtain different results of data-driven models

To perform image feature extraction:
```console
python src/image_feature_extractor.py --dataset=ph2 --batch_size=10 --n_jobs=4 --extract_features=True
```

To perform embedding feature extraction using ResNet50 CNN model:
```console
python src/train_resnet.py --dataset=ph2 --n_seeds=5 --n_epochs=100 --batch_size=32 --n_epochs_early_stop=15 --patience_lr=5 --initial_lr=1e-4 --extract_embeddings=yes
```

To perform ensemble feature selection :
```console
python src/feature_selection.py --dataset=ph2 --modality=statistics --n_boots=100 --fs=relief --agg_func=mean --n_seeds=5
```

To train ML models with CTGAN and generate plots for metrics obtained with mean and standard deviation:
```console
python src/over.py --dataset=ph2 --modality=statistics --classifier=dt --type_sampling=over --oversampler=ctgan --type_encoding=standard --n_seeds=5 --n_epochs=100 --batch_size=50 --imbalance_ratio=0,7 --graphics=yes
```

To perform bootstrap resampling as a nonparametric statistical test and provide statistical characterization of the selected features by ensemble feature selection:
```console
python src/bootstrap_interpretability.py --dataset=ph2 --type_sampling=over --oversampler=ctgan --type_encoding=standard  --n_epochs=100 --batch_size=50 --classifier=dt --ir=0.7 --seed=0 --bootstrap_graphic=yes
```

To provide interpretability with SHAP on the test set:
```console
python src/shap_interpretability.py --dataset=ph2 --type_sampling=over --oversampler=ctgan --type_encoding=standard  --n_epochs=100 --batch_size=50 --classifier=dt --ir=0.7 --seed=0 
```


