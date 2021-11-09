# Linearized HSMM (LHSMM) for Video Modeling

This repo provides instructions for reproducing our LHSMM results in ``Linearized Structured Models''. Note that our repo is based on [action-segmentation](https://github.com/dpfried/action-segmentation) (Fried et al. 2020, [Learning to Segment Actions from Observation and Narration](https://arxiv.org/abs/2005.03684)).


## Dependencies

* Python 3.8
* PyTorch 1.8

First, we need to install our variant of [pytorch-struct](https://github.com/harvardnlp/pytorch-struct) to use our efficient HSMM inference algorithm for linear kernels:

```
cd pytorch_struct
python setup.py install
```

## Data Setup

1. Download and unpack the CrossTask dataset of Zhukov et al.:

```
cd data
mkdir crosstask
cd crosstask
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_features.zip
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_constraints.zip
unzip '*.zip'
```

2. Preprocess the features with PCA. In the repository's root folder, run

```
PYTHONPATH="src/":$PYTHONPATH python src/data/crosstask.py
```

This should generate the folder `data/crosstask/crosstask_processed/crosstask_primary_pca-200_with-bkg_by-task`.

## Training and Evaluation

The below command kicks off training. Note that both validation and test log likelihoods are printed so we don't have a separate test script. Since we are dealing with 1024 features, training might take up to 2 days (tested on a 32GB Nvidia V100 GPU).

```
CUDA_VISIBLE_DEVICES=0 ./run_crosstask_i3d-resnet-audio.sh compound_unsup_semimarkov_tss_gn-10_ep-4_L-1024_N-16 --features pca --classifier semimarkov --training unsupervised --sm_supervised_method gradient-based --mix_tasks --crosstask_training_data primary --max_grad_norm 10 --epochs 4 --task_specific_steps --cuda --no_merge_classes --n_classes 1024 --num_features 16  --batch_size 1 --batch_accumulation 5 --print_every 1 > log.kerneltrain.L1024.N16.bsz1.acc5 2>&1&
```
