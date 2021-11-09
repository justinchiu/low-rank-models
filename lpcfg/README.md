# Low-Rank PCFG (LPCFG) for Language Modeling

This repo provides instructions for reproducing our LPCFG results in
`Low-Rank Constraints for Fast Inference in Structured Models`.
Note that our repo is based on [compound-pcfg](https://github.com/harvardnlp/compound-pcfg/) and [cpcfg](https://github.com/zhaoyanpeng/cpcfg) (Kim et al 2019, [Compound Probabilistic Context-Free Grammars for Grammar Induction](https://www.aclweb.org/anthology/P19-1228)).

## Dependencies

We need a customized version of [pytorch-struct](https://github.com/harvardnlp/pytorch-struct):

```
cd pytorch-struct
python setup.py install
```

Besides, we need `genbmm`:

```
!pip install -qU git+https://github.com/harvardnlp/genbmm
```

## Data

First, download data provided by [compound-pcfg](https://github.com/harvardnlp/compound-pcfg) [here](https://drive.google.com/file/d/1m4ssitfkWcDSxAE6UYidrP6TlUctSG2D/view?usp=sharing).

Then run

```
python process_ptb.py --ptb_path PATH-TO-PTB/parsed/mrg/wsj --output_path data
```
Now run the preprocessing script
```
python preprocess.py --trainfile data/ptb-train.txt --valfile data/ptb-valid.txt 
--testfile data/ptb-test.txt --outputfile data/ptb --vocabsize 10000 --lowercase 1 --replace_num 1
```

## Training

```
CUDA_VISIBLE_DEVICES=0 python train_fast.py --temperature 1 --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --band 0 --save_path 100_200_rff_features_32_2e3 --z_dim 0 --model_type 19th --num_epochs 15 --lr 2e-3 --num_features 32 --no_argmax True --nt_states 100 --t_states 200 --accumulate 4 > 100_200_log.rff_features32.acc4.genbmm.lr2e3 2>&1 &
```

## Evaluation

```
CUDA_VISIBLE_DEVICES=0 python eval_best.py --model_file 100_200_rff_features_32_2e3/best.pt --data_file data/ptb-test.txt --out_file "./test.pred" --gold_out_file "./test.gold"
```
