run_hmm () {
    WANDB_MODE=dryrun python main.py --lr 0.001 --column_dropout 0 --transition_dropout 0.1 --feature_dropout 0.1 --dropout_type state --model blhmm --bsz 256 --num_classes 16384 --emb_dim 256 --hidden_dim 256 --dataset ptb --iterator bucket --parameterization softmax --transmlp 0
}

run_lhmm () {
    WANDB_MODE=dryrun python main.py --lr 0.001 --column_dropout 1 --transition_dropout 0.1 --feature_dropout 0.1 --dropout_type state --model blhmm --bsz 256 --num_classes 16384 --emb_dim 256 --hidden_dim 256 --dataset ptb --iterator bucket --parameterization smp --projection_method static --update_proj 1 --num_features 1024 --anti 0 --l2norm 0 --sm_emit 1 --eval_bsz 256 --num_epochs 30 --band 0 --eff 1
}

run_band () {
    WANDB_MODE=dryrun python main.py --lr 0.001 --column_dropout 1 --transition_dropout 0.1 --feature_dropout 0.1 --dropout_type state --model bandedhmm --bsz 256 --num_classes 16384 --emb_dim 256 --hidden_dim 256 --dataset ptb --iterator bucket --parameterization smp --projection_method static --update_proj 1 --num_features 1024 --anti 0 --l2norm 0 --sm_emit 1 --eval_bsz 256 --num_epochs 30 --band 512 --eff 1
}

run_sehmm () {
    WANDB_MODE=dryrun python main.py --lr 0.001 --column_dropout 0 --transition_dropout 0.1 --dropout_type state --model sblhmm --bsz 256 --num_classes 16384 --emb_dim 256 --hidden_dim 256 --dataset ptb --iterator bucket --parameterization softmax --l2norm 0 --eff 0 --states_per_word 128 --train_spw 115 --assignment brown --num_clusters 128 --transmlp 0
}
