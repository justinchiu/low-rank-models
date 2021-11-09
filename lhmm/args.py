from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    # dataset
    parser.add_argument("--dataset",
        choices=["ptb", "wikitext2", "wsj"], default="ptb")
    parser.add_argument("--iterator", choices=["bucket", "bptt"], default="bucket")
    # learning args
    parser.add_argument("--bsz", default=512, type=int,)
    parser.add_argument("--eval_bsz", default=512, type=int,)
    parser.add_argument("--bsz_fn", choices=["tokens", "sentences"], default="tokens",)
    parser.add_argument("--lr", default=1e-3, type=float,)
    parser.add_argument("--clip", default=5, type=float,)
    parser.add_argument("--beta1", default=0.9, type=float,)
    parser.add_argument("--beta2", default=0.999, type=float,)
    parser.add_argument("--wd", default=0.000, type=float,)
    parser.add_argument("--decay", default=4, type=float,)
    parser.add_argument("--optimizer",
        choices=["adamw", "sgd"],
        default="adamw",)
    parser.add_argument("--schedule",
        choices=["reducelronplateau", "noam", "none"],
        default="reducelronplateau",)
    parser.add_argument("--warmup_steps", default=500, type=int,)
    parser.add_argument("--patience", default=4, type=int,)
    parser.add_argument("--bptt", default=32, type=int, )
    parser.add_argument("--num_epochs", default=50, type=int,)
    parser.add_argument("--num_checks", default=4, type=int,)
    parser.add_argument("--report_every", default=5000,)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--devid", default=0, type=int,)
    parser.add_argument("--aux_devid", default=1, type=int,)
    parser.add_argument("--model", choices=[
        "shmm", "dhmm", "chmm", "hmm", "lstm", "ff",
        "mshmm", "dshmm",
        "factoredhmm",
        "lhmm",
        "sparsekernelhmm",
        "blhmm",
        "bandedhmm",
        "sblhmm",
        "msblhmm",
    ], default="chmm",)
    parser.add_argument("--seed", default=1111, type=int,)
    parser.add_argument("--eval_only", default="",)

    parser.add_argument("--timing", default=0, type=int,)
    parser.add_argument("--no_shuffle_train", default=0, type=int, help="don't shuffle if > 0")
    parser.add_argument("--reset_eos", default=0, type=int, help="reset state at eos if > 0")
    parser.add_argument("--flat_clusters", default=0, type=int, help="flat clusters if > 0")

    parser.add_argument("--chp_theta", default=0, type=int,
        help="checkpoint hmm parameters if > 0")

    parser.add_argument("--param", choices=["neural", "scalar"], default="neural")
    parser.add_argument("--emit", choices=[
        "word", "char",
    ], default="word",)
    parser.add_argument("--emit_dims", type=int, nargs="+",
        help="use res layers and concat if > 0")
    parser.add_argument("--num_highway", type=int, default=0, help="num highway layers in emit")
    parser.add_argument("--state", choices=[
        "ind", "fac",
    ], default="ind",)

    add_nn_args(parser)

    # add chmm args
    add_chmm_args(parser)
    add_dhmm_args(parser)

    # add linear args
    add_linear_args(parser)

    add_music_args(parser)

    return parser.parse_args()

def add_nn_args(parser):
    parser.add_argument_group("nn")
    parser.add_argument("--tie_weights", default=0, type=int,)
    parser.add_argument("--ngrams", default=5, type=int,)

def add_chmm_args(parser):
    parser.add_argument_group("chmm")
    parser.add_argument("--num_layers", default=2, type=int,)
    parser.add_argument("--emb_dim", default=256, type=int,)
    parser.add_argument("--hidden_dim", default=256, type=int,)
    parser.add_argument("--char_dim", default=16, type=int,)
    parser.add_argument("--dropout", default=0, type=float,)
    parser.add_argument("--num_classes", default=16384, type=int,)
    parser.add_argument("--words_per_state", default=512, type=int,)
    parser.add_argument("--states_per_word", default=128, type=int,)
    parser.add_argument("--train_spw", default=64, type=int, help="number of train states")
    parser.add_argument("--ffnn", default="", type=str, help="default is oldres")
    parser.add_argument("--tw", default="", type=str, help="default is no weight tying")
    parser.add_argument("--transition_dropout", default=0, type=float,)
    parser.add_argument("--column_dropout", default=0, type=int, help="0 = no coldrop")
    parser.add_argument("--start_dropout", default=0, type=float,)
    parser.add_argument("--dropout_type", choices=[
        "unstructured", "state", "cluster", "none",
    ])
    parser.add_argument("--word_dropout", default=0, type=float, help="use uniform emission [0,1]")
    parser.add_argument("--assignment", choices=[
        "brown", "unevenbrown",
        "word2vec",
        "uniform",
        "none",
    ], default="brown",)
    parser.add_argument("--num_clusters", default=128, type=int, help="number of brown clusters")
    parser.add_argument("--num_common", default=0, type=int, help="top k common words (only unevenbrown)")
    parser.add_argument("--num_common_states", default=0, type=int, help="number of common states (only unevenbrown)")
    parser.add_argument("--states_per_common", default=0, type=int, help="repeat each common word (only unevenbrown)")

def add_dhmm_args(parser):
    parser.add_argument_group("dhmm")
    parser.add_argument("--assignment_noise", choices=["gumbel", "none"], default="gumbel",)
    parser.add_argument("--noise_anneal_steps", default=0, type=int, help="0 = no anneal")
    parser.add_argument("--keep_counts", default=0, type=int,)
    parser.add_argument("--log_counts", default=0, type=int)
    parser.add_argument("--count_posterior", choices=[], default="",)
    parser.add_argument("--posterior_weight", type=float, default=1,)

def add_linear_args(parser):
    parser.add_argument_group("linear")
    parser.add_argument("--parameterization", default="softmax",
        choices = [
            "softmax",
            "smp",
        ],
    )
    parser.add_argument("--update_projection", default=0, type=int)
    parser.add_argument("--projection_method",
        choices = ["static", "random"],
        default = "static",
    )
    parser.add_argument("--learn_temp", default="none",
        choices = ["mul", "add", "none"],
    )

    parser.add_argument("--sm_trans", default=0, type=int, help="force sm trans")
    parser.add_argument("--sm_emit", default=0, type=int, help="force sm emit")
    parser.add_argument("--no_layernorm", default=0, type=int, help="remove layernorm")
    parser.add_argument("--rff_method", choices=["log", "exp", "relu"], default="log")
    parser.add_argument("--num_features", default=512, type=int, help="num features")
    parser.add_argument("--no_shift", default=0, type=int, help="remove shift in proj")
    parser.add_argument("--anti", default=0, type=int, help="antithetic sample for proj")
    parser.add_argument("--l2norm", default=0, type=int, help="l2 norm embeddings")
    parser.add_argument("--diffproj", default=0, type=int, help="use different projections")
    parser.add_argument("--transmlp", default=0, type=int, help="use transition mlp")
    parser.add_argument("--eff", default=0, type=int, help="use efficient implementation of fwd")

    parser.add_argument("--band", default=0, type=int, help="upper and lower band length")
    parser.add_argument("--band_only", default=0, type=int, help="only use band")
    parser.add_argument("--band_method", default="sum",
        choices=["sum","product", "only", "none"],
        help="band mixture method (sum or product of experts, band only, or no band)")

    parser.add_argument("--regularize_eigenvalue", default="none",
        choices=["min","ratio", "max", "sum", "prod", "none"],
        help="regularize eigenvalues")

    parser.add_argument("--regularize_cols", default=0., type=float,
        help="regularize column of transition matrix")
    parser.add_argument("--regularize_pairs", default=0, type=int,
        help="regularize rows of transition matrix to be different")

    # May experiments
    parser.add_argument("--regularize_transition_entropy", default=0., type=float,
        help="todo")
    parser.add_argument("--regularize_transition_marginal_entropy", default=0., type=float,
        help="todo")
    # emission
    parser.add_argument("--emission_dropout", default=0., type=float,
        help="todo")
    parser.add_argument("--emission_topk", default=0., type=int,
        help="todo")
    parser.add_argument("--emission_sparsemax", default=0., type=int,
        help="todo")
    # matching 
    parser.add_argument("--frozen_pretrained_transition",
        help="todo")
    # explicit refinements
    parser.add_argument("--explicit_refinement", type=int,
        help="todo")


    parser.add_argument("--feature_dropout", default=0, type=float,)

    parser.add_argument("--tie_start", default=1, type=int, help="tie start embedding")

    parser.add_argument("--dbg_double", default=0, type=int, help="double precision (for testing)")
    parser.add_argument("--dbg_sparse", default=0, type=int, help="debug sparse kernel hmm")

def add_music_args(parser):
    parser.add_argument("--scalar_emit", default=0, type=int, help="use scalar param emit")
    parser.add_argument("--music_dataset",
        choices=["jsb", "muse", "nottingham", "piano"], default="jsb")
