
import sys
import random

import yaml

import numpy as np
import torch as th

class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack


def set_seed(seed):
    """Sets random seed everywhere."""
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_config(path, device):
    with open(path, "r") as f:
        pack = Pack(yaml.load(f, Loader = yaml.Loader))
        pack.device = device
        return pack

def get_name(config, music=False):
    return "_".join([
        config.dataset if not music else config.music_dataset,
        config.iterator,
        config.model,
        f"k{config.num_classes}",
        #f"wps{config.words_per_state}",
        #f"spw{config.states_per_word}",
        #f"tspw{config.train_spw}",
        #f"ff{config.ffnn}",
        f"ed{config.emb_dim}",
        f"d{config.hidden_dim}",
        #f"cd{config.char_dim}",
        #f"dp{config.dropout}",
        f"bs{config.band}",
        f"bm{config.band_method}",
        f"tdp{config.transition_dropout}",
        f"fdp{config.feature_dropout}",
        #f"cdp{config.column_dropout}",
        #f"sdp{config.start_dropout}",
        f"dt{config.dropout_type}",
        #f"wd{config.word_dropout}",
        #config.bsz_fn,
        f"b{config.bsz}",
        config.optimizer,
        f"lr{config.lr}",
        f"c{config.clip}",
        f"tw{config.tw}",
        #f"nas{config.noise_anneal_steps}",
        #f"pw{config.posterior_weight}",
        #f"as{config.assignment}",
        #f"nb{config.num_clusters}",
        #f"nc{config.num_common}",
        #f"ncs{config.num_common_states}",
        #f"spc{config.states_per_common}",
        f"n{config.ngrams}",
        f"r{config.reset_eos}",
        f"ns{config.no_shuffle_train}",
        #f"fc{config.flat_clusters}",
        f"e{config.emit}",
        #f"ed{'-'.join(str(x) for x in config.emit_dims) if config.emit_dims is not None else 'none'}",
        #f"nh{config.num_highway}",
        f"s{config.state}",
        f"ts{config.tie_start}",
        f"p{config.parameterization}",
        f"up{config.update_projection}",
        f"lt{config.learn_temp}",
        f"pm{config.projection_method}",
        f"st{config.sm_trans}",
        f"se{config.sm_emit}",
        f"tm{config.transmlp}",
        f"rm{config.rff_method}",
        f"nf{config.num_features}",
        f"ns{config.no_shift}",
        f"a{config.anti}",
        f"l2{config.l2norm}",
        f"dfp{config.diffproj}",
        f"eff{config.eff}",
        #f"re{config.regularize_eigenvalue}",
        #f"rc{config.regularize_cols}",
        #f"rp{config.regularize_pairs}",
        #f"tme{config.regularize_transition_marginal_entropy}",
    ])

def get_mask_lengths(text, V):
    mask = text != V.stoi["<pad>"]
    lengths = mask.sum(-1)
    n_tokens = mask.sum()
    return mask, lengths, n_tokens

# randomly assign M states to each word (out of K)
# states have at most L words (sparsity is nice)
def assign_states(num_states, states_per_word, num_words, words_per_state):
    word2state = [[] for _ in range(num_words)]
    state2word = [[] for _ in range(num_states)]
    for w in range(num_words):
        perm = np.random.permutation(num_states)
        #print([len(state2word[x]) for x in range(num_states)])
        #print(sum([len(state2word[x]) for x in range(num_states)]))
        #print(sum([len(state2word[x]) < words_per_state for x in range(num_states)]))
        i = 0
        while len(word2state[w]) < states_per_word:
            try:
                s = perm[i]
            except:
                print("Try again with more states or words_per_state")
                sys.exit()
            if len(state2word[s]) < words_per_state:
                word2state[w].append(s)
                state2word[s].append(w)
            i += 1
    # pad state2word to words_per_state
    for s in range(num_states):
        while len(state2word[s]) < words_per_state:
            state2word[s].append(num_words)
    return np.array(word2state), np.array(state2word)

# slower but avoids degenerate solutions better?
def assign_states2(num_states, states_per_word, num_words, words_per_state):
    word2state = [[] for _ in range(num_words)]
    state2word = [[] for _ in range(num_states)]
    for i in range(states_per_word):
        for w in range(num_words):
            p = np.array([
                (words_per_state - len(state2word[s])) if s not in word2state[w] else 0
                for s in range(num_states)
            ])
            p = p / p.sum()
            s = np.random.choice(num_states, p = p)
            word2state[w].append(s)
            state2word[s].append(w)
    # pad state2word to words_per_state
    for s in range(num_states):
        while len(state2word[s]) < words_per_state:
            state2word[s].append(num_words)
    return np.array(word2state), np.array(state2word)

def assign_states3(num_states, states_per_word, num_words, words_per_state):
    word2state = [[] for _ in range(num_words)]
    state2word = [[] for _ in range(num_states)]
    perm = np.random.permutation(num_states * words_per_state)
    splits = np.split(perm, [states_per_word * x for x in range(1, num_words+1)])
    for word, split in enumerate(splits):
        if len(split) != states_per_word:
            break
        for state_flat in split:
            #state = state_flat % num_states
            state = state_flat // words_per_state
            word2state[word].append(state)
            state2word[state].append(word)
    for s in range(num_states):
        while len(state2word[s]) < words_per_state:
            state2word[s].append(num_words)
    return np.array(word2state), np.array(state2word)

def log_eye(K, dtype, device):
    x = th.empty(K, K, dtype = dtype, device = device)
    x.fill_(float("-inf"))
    x.diagonal().fill_(0)
    return x

def plot_counts(counts):
    import matplotlib.pyplot as plt
    num_c, num_w = counts.shape
    words = [
        13, 29, 67, 111, 131, 171, 373, 567, 700, 800,
        5617,5053,5601,5756,1482,7443,3747,8314,11,3722,7637,7916,3376,7551,
        5391,9072,230,9244,6869,441,1076,7093,1845,201,1386,6738,2840,4909,
    ]
    counts = counts[:, words]
    fig, axs = plt.subplots(1, 3)
    axs[0].spy(counts, precision=0.0001, markersize=1, aspect="auto")
    axs[1].spy(counts, precision=0.001, markersize=1, aspect="auto")
    axs[2].spy(counts, precision=0.01, markersize=1, aspect="auto")
    return plt

def print_gpu_mem():
    print(f"Max mem allocated {th.cuda.max_memory_allocated() / 2 ** 30:.2f}")
    print(f"Max mem cached {th.cuda.max_memory_cached() / 2 ** 30:.2f}")


def dump_transition(hmm):
    transition = hmm.transition()
    name = get_name(hmm.config)
    np.save(f"transitions/{name}-transition.npy", transition.detach().cpu().numpy())

def dump_svd(model):
    name = get_name(model.config)
    start, transition, emission = model.compute_parameters(model.word2state)
    myt = model.transition(print_max=True)
    mye = emission

    _,Ts,_ = myt.exp().svd(compute_uv=False)
    _,Es,_ = mye.exp().svd(compute_uv=False)

    """
    Tdata = [[i,v] for i,v in enumerate(Ts.cpu().detach().numpy())]
    Edata = [[i,v] for i,v in enumerate(Es.cpu().detach().numpy())]

    table = wandb.Table(data=data, columns = ["index", "value"]) 
    wandb.log({
        "transition_entropy": Ht,
        "emission_entropy": He,
        "svd": wandb.plot.scatter(table, "index", "value", title="Singular Values"),
    }, step=WANDB_STEP)
    """

    np.save(f"svd/{name}-svd-transition.npy", Ts.detach().cpu().numpy())
    np.save(f"svd/{name}-svd-emission.npy", Es.detach().cpu().numpy())

