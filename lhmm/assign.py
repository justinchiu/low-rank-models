
import time as timep

import sys

from collections import defaultdict

import numpy as np
import torch as th


# randomly assign M states to each word (out of K)
# states have at most L words (sparsity is nice)
def assign_states(num_states, states_per_word, num_words):
    word2state = [[] for _ in range(num_words)]
    for w in range(num_words):
        perm = np.random.permutation(num_states)[:states_per_word]
        word2state[w] = perm
    return np.array(word2state)

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

def convert_w2s(word2state, num_states):
    # invert word to state mapping
    # assumes the last word is a padding word
    state2word = [[] for _ in range(num_states)]
    num_wordsp1, states_per_word = word2state.shape
    num_words = num_wordsp1 - 1
    for word in range(num_words):
        for idx in range(states_per_word):
            state = word2state[word, idx]
            state2word[state].append(word)
    words_per_state = max(len(x) for x in state2word)
    for s in range(num_states):
        l = len(state2word[s])
        if l < words_per_state:
            state2word[s].extend([num_words] * (words_per_state - l))
    return th.tensor(state2word, dtype=word2state.dtype, device=word2state.device)

def perturb_kmax_old(potentials, noise_dist, k):
    s = timep.time()
    num_states, num_words = potentials.shape
    perturbed_scores = potentials + noise_dist.sample(potentials.shape).squeeze(-1)
    print(f"sample add: {timep.time() - s:.3f}")
    s = timep.time()
    scores, idx = perturbed_scores.t().topk(k, dim=1)
    print(f"topk: {timep.time() - s:.3f}")
    s = timep.time()
    # idx is word2state
    word2state = idx
    state2word = convert_w2s(word2state, num_states)
    print(f"convert: {timep.time() - s:.3f}")
    return word2state, state2word

def perturb_kmax(potentials, noise_dist, k):
    num_states, num_words = potentials.shape
    perturbed_scores = potentials + noise_dist.sample(potentials.shape).squeeze(-1)
    # always topk on inner dim
    scores, idx = perturbed_scores.t().topk(k, dim=1)
    # return word2state
    return idx

def read_pmi_clusters():
    pass

def read_lm_clusters(V, path="clusters/lm-128/paths"):
    with open(path, "r") as f:
        word2cluster = {}
        word_counts = []
        cluster2word = defaultdict(list)
        cluster2id = {}
        id = 0
        for line in f:
            cluster, word, count = line.strip().split()
            if cluster not in cluster2id:
                cluster2id[cluster] = id
                id += 1
            cluster_id = cluster2id[cluster]
            word2cluster[V[word]] = cluster_id
            cluster2word[cluster_id].append(V[word])
            word_counts.append((V[word], int(count)))
        print(f"Read {id} clusters from {path}")
        return (
            word2cluster,
            sorted(word_counts, key=lambda x: x[1], reverse=True),
            dict(cluster2word),
        )

def assign_states_brown_wrong(
    num_states, states_per_word,
    word2cluster, word_counts, cluster2word,
):
    num_words = max(word2cluster.keys()) + 1
    num_clusters = len(set(word2cluster.values()))
    # number of words that get their set of own states
    # each of these words gets states_per_word states
    num_singletons = (num_states - num_clusters - 1) // states_per_word
    # need to add pad=1, eos=3
    singleton_words = (
        [x[0] for x in word_counts[:num_singletons-2]]
        + [1, 3]
    )
    word2state = np.ndarray((num_words, states_per_word,), dtype=np.int64)
    singleton_idx = 0
    for word in range(0, num_words):
        if word in singleton_words:
            word2state[word] = range(
                states_per_word * singleton_idx,
                states_per_word * (singleton_idx + 1),
            )
            singleton_idx += 1
        else:
            word2state[word, 0] = word2cluster[word] + num_singletons * states_per_word
            word2state[word, 1:] = num_states - 1
    return word2state

def assign_states_brown(
    num_states, word2cluster, V,
    states_per_word, 
):
    # must have num_states = num_clusters * num_repeats 
    num_words = len(V)
    # assume this is less than num_states // states_per_word
    num_clusters = len(set(word2cluster.values()))
    #states_per_word = num_states // num_clusters

    word2state = np.ndarray((num_words, states_per_word,), dtype=np.int64)
    for word in range(0, num_words):
        # try putting in last cluster?
        #cluster = word2cluster[word] if word in word2cluster else num_clusters + 1
        cluster = word2cluster[word] if word in word2cluster else num_clusters-1
        word2state[word] = range(
            states_per_word * cluster,
            states_per_word * (cluster + 1),
        )
    return word2state

def assign_states_uneven_brown(
    num_states, word2cluster, V,
    states_per_word, # will go to brown clusters
    word_counts,
    num_common,
    num_common_states,
    states_per_common, # will go to commons
):
    num_words = len(V)
    # assume this is less than num_states // states_per_word
    num_clusters = len(set(word2cluster.values())) + 1
    states_per_cluster = states_per_word - states_per_common
    num_cluster_states = num_clusters * states_per_cluster
    num_common_states = num_states - num_cluster_states
    # num_states cluster is padding

    # for now, assume states_per_word > states_per_common
    word2state = np.ndarray((num_words, states_per_word,), dtype=np.int64)
    word2state.fill(num_states-1)

    # first do common words
    common_words = set(x[0] for x in word_counts[:num_common])
    for word in common_words:
        word2state[word, :states_per_common] = np.random.permutation(num_common_states)[:states_per_common]

    # then do clusters
    for word in range(0, num_words):
        cluster = word2cluster[word] if word in word2cluster else num_clusters - 1
        word2state[word,states_per_common:states_per_common+states_per_cluster] = range(
            num_common_states + states_per_cluster * cluster,
            num_common_states + states_per_cluster * (cluster + 1),
        )

    return word2state

def assign_states_brown_cluster(
    num_states, word2cluster, V,
    states_per_word,
    states_per_word_d,
):
    # must have num_states = num_clusters * num_repeats 
    num_words = len(V)
    # assume this is less than num_states // states_per_word
    num_clusters = len(set(word2cluster.values()))
    #states_per_word = num_states // num_clusters
    w2c = np.ndarray(len(V), dtype=np.int64)
    for word in range(len(V)):
        w2c[word] = (word2cluster[word]
            if word in word2cluster
            else num_clusters-1
        )
    cluster2state = np.ndarray((num_clusters, states_per_word), dtype=np.int64)
    for c in range(0, num_clusters):
        cluster2state[c] = range(
            states_per_word * c,
            states_per_word * (c + 1),
        )
    word2state = cluster2state[w2c]
    # the dropped cluster to words after reindexing
    # assume states per word // 2
    c2sw_d = th.LongTensor([
        list(range(c * states_per_word_d, (c+1) * states_per_word_d))
        for c in range(num_clusters)
    ])
    return word2state, cluster2state, w2c, c2sw_d


if __name__ == "__main__":
    num_states = int(2 ** 14)
    states_per_word = int(128)
    num_words = int(1e4)
    words_per_state = int(512)
    word2state, state2word = assign_states(num_states, states_per_word, num_words, words_per_state)

    word_state_score = th.randn(num_words, num_states, requires_grad=True)
    gumbel_noise = th.distributions.Gumbel(0, 1)
    word2state, state2word = perturb_kmax(word_state_score, gumbel_noise, states_per_word)

    import pdb; pdb.set_trace()
