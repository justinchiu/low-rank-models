import os
import time as timep

import torch as th
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

import torch_struct as ts

from genbmm import logbmm

from .misc import ResLayer, LogDropoutM

from utils import Pack

from assign import read_lm_clusters, assign_states_brown_cluster

from .linear_utils import get_2d_array, project_logits

def trans(s):
    return s.transpose(-2, -1).contiguous()

# Sparse Bmm Linear Hmm Language model
class SblHmmLm(nn.Module):
    def __init__(self, V, config):
        super(SblHmmLm, self).__init__()

        #self.i = 0

        self.config = config
        self.V = V
        self.device = config.device


        self.sm_emit = config.sm_emit
        self.sm_trans = config.sm_trans

        self.timing = config.timing > 0
        self.eff = config.eff

        self.C = config.num_classes
        self.D = config.num_features
        self.transmlp = config.transmlp

        self.hidden_dim = config.hidden_dim

        self.learn_temp = config.learn_temp
        if self.learn_temp == "mul":
            self.temp = nn.Parameter(th.FloatTensor([1]))

        # init parameters

        # p(z0)
        self.tie_start = config.tie_start
        self.start_emb = nn.Parameter(
            th.randn(config.hidden_dim),
        )
        self.start_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            ResLayer(config.hidden_dim, config.hidden_dim),
            ResLayer(config.hidden_dim, config.hidden_dim),
        )
        self.next_start_emb = nn.Parameter(
            th.randn(config.hidden_dim),
        )
        assert self.tie_start, "Needs tie_start to be correct"
        """
        if self.tie_start:
            # to prevent changing results, which previously had this bug
            # that was never seen since this parameter is not used
            # if start is tied.
            self.next_start_emb = nn.Parameter(
                th.randn(config.hidden_dim),
            )
        else:
            self.next_start_emb = nn.Parameter(
                th.randn(self.C, config.hidden_dim),
            )
        """

        # p(zt | zt-1)
        self.state_emb = nn.Parameter(
            th.randn(self.C, config.hidden_dim),
        )
        if self.transmlp:
            self.trans_mlp = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                ResLayer(config.hidden_dim, config.hidden_dim),
                ResLayer(config.hidden_dim, config.hidden_dim),
            )
        self.next_state_emb = nn.Parameter(
            th.randn(self.C, config.hidden_dim),
        )

        # p(xt | zt)
        self.preterminal_emb = nn.Parameter(
            th.randn(self.C, config.hidden_dim),
        )
        self.terminal_mlp = nn.Sequential(
            ResLayer(config.hidden_dim, config.hidden_dim),
            ResLayer(config.hidden_dim, config.hidden_dim),
        )
        self.terminal_emb = nn.Parameter(
            th.randn(len(V), config.hidden_dim)
        )

        self.transition_dropout = config.transition_dropout
        self.feature_dropout = config.feature_dropout
        self.log_dropout = LogDropoutM(config.transition_dropout)
        self.dropout_type = config.dropout_type

        # init
        for p in self.parameters():
            if p.dim() > 1:
                th.nn.init.xavier_uniform_(p)

        # log-linear or linear, etc
        self.parameterization = config.parameterization
        self.l2norm = config.l2norm
        self.anti = config.anti
        self.diffproj = config.diffproj
        if self.parameterization == "smp":
            if config.projection_method == "static":
                self._projection = nn.Parameter(self.init_proj())
                if not config.update_projection:
                    self._projection.requires_grad = False
                if self.diffproj:
                    self._projection_emit = nn.Parameter(self.init_proj())
                    if not config.update_projection:
                        self._projection_emit.requires_grad = False
            self.projection_method = config.projection_method

        self.init_partitions(config)

    def init_partitions(self, config):
        self.num_clusters = config.num_clusters

        self.words_per_state = config.words_per_state
        self.states_per_word = config.states_per_word
        self.train_states_per_word = config.train_spw
        self.states_per_word_d = config.train_spw


        path = f"clusters/lm-{self.num_clusters}/paths"

        word2cluster, word_counts, cluster2word = read_lm_clusters(
            self.V,
            path=path,
        )
        self.word_counts = word_counts

        assert self.states_per_word * self.num_clusters <= self.C

        word2state = None
        if config.assignment == "brown":
            (
                word2state,
                cluster2state,
                word2cluster,
                c2sw_d,
            ) = assign_states_brown_cluster(
                self.C,
                word2cluster,
                self.V,
                self.states_per_word,
                self.states_per_word_d,
            )
        else:
            raise ValueError(f"No such assignment {config.assignment}")

        # need to save this with model
        self.register_buffer("word2state", th.from_numpy(word2state))
        self.register_buffer("cluster2state", th.from_numpy(cluster2state))
        self.register_buffer("word2cluster", th.from_numpy(word2cluster))
        self.register_buffer("c2sw_d", c2sw_d)
        self.register_buffer("word2state_d", self.c2sw_d[self.word2cluster])

        self.a = (th.arange(0, len(self.V))[:, None]
            .expand(len(self.V), self.states_per_word)
            .contiguous()
            .view(-1)
            .to(self.device)
        )
        self.v = th.ones((len(self.V)) * self.states_per_word).to(self.device)


        self.ad = (th.arange(0, len(self.V))[:, None]
            .expand(len(self.V), self.states_per_word_d)
            .contiguous()
            .view(-1)
            .to(self.device)
        )
        self.vd = th.ones((len(self.V)) * self.states_per_word_d).to(self.device)


    def init_state(self, bsz):
        return self.start.unsqueeze(0).expand(bsz, self.C)

    def init_proj(self):
        #if self.config.rff_method == "relu":
            #return th.nn.init.xavier_uniform_(th.empty(self.config.hidden_dim, self.config.num_features)).to(self.device)
        if not self.anti:
            return get_2d_array(self.config.num_features, self.config.hidden_dim).t().to(self.device)
        else:
            projection_matrix = get_2d_array(
                self.config.num_features//2, self.config.hidden_dim).t().to(self.device)
            return th.cat([projection_matrix, -projection_matrix], -1)

    @property
    def projection(self):
        if self.projection_method == "static":
            pass
        elif self.projection_method == "random":
            self._projection = nn.Parameter(
                self.init_proj()
            )
            self._projection.requires_grad = False
        else:
            raise ValueError(f"Invalid projection_method: {self.projection_method}")
        return self._projection

    @property
    def projection_emit(self):
        if self.projection_method == "static":
            pass
        elif self.projection_method == "random":
            self._projection_emit = nn.Parameter(
                self.init_proj()
            )
            self._projection_emit.requires_grad = False
        else:
            raise ValueError(f"Invalid projection_method: {self.projection_method}")
        return self._projection_emit

    def start(self, states=None, feat_mask=None):
        keep_feat_mask = ~feat_mask if feat_mask is not None else None
        #return self.start_mlp(self.start_emb).log_softmax(-1)
        fx = self.start_mlp(self.start_emb)
        fy = self.next_state_emb if self.tie_start else self.next_start_emb

        if self.parameterization == "softmax" or self.sm_trans:
            logits = fx @ fy.T if states is None else fx @ fy[states].T
            if self.learn_temp == "mul":
                logits = logits * self.temp
            return logits.log_softmax(-1)
        elif self.parameterization == "smp" and not self.sm_trans:
            fy = self.next_state_emb if states is None else self.next_state_emb[states]
            if self.l2norm:
                fx = fx / fx.norm(dim=-1, keepdim=True)
                fy = fy / fy.norm(dim=-1, keepdim=True)

            projection = self.projection if keep_feat_mask is None else self.projection[:,keep_feat_mask]
            if self.learn_temp == "mul":
                projection = projection * self.temp

            logits = project_logits(
                fx[None, None],
                fy[None],
                projection,
                rff_method = self.config.rff_method,
            )[0,0]
            return logits.log_softmax(-1)
        else:
            raise ValueError(f"Invalid parameterization: {self.parameterization}")


    def transition(self, states=None, feat_mask=None, print_max=None):
        keep_feat_mask = ~feat_mask if feat_mask is not None else None

        fx = self.state_emb if states is None else self.state_emb[states]
        fx = self.trans_mlp(fx) if self.transmlp else fx
        fy = self.next_state_emb if states is None else self.next_state_emb[states]
        if self.l2norm:
            fx = fx / fx.norm(dim=-1, keepdim=True)
            fy = fy / fy.norm(dim=-1, keepdim=True)

        if self.parameterization == "softmax" or self.sm_trans:
            logits = fx @ fy.T
            #logits = logits.masked_fill(logits != logits, float("-inf"))
            if self.learn_temp == "mul":
                logits = logits * self.temp
            return logits.log_softmax(-1)
        elif self.parameterization == "smp" and not self.sm_trans:
            projection = (self.projection
                if keep_feat_mask is None
                else self.projection[:,keep_feat_mask]
            )
            if self.learn_temp == "mul":
                projection = projection * self.temp

            logits = project_logits(
                fx[None],
                fy[None],
                projection,
                rff_method = self.config.rff_method,
                fast = False, # save memory by using genbmm.logbmm
            )[0]
            #import pdb; pdb.set_trace()
            return logits.log_softmax(-1)
        else:
            raise ValueError(f"Invalid parameterization: {self.parameterization}")

    def emission(self, states=None):
        fx = self.terminal_mlp(self.preterminal_emb
            if states is None else self.preterminal_emb[states])
        if self.parameterization == "softmax" or self.sm_emit:
            logits = fx @ self.terminal_emb.T
        elif self.parameterization == "smp" and not self.sm_emit:
            # renormalize, important
            if self.l2norm:
                fx = fx / fx.norm(dim=-1, keepdim=True)
                fy = self.terminal_emb / self.terminal_emb.norm(dim=-1, keepdim=True)
            else:
                fy = self.terminal_emb

            logits = project_logits(
                fx[None],
                fy[None],
                self.projection_emit if self.diffproj else self.projection,
            )[0]
        else:
            raise ValueError(f"Invalid parameterization: {self.parameterization}")

        a = self.a if states is None else self.ad
        v = self.v if states is None else self.vd
        word2state = self.word2state if states is None else self.word2state_d

        i = th.stack([word2state.view(-1), a])
        C = logits.shape[0]
        sparse = th.sparse.ByteTensor(i, v, th.Size([C, len(self.V)]))
        mask = sparse.to_dense().bool().to(logits.device)
        log_probs = logits.masked_fill_(~mask, float("-inf")).log_softmax(-1)
        return log_probs

    def forward(self, inputs, state=None):
        raise NotImplementedError
        # forall x, p(X = x)
        pass

    def compute_parameters(self,
        word2state=None,
        states=None, word_mask=None,       
        lpz=None, last_states=None,         
    ):
        # TODO: return struct instead of passing around distributions
        if self.eff:
            return self.compute_rff_parameters()

        transition = self.transition()

        if lpz is not None:
            start = (lpz[:,:,None] + transition[None]).logsumexp(1)
        else:
            start = self.start()

        emission = self.emission()
        return start, transition, emission

    def compute_loss(                                           
        self,
        text, start, transition, emission, word2state=None,
        mask=None, lengths=None,
        keep_counts = False,
    ):
        if self.eff:
            # return two things, losses struct and next state vec
            return self.compute_rff_loss(
                text, start, transition, emission,
                word2state=word2state,
                mask=mask, lengths=lengths,
            )

        N, T = text.shape
        transition = transition.exp()

        state_t = word2state[text]
        p_emit = emission[
            state_t,
            text[:,:,None],
        ]

        transitions = transition[state_t[:,:-1,:,None], state_t[:,1:,None,:]]

        alphas_bmm = []
        evidences_bmm = []
        #alpha_un = start + p_emit[:,0] # {N} x C
        alpha_un = start[state_t[:,0]] + p_emit[:,0] # {N} x S
        Ot = alpha_un.logsumexp(-1, keepdim=True)
        alpha = (alpha_un - Ot).exp()
        alphas_bmm.append(alpha)
        evidences_bmm.append(Ot)
        for t in range(T-1):
            # logbmm
            #alpha = (alpha[:,:,None] + transition[None] + p_emit[:,t+1,None,:]).logsumexp(-2)
            #alpha_un = (alpha @ transition).log() + p_emit[:,t+1]
            alpha_un = (alpha[:,None] @ transitions[:,t])[:,0].log() + p_emit[:,t+1]
            Ot = alpha_un.logsumexp(-1, keepdim=True)
            alpha = (alpha_un - Ot).exp()
            alphas_bmm.append(alpha)
            evidences_bmm.append(Ot)
        O = th.cat(evidences_bmm, -1)
        evidence = O[mask].sum(-1)
        #import pdb; pdb.set_trace()
        return Pack(
            elbo = None,
            evidence = evidence,
            loss = evidence,
        ), alpha.log()


    def score(self, text, lpz=None, last_states=None, mask=None, lengths=None):
        N, T = text.shape

        if self.training and self.dropout_type != "none":
            #I = (th.distributions.Gumbel(self.zero, self.one)
            I = (th.distributions.Gumbel(0, 1)
                .sample(self.cluster2state.shape)
                .squeeze(-1)
                .topk(self.train_states_per_word, dim=-1)
                .indices
            ).to(self.device) # just do it on cpu for now
            states = self.cluster2state.gather(1, I).view(-1)

            feat_mask = (th.empty(self.D, device=self.device)
                .bernoulli_(self.feature_dropout)
                .bool()
            )

            word2state = self.word2state_d
        else:
            states = None
            feat_mask = None
            word2state = self.word2state

        #transition_logits = self.transition_logits()
        #transition = self.mask_transition(transition_logits, transition_mask)
        transition = self.transition(states, feat_mask).exp()
        emission = self.emission(states)
        #self.i += 1
        #if self.i > 1000:
            #import pdb; pdb.set_trace()

        if lpz is not None:
            raise NotImplementedError
            # have to handle masking, but ok for now since not bptt.
            start = (lpz[:,:,None] + transition[None]).logsumexp(1)
        else:
            start = self.start(states)
            #start_logits = self.start_logits()
            #start = self.mask_start(start_logits, start_mask)

        state_t = word2state[text]
        p_emit = emission[
            state_t,
            text[:,:,None],
        ]

        transitions = transition[state_t[:,:-1,:,None], state_t[:,1:,None,:]]

        alphas_bmm = []
        evidences_bmm = []
        #alpha_un = start + p_emit[:,0] # {N} x C
        alpha_un = start[state_t[:,0]] + p_emit[:,0] # {N} x S
        Ot = alpha_un.logsumexp(-1, keepdim=True)
        alpha = (alpha_un - Ot).exp()
        alphas_bmm.append(alpha)
        evidences_bmm.append(Ot)
        for t in range(T-1):
            # logbmm
            #alpha = (alpha[:,:,None] + transition[None] + p_emit[:,t+1,None,:]).logsumexp(-2)
            #alpha_un = (alpha @ transition).log() + p_emit[:,t+1]
            alpha_un = (alpha[:,None] @ transitions[:,t])[:,0].log() + p_emit[:,t+1]
            Ot = alpha_un.logsumexp(-1, keepdim=True)
            alpha = (alpha_un - Ot).exp()
            alphas_bmm.append(alpha)
            evidences_bmm.append(Ot)
        O = th.cat(evidences_bmm, -1)
        evidence = O[mask].sum(-1)

        return Pack(
            elbo = None,
            evidence = evidence,
            loss = evidence,
        ), alpha.log(), None

    def score_rff(self, text, lpz=None, last_states=None, mask=None, lengths=None):
        N, T = text.shape
        C = self.C
        D = self.D

        if self.training and self.dropout_type != "none":
            I = (th.distributions.Gumbel(self.zero, self.one)
                .sample(self.cluster2state.shape)
                .squeeze(-1)
                .topk(self.train_states_per_word, dim=-1)
                .indices
            )
            states = self.cluster2state.gather(1, I).view(-1)

            feat_mask = (th.empty(self.D, device=self.device)
                .bernoulli_(self.feature_dropout)
                .bool()
            )

            word2state = self.word2state_d
        else:
            states = None
            feat_mask = None
            word2state = self.word2state


        if lpz is not None:
            start = lpz
        else:
            start = self.start(states, feat_mask)
        if self.timing:
            start_ = timep.time()
        emission = self.emission(states)
        if self.timing:
            print(f"total emit time: {timep.time() - start_}")
            start_ = timep.time()

        # gather emission
        # N x T x C
        #num_states = self.C if states is None else self.train_states_per_word
        state_t = word2state[text]

        logp_emit = emission[
            state_t,
            text[:,:,None],
        ]

        if self.timing:
            print(f"total emit index time: {timep.time() - start_}")
            start_ = timep.time()

        state_emb = self.state_emb if states is None else self.state_emb[states]
        if self.transmlp:
            state_emb = self.trans_mlp(state_emb)
        next_state_emb = self.next_state_emb if states is None else self.next_state_emb[states]
        if self.l2norm:
            state_emb = state_emb / state_emb.norm(dim=-1, keepdim=True)
            next_state_emb = next_state_emb / next_state_emb.norm(dim=-1, keepdim=True)

        # sum vectors and sum matrices
        projection = self.projection if feat_mask is None else self.projection[:,~feat_mask]
        if self.learn_temp == "mul":
            projection = projection * self.temp

        log_phi_w = state_emb @ projection
        log_phi_u = next_state_emb @ projection

        # Todo: abstract away performer kernel
        #log_phi_w = state_emb @ projection - state_emb.square().sum(-1, keepdim=True) / 2
        #log_phi_u = next_state_emb @ projection - next_state_emb.square().sum(-1, keepdim=True) / 2

        # O(CD)
        log_denominator = (log_phi_w + log_phi_u.logsumexp(0, keepdim=True)).logsumexp(-1)
        # O(CD)
        normed_log_phi_w = log_phi_w - log_denominator[:,None]

        normalized_phi_w = normed_log_phi_w.exp()
        phi_u = log_phi_u.exp()
        if self.timing:
            print(f"total proj time: {timep.time() - start_}")
            start_ = timep.time()

        left_proj = normalized_phi_w[state_t[:,:-1]]
        right_proj = phi_u[state_t[:,1:]]

        if self.timing:
            print(f"total index proj time: {timep.time() - start_}")
            start_ = timep.time()

        alphas = []
        Os = []

        alpha_un = start[state_t[:,0]] + logp_emit[:,0]
        Ot = alpha_un.logsumexp(-1, keepdim=True)
        alpha = (alpha_un - Ot).exp()
        alphas.append(alpha)
        Os.append(Ot)
        for t in range(T-1):
            gamma = alpha[:,None] @ left_proj[:,t]
            alpha_un = logp_emit[:,t+1] + (gamma @ right_proj[:,t].transpose(-1,-2))[:,0].log()
            Ot = alpha_un.logsumexp(-1, keepdim=True)
            alpha = (alpha_un - Ot).exp()

            alphas.append(alpha)
            Os.append(Ot)
        O = th.cat(Os, -1)
        evidence = O[mask].sum()
        if self.timing:
            print(f"total inference time: {timep.time() - start_}")
            start_ = timep.time()

        return Pack(
            elbo = None,
            evidence = evidence,
            loss = evidence,
        ), alpha.log(), None


    def compute_rff_parameters(self):
        state_emb = self.state_emb
        next_state_emb = self.next_state_emb
        if self.transmlp:
            state_emb = self.trans_mlp(state_emb)
        if self.l2norm:
            state_emb = self.state_emb / self.state_emb.norm(dim=-1, keepdim=True)
            next_state_emb = self.next_state_emb / self.next_state_emb.norm(dim=-1, keepdim=True)

        # sum vectors and sum matrices
        projection = self.projection
        if self.learn_temp == "mul":
            projection = projection * self.temp
        log_phi_w = state_emb @ projection
        log_phi_u = next_state_emb @ projection

        # TODO: abstract away perfomer kernel
        #log_phi_w = state_emb @ projection - state_emb.square().sum(-1, keepdim=True) / 2
        #log_phi_u = next_state_emb @ projection - next_state_emb.square().sum(-1, keepdim=True) / 2

        log_denominator = (log_phi_w + log_phi_u.logsumexp(0, keepdim=True)).logsumexp(-1)
        normed_log_phi_w = log_phi_w - log_denominator[:, None]

        start = self.start()
        emission = self.emission()

        return start, (normed_log_phi_w.exp(), log_phi_u.exp()), emission

    def compute_rff_loss(
        self,
        text, start, transition, emission,
        word2state=None,
        mask=None, lengths=None,
    ):
        N, T = text.shape
        normalized_phi_w, phi_u = transition

        # gather emission
        # N x T x C
        state_t = word2state[text]
        logp_emit = emission[
            state_t,
            text[:,:,None],
        ]

        left_proj = normalized_phi_w[state_t[:,:-1]]
        right_proj = phi_u[state_t[:,1:]]

        alphas = []
        Os = []

        alpha_un = start[state_t[:,0]] + logp_emit[:,0]
        Ot = alpha_un.logsumexp(-1, keepdim=True)
        alpha = (alpha_un - Ot).exp()
        alphas.append(alpha)
        Os.append(Ot)
        for t in range(T-1):
            gamma = alpha[:,None] @ left_proj[:,t]
            alpha_un = logp_emit[:,t+1] + (gamma @ right_proj[:,t].transpose(-1,-2))[:,0].log()
            Ot = alpha_un.logsumexp(-1, keepdim=True)
            alpha = (alpha_un - Ot).exp()

            alphas.append(alpha)
            Os.append(Ot)
        O = th.cat(Os, -1)
        evidence = O[mask].sum()

        return Pack(
            elbo = None,
            evidence = evidence,
            loss = evidence,
        ), alpha.log()

