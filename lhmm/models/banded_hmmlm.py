import os
import time as timep

import torch as th
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from .misc import ResLayer, LogDropoutM

from utils import Pack

from genbmm import BandedMatrix, logbmm

from .linear_utils import get_2d_array, project_logits, logbbmv, bbmv

def trans(s):
    return s.transpose(-2, -1).contiguous()

# use bmm for linear hmm
class BandedHmmLm(nn.Module):
    def __init__(self, V, config):
        super(BandedHmmLm, self).__init__()

        self.config = config
        self.V = V
        self.device = config.device

        self.sm_emit = config.sm_emit
        self.sm_trans = config.sm_trans
        self.transmlp = config.transmlp

        self.timing = config.timing > 0
        self.eff = config.eff

        self.C = config.num_classes
        self.D = config.num_features
        self.K = config.band
        self.K1K = 2 * self.K + 1
        self.band_method = config.band_method

        self.word2state = None

        self.hidden_dim = config.hidden_dim

        self.learn_temp = config.learn_temp
        if self.learn_temp == "mul":
            self.temp = nn.Parameter(th.FloatTensor([1]))
        elif self.learn_temp == "add":
            self.temp = nn.Parameter(th.FloatTensor([0]))

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

        # maybe switch to row?
        self.col_banded_transition = nn.Parameter(
            th.randn(self.C, self.K1K),
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

        # zeros of banded matrix
        self.band_fill_value = (float("-inf")
            if self.band_method == "sum" or self.band_method == "only"
            else 0.)
        # initialize band to be competitive with kernel logits
        if self.band_method == "sum":
            # HACKY
            with th.no_grad():
                self.col_banded_transition.data += 30

    def init_state(self, bsz):
        return self.start.unsqueeze(0).expand(bsz, self.C)

    def init_proj(self):
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

    def start(self, mask=None, feat_mask=None):
        keep_mask = ~mask if mask is not None else None
        keep_feat_mask = ~feat_mask if feat_mask is not None else None

        fx = self.start_mlp(self.start_emb)
        fy = self.next_state_emb if self.tie_start else self.next_start_emb

        if self.parameterization == "softmax" or self.sm_trans:
            logits = fx @ fy.T if mask is None else fx @ fy[keep_mask].T
            if self.learn_temp == "mul":
                logits = logits * self.temp
            return logits.log_softmax(-1)
        elif self.parameterization == "smp" and not self.sm_trans:
            fy = self.next_state_emb if keep_mask is None else self.next_state_emb[keep_mask]
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


    def transition(self, mask=None, feat_mask=None, print_max=False):
        keep_mask = ~mask if mask is not None else None
        keep_feat_mask = ~feat_mask if feat_mask is not None else None

        fx = self.state_emb if keep_mask is None else self.state_emb[keep_mask]
        fx = self.trans_mlp(fx) if self.transmlp else fx
        fy = self.next_state_emb if keep_mask is None else self.next_state_emb[keep_mask]
        if self.l2norm:
            fx = fx / fx.norm(dim=-1, keepdim=True)
            fy = fy / fy.norm(dim=-1, keepdim=True)

        if self.parameterization == "softmax" or self.sm_trans:
            logits = fx @ fy.T
            if self.learn_temp == "mul":
                logits = logits * self.temp
        elif self.parameterization == "smp" and not self.sm_trans:
            projection = self.projection if keep_feat_mask is None else self.projection[:,keep_feat_mask]
            # important to renormalize. maybe move this into project_logits
            if self.learn_temp == "mul":
                projection = projection * self.temp
            logits = logbmm((fx @ projection)[None], (fy @ projection).T.contiguous()[None])[0]
        else:
            raise ValueError(f"Invalid parameterization: {self.parameterization}")

        if self.K > 0:
            # add the diagonal
            dense_banded_transition = BandedMatrix(
                self.col_banded_transition[None], self.K, self.K,
                fill=self.band_fill_value,
            ).to_dense()[0]
            if print_max:
                lmax = logits.max().item()
                bmax = self.col_banded_transition.max().item()
                print(f"lmax {lmax:.2f} | bmax {bmax:.2f}")
                #import pdb; pdb.set_trace()

            if self.band_method == "sum":
                logits = logits.logaddexp(dense_banded_transition)
            elif self.band_method == "product":
                logits = logits + dense_banded_transition
            elif self.band_method == "only":
                logits = dense_banded_transition
            elif self.band_method == "none":
                pass
            else:
                raise ValueError(f"Invalid band_method: {self.band_method}")

        return logits.log_softmax(-1)

    def emission(self, mask=None):
        keep_mask = ~mask if mask is not None else None
        fx = self.terminal_mlp(self.preterminal_emb
            if mask is None else self.preterminal_emb[keep_mask])
        if self.parameterization == "softmax" or self.sm_emit:
            return (fx @ self.terminal_emb.T).log_softmax(-1)
        elif self.parameterization == "smp" and not self.sm_emit:
            # renormalize, important
            if self.l2norm:
                fx = fx / fx.norm(dim=-1, keepdim=True)
                fy = self.terminal_emb / self.terminal_emb.norm(dim=-1, keepdim=True)
            else:
                fy = self.terminal_emb

            return project_logits(
                fx[None],
                fy[None],
                self.projection_emit if self.diffproj else self.projection,
            )[0].log_softmax(-1)
        else:
            raise ValueError(f"Invalid parameterization: {self.parameterization}")

    def forward(self, inputs, state=None):
        raise NotImplementedError
        # forall x, p(X = x)
        pass

    def log_potentials(self, text, states=None, lpz=None, last_states=None,):
        log_potentials = ts.LinearChain.hmm(
            transition = self.transition().t(),
            emission = self.emission().t(),
            init = self.start(),
            observations = text,
            semiring = ts.LogSemiring,
        )
        return log_potentials

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

        p_emit = emission[
            th.arange(self.C)[None,None],
            text[:,:,None],
        ]

        alphas_bmm = []
        evidences_bmm = []
        alpha_un = start + p_emit[:,0] # {N} x C
        Ot = alpha_un.logsumexp(-1, keepdim=True)
        alpha = (alpha_un - Ot).exp()
        alphas_bmm.append(alpha)
        evidences_bmm.append(Ot)
        for t in range(T-1):
            # logbmm
            #alpha = (alpha[:,:,None] + transition[None] + p_emit[:,t+1,None,:]).logsumexp(-2)
            alpha_un = (alpha @ transition).log() + p_emit[:,t+1]
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
        ), alpha.log()


    def score(self, text, lpz=None, last_states=None, mask=None, lengths=None):
        N, T = text.shape

        start_mask, transition_mask, feat_mask = None, None, None
        if not self.training or self.dropout_type == "none" or self.dropout_type is None:
            # no dropout
            pass
        elif self.dropout_type == "transition":
            raise NotImplementedError
            transition_mask = (th.empty(self.C, self.C, device=self.device)
                .fill_(self.transition_dropout)
                .bernoulli_()
                .bool()
            )
        elif self.dropout_type == "starttransition":
            raise NotImplementedError
            transition_mask = (th.empty(self.C, self.C, device=self.device)
                .fill_(self.transition_dropout)
                .bernoulli_()
                .bool()
            )
            start_mask = (th.empty(self.C, device=self.device)
                .fill_(self.transition_dropout)
                .bernoulli_()
                .bool()
            )
        elif self.dropout_type == "column":
            raise NotImplementedError
            transition_mask = (th.empty(self.C, device=self.device)
                .fill_(self.transition_dropout)
                .bernoulli_()
                .bool()
            )
        elif self.dropout_type == "startcolumn":
            raise NotImplementedError
            transition_mask = (th.empty(self.C, device=self.device)
                .fill_(self.transition_dropout)
                .bernoulli_()
                .bool()
            )
            start_mask = (th.empty(self.C, device=self.device)
                .fill_(self.transition_dropout)
                .bernoulli_(self.transition_dropout)
                .bool()
            )
        elif self.dropout_type == "state":
            m = (th.empty(self.C, device=self.device)
                .bernoulli_(self.transition_dropout)
                .bool()
            )
            start_mask, transition_mask = m, m
            feat_mask = (th.empty(self.D, device=self.device)
                .bernoulli_(self.feature_dropout)
                .bool()
            )
        else:
            raise ValueError(f"Unsupported dropout type {self.dropout_type}")

        transition = self.transition(transition_mask, feat_mask).exp()
        emission = self.emission(transition_mask)

        if lpz is not None:
            raise NotImplementedError
            # have to handle masking, but ok for now since not bptt.
            start = (lpz[:,:,None] + transition[None]).logsumexp(1)
        else:
            start = self.start(start_mask)

        num_states = self.C if transition_mask is None else (~transition_mask).sum().item()
        p_emit = emission[
            th.arange(num_states)[None,None],
            text[:,:,None],
        ]

        alphas_bmm = []
        evidences_bmm = []
        alpha_un = start + p_emit[:,0] # {N} x C
        Ot = alpha_un.logsumexp(-1, keepdim=True)
        alpha = (alpha_un - Ot).exp()
        alphas_bmm.append(alpha)
        evidences_bmm.append(Ot)
        for t in range(T-1):
            # logbmm
            #alpha = (alpha[:,:,None] + transition[None] + p_emit[:,t+1,None,:]).logsumexp(-2)
            alpha_un = (alpha @ transition).log() + p_emit[:,t+1]
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

        start_mask, transition_mask, feat_mask = None, None, None
        if not self.training or self.dropout_type == "none" or self.dropout_type is None:
            # no dropout
            pass
        elif self.dropout_type == "state":
            m = (th.empty(self.C, device=self.device)
                .bernoulli_(self.transition_dropout)
                .bool()
            )
            start_mask, transition_mask = m, m
            feat_mask = (th.empty(self.D, device=self.device)
                .bernoulli_(self.feature_dropout)
                .bool()
            )
        else:
            raise ValueError(f"Unsupported dropout type {self.dropout_type}")

        if lpz is not None:
            start = lpz
        else:
            # no speedup from dropout
            #start = self.start(start_mask, feat_mask)
            start = self.start()
        if self.timing:
            start_ = timep.time()
        # no from dropout
        #emission = self.emission(transition_mask)
        emission = self.emission()
        if transition_mask is not None:
            # mask out emission
            emission = emission.masked_fill(transition_mask[:,None], float("-inf"))
        if self.timing:
            print(f"total emit time: {timep.time() - start_}")
            start_ = timep.time()

        # gather emission
        # N x T x C
        #num_states = self.C if transition_mask is None else (~transition_mask).sum().item()
        num_states = self.C
        logp_emit = emission[
            th.arange(num_states)[None,None],
            text[:,:,None],
        ]
        if self.timing:
            print(f"total emit index time: {timep.time() - start_}")
            start_ = timep.time()

        #state_emb = self.state_emb if transition_mask is None else self.state_emb[~transition_mask]
        #next_state_emb = self.next_state_emb if transition_mask is None else self.next_state_emb[~transition_mask]
        state_emb = self.state_emb
        next_state_emb = self.next_state_emb
        if self.l2norm:
            state_emb = state_emb / state_emb.norm(dim=-1, keepdim=True)
            next_state_emb = next_state_emb / next_state_emb.norm(dim=-1, keepdim=True)

        # sum vectors and sum matrices
        projection = self.projection if feat_mask is None else self.projection[:,~feat_mask]
        if self.learn_temp == "mul":
            projection = projection * self.temp
        log_phi_w = (state_emb @ projection)
        log_phi_u = next_state_emb @ projection
        # TODO: performer kernel, abstract away
        #log_phi_w = state_emb @ projection - state_emb.square().sum(-1, keepdim=True) / 2
        #log_phi_u = next_state_emb @ projection - next_state_emb.square().sum(-1, keepdim=True) / 2

        if self.K > 0:
            row_banded_transition = BandedMatrix(
                self.col_banded_transition[None], self.K, self.K,
                #fill = float("-inf"),
                fill = -1e5,
            ).transpose().data[0]

        # O(CD)
        log_denominator = (log_phi_w + log_phi_u.logsumexp(0, keepdim=True)).logsumexp(-1)
        if self.K > 0:
            log_denominator = log_denominator.logaddexp(row_banded_transition.logsumexp(-1))
        normed_log_phi_w = log_phi_w - log_denominator[:,None]

        normalized_phi_w = normed_log_phi_w.exp()
        phi_u = log_phi_u.exp()

        # log space
        #normed_banded_transition = row_banded_transition - log_denominator[:,None]
        # prob space
        if self.K > 0:
            normed_banded_transition = (row_banded_transition - log_denominator[:,None]).exp()

            normed_col_banded_transition = BandedMatrix(
                normed_banded_transition[None],
                self.K, self.K,
                #fill = float("-inf"),
                #fill = -1e5, # <= use this one if log
                fill = 0, # <= use this one if prob
            ).transpose().data[0]

        alphas = []
        Os = []
        #alpha = start * p_emit[:,0] # {N} x C
        alpha_un = start + logp_emit[:,0]
        Ot = alpha_un.logsumexp(-1, keepdim=True)
        log_alpha = alpha_un - Ot
        alpha = log_alpha.exp()
        alphas.append(alpha)
        Os.append(Ot)
        for t in range(T-1):
            gamma = alpha @ normalized_phi_w
            alpha_un = (gamma @ phi_u.T).log()

            #log_band_alpha = logbbmv(log_alpha, normed_col_banded_transition, self.K)
            #import pdb; pdb.set_trace()
            if self.K > 0:
                log_band_alpha = bbmv(alpha, normed_col_banded_transition, self.K).log()

                alpha_un1 = alpha_un.logaddexp(log_band_alpha)
                alpha_un2 = logp_emit[:,t+1] + alpha_un1
            else:
                alpha_un2 = logp_emit[:,t+1] + alpha_un

            Ot = alpha_un2.logsumexp(-1, keepdim=True)
            log_alpha = alpha_un2 - Ot
            alpha = log_alpha.exp()

            alphas.append(alpha)
            Os.append(Ot)
        O = th.cat(Os, -1)
        evidence = O[mask].sum()
        #import pdb; pdb.set_trace()

        return Pack(
            elbo = None,
            evidence = evidence,
            loss = evidence,
        ), alpha.log(), None
        # ^ alpha.log is wrong, need to use lengths to index


    def compute_rff_parameters(self):
        if self.l2norm:
            state_emb = self.state_emb / self.state_emb.norm(dim=-1, keepdim=True)
            next_state_emb = self.next_state_emb / self.next_state_emb.norm(dim=-1, keepdim=True)
        else:
            state_emb = self.state_emb
            next_state_emb = self.next_state_emb

        # sum vectors and sum matrices
        projection = self.projection
        if self.learn_temp == "mul":
            projection = projection * self.temp
        log_phi_w = state_emb @ projection
        log_phi_u = next_state_emb @ projection

        # TODO: performer kernel, abstract away
        #log_phi_w = state_emb @ projection - state_emb.square().sum(-1, keepdim=True) / 2
        #log_phi_u = next_state_emb @ projection - next_state_emb.square().sum(-1, keepdim=True) / 2

        row_banded_transition = BandedMatrix(
            self.col_banded_transition[None], self.K, self.K,
            fill = float("-inf")
        ).transpose().data[0]

        # O(CD)
        log_denominator = (log_phi_w + log_phi_u.logsumexp(0, keepdim=True)).logsumexp(-1)
        if self.K > 0:
            log_denominator = log_denominator.logaddexp(row_banded_transition.logsumexp(-1))
        normed_log_phi_w = log_phi_w - log_denominator[:, None]

        normed_banded_transition = row_banded_transition - log_denominator[:,None]
        normed_banded_transition = normed_banded_transition.exp()
        normed_col_banded_transition = BandedMatrix(
            normed_banded_transition[None],
            self.K, self.K,
            #fill = float("-inf"),
            fill = 0,
        ).transpose().data[0]

        start = self.start()
        emission = self.emission()

        return (
            start,
            (normed_log_phi_w.exp(), log_phi_u.exp(), normed_col_banded_transition),
            emission,
        )

    def compute_rff_loss(
        self,
        text, start, transition, emission,
        word2state=None,
        mask=None, lengths=None,
    ):
        N, T = text.shape
        normalized_phi_w, phi_u, normed_col_banded_transition = transition

        # gather emission
        # N x T x C
        logp_emit = emission[
            th.arange(self.C)[None,None],
            text[:,:,None],
        ]

        alphas = []
        Os = []
        #alpha = start * p_emit[:,0] # {N} x C
        alpha_un = start + logp_emit[:,0]
        Ot = alpha_un.logsumexp(-1, keepdim=True)
        log_alpha = alpha_un - Ot
        alpha = log_alpha.exp()
        alphas.append(alpha)
        Os.append(Ot)
        for t in range(T-1):
            gamma = alpha @ normalized_phi_w
            alpha_un = (gamma @ phi_u.T).log()

            if self.K > 0:
                #log_band_alpha = logbbmv(log_alpha, normed_col_banded_transition, self.K)
                log_band_alpha = bbmv(alpha, normed_col_banded_transition, self.K).log()
                alpha_un1 = alpha_un.logaddexp(log_band_alpha)
                alpha_un2 = logp_emit[:,t+1] + alpha_un1
            else:
                alpha_un2 = logp_emit[:,t+1] + alpha_un

            Ot = alpha_un2.logsumexp(-1, keepdim=True)
            log_alpha = alpha_un2 - Ot
            alpha = log_alpha.exp()

            alphas.append(alpha)
            Os.append(Ot)
        O = th.cat(Os, -1)
        evidence = O[mask].sum()

        return Pack(
            elbo = None,
            evidence = evidence,
            loss = evidence,
        ), alpha.log()
        # ^ alpha wrong, need to index with lengths
