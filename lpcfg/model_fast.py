import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import genbmm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

use_eff = True

class ResLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.linear(x) + x

class CPCFG(torch.nn.Module):
    def __init__(self, V, NT, T, *args, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256, **kwargs): 
        super(CPCFG, self).__init__()
        assert z_dim >= 0
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.share_term = kwargs.get("share_term", False)
        self.share_rule = kwargs.get("share_rule", False)
        self.share_root = kwargs.get("share_root", False)
        self.wo_enc_emb = kwargs.get("wo_enc_emb", False)

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim if self.share_rule else s_dim + z_dim
        self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if self.share_root else s_dim + z_dim
        self.root_mlp = nn.Sequential(nn.Linear(root_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))
        if z_dim > 0:
            i_dim = w_dim
            self.enc_emb = lambda x: x 
            if not self.wo_enc_emb:
                self.enc_emb = nn.Embedding(V, w_dim)
                self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                    bidirectional=True, num_layers=1, batch_first=True)
                i_dim = h_dim * 2
            self.enc_out = nn.Linear(i_dim, z_dim * 2)

        term_dim = s_dim if self.share_term else s_dim + z_dim
        self.term_mlp = nn.Sequential(nn.Linear(term_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    def forward(self, x, lengths, *args, txt=None, txt_lengths=None, use_mean=False, **kwargs):
        """ x, lengths: word ids; txt, txt_lengths: sub-word ids """
        b, n = x.shape[:2]
        batch_size = b 
        if self.z_dim > 0:
            max_pooling = kwargs.get("max_pooling", True)
            enforce_sorted = kwargs.get("enforce_sorted", False)
            item = (x, lengths, txt, txt_lengths) + args if txt is not None else x
            mean, lvar = self.enc(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
        else:
            z = torch.zeros(b, 1).cuda()
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0 and not self.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0 and not self.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            if self.z_dim > 0 and not self.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim # bsz, NT, z
                )
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1) # bsz, NT, H+z
            rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            return rule_prob

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        return (terms_ll, rules_ll, roots_ll), kl 

class CPCFG2(torch.nn.Module):
    def __init__(self, V, NT, T, *args, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256, **kwargs): 
        super(CPCFG2, self).__init__()
        assert z_dim >= 0
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.share_term = kwargs.get("share_term", False)
        self.share_rule = kwargs.get("share_rule", False)
        self.share_root = kwargs.get("share_root", False)
        self.wo_enc_emb = kwargs.get("wo_enc_emb", False)

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim if self.share_rule else s_dim + z_dim
        #self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if self.share_root else s_dim + z_dim
        self.root_mlp = nn.Sequential(nn.Linear(root_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))
        self.b_emb = nn.Parameter(torch.randn(self.NT_T, s_dim))
        self.c_emb = nn.Parameter(torch.randn(self.NT_T, s_dim))
        self.b_mlp = nn.Sequential(nn.Linear(rule_dim+s_dim, s_dim),
                                   ResLayer(s_dim, s_dim),
                                   ResLayer(s_dim, s_dim),
                                   nn.Linear(s_dim, s_dim))
        self.c_mlp = nn.Sequential(nn.Linear(rule_dim+s_dim, s_dim),
                                   ResLayer(s_dim, s_dim),
                                   ResLayer(s_dim, s_dim),
                                   nn.Linear(s_dim, s_dim))
        if z_dim > 0:
            i_dim = w_dim
            self.enc_emb = lambda x: x 
            if not self.wo_enc_emb:
                self.enc_emb = nn.Embedding(V, w_dim)
                self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                    bidirectional=True, num_layers=1, batch_first=True)
                i_dim = h_dim * 2
            self.enc_out = nn.Linear(i_dim, z_dim * 2)

        term_dim = s_dim if self.share_term else s_dim + z_dim
        self.term_mlp = nn.Sequential(nn.Linear(term_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    def forward(self, x, lengths, *args, txt=None, txt_lengths=None, use_mean=False, **kwargs):
        """ x, lengths: word ids; txt, txt_lengths: sub-word ids """
        b, n = x.shape[:2]
        batch_size = b 
        if self.z_dim > 0:
            max_pooling = kwargs.get("max_pooling", True)
            enforce_sorted = kwargs.get("enforce_sorted", False)
            item = (x, lengths, txt, txt_lengths) + args if txt is not None else x
            mean, lvar = self.enc(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
        else:
            z = torch.zeros(b, 1).cuda()
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0 and not self.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0 and not self.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            if self.z_dim > 0 and not self.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim # bsz, NT, z
                )
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1) # bsz, NT, H+z
            A_emb = nonterm_emb.unsqueeze(2).expand(-1, -1, self.NT_T, -1) # bsz, NT, NT_T, H+z
            B_emb = self.b_emb.unsqueeze(0).expand(b, -1, -1).unsqueeze(1).expand(-1, self.NT, -1, -1) # bsz, NT, NT_T, H
            A_B_emb = torch.cat([A_emb, B_emb], -1) # bsz, NT, NT_T, H+z+H
            A_B_emb = self.b_mlp(A_B_emb) # bsz, NT, NT_T, H
            C_emb = self.c_emb.unsqueeze(0).expand(b, -1, -1).unsqueeze(1).expand(-1, self.NT, -1, -1) # bsz, NT, NT_T, H
            A_C_emb = torch.cat([A_emb, C_emb], -1) # bsz, NT, NT_T, H+z+H
            A_C_emb = self.c_mlp(A_C_emb).transpose(-1, -2) # bsz, NT, H, NT_T

            A_BC = torch.matmul(A_B_emb, A_C_emb).view(b, self.NT, self.NT_T**2) # bsz, NT, NT_T*NT_T
            rule_prob = F.log_softmax(A_BC, -1)

            #rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            return rule_prob

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        return (terms_ll, rules_ll, roots_ll), kl 

class CPCFG3(torch.nn.Module):
    def __init__(self, V, NT, T, *args, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256, **kwargs): 
        super(CPCFG3, self).__init__()
        assert z_dim >= 0
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.share_term = kwargs.get("share_term", False)
        self.share_rule = kwargs.get("share_rule", False)
        self.share_root = kwargs.get("share_root", False)
        self.wo_enc_emb = kwargs.get("wo_enc_emb", False)

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim if self.share_rule else s_dim + z_dim
        #self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if self.share_root else s_dim + z_dim
        self.root_mlp = nn.Sequential(nn.Linear(root_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))
        self.ab_emb = nn.Parameter(torch.randn(self.NT * self.NT_T, s_dim))
        self.ac_emb = nn.Parameter(torch.randn(self.NT * self.NT_T, s_dim))
        #self.b_mlp = nn.Sequential(nn.Linear(rule_dim+s_dim, s_dim),
        #                           ResLayer(s_dim, s_dim),
        #                           ResLayer(s_dim, s_dim),
        #                           nn.Linear(s_dim, s_dim))
        #self.c_mlp = nn.Sequential(nn.Linear(rule_dim+s_dim, s_dim),
        #                           ResLayer(s_dim, s_dim),
        #                           ResLayer(s_dim, s_dim),
        #                           nn.Linear(s_dim, s_dim))
        if z_dim > 0:
            assert False, z_dim
            i_dim = w_dim
            self.enc_emb = lambda x: x 
            if not self.wo_enc_emb:
                self.enc_emb = nn.Embedding(V, w_dim)
                self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                    bidirectional=True, num_layers=1, batch_first=True)
                i_dim = h_dim * 2
            self.enc_out = nn.Linear(i_dim, z_dim * 2)

        term_dim = s_dim if self.share_term else s_dim + z_dim
        self.term_mlp = nn.Sequential(nn.Linear(term_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    def forward(self, x, lengths, *args, txt=None, txt_lengths=None, use_mean=False, **kwargs):
        """ x, lengths: word ids; txt, txt_lengths: sub-word ids """
        b, n = x.shape[:2]
        batch_size = b 
        if self.z_dim > 0:
            max_pooling = kwargs.get("max_pooling", True)
            enforce_sorted = kwargs.get("enforce_sorted", False)
            item = (x, lengths, txt, txt_lengths) + args if txt is not None else x
            mean, lvar = self.enc(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
        else:
            z = torch.zeros(b, 1).cuda()
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0 and not self.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0 and not self.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            if self.z_dim > 0 and not self.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim # bsz, NT, z
                )
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1) # bsz, NT, H+z
            assert self.z_dim == 0, self.z_dim
            #A_emb = nonterm_emb.unsqueeze(2).expand(-1, -1, self.NT_T, -1) # bsz, NT, NT_T, H+z
            #B_emb = self.b_emb.unsqueeze(0).expand(b, -1, -1).unsqueeze(1).expand(-1, self.NT, -1, -1) # bsz, NT, NT_T, H
            #A_B_emb = torch.cat([A_emb, B_emb], -1) # bsz, NT, NT_T, H+z+H
            #A_B_emb = self.b_mlp(A_B_emb) # bsz, NT, NT_T, H
            #C_emb = self.c_emb.unsqueeze(0).expand(b, -1, -1).unsqueeze(1).expand(-1, self.NT, -1, -1) # bsz, NT, NT_T, H
            #A_C_emb = torch.cat([A_emb, C_emb], -1) # bsz, NT, NT_T, H+z+H
            #A_C_emb = self.c_mlp(A_C_emb).transpose(-1, -2) # bsz, NT, H, NT_T

            #A_BC = torch.matmul(A_B_emb, A_C_emb).view(b, self.NT, self.NT_T**2) # bsz, NT, NT_T*NT_T
            #rule_prob = F.log_softmax(A_BC, -1)
            AB_emb = self.ab_emb.unsqueeze(0).expand(b, -1, -1).contiguous().view(b*self.NT, self.NT_T, -1) # bsz*NT, NT_T, s
            AC_emb = self.ac_emb.unsqueeze(0).expand(b, -1, -1).contiguous().view(b*self.NT, self.NT_T, -1).transpose(-1, -2) # bsz*NT, s, NT_T
            A_BC = torch.bmm(AB_emb, AC_emb).view(b, self.NT, -1) # bsz, NT, NT_T * NT_T
            rule_prob = F.log_softmax(A_BC, -1)

            #rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            return rule_prob

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        return (terms_ll, rules_ll, roots_ll), kl 


import torch
import torch.nn as nn
import math

def nonnegative_softmax_kernel_feature_creator(
        data: torch.Tensor,
        projection_matrix: torch.Tensor,
        is_query: bool,
        return_log2: bool=False,
        return_log: bool=False,
        relu: bool=False,
        nonorm: bool=False,
        eps: float=0.0001):
    """
    Constructs nonnegative kernel features for fast softmax attention.
    Args:
      data: input for which features are computes
      projection_matrix: random matrix used to compute features
      is_query: predicate indicating whether input data corresponds to queries or
        keys
      eps: numerical stabilizer.
    Returns:
      Random features for fast softmax attention.
    """

    ratio = 1.0 / math.sqrt(projection_matrix.shape[0])

    bsz = data.size(0)
   
    projection = projection_matrix.unsqueeze(0).expand(bsz, -1, -1)

    # Compute wx
    # data:       bsz, len, D
    # projection: bsz, D, #features
    data_dash = torch.bmm(
        data,
        projection
    ) # bsz, len, #features

    # Compute ||x||^2/2
    diag_data = data * data
    diag_data = torch.sum(diag_data, -1) # (bsz, len) ||x||^2
    diag_data = diag_data / 2.0
    diag_data = diag_data.unsqueeze(-1) # bsz, len, 1

    #if return_log2:
    #    if is_query:
    #        # for each query, we can independently scale to avoid overflows
    #        return data_dash - diag_data - torch.max(data_dash, dim=-1, keepdim=True)[0]) + math.log(eps)
    #    else:
    #        # for keys, we need to use the same normalizer to avoid overflows
    #        return data_dash - diag_data - torch.max(data_dash) + math.log(eps)
    if relu:
        assert not return_log, return_log
    if return_log:
        if nonorm:
            return data_dash + math.log(ratio)
        return data_dash - diag_data + math.log(ratio)

    # Compute exp(wx - ||x||^2/2)  
    # (Lemma 1, SM(x, y) = E_{w~N(0,I)} exp(wx - ||x||^2/2) exp(wy - ||y||^2/2))
    if is_query:
        if relu:
            data_dash = ratio * (torch.relu(data_dash - diag_data) + eps)
        else:
            # for each query, we can independently scale to avoid overflows
            data_dash = ratio * (
                torch.exp(data_dash - diag_data -
                        torch.max(data_dash, dim=-1, keepdim=True)[0]) + eps)
    else:
        if relu:
            data_dash = ratio * (torch.relu(data_dash - diag_data) + eps)
        else:
            data_dash = ratio * (
                torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    return data_dash

kernel = nonnegative_softmax_kernel_feature_creator

def get_2d_array(nb_rows, nb_columns, scaling=0):
  nb_full_blocks = int(nb_rows / nb_columns)
  block_list = []
  #rng = self.key
  for _ in range(nb_full_blocks):
    #rng, rng_input = jax.random.split(rng)
    unstructured_block = torch.randn(nb_columns, nb_columns)
    q, _ = torch.qr(unstructured_block)
    q = q.T
    block_list.append(q)
  remaining_rows = nb_rows - nb_full_blocks * nb_columns
  if remaining_rows > 0:
    unstructured_block = torch.randn(nb_columns, nb_columns)
    q, _ = torch.qr(unstructured_block)
    q = q.T
    block_list.append(q[0:remaining_rows])
  final_matrix = torch.cat(block_list, 0)
  #print (final_matrix.size())

  if scaling == 0:
    multiplier = torch.norm(
        torch.randn(nb_rows, nb_columns), dim=-1).view(-1, 1)
  elif scaling == 1:
    multiplier = torch.sqrt(float(nb_columns)) * torch.ones((nb_rows))
  else:
    raise ValueError('Scaling must be one of {0, 1}. Was %s' % scaling)

  return multiplier * final_matrix

class CPCFGFixProj(torch.nn.Module):
    def __init__(self, V, NT, T, *args, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256,
                 num_features=0, **kwargs): 
        super(CPCFGFixProj, self).__init__()
        assert z_dim >= 0
        self.num_features = num_features
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.share_term = kwargs.get("share_term", False)
        self.share_rule = kwargs.get("share_rule", False)
        self.share_root = kwargs.get("share_root", False)
        self.wo_enc_emb = kwargs.get("wo_enc_emb", False)

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim if self.share_rule else s_dim + z_dim
        self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if self.share_root else s_dim + z_dim
        self.root_mlp = nn.Sequential(nn.Linear(root_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))

        projection_matrix = get_2d_array(num_features//2, rule_dim).transpose(0,1)
        self.projection_matrix_anti = torch.cat([projection_matrix, -projection_matrix], -1).cuda()
        self.projection_matrix_anti.requires_grad = False


        if z_dim > 0:
            i_dim = w_dim
            self.enc_emb = lambda x: x 
            if not self.wo_enc_emb:
                self.enc_emb = nn.Embedding(V, w_dim)
                self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                    bidirectional=True, num_layers=1, batch_first=True)
                i_dim = h_dim * 2
            self.enc_out = nn.Linear(i_dim, z_dim * 2)

        term_dim = s_dim if self.share_term else s_dim + z_dim
        self.term_mlp = nn.Sequential(nn.Linear(term_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    def forward(self, x, lengths, *args, txt=None, txt_lengths=None, use_mean=False, **kwargs):
        """ x, lengths: word ids; txt, txt_lengths: sub-word ids """
        b, n = x.shape[:2]
        batch_size = b 
        if self.z_dim > 0:
            max_pooling = kwargs.get("max_pooling", True)
            enforce_sorted = kwargs.get("enforce_sorted", False)
            item = (x, lengths, txt, txt_lengths) + args if txt is not None else x
            mean, lvar = self.enc(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
        else:
            z = torch.zeros(b, 1).cuda()
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0 and not self.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0 and not self.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            if self.z_dim > 0 and not self.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim # bsz, NT, z
                )
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1) # bsz, NT, H+z

            #import pdb; pdb.set_trace()
            #rule_prob_1 = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            nonterm_emb_features = kernel(nonterm_emb, self.projection_matrix_anti.detach(), is_query=True, eps=0.0001) # bsz, NT, r
            bc_emb = self.rule_mlp.weight.unsqueeze(0) # 1, NT_T^2, H+z
            bc_emb_features = kernel(bc_emb, self.projection_matrix_anti.detach(), is_query=False, eps=0.0001) # 1, NT_T^2, r
            rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
            rule_prob = rule_prob / rule_prob.sum(-1, keepdim=True)
            rule_prob = rule_prob.log()


            #rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            with torch.no_grad():
                rule_prob_gt = torch.matmul(nonterm_emb, bc_emb.transpose(-1, -2)) # bsz, NT, NT_T^2
                rule_prob_gt = F.log_softmax(rule_prob_gt, -1)
            return (rule_prob, rule_prob_gt)

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        return (terms_ll, rules_ll, roots_ll), kl 

class CPCFGProj(torch.nn.Module):
    def __init__(self, V, NT, T, *args, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256,
                 num_features=0, **kwargs): 
        super(CPCFGProj, self).__init__()
        assert z_dim >= 0
        self.num_features = num_features
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.share_term = kwargs.get("share_term", False)
        self.share_rule = kwargs.get("share_rule", False)
        self.share_root = kwargs.get("share_root", False)
        self.wo_enc_emb = kwargs.get("wo_enc_emb", False)

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim if self.share_rule else s_dim + z_dim
        self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if self.share_root else s_dim + z_dim
        self.root_mlp = nn.Sequential(nn.Linear(root_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))

        projection_matrix = get_2d_array(num_features//2, rule_dim).transpose(0,1)
        self.projection_matrix_anti = nn.Parameter(torch.cat([projection_matrix, -projection_matrix], -1))
        #self.projection_matrix_anti.requires_grad = False


        if z_dim > 0:
            i_dim = w_dim
            self.enc_emb = lambda x: x 
            if not self.wo_enc_emb:
                self.enc_emb = nn.Embedding(V, w_dim)
                self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                    bidirectional=True, num_layers=1, batch_first=True)
                i_dim = h_dim * 2
            self.enc_out = nn.Linear(i_dim, z_dim * 2)

        term_dim = s_dim if self.share_term else s_dim + z_dim
        self.term_mlp = nn.Sequential(nn.Linear(term_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        self._initialize()

    def _initialize(self):
        #import pdb; pdb.set_trace()
        for n, p in self.named_parameters():
            print (n)
            if p.dim() > 1 and ('projection_matrix' not in n):
                torch.nn.init.xavier_uniform_(p)
        #import pdb; pdb.set_trace()

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    def forward(self, x, lengths, *args, txt=None, txt_lengths=None, use_mean=False, **kwargs):
        """ x, lengths: word ids; txt, txt_lengths: sub-word ids """
        b, n = x.shape[:2]
        batch_size = b 
        if self.z_dim > 0:
            max_pooling = kwargs.get("max_pooling", True)
            enforce_sorted = kwargs.get("enforce_sorted", False)
            item = (x, lengths, txt, txt_lengths) + args if txt is not None else x
            mean, lvar = self.enc(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
        else:
            z = torch.zeros(b, 1).cuda()
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0 and not self.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0 and not self.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            if self.z_dim > 0 and not self.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim # bsz, NT, z
                )
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1) # bsz, NT, H+z

            #import pdb; pdb.set_trace()
            #rule_prob_1 = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            nonterm_emb_features = kernel(nonterm_emb, self.projection_matrix_anti, is_query=True, eps=0.0001) # bsz, NT, r
            bc_emb = self.rule_mlp.weight.unsqueeze(0) # 1, NT_T^2, H+z
            bc_emb_features = kernel(bc_emb, self.projection_matrix_anti, is_query=False, eps=0.0001) # 1, NT_T^2, r
            rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
            rule_prob = rule_prob / rule_prob.sum(-1, keepdim=True)
            rule_prob = rule_prob.log()


            #rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            with torch.no_grad():
                rule_prob_gt = torch.matmul(nonterm_emb, bc_emb.transpose(-1, -2)) # bsz, NT, NT_T^2
                rule_prob_gt = F.log_softmax(rule_prob_gt, -1)
            return (rule_prob, rule_prob_gt)

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        return (terms_ll, rules_ll, roots_ll), kl 

class CPCFGProjDebug(torch.nn.Module):
    def __init__(self, V, NT, T, *args, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256,
                 num_features=0, **kwargs): 
        super(CPCFGProjDebug, self).__init__()
        assert z_dim >= 0
        self.num_features = num_features
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.share_term = kwargs.get("share_term", False)
        self.share_rule = kwargs.get("share_rule", False)
        self.share_root = kwargs.get("share_root", False)
        self.wo_enc_emb = kwargs.get("wo_enc_emb", False)

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim if self.share_rule else s_dim + z_dim
        self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if self.share_root else s_dim + z_dim
        self.root_mlp = nn.Sequential(nn.Linear(root_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))

        projection_matrix = get_2d_array(num_features//2, rule_dim).transpose(0,1)
        self.projection_matrix_anti = nn.Parameter(torch.cat([projection_matrix, -projection_matrix], -1))
        #self.projection_matrix_anti.requires_grad = False


        if z_dim > 0:
            i_dim = w_dim
            self.enc_emb = lambda x: x 
            if not self.wo_enc_emb:
                self.enc_emb = nn.Embedding(V, w_dim)
                self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                    bidirectional=True, num_layers=1, batch_first=True)
                i_dim = h_dim * 2
            self.enc_out = nn.Linear(i_dim, z_dim * 2)

        term_dim = s_dim if self.share_term else s_dim + z_dim
        self.term_mlp = nn.Sequential(nn.Linear(term_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        self._initialize()

    def _initialize(self):
        #import pdb; pdb.set_trace()
        for n, p in self.named_parameters():
            print (n)
            if p.dim() > 1 and ('projection_matrix' not in n):
                torch.nn.init.xavier_uniform_(p)
        #import pdb; pdb.set_trace()

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    def forward(self, x, lengths, *args, txt=None, txt_lengths=None, use_mean=False, **kwargs):
        """ x, lengths: word ids; txt, txt_lengths: sub-word ids """
        b, n = x.shape[:2]
        batch_size = b 
        if self.z_dim > 0:
            max_pooling = kwargs.get("max_pooling", True)
            enforce_sorted = kwargs.get("enforce_sorted", False)
            item = (x, lengths, txt, txt_lengths) + args if txt is not None else x
            mean, lvar = self.enc(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
        else:
            z = torch.zeros(b, 1).cuda()
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0 and not self.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0 and not self.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            if self.z_dim > 0 and not self.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim # bsz, NT, z
                )
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1) # bsz, NT, H+z

            #import pdb; pdb.set_trace()
            #rule_prob_1 = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
# nonterm_emb: bsz, NT, H+z
# bc_emb: bsz, NT_T^2, H+z
            #nonterm_emb_features = kernel(nonterm_emb, self.projection_matrix_anti, is_query=True, eps=0.0001) # bsz, NT, r
            bc_emb = self.rule_mlp.weight.unsqueeze(0) # 1, NT_T^2, H+z
            #bc_emb_features = kernel(bc_emb, self.projection_matrix_anti, is_query=False, eps=0.0001) # 1, NT_T^2, r
            #rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
            #rule_prob = rule_prob / rule_prob.sum(-1, keepdim=True)
            #rule_prob = rule_prob.log()
            rule_prob = torch.matmul(nonterm_emb, bc_emb.transpose(-1, -2)) # bsz, NT, NT_T^2
            rule_prob = F.log_softmax(rule_prob, -1)


            #rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            return rule_prob

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        return (terms_ll, rules_ll, roots_ll), kl 

class CPCFGProj2(torch.nn.Module):
    def __init__(self, V, NT, T, *args, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256,
                 num_features=0, **kwargs): 
        super(CPCFGProj2, self).__init__()
        assert z_dim >= 0
        self.num_features = num_features
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.share_term = kwargs.get("share_term", False)
        self.share_rule = kwargs.get("share_rule", False)
        self.share_root = kwargs.get("share_root", False)
        self.wo_enc_emb = kwargs.get("wo_enc_emb", False)

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim if self.share_rule else s_dim + z_dim
        self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if self.share_root else s_dim + z_dim
        self.root_mlp = nn.Sequential(nn.Linear(root_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))

        projection_matrix = get_2d_array(num_features//2, rule_dim).transpose(0,1)
        self.projection_matrix_anti = nn.Parameter(torch.cat([projection_matrix, -projection_matrix], -1))
        #self.projection_matrix_anti.requires_grad = False


        if z_dim > 0:
            i_dim = w_dim
            self.enc_emb = lambda x: x 
            if not self.wo_enc_emb:
                self.enc_emb = nn.Embedding(V, w_dim)
                self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                    bidirectional=True, num_layers=1, batch_first=True)
                i_dim = h_dim * 2
            self.enc_out = nn.Linear(i_dim, z_dim * 2)

        term_dim = s_dim if self.share_term else s_dim + z_dim
        self.term_mlp = nn.Sequential(nn.Linear(term_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        self._initialize()

    def _initialize(self):
        #import pdb; pdb.set_trace()
        for n, p in self.named_parameters():
            print (n)
            if p.dim() > 1 and ('projection_matrix' not in n):
                torch.nn.init.xavier_uniform_(p)
        #import pdb; pdb.set_trace()

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    def forward(self, x, lengths, *args, txt=None, txt_lengths=None, use_mean=False, **kwargs):
        """ x, lengths: word ids; txt, txt_lengths: sub-word ids """
        b, n = x.shape[:2]
        batch_size = b 
        if self.z_dim > 0:
            max_pooling = kwargs.get("max_pooling", True)
            enforce_sorted = kwargs.get("enforce_sorted", False)
            item = (x, lengths, txt, txt_lengths) + args if txt is not None else x
            mean, lvar = self.enc(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
        else:
            z = torch.zeros(b, 1).cuda()
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0 and not self.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0 and not self.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            if self.z_dim > 0 and not self.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim # bsz, NT, z
                )
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1) # bsz, NT, H+z

            #import pdb; pdb.set_trace()
            #rule_prob_1 = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            nonterm_emb_features = kernel(nonterm_emb, self.projection_matrix_anti, is_query=True, eps=0.0001, return_log=True) # bsz, NT, r
            bc_emb = self.rule_mlp.weight.unsqueeze(0) # 1, NT_T^2, H+z
            bc_emb_features = kernel(bc_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=True) # 1, NT_T^2, r
            #import pdb; pdb.set_trace()
# bsz, NT, 1, r
# bsz, 1, NT_T^2, r

            rule_prob = (
                     nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:]
                     ).logsumexp(-1).log_softmax(-1)

            #rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
            #rule_prob = rule_prob / rule_prob.sum(-1, keepdim=True)
            #rule_prob = rule_prob.log()


            #rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            with torch.no_grad():
                rule_prob_gt = torch.matmul(nonterm_emb, bc_emb.transpose(-1, -2)) # bsz, NT, NT_T^2
                rule_prob_gt = F.log_softmax(rule_prob_gt, -1)
            return (rule_prob, rule_prob_gt)

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        return (terms_ll, rules_ll, roots_ll), kl 

class CPCFGProjb(torch.nn.Module):
    def __init__(self, V, NT, T, *args, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256,
                 num_features=0, 
                 relu=False,
                 norm_square=0,
                 unit_norm=False,
                 temperature=1,
                 **kwargs): 
        super(CPCFGProjb, self).__init__()
        assert z_dim >= 0
        self.num_features = num_features
        self.relu = relu
        self.unit_norm = unit_norm
        self.temperature = temperature
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.share_term = kwargs.get("share_term", False)
        self.share_rule = kwargs.get("share_rule", False)
        self.share_root = kwargs.get("share_root", False)
        self.wo_enc_emb = kwargs.get("wo_enc_emb", False)

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim if self.share_rule else s_dim + z_dim
        self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if self.share_root else s_dim + z_dim
        self.root_mlp = nn.Sequential(nn.Linear(root_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))

        projection_matrix = get_2d_array(num_features//2, rule_dim).transpose(0,1)
        if temperature != 1:
            projection_matrix = projection_matrix * math.sqrt(temperature)
        self.projection_matrix_anti = nn.Parameter(torch.cat([projection_matrix, -projection_matrix], -1))
        #self.projection_matrix_anti.requires_grad = False


        if z_dim > 0:
            i_dim = w_dim
            self.enc_emb = lambda x: x 
            if not self.wo_enc_emb:
                self.enc_emb = nn.Embedding(V, w_dim)
                self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                    bidirectional=True, num_layers=1, batch_first=True)
                i_dim = h_dim * 2
            self.enc_out = nn.Linear(i_dim, z_dim * 2)

        term_dim = s_dim if self.share_term else s_dim + z_dim
        self.term_mlp = nn.Sequential(nn.Linear(term_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        self._initialize()
        if self.unit_norm:
            self._normalize()

    def _normalize(self):
        assert self.unit_norm
        self.rule_mlp.weight.data.div_(self.rule_mlp.weight.data.norm(dim=-1, keepdim=True))
        self.nonterm_emb.data.div_(self.nonterm_emb.data.norm(dim=-1, keepdim=True))
        #import pdb; pdb.set_trace()

    def _initialize(self):
        #import pdb; pdb.set_trace()
        for n, p in self.named_parameters():
            print (n)
            if p.dim() > 1 and ('projection_matrix' not in n):
                torch.nn.init.xavier_uniform_(p)
        #import pdb; pdb.set_trace()

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    def forward(self, x, lengths, *args, txt=None, txt_lengths=None, use_mean=False, **kwargs):
        """ x, lengths: word ids; txt, txt_lengths: sub-word ids """
        b, n = x.shape[:2]
        batch_size = b 
        if self.z_dim > 0:
            max_pooling = kwargs.get("max_pooling", True)
            enforce_sorted = kwargs.get("enforce_sorted", False)
            item = (x, lengths, txt, txt_lengths) + args if txt is not None else x
            mean, lvar = self.enc(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
        else:
            z = torch.zeros(b, 1).cuda()
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0 and not self.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0 and not self.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            if self.z_dim > 0 and not self.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim # bsz, NT, z
                )
                assert False
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1) # bsz, NT, H+z

            #import pdb; pdb.set_trace()
            #rule_prob_1 = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            bc_emb = self.rule_mlp.weight.unsqueeze(0) # 1, NT_T^2, H+z
            return_log = True
            if self.relu:
                return_log = False
            if self.unit_norm:
                nonterm_emb = nonterm_emb / nonterm_emb.norm(dim=-1, keepdim=True)
                bc_emb = bc_emb / bc_emb.norm(dim=-1, keepdim=True)
            nonterm_emb_features = kernel(nonterm_emb, self.projection_matrix_anti, is_query=True, eps=0.0001, return_log=return_log, relu=self.relu) # bsz, NT, r
            bc_emb_bias = self.rule_mlp.bias.view(1, 1, -1) # 1, 1, NT_T^2
            bc_emb_features = kernel(bc_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT_T^2, r
            #import pdb; pdb.set_trace()
# bsz, NT, 1, r
# bsz, 1, NT_T^2, r

# bsz, NT, NT_T^2

            #import pdb; pdb.set_trace()
            if not self.relu:
                rule_prob = ((
                         nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:]
                         ).logsumexp(-1) + bc_emb_bias).log_softmax(-1)
            else:
                rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
                rule_prob_log = rule_prob.log() + bc_emb_bias
                rule_prob = rule_prob_log.log_softmax(-1)
            #rule_prob2 = ((
            #         nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] + bc_emb_bias.unsqueeze(-1)
            #         ).logsumexp(-1)).log_softmax(-1)
            #import pdb; pdb.set_trace()
            #rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
            #rule_prob = rule_prob / rule_prob.sum(-1, keepdim=True)
            #rule_prob = rule_prob.log()


            #rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            with torch.no_grad():
                rule_prob_gt = torch.matmul(nonterm_emb, bc_emb.transpose(-1, -2)) + bc_emb_bias # bsz, NT, NT_T^2
                rule_prob_gt = F.log_softmax(rule_prob_gt, -1)
            return (rule_prob, rule_prob_gt, bc_emb_bias, bc_emb_features, nonterm_emb_features)

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        bc_emb_bias, bc_emb_features, nonterm_emb_features = rules_ll[2:]
        return (terms_ll, rules_ll[:2], roots_ll), kl 
        if not self.relu: # this is actually slower
            return (terms_ll, rules_ll[:2], roots_ll, bc_emb_bias, bc_emb_features, nonterm_emb_features), kl 
        else:
            return (terms_ll, rules_ll[:2], roots_ll), kl 

class CPCFGProjDebugBand(torch.nn.Module):
    def __init__(self, V, NT, T, *args, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256,
                 unit_norm=False,
                 num_features=0, **kwargs): 
        super(CPCFGProjDebugBand, self).__init__()
        assert z_dim >= 0
        self.num_features = num_features
        self.unit_norm = unit_norm
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.share_term = kwargs.get("share_term", False)
        self.share_rule = kwargs.get("share_rule", False)
        self.share_root = kwargs.get("share_root", False)
        self.wo_enc_emb = kwargs.get("wo_enc_emb", False)

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim if self.share_rule else s_dim + z_dim
        self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if self.share_root else s_dim + z_dim
        self.root_mlp = nn.Sequential(nn.Linear(root_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))

        projection_matrix = get_2d_array(num_features//2, rule_dim).transpose(0,1)
        self.projection_matrix_anti = nn.Parameter(torch.cat([projection_matrix, -projection_matrix], -1))
        #self.projection_matrix_anti.requires_grad = False


        if z_dim > 0:
            i_dim = w_dim
            self.enc_emb = lambda x: x 
            if not self.wo_enc_emb:
                self.enc_emb = nn.Embedding(V, w_dim)
                self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                    bidirectional=True, num_layers=1, batch_first=True)
                i_dim = h_dim * 2
            self.enc_out = nn.Linear(i_dim, z_dim * 2)

        term_dim = s_dim if self.share_term else s_dim + z_dim
        self.term_mlp = nn.Sequential(nn.Linear(term_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        self._initialize()
        if self.unit_norm:
            self._normalize()

    def _normalize(self):
        assert self.unit_norm
        self.rule_mlp.weight.data.div_(self.rule_mlp.weight.data.norm(dim=-1, keepdim=True))
        self.nonterm_emb.data.div_(self.nonterm_emb.data.norm(dim=-1, keepdim=True))
        #import pdb; pdb.set_trace()

    def _initialize(self):
        #import pdb; pdb.set_trace()
        for n, p in self.named_parameters():
            print (n)
            if p.dim() > 1 and ('projection_matrix' not in n):
                torch.nn.init.xavier_uniform_(p)
        #import pdb; pdb.set_trace()

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    def forward(self, x, lengths, *args, txt=None, txt_lengths=None, use_mean=False, **kwargs):
        """ x, lengths: word ids; txt, txt_lengths: sub-word ids """
        b, n = x.shape[:2]
        batch_size = b 
        if self.z_dim > 0:
            max_pooling = kwargs.get("max_pooling", True)
            enforce_sorted = kwargs.get("enforce_sorted", False)
            item = (x, lengths, txt, txt_lengths) + args if txt is not None else x
            mean, lvar = self.enc(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
        else:
            z = torch.zeros(b, 1).cuda()
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0 and not self.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0 and not self.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            if self.z_dim > 0 and not self.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim # bsz, NT, z
                )
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1) # bsz, NT, H+z

            #import pdb; pdb.set_trace()
            #rule_prob_1 = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
# nonterm_emb: bsz, NT, H+z
# bc_emb: bsz, NT_T^2, H+z
            #nonterm_emb_features = kernel(nonterm_emb, self.projection_matrix_anti, is_query=True, eps=0.0001) # bsz, NT, r
            bc_emb = self.rule_mlp.weight.unsqueeze(0) # 1, NT_T^2, H+z
            #bc_emb_features = kernel(bc_emb, self.projection_matrix_anti, is_query=False, eps=0.0001) # 1, NT_T^2, r
            #rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
            #rule_prob = rule_prob / rule_prob.sum(-1, keepdim=True)
            #rule_prob = rule_prob.log()
            if self.unit_norm:
                nonterm_emb = nonterm_emb / nonterm_emb.norm(dim=-1, keepdim=True)
                bc_emb = bc_emb / bc_emb.norm(dim=-1, keepdim=True)
            rule_prob = torch.matmul(nonterm_emb, bc_emb.transpose(-1, -2)) # bsz, NT, NT_T^2
            #rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            if self.band > 0:
                #import pdb; pdb.set_trace()
                x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                diff = torch.abs(x_ids - y_ids)
                BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                band_BC_bc = (diff < self.band) & BC_bc
                band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                band = band_BC_bc | band_Bc | band_bC
                band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            elif self.band < 0:
                #import pdb; pdb.set_trace()
                #x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                #y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                #diff = torch.abs(x_ids - y_ids)
                #BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                #Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                #bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                #band_BC_bc = (diff < self.band) & BC_bc
                #band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                #band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                #band = band_BC_bc | band_Bc | band_bC
                ids = list(range(self.NT_T**2))
                import random
                random.seed(1234)
                random.shuffle(ids)
                ids = ids[(((-self.band-1)*2+1)*2*self.NT_T):]
                #rule_prob[ids] = -float('inf')
                band = torch.zeros(self.NT_T, self.NT_T).cuda()
                band = band.view(-1)
                band[ids] = -float('inf')
                #band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)

            rule_prob = F.log_softmax(rule_prob, -1)


            #rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            return rule_prob

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        self.rules_ll = rules_ll[0]
        return (terms_ll, rules_ll, roots_ll), kl 

class CPCFGProjbBand(torch.nn.Module):
    def __init__(self, V, NT, T, *args, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256,
                 num_features=0, 
                 relu=False,
                 norm_square=0,
                 unit_norm=False,
                 temperature=1,
                 **kwargs): 
        super(CPCFGProjbBand, self).__init__()
        assert z_dim >= 0
        self.num_features = num_features
        self.relu = relu
        self.unit_norm = unit_norm
        self.temperature = temperature
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.genbmm = False
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.share_term = kwargs.get("share_term", False)
        self.share_rule = kwargs.get("share_rule", False)
        self.share_root = kwargs.get("share_root", False)
        self.wo_enc_emb = kwargs.get("wo_enc_emb", False)

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim if self.share_rule else s_dim + z_dim
        self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if self.share_root else s_dim + z_dim
        self.root_mlp = nn.Sequential(nn.Linear(root_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))

        projection_matrix = get_2d_array(num_features//2, rule_dim).transpose(0,1)
        if temperature != 1:
            projection_matrix = projection_matrix * math.sqrt(temperature)
        self.projection_matrix_anti = nn.Parameter(torch.cat([projection_matrix, -projection_matrix], -1))
        #self.projection_matrix_anti.requires_grad = False


        if z_dim > 0:
            i_dim = w_dim
            self.enc_emb = lambda x: x 
            if not self.wo_enc_emb:
                self.enc_emb = nn.Embedding(V, w_dim)
                self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                    bidirectional=True, num_layers=1, batch_first=True)
                i_dim = h_dim * 2
            self.enc_out = nn.Linear(i_dim, z_dim * 2)

        term_dim = s_dim if self.share_term else s_dim + z_dim
        self.term_mlp = nn.Sequential(nn.Linear(term_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        self._initialize()
        if self.unit_norm:
            self._normalize()

    def _normalize(self):
        assert self.unit_norm
        self.rule_mlp.weight.data.div_(self.rule_mlp.weight.data.norm(dim=-1, keepdim=True))
        self.nonterm_emb.data.div_(self.nonterm_emb.data.norm(dim=-1, keepdim=True))
        #import pdb; pdb.set_trace()

    def _initialize(self):
        #import pdb; pdb.set_trace()
        for n, p in self.named_parameters():
            print (n)
            if p.dim() > 1 and ('projection_matrix' not in n):
                torch.nn.init.xavier_uniform_(p)
        #import pdb; pdb.set_trace()

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    def forward(self, x, lengths, *args, txt=None, txt_lengths=None, use_mean=False, **kwargs):
        """ x, lengths: word ids; txt, txt_lengths: sub-word ids """
        b, n = x.shape[:2]
        batch_size = b 
        if self.z_dim > 0:
            max_pooling = kwargs.get("max_pooling", True)
            enforce_sorted = kwargs.get("enforce_sorted", False)
            item = (x, lengths, txt, txt_lengths) + args if txt is not None else x
            mean, lvar = self.enc(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
        else:
            z = torch.zeros(b, 1).cuda()
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0 and not self.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0 and not self.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            if self.z_dim > 0 and not self.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim # bsz, NT, z
                )
                assert False
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1) # bsz, NT, H+z

            #import pdb; pdb.set_trace()
            #rule_prob_1 = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            bc_emb = self.rule_mlp.weight.unsqueeze(0) # 1, NT_T^2, H+z
            return_log = True
            if self.relu:
                return_log = False
            if self.unit_norm:
                nonterm_emb = nonterm_emb / nonterm_emb.norm(dim=-1, keepdim=True)
                bc_emb = bc_emb / bc_emb.norm(dim=-1, keepdim=True)
            nonterm_emb_features = kernel(nonterm_emb, self.projection_matrix_anti, is_query=True, eps=0.0001, return_log=return_log, relu=self.relu) # bsz, NT, r
            bc_emb_bias = self.rule_mlp.bias.view(1, 1, -1) # 1, 1, NT_T^2
            bc_emb_features = kernel(bc_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT_T^2, r
            #import pdb; pdb.set_trace()
# bsz, NT, 1, r
# bsz, 1, NT_T^2, r

# bsz, NT, NT_T^2

            #import pdb; pdb.set_trace()
            if not self.relu:
                if not self.genbmm:
                    rule_prob = (
                             nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT_T^2, r
                             ).logsumexp(-1) + bc_emb_bias#.log_softmax(-1) # bsz, NT, NT_T^2
                else:
                    bc_emb_features_2 = bc_emb_features.expand(b, -1, -1).transpose(-1, -2).contiguous() # b, r, NT_T^2    # b, NT, r
                    rule_prob = genbmm.logbmm(nonterm_emb_features, bc_emb_features_2) + bc_emb_bias  # b, NT, NT_T^2
            else:
                rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
                rule_prob_log = rule_prob.log() + bc_emb_bias
                #rule_prob = rule_prob_log.log_softmax(-1)
            #rule_prob2 = ((
            #         nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] + bc_emb_bias.unsqueeze(-1)
            #         ).logsumexp(-1)).log_softmax(-1)
            #import pdb; pdb.set_trace()
            #rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
            #rule_prob = rule_prob / rule_prob.sum(-1, keepdim=True)
            #rule_prob = rule_prob.log()


            if self.band > 0:
                #import pdb; pdb.set_trace()
                x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                diff = torch.abs(x_ids - y_ids)
                BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                band_BC_bc = (diff < self.band) & BC_bc
                band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                band = band_BC_bc | band_Bc | band_bC
                band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            elif self.band < 0:
                #import pdb; pdb.set_trace()
                #x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                #y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                #diff = torch.abs(x_ids - y_ids)
                #BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                #Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                #bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                #band_BC_bc = (diff < self.band) & BC_bc
                #band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                #band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                #band = band_BC_bc | band_Bc | band_bC
                ids = list(range(self.NT_T**2))
                import random
                random.seed(1234)
                random.shuffle(ids)
                ids = ids[(((-self.band-1)*2+1)*2*self.NT_T):]
                #rule_prob[ids] = -float('inf')
                band = torch.zeros(self.NT_T, self.NT_T).cuda()
                band = band.view(-1)
                band[ids] = -float('inf')
                #band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            #rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            rule_prob = F.log_softmax(rule_prob, -1)
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            with torch.no_grad():
                rule_prob_gt = torch.matmul(nonterm_emb, bc_emb.transpose(-1, -2)) + bc_emb_bias # bsz, NT, NT_T^2
                rule_prob_gt = F.log_softmax(rule_prob_gt, -1)
            return (rule_prob, rule_prob_gt, bc_emb_bias, bc_emb_features, nonterm_emb_features)

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        bc_emb_bias, bc_emb_features, nonterm_emb_features = rules_ll[2:]
        self.rules_ll = rules_ll[0][0]
        if False and (not self.relu):
            return (terms_ll, rules_ll[:2], roots_ll, bc_emb_bias, bc_emb_features, nonterm_emb_features), kl 
        else:
            return (terms_ll, rules_ll[:2], roots_ll), kl 

class CPCFGProjbBandNoanti(torch.nn.Module):
    def __init__(self, V, NT, T, *args, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256,
                 num_features=0, 
                 relu=False,
                 norm_square=0,
                 unit_norm=False,
                 temperature=1,
                 **kwargs): 
        super(CPCFGProjbBandNoanti, self).__init__()
        assert z_dim >= 0
        self.num_features = num_features
        self.relu = relu
        self.unit_norm = unit_norm
        self.temperature = temperature
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.genbmm = False
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.share_term = kwargs.get("share_term", False)
        self.share_rule = kwargs.get("share_rule", False)
        self.share_root = kwargs.get("share_root", False)
        self.wo_enc_emb = kwargs.get("wo_enc_emb", False)

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim if self.share_rule else s_dim + z_dim
        self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if self.share_root else s_dim + z_dim
        self.root_mlp = nn.Sequential(nn.Linear(root_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))

        projection_matrix = get_2d_array(num_features, rule_dim).transpose(0,1)
        if temperature != 1:
            projection_matrix = projection_matrix * math.sqrt(temperature)
        self.projection_matrix_anti = nn.Parameter(projection_matrix)
        #self.projection_matrix_anti.requires_grad = False


        if z_dim > 0:
            i_dim = w_dim
            self.enc_emb = lambda x: x 
            if not self.wo_enc_emb:
                self.enc_emb = nn.Embedding(V, w_dim)
                self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                    bidirectional=True, num_layers=1, batch_first=True)
                i_dim = h_dim * 2
            self.enc_out = nn.Linear(i_dim, z_dim * 2)

        term_dim = s_dim if self.share_term else s_dim + z_dim
        self.term_mlp = nn.Sequential(nn.Linear(term_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        self._initialize()
        if self.unit_norm:
            self._normalize()

    def _normalize(self):
        assert self.unit_norm
        self.rule_mlp.weight.data.div_(self.rule_mlp.weight.data.norm(dim=-1, keepdim=True))
        self.nonterm_emb.data.div_(self.nonterm_emb.data.norm(dim=-1, keepdim=True))
        #import pdb; pdb.set_trace()

    def _initialize(self):
        #import pdb; pdb.set_trace()
        for n, p in self.named_parameters():
            print (n)
            if p.dim() > 1 and ('projection_matrix' not in n):
                torch.nn.init.xavier_uniform_(p)
        #import pdb; pdb.set_trace()

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    def forward(self, x, lengths, *args, txt=None, txt_lengths=None, use_mean=False, **kwargs):
        """ x, lengths: word ids; txt, txt_lengths: sub-word ids """
        b, n = x.shape[:2]
        batch_size = b 
        if self.z_dim > 0:
            max_pooling = kwargs.get("max_pooling", True)
            enforce_sorted = kwargs.get("enforce_sorted", False)
            item = (x, lengths, txt, txt_lengths) + args if txt is not None else x
            mean, lvar = self.enc(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
        else:
            z = torch.zeros(b, 1).cuda()
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0 and not self.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0 and not self.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            if self.z_dim > 0 and not self.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim # bsz, NT, z
                )
                assert False
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1) # bsz, NT, H+z

            #import pdb; pdb.set_trace()
            #rule_prob_1 = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            bc_emb = self.rule_mlp.weight.unsqueeze(0) # 1, NT_T^2, H+z
            return_log = True
            if self.relu:
                return_log = False
            if self.unit_norm:
                nonterm_emb = nonterm_emb / nonterm_emb.norm(dim=-1, keepdim=True)
                bc_emb = bc_emb / bc_emb.norm(dim=-1, keepdim=True)
            nonterm_emb_features = kernel(nonterm_emb, self.projection_matrix_anti, is_query=True, eps=0.0001, return_log=return_log, relu=self.relu) # bsz, NT, r
            bc_emb_bias = self.rule_mlp.bias.view(1, 1, -1) # 1, 1, NT_T^2
            bc_emb_features = kernel(bc_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT_T^2, r
            #import pdb; pdb.set_trace()
# bsz, NT, 1, r
# bsz, 1, NT_T^2, r

# bsz, NT, NT_T^2

            #import pdb; pdb.set_trace()
            if not self.relu:
                if not self.genbmm:
                    rule_prob = (
                             nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT_T^2, r
                             ).logsumexp(-1) + bc_emb_bias#.log_softmax(-1) # bsz, NT, NT_T^2
                else:
                    bc_emb_features_2 = bc_emb_features.expand(b, -1, -1).transpose(-1, -2).contiguous() # b, r, NT_T^2    # b, NT, r
                    rule_prob = genbmm.logbmm(nonterm_emb_features, bc_emb_features_2) + bc_emb_bias  # b, NT, NT_T^2
            else:
                rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
                rule_prob_log = rule_prob.log() + bc_emb_bias
                #rule_prob = rule_prob_log.log_softmax(-1)
            #rule_prob2 = ((
            #         nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] + bc_emb_bias.unsqueeze(-1)
            #         ).logsumexp(-1)).log_softmax(-1)
            #import pdb; pdb.set_trace()
            #rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
            #rule_prob = rule_prob / rule_prob.sum(-1, keepdim=True)
            #rule_prob = rule_prob.log()


            if self.band > 0:
                #import pdb; pdb.set_trace()
                x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                diff = torch.abs(x_ids - y_ids)
                BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                band_BC_bc = (diff < self.band) & BC_bc
                band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                band = band_BC_bc | band_Bc | band_bC
                band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            elif self.band < 0:
                #import pdb; pdb.set_trace()
                #x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                #y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                #diff = torch.abs(x_ids - y_ids)
                #BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                #Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                #bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                #band_BC_bc = (diff < self.band) & BC_bc
                #band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                #band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                #band = band_BC_bc | band_Bc | band_bC
                ids = list(range(self.NT_T**2))
                import random
                random.seed(1234)
                random.shuffle(ids)
                ids = ids[(((-self.band-1)*2+1)*2*self.NT_T):]
                #rule_prob[ids] = -float('inf')
                band = torch.zeros(self.NT_T, self.NT_T).cuda()
                band = band.view(-1)
                band[ids] = -float('inf')
                #band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            #rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            rule_prob = F.log_softmax(rule_prob, -1)
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            with torch.no_grad():
                rule_prob_gt = torch.matmul(nonterm_emb, bc_emb.transpose(-1, -2)) + bc_emb_bias # bsz, NT, NT_T^2
                rule_prob_gt = F.log_softmax(rule_prob_gt, -1)
            return (rule_prob, rule_prob_gt, bc_emb_bias, bc_emb_features, nonterm_emb_features)

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        bc_emb_bias, bc_emb_features, nonterm_emb_features = rules_ll[2:]
        self.rules_ll = rules_ll[0][0]
        if False and (not self.relu):
            return (terms_ll, rules_ll[:2], roots_ll, bc_emb_bias, bc_emb_features, nonterm_emb_features), kl 
        else:
            return (terms_ll, rules_ll[:2], roots_ll), kl 

class CPCFGProjbBandNonorm(torch.nn.Module):
    def __init__(self, V, NT, T, *args, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256,
                 num_features=0, 
                 relu=False,
                 norm_square=0,
                 unit_norm=False,
                 temperature=1,
                 **kwargs): 
        super(CPCFGProjbBandNonorm, self).__init__()
        assert z_dim >= 0
        self.num_features = num_features
        self.relu = relu
        self.unit_norm = unit_norm
        self.temperature = temperature
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.genbmm = False
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.share_term = kwargs.get("share_term", False)
        self.share_rule = kwargs.get("share_rule", False)
        self.share_root = kwargs.get("share_root", False)
        self.wo_enc_emb = kwargs.get("wo_enc_emb", False)

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim if self.share_rule else s_dim + z_dim
        self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if self.share_root else s_dim + z_dim
        self.root_mlp = nn.Sequential(nn.Linear(root_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))

        projection_matrix = get_2d_array(num_features//2, rule_dim).transpose(0,1)
        if temperature != 1:
            projection_matrix = projection_matrix * math.sqrt(temperature)
        self.projection_matrix_anti = nn.Parameter(torch.cat([projection_matrix, -projection_matrix], -1))
        #self.projection_matrix_anti.requires_grad = False


        if z_dim > 0:
            i_dim = w_dim
            self.enc_emb = lambda x: x 
            if not self.wo_enc_emb:
                self.enc_emb = nn.Embedding(V, w_dim)
                self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                    bidirectional=True, num_layers=1, batch_first=True)
                i_dim = h_dim * 2
            self.enc_out = nn.Linear(i_dim, z_dim * 2)

        term_dim = s_dim if self.share_term else s_dim + z_dim
        self.term_mlp = nn.Sequential(nn.Linear(term_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        self._initialize()
        if self.unit_norm:
            self._normalize()

    def _normalize(self):
        assert self.unit_norm
        self.rule_mlp.weight.data.div_(self.rule_mlp.weight.data.norm(dim=-1, keepdim=True))
        self.nonterm_emb.data.div_(self.nonterm_emb.data.norm(dim=-1, keepdim=True))
        #import pdb; pdb.set_trace()

    def _initialize(self):
        #import pdb; pdb.set_trace()
        for n, p in self.named_parameters():
            print (n)
            if p.dim() > 1 and ('projection_matrix' not in n):
                torch.nn.init.xavier_uniform_(p)
        #import pdb; pdb.set_trace()

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    def forward(self, x, lengths, *args, txt=None, txt_lengths=None, use_mean=False, **kwargs):
        """ x, lengths: word ids; txt, txt_lengths: sub-word ids """
        b, n = x.shape[:2]
        batch_size = b 
        if self.z_dim > 0:
            max_pooling = kwargs.get("max_pooling", True)
            enforce_sorted = kwargs.get("enforce_sorted", False)
            item = (x, lengths, txt, txt_lengths) + args if txt is not None else x
            mean, lvar = self.enc(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
        else:
            z = torch.zeros(b, 1).cuda()
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0 and not self.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0 and not self.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            if self.z_dim > 0 and not self.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim # bsz, NT, z
                )
                assert False
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1) # bsz, NT, H+z

            #import pdb; pdb.set_trace()
            #rule_prob_1 = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            bc_emb = self.rule_mlp.weight.unsqueeze(0) # 1, NT_T^2, H+z
            return_log = True
            if self.relu:
                return_log = False
            if self.unit_norm:
                nonterm_emb = nonterm_emb / nonterm_emb.norm(dim=-1, keepdim=True)
                bc_emb = bc_emb / bc_emb.norm(dim=-1, keepdim=True)
            nonterm_emb_features = kernel(nonterm_emb, self.projection_matrix_anti, is_query=True, eps=0.0001, return_log=return_log, relu=self.relu, nonorm=True) # bsz, NT, r
            bc_emb_bias = self.rule_mlp.bias.view(1, 1, -1) # 1, 1, NT_T^2
            bc_emb_features = kernel(bc_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu, nonorm=True) # 1, NT_T^2, r
            #import pdb; pdb.set_trace()
# bsz, NT, 1, r
# bsz, 1, NT_T^2, r

# bsz, NT, NT_T^2

            #import pdb; pdb.set_trace()
            if not self.relu:
                if not self.genbmm:
                    rule_prob = (
                             nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT_T^2, r
                             ).logsumexp(-1) + bc_emb_bias#.log_softmax(-1) # bsz, NT, NT_T^2
                else:
                    bc_emb_features_2 = bc_emb_features.expand(b, -1, -1).transpose(-1, -2).contiguous() # b, r, NT_T^2    # b, NT, r
                    rule_prob = genbmm.logbmm(nonterm_emb_features, bc_emb_features_2) + bc_emb_bias  # b, NT, NT_T^2
            else:
                rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
                rule_prob_log = rule_prob.log() + bc_emb_bias
                #rule_prob = rule_prob_log.log_softmax(-1)
            #rule_prob2 = ((
            #         nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] + bc_emb_bias.unsqueeze(-1)
            #         ).logsumexp(-1)).log_softmax(-1)
            #import pdb; pdb.set_trace()
            #rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
            #rule_prob = rule_prob / rule_prob.sum(-1, keepdim=True)
            #rule_prob = rule_prob.log()


            if self.band > 0:
                #import pdb; pdb.set_trace()
                x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                diff = torch.abs(x_ids - y_ids)
                BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                band_BC_bc = (diff < self.band) & BC_bc
                band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                band = band_BC_bc | band_Bc | band_bC
                band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            elif self.band < 0:
                #import pdb; pdb.set_trace()
                #x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                #y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                #diff = torch.abs(x_ids - y_ids)
                #BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                #Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                #bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                #band_BC_bc = (diff < self.band) & BC_bc
                #band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                #band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                #band = band_BC_bc | band_Bc | band_bC
                ids = list(range(self.NT_T**2))
                import random
                random.seed(1234)
                random.shuffle(ids)
                ids = ids[(((-self.band-1)*2+1)*2*self.NT_T):]
                #rule_prob[ids] = -float('inf')
                band = torch.zeros(self.NT_T, self.NT_T).cuda()
                band = band.view(-1)
                band[ids] = -float('inf')
                #band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            #rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            rule_prob = F.log_softmax(rule_prob, -1)
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            with torch.no_grad():
                rule_prob_gt = torch.matmul(nonterm_emb, bc_emb.transpose(-1, -2)) + bc_emb_bias # bsz, NT, NT_T^2
                rule_prob_gt = F.log_softmax(rule_prob_gt, -1)
            return (rule_prob, rule_prob_gt, bc_emb_bias, bc_emb_features, nonterm_emb_features)

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        bc_emb_bias, bc_emb_features, nonterm_emb_features = rules_ll[2:]
        self.rules_ll = rules_ll[0][0]
        if False and (not self.relu):
            return (terms_ll, rules_ll[:2], roots_ll, bc_emb_bias, bc_emb_features, nonterm_emb_features), kl 
        else:
            return (terms_ll, rules_ll[:2], roots_ll), kl 

class CPCFGProjbBandSoftmax(torch.nn.Module):
    def __init__(self, V, NT, T, *args, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256,
                 num_features=0, 
                 relu=False,
                 norm_square=0,
                 unit_norm=False,
                 temperature=1,
                 **kwargs): 
        super(CPCFGProjbBandSoftmax, self).__init__()
        assert z_dim >= 0
        self.num_features = num_features
        self.relu = relu
        self.unit_norm = unit_norm
        self.temperature = temperature
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.genbmm = False
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.share_term = kwargs.get("share_term", False)
        self.share_rule = kwargs.get("share_rule", False)
        self.share_root = kwargs.get("share_root", False)
        self.wo_enc_emb = kwargs.get("wo_enc_emb", False)

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim if self.share_rule else s_dim + z_dim
        self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if self.share_root else s_dim + z_dim
        self.root_mlp = nn.Sequential(nn.Linear(root_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))

        projection_matrix = get_2d_array(num_features//2, rule_dim).transpose(0,1)
        if temperature != 1:
            projection_matrix = projection_matrix * math.sqrt(temperature)
        self.projection_matrix_anti = nn.Parameter(torch.cat([projection_matrix, -projection_matrix], -1))
        #self.projection_matrix_anti.requires_grad = False


        if z_dim > 0:
            i_dim = w_dim
            self.enc_emb = lambda x: x 
            if not self.wo_enc_emb:
                self.enc_emb = nn.Embedding(V, w_dim)
                self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                    bidirectional=True, num_layers=1, batch_first=True)
                i_dim = h_dim * 2
            self.enc_out = nn.Linear(i_dim, z_dim * 2)

        term_dim = s_dim if self.share_term else s_dim + z_dim
        self.term_mlp = nn.Sequential(nn.Linear(term_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        self._initialize()
        if self.unit_norm:
            self._normalize()

    def _normalize(self):
        assert self.unit_norm
        self.rule_mlp.weight.data.div_(self.rule_mlp.weight.data.norm(dim=-1, keepdim=True))
        self.nonterm_emb.data.div_(self.nonterm_emb.data.norm(dim=-1, keepdim=True))
        #import pdb; pdb.set_trace()

    def _initialize(self):
        #import pdb; pdb.set_trace()
        for n, p in self.named_parameters():
            print (n)
            if p.dim() > 1 and ('projection_matrix' not in n):
                torch.nn.init.xavier_uniform_(p)
        #import pdb; pdb.set_trace()

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    def forward(self, x, lengths, *args, txt=None, txt_lengths=None, use_mean=False, **kwargs):
        """ x, lengths: word ids; txt, txt_lengths: sub-word ids """
        b, n = x.shape[:2]
        batch_size = b 
        if self.z_dim > 0:
            max_pooling = kwargs.get("max_pooling", True)
            enforce_sorted = kwargs.get("enforce_sorted", False)
            item = (x, lengths, txt, txt_lengths) + args if txt is not None else x
            mean, lvar = self.enc(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
        else:
            z = torch.zeros(b, 1).cuda()
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0 and not self.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0 and not self.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            if self.z_dim > 0 and not self.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim # bsz, NT, z
                )
                assert False
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1) # bsz, NT, H+z

            #import pdb; pdb.set_trace()
            #rule_prob_1 = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            ball_call_emb = self.rule_mlp.weight.unsqueeze(0) # 1, NT_T^2, H+z
            return_log = True
            if self.relu:
                return_log = False
            if self.unit_norm:
                nonterm_emb = nonterm_emb / nonterm_emb.norm(dim=-1, keepdim=True)
                ball_call_emb = ball_call_emb / ball_call_emb.norm(dim=-1, keepdim=True)

            ball_call_emb_bias = self.rule_mlp.bias.view(1, 1, self.NT_T, self.NT_T) # 1, 1, NT_T^2
            bterm_cterm_emb_bias = ball_call_emb_bias[:, :, self.NT:, self.NT:].contiguous().view(1, 1, -1) # 1, 1, T^2
            b_c_emb_bias = ball_call_emb_bias[:, :, :self.NT, :self.NT].contiguous().view(1, 1, -1) # 1, 1, T^2
            bterm_c_emb_bias = ball_call_emb_bias[:, :, self.NT:, :self.NT].contiguous().view(1, 1, -1) # 1, 1, T^2
            b_cterm_emb_bias = ball_call_emb_bias[:, :, :self.NT, self.NT:].contiguous().view(1, 1, -1) # 1, 1, T^2

            ball_call_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)
            b_c_emb = ball_call_emb[:, :self.NT, :self.NT, :].contiguous().view(1, self.NT**2, -1) # 1, NT^2, H
            bterm_c_emb = ball_call_emb[:, self.NT:, :self.NT, :].contiguous().view(1, self.T*self.NT, -1) # 1, T*NT, H
            b_cterm_emb = ball_call_emb[:, :self.NT, self.NT:, :].contiguous().view(1, self.NT*self.T, -1) # 1, NT*T, H

            #import pdb; pdb.set_trace()

            bterm_cterm_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)[:, self.NT:, self.NT:, :].contiguous().view(1, self.T**2, -1).transpose(-1, -2).contiguous() # 1, H, T^2
            bterm_cterm_rule_prob = torch.matmul(nonterm_emb, bterm_cterm_emb) + bterm_cterm_emb_bias # bsz, NT, T^2
            bterm_cterm_rule_prob = bterm_cterm_rule_prob.view(b, self.NT, self.T, self.T)

            a_emb_features = kernel(nonterm_emb, self.projection_matrix_anti, is_query=True, eps=0.0001, return_log=return_log, relu=self.relu) # bsz, NT, r

            #bc_emb_features = kernel(bc_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT_T^2, r
            b_c_emb_features = kernel(b_c_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT^2, r
            bterm_c_emb_features = kernel(bterm_c_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, T*NT, r
            b_cterm_emb_features = kernel(b_cterm_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT*T, r
            #import pdb; pdb.set_trace()
# bsz, NT, 1, r
# bsz, 1, NT_T^2, r

# bsz, NT, NT_T^2

            #import pdb; pdb.set_trace()
            if not self.relu:
                if not self.genbmm:
                    #rule_prob = (
                    #         nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT_T^2, r
                    #         ).logsumexp(-1) # bsz, NT, NT_T^2
                    b_c_rule_prob = (
                             a_emb_features[:,:,None,:] + b_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT^2, r
                             ).logsumexp(-1) # bsz, NT, NT^2
                    b_c_rule_prob = b_c_rule_prob + b_c_emb_bias
                    b_c_rule_prob = b_c_rule_prob.view(b, self.NT, self.NT, self.NT)
                    bterm_c_rule_prob = (
                             a_emb_features[:,:,None,:] + bterm_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, T*NT, r
                             ).logsumexp(-1) # bsz, NT, T*NT
                    bterm_c_rule_prob = bterm_c_rule_prob + bterm_c_emb_bias
                    bterm_c_rule_prob = bterm_c_rule_prob.view(b, self.NT, self.T, self.NT)
                    b_cterm_rule_prob = (
                             a_emb_features[:,:,None,:] + b_cterm_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT*T, r
                             ).logsumexp(-1) # bsz, NT, NT*T
                    b_cterm_rule_prob = b_cterm_rule_prob + b_cterm_emb_bias
                    b_cterm_rule_prob = b_cterm_rule_prob.view(b, self.NT, self.NT, self.T)

                    rule_prob = torch.cat (   (torch.cat ((b_c_rule_prob, b_cterm_rule_prob), -1), torch.cat ((bterm_c_rule_prob, bterm_cterm_rule_prob), -1)), -2)


                    rule_prob = rule_prob.view(b, self.NT, self.NT_T**2)
                    #rule_prob[:, :, self.NT:, self.NT:] = a_berm_cterm
                    #rule_prob = rule_prob + bc_emb_bias#.log_softmax(-1) # bsz, NT, NT_T^2
                else:
                    assert False
                    bc_emb_features_2 = bc_emb_features.expand(b, -1, -1).transpose(-1, -2).contiguous() # b, r, NT_T^2    # b, NT, r
                    rule_prob = genbmm.logbmm(nonterm_emb_features, bc_emb_features_2) + bc_emb_bias  # b, NT, NT_T^2
            else:
                rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
                rule_prob_log = rule_prob.log() + bc_emb_bias
                #rule_prob = rule_prob_log.log_softmax(-1)
            #rule_prob2 = ((
            #         nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] + bc_emb_bias.unsqueeze(-1)
            #         ).logsumexp(-1)).log_softmax(-1)
            #import pdb; pdb.set_trace()
            #rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
            #rule_prob = rule_prob / rule_prob.sum(-1, keepdim=True)
            #rule_prob = rule_prob.log()


            if self.band > 0:
                #import pdb; pdb.set_trace()
                x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                diff = torch.abs(x_ids - y_ids)
                BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                band_BC_bc = (diff < self.band) & BC_bc
                band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                band = band_BC_bc | band_Bc | band_bC
                band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            elif self.band < 0:
                #import pdb; pdb.set_trace()
                #x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                #y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                #diff = torch.abs(x_ids - y_ids)
                #BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                #Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                #bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                #band_BC_bc = (diff < self.band) & BC_bc
                #band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                #band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                #band = band_BC_bc | band_Bc | band_bC
                ids = list(range(self.NT_T**2))
                import random
                random.seed(1234)
                random.shuffle(ids)
                ids = ids[(((-self.band-1)*2+1)*2*self.NT_T):]
                #rule_prob[ids] = -float('inf')
                band = torch.zeros(self.NT_T, self.NT_T).cuda()
                band = band.view(-1)
                band[ids] = -float('inf')
                #band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            #rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            rule_prob = F.log_softmax(rule_prob, -1)
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            with torch.no_grad():
                rule_prob_gt = torch.matmul(nonterm_emb, ball_call_emb.view(1, self.NT_T**2, -1).transpose(-1, -2)) + ball_call_emb_bias.view(1, 1, -1) # bsz, NT, NT_T^2
                rule_prob_gt = F.log_softmax(rule_prob_gt, -1)
            #return (rule_prob, rule_prob_gt, ball_call_emb_bias, ball_call_emb_features, nonterm_emb_features)
            return (rule_prob, rule_prob_gt, None, None, None)

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        bc_emb_bias, bc_emb_features, nonterm_emb_features = rules_ll[2:]
        self.rules_ll = rules_ll[0][0]
        if False and (not self.relu):
            return (terms_ll, rules_ll[:2], roots_ll, bc_emb_bias, bc_emb_features, nonterm_emb_features), kl 
        else:
            return (terms_ll, rules_ll[:2], roots_ll), kl 

class CPCFGProjbBandSoftmaxNoshare(torch.nn.Module):
    def __init__(self, V, NT, T, *args, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256,
                 num_features=0, 
                 relu=False,
                 norm_square=0,
                 unit_norm=False,
                 temperature=1,
                 **kwargs): 
        super(CPCFGProjbBandSoftmaxNoshare, self).__init__()
        assert z_dim >= 0
        self.num_features = num_features
        self.relu = relu
        self.unit_norm = unit_norm
        self.temperature = temperature
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.genbmm = False
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.share_term = kwargs.get("share_term", False)
        self.share_rule = kwargs.get("share_rule", False)
        self.share_root = kwargs.get("share_root", False)
        self.wo_enc_emb = kwargs.get("wo_enc_emb", False)

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.nonterm_emb_softmax = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim if self.share_rule else s_dim + z_dim
        self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if self.share_root else s_dim + z_dim
        self.root_mlp = nn.Sequential(nn.Linear(root_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))

        projection_matrix = get_2d_array(num_features//2, rule_dim).transpose(0,1)
        if temperature != 1:
            projection_matrix = projection_matrix * math.sqrt(temperature)
        self.projection_matrix_anti = nn.Parameter(torch.cat([projection_matrix, -projection_matrix], -1))
        #self.projection_matrix_anti.requires_grad = False


        if z_dim > 0:
            i_dim = w_dim
            self.enc_emb = lambda x: x 
            if not self.wo_enc_emb:
                self.enc_emb = nn.Embedding(V, w_dim)
                self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                    bidirectional=True, num_layers=1, batch_first=True)
                i_dim = h_dim * 2
            self.enc_out = nn.Linear(i_dim, z_dim * 2)

        term_dim = s_dim if self.share_term else s_dim + z_dim
        self.term_mlp = nn.Sequential(nn.Linear(term_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        self._initialize()
        if self.unit_norm:
            self._normalize()

    def _normalize(self):
        assert self.unit_norm
        self.rule_mlp.weight.data.div_(self.rule_mlp.weight.data.norm(dim=-1, keepdim=True))
        self.nonterm_emb.data.div_(self.nonterm_emb.data.norm(dim=-1, keepdim=True))
        #import pdb; pdb.set_trace()

    def _initialize(self):
        #import pdb; pdb.set_trace()
        for n, p in self.named_parameters():
            print (n)
            if p.dim() > 1 and ('projection_matrix' not in n):
                torch.nn.init.xavier_uniform_(p)
        #import pdb; pdb.set_trace()

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    def forward(self, x, lengths, *args, txt=None, txt_lengths=None, use_mean=False, **kwargs):
        """ x, lengths: word ids; txt, txt_lengths: sub-word ids """
        b, n = x.shape[:2]
        batch_size = b 
        if self.z_dim > 0:
            max_pooling = kwargs.get("max_pooling", True)
            enforce_sorted = kwargs.get("enforce_sorted", False)
            item = (x, lengths, txt, txt_lengths) + args if txt is not None else x
            mean, lvar = self.enc(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
        else:
            z = torch.zeros(b, 1).cuda()
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0 and not self.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0 and not self.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            nonterm_emb_softmax = self.nonterm_emb_softmax.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            if self.z_dim > 0 and not self.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim # bsz, NT, z
                )
                assert False
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1) # bsz, NT, H+z

            #import pdb; pdb.set_trace()
            #rule_prob_1 = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            ball_call_emb = self.rule_mlp.weight.unsqueeze(0) # 1, NT_T^2, H+z
            return_log = True
            if self.relu:
                return_log = False
            if self.unit_norm:
                nonterm_emb = nonterm_emb / nonterm_emb.norm(dim=-1, keepdim=True)
                ball_call_emb = ball_call_emb / ball_call_emb.norm(dim=-1, keepdim=True)

            ball_call_emb_bias = self.rule_mlp.bias.view(1, 1, self.NT_T, self.NT_T) # 1, 1, NT_T^2
            bterm_cterm_emb_bias = ball_call_emb_bias[:, :, self.NT:, self.NT:].contiguous().view(1, 1, -1) # 1, 1, T^2
            b_c_emb_bias = ball_call_emb_bias[:, :, :self.NT, :self.NT].contiguous().view(1, 1, -1) # 1, 1, T^2
            bterm_c_emb_bias = ball_call_emb_bias[:, :, self.NT:, :self.NT].contiguous().view(1, 1, -1) # 1, 1, T^2
            b_cterm_emb_bias = ball_call_emb_bias[:, :, :self.NT, self.NT:].contiguous().view(1, 1, -1) # 1, 1, T^2

            ball_call_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)
            b_c_emb = ball_call_emb[:, :self.NT, :self.NT, :].contiguous().view(1, self.NT**2, -1) # 1, NT^2, H
            bterm_c_emb = ball_call_emb[:, self.NT:, :self.NT, :].contiguous().view(1, self.T*self.NT, -1) # 1, T*NT, H
            b_cterm_emb = ball_call_emb[:, :self.NT, self.NT:, :].contiguous().view(1, self.NT*self.T, -1) # 1, NT*T, H

            #import pdb; pdb.set_trace()

            bterm_cterm_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)[:, self.NT:, self.NT:, :].contiguous().view(1, self.T**2, -1).transpose(-1, -2).contiguous() # 1, H, T^2
            bterm_cterm_rule_prob = torch.matmul(nonterm_emb_softmax, bterm_cterm_emb) + bterm_cterm_emb_bias # bsz, NT, T^2
            bterm_cterm_rule_prob = bterm_cterm_rule_prob.view(b, self.NT, self.T, self.T)

            a_emb_features = kernel(nonterm_emb, self.projection_matrix_anti, is_query=True, eps=0.0001, return_log=return_log, relu=self.relu) # bsz, NT, r

            #bc_emb_features = kernel(bc_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT_T^2, r
            b_c_emb_features = kernel(b_c_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT^2, r
            bterm_c_emb_features = kernel(bterm_c_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, T*NT, r
            b_cterm_emb_features = kernel(b_cterm_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT*T, r
            #import pdb; pdb.set_trace()
# bsz, NT, 1, r
# bsz, 1, NT_T^2, r

# bsz, NT, NT_T^2

            #import pdb; pdb.set_trace()
            if not self.relu:
                if not self.genbmm:
                    #rule_prob = (
                    #         nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT_T^2, r
                    #         ).logsumexp(-1) # bsz, NT, NT_T^2
                    b_c_rule_prob = (
                             a_emb_features[:,:,None,:] + b_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT^2, r
                             ).logsumexp(-1) # bsz, NT, NT^2
                    b_c_rule_prob = b_c_rule_prob + b_c_emb_bias
                    b_c_rule_prob = b_c_rule_prob.view(b, self.NT, self.NT, self.NT)
                    bterm_c_rule_prob = (
                             a_emb_features[:,:,None,:] + bterm_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, T*NT, r
                             ).logsumexp(-1) # bsz, NT, T*NT
                    bterm_c_rule_prob = bterm_c_rule_prob + bterm_c_emb_bias
                    bterm_c_rule_prob = bterm_c_rule_prob.view(b, self.NT, self.T, self.NT)
                    b_cterm_rule_prob = (
                             a_emb_features[:,:,None,:] + b_cterm_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT*T, r
                             ).logsumexp(-1) # bsz, NT, NT*T
                    b_cterm_rule_prob = b_cterm_rule_prob + b_cterm_emb_bias
                    b_cterm_rule_prob = b_cterm_rule_prob.view(b, self.NT, self.NT, self.T)

                    rule_prob = torch.cat (   (torch.cat ((b_c_rule_prob, b_cterm_rule_prob), -1), torch.cat ((bterm_c_rule_prob, bterm_cterm_rule_prob), -1)), -2)


                    rule_prob = rule_prob.view(b, self.NT, self.NT_T**2)
                    #rule_prob[:, :, self.NT:, self.NT:] = a_berm_cterm
                    #rule_prob = rule_prob + bc_emb_bias#.log_softmax(-1) # bsz, NT, NT_T^2
                else:
                    assert False
                    bc_emb_features_2 = bc_emb_features.expand(b, -1, -1).transpose(-1, -2).contiguous() # b, r, NT_T^2    # b, NT, r
                    rule_prob = genbmm.logbmm(nonterm_emb_features, bc_emb_features_2) + bc_emb_bias  # b, NT, NT_T^2
            else:
                rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
                rule_prob_log = rule_prob.log() + bc_emb_bias
                #rule_prob = rule_prob_log.log_softmax(-1)
            #rule_prob2 = ((
            #         nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] + bc_emb_bias.unsqueeze(-1)
            #         ).logsumexp(-1)).log_softmax(-1)
            #import pdb; pdb.set_trace()
            #rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
            #rule_prob = rule_prob / rule_prob.sum(-1, keepdim=True)
            #rule_prob = rule_prob.log()


            if self.band > 0:
                #import pdb; pdb.set_trace()
                x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                diff = torch.abs(x_ids - y_ids)
                BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                band_BC_bc = (diff < self.band) & BC_bc
                band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                band = band_BC_bc | band_Bc | band_bC
                band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            elif self.band < 0:
                #import pdb; pdb.set_trace()
                #x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                #y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                #diff = torch.abs(x_ids - y_ids)
                #BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                #Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                #bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                #band_BC_bc = (diff < self.band) & BC_bc
                #band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                #band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                #band = band_BC_bc | band_Bc | band_bC
                ids = list(range(self.NT_T**2))
                import random
                random.seed(1234)
                random.shuffle(ids)
                ids = ids[(((-self.band-1)*2+1)*2*self.NT_T):]
                #rule_prob[ids] = -float('inf')
                band = torch.zeros(self.NT_T, self.NT_T).cuda()
                band = band.view(-1)
                band[ids] = -float('inf')
                #band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            #rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            rule_prob = F.log_softmax(rule_prob, -1)
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            with torch.no_grad():
                rule_prob_gt = torch.matmul(nonterm_emb, ball_call_emb.view(1, self.NT_T**2, -1).transpose(-1, -2)) + ball_call_emb_bias.view(1, 1, -1) # bsz, NT, NT_T^2
                rule_prob_gt = F.log_softmax(rule_prob_gt, -1)
            #return (rule_prob, rule_prob_gt, ball_call_emb_bias, ball_call_emb_features, nonterm_emb_features)
            return (rule_prob, rule_prob_gt, None, None, None)

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        bc_emb_bias, bc_emb_features, nonterm_emb_features = rules_ll[2:]
        self.rules_ll = rules_ll[0][0]
        if False and (not self.relu):
            return (terms_ll, rules_ll[:2], roots_ll, bc_emb_bias, bc_emb_features, nonterm_emb_features), kl 
        else:
            return (terms_ll, rules_ll[:2], roots_ll), kl 
class CPCFGProjbBandSoftmaxNoshareNoanti(torch.nn.Module):
    def __init__(self, V, NT, T, *args, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256,
                 num_features=0, 
                 relu=False,
                 norm_square=0,
                 unit_norm=False,
                 temperature=1,
                 **kwargs): 
        super(CPCFGProjbBandSoftmaxNoshareNoanti, self).__init__()
        assert z_dim >= 0
        self.num_features = num_features
        self.relu = relu
        self.unit_norm = unit_norm
        self.temperature = temperature
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.genbmm = False
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.share_term = kwargs.get("share_term", False)
        self.share_rule = kwargs.get("share_rule", False)
        self.share_root = kwargs.get("share_root", False)
        self.wo_enc_emb = kwargs.get("wo_enc_emb", False)

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.nonterm_emb_softmax = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim if self.share_rule else s_dim + z_dim
        self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if self.share_root else s_dim + z_dim
        self.root_mlp = nn.Sequential(nn.Linear(root_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))

        projection_matrix = get_2d_array(num_features, rule_dim).transpose(0,1)
        if temperature != 1:
            projection_matrix = projection_matrix * math.sqrt(temperature)
        self.projection_matrix_anti = nn.Parameter(projection_matrix)
        #self.projection_matrix_anti.requires_grad = False


        if z_dim > 0:
            i_dim = w_dim
            self.enc_emb = lambda x: x 
            if not self.wo_enc_emb:
                self.enc_emb = nn.Embedding(V, w_dim)
                self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                    bidirectional=True, num_layers=1, batch_first=True)
                i_dim = h_dim * 2
            self.enc_out = nn.Linear(i_dim, z_dim * 2)

        term_dim = s_dim if self.share_term else s_dim + z_dim
        self.term_mlp = nn.Sequential(nn.Linear(term_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        self._initialize()
        if self.unit_norm:
            self._normalize()

    def _normalize(self):
        assert self.unit_norm
        self.rule_mlp.weight.data.div_(self.rule_mlp.weight.data.norm(dim=-1, keepdim=True))
        self.nonterm_emb.data.div_(self.nonterm_emb.data.norm(dim=-1, keepdim=True))
        #import pdb; pdb.set_trace()

    def _initialize(self):
        #import pdb; pdb.set_trace()
        for n, p in self.named_parameters():
            print (n)
            if p.dim() > 1 and ('projection_matrix' not in n):
                torch.nn.init.xavier_uniform_(p)
        #import pdb; pdb.set_trace()

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    def forward(self, x, lengths, *args, txt=None, txt_lengths=None, use_mean=False, **kwargs):
        """ x, lengths: word ids; txt, txt_lengths: sub-word ids """
        b, n = x.shape[:2]
        batch_size = b 
        if self.z_dim > 0:
            max_pooling = kwargs.get("max_pooling", True)
            enforce_sorted = kwargs.get("enforce_sorted", False)
            item = (x, lengths, txt, txt_lengths) + args if txt is not None else x
            mean, lvar = self.enc(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
        else:
            z = torch.zeros(b, 1).cuda()
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0 and not self.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0 and not self.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            nonterm_emb_softmax = self.nonterm_emb_softmax.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            if self.z_dim > 0 and not self.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim # bsz, NT, z
                )
                assert False
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1) # bsz, NT, H+z

            #import pdb; pdb.set_trace()
            #rule_prob_1 = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            ball_call_emb = self.rule_mlp.weight.unsqueeze(0) # 1, NT_T^2, H+z
            return_log = True
            if self.relu:
                return_log = False
            if self.unit_norm:
                nonterm_emb = nonterm_emb / nonterm_emb.norm(dim=-1, keepdim=True)
                ball_call_emb = ball_call_emb / ball_call_emb.norm(dim=-1, keepdim=True)

            ball_call_emb_bias = self.rule_mlp.bias.view(1, 1, self.NT_T, self.NT_T) # 1, 1, NT_T^2
            bterm_cterm_emb_bias = ball_call_emb_bias[:, :, self.NT:, self.NT:].contiguous().view(1, 1, -1) # 1, 1, T^2
            b_c_emb_bias = ball_call_emb_bias[:, :, :self.NT, :self.NT].contiguous().view(1, 1, -1) # 1, 1, T^2
            bterm_c_emb_bias = ball_call_emb_bias[:, :, self.NT:, :self.NT].contiguous().view(1, 1, -1) # 1, 1, T^2
            b_cterm_emb_bias = ball_call_emb_bias[:, :, :self.NT, self.NT:].contiguous().view(1, 1, -1) # 1, 1, T^2

            ball_call_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)
            b_c_emb = ball_call_emb[:, :self.NT, :self.NT, :].contiguous().view(1, self.NT**2, -1) # 1, NT^2, H
            bterm_c_emb = ball_call_emb[:, self.NT:, :self.NT, :].contiguous().view(1, self.T*self.NT, -1) # 1, T*NT, H
            b_cterm_emb = ball_call_emb[:, :self.NT, self.NT:, :].contiguous().view(1, self.NT*self.T, -1) # 1, NT*T, H

            #import pdb; pdb.set_trace()

            bterm_cterm_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)[:, self.NT:, self.NT:, :].contiguous().view(1, self.T**2, -1).transpose(-1, -2).contiguous() # 1, H, T^2
            bterm_cterm_rule_prob = torch.matmul(nonterm_emb_softmax, bterm_cterm_emb) + bterm_cterm_emb_bias # bsz, NT, T^2
            bterm_cterm_rule_prob = bterm_cterm_rule_prob.view(b, self.NT, self.T, self.T)

            a_emb_features = kernel(nonterm_emb, self.projection_matrix_anti, is_query=True, eps=0.0001, return_log=return_log, relu=self.relu) # bsz, NT, r

            #bc_emb_features = kernel(bc_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT_T^2, r
            b_c_emb_features = kernel(b_c_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT^2, r
            bterm_c_emb_features = kernel(bterm_c_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, T*NT, r
            b_cterm_emb_features = kernel(b_cterm_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT*T, r
            #import pdb; pdb.set_trace()
# bsz, NT, 1, r
# bsz, 1, NT_T^2, r

# bsz, NT, NT_T^2

            #import pdb; pdb.set_trace()
            if not self.relu:
                if not self.genbmm:
                    #rule_prob = (
                    #         nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT_T^2, r
                    #         ).logsumexp(-1) # bsz, NT, NT_T^2
                    b_c_rule_prob = (
                             a_emb_features[:,:,None,:] + b_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT^2, r
                             ).logsumexp(-1) # bsz, NT, NT^2
                    b_c_rule_prob = b_c_rule_prob + b_c_emb_bias
                    b_c_rule_prob = b_c_rule_prob.view(b, self.NT, self.NT, self.NT)
                    bterm_c_rule_prob = (
                             a_emb_features[:,:,None,:] + bterm_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, T*NT, r
                             ).logsumexp(-1) # bsz, NT, T*NT
                    bterm_c_rule_prob = bterm_c_rule_prob + bterm_c_emb_bias
                    bterm_c_rule_prob = bterm_c_rule_prob.view(b, self.NT, self.T, self.NT)
                    b_cterm_rule_prob = (
                             a_emb_features[:,:,None,:] + b_cterm_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT*T, r
                             ).logsumexp(-1) # bsz, NT, NT*T
                    b_cterm_rule_prob = b_cterm_rule_prob + b_cterm_emb_bias
                    b_cterm_rule_prob = b_cterm_rule_prob.view(b, self.NT, self.NT, self.T)

                    rule_prob = torch.cat (   (torch.cat ((b_c_rule_prob, b_cterm_rule_prob), -1), torch.cat ((bterm_c_rule_prob, bterm_cterm_rule_prob), -1)), -2)


                    rule_prob = rule_prob.view(b, self.NT, self.NT_T**2)
                    #rule_prob[:, :, self.NT:, self.NT:] = a_berm_cterm
                    #rule_prob = rule_prob + bc_emb_bias#.log_softmax(-1) # bsz, NT, NT_T^2
                else:
# bsz, NT, r   1, NT^2, r
                    b_c_rule_prob = genbmm.logbmm(a_emb_features, b_c_emb_features.transpose(-1, -2).contiguous())
                    #b_c_rule_prob = (
                    #         a_emb_features[:,:,None,:] + b_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT^2, r
                    #         ).logsumexp(-1) # bsz, NT, NT^2
                    b_c_rule_prob = b_c_rule_prob + b_c_emb_bias
                    b_c_rule_prob = b_c_rule_prob.view(b, self.NT, self.NT, self.NT)
                    bterm_c_rule_prob = genbmm.logbmm(a_emb_features, bterm_c_emb_features.transpose(-1, -2).contiguous())
                    #bterm_c_rule_prob = (
                    #         a_emb_features[:,:,None,:] + bterm_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, T*NT, r
                    #         ).logsumexp(-1) # bsz, NT, T*NT
                    bterm_c_rule_prob = bterm_c_rule_prob + bterm_c_emb_bias
                    bterm_c_rule_prob = bterm_c_rule_prob.view(b, self.NT, self.T, self.NT)
                    b_cterm_rule_prob = genbmm.logbmm(a_emb_features, b_cterm_emb_features.transpose(-1, -2).contiguous())
                    #b_cterm_rule_prob = (
                    #         a_emb_features[:,:,None,:] + b_cterm_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT*T, r
                    #         ).logsumexp(-1) # bsz, NT, NT*T
                    b_cterm_rule_prob = b_cterm_rule_prob + b_cterm_emb_bias
                    b_cterm_rule_prob = b_cterm_rule_prob.view(b, self.NT, self.NT, self.T)

                    rule_prob = torch.cat (   (torch.cat ((b_c_rule_prob, b_cterm_rule_prob), -1), torch.cat ((bterm_c_rule_prob, bterm_cterm_rule_prob), -1)), -2)


                    rule_prob = rule_prob.view(b, self.NT, self.NT_T**2)
                    #assert False
                    #bc_emb_features_2 = bc_emb_features.expand(b, -1, -1).transpose(-1, -2).contiguous() # b, r, NT_T^2    # b, NT, r
                    #rule_prob = genbmm.logbmm(nonterm_emb_features, bc_emb_features_2) + bc_emb_bias  # b, NT, NT_T^2
            else:
                rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
                rule_prob_log = rule_prob.log() + bc_emb_bias
                #rule_prob = rule_prob_log.log_softmax(-1)
            #rule_prob2 = ((
            #         nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] + bc_emb_bias.unsqueeze(-1)
            #         ).logsumexp(-1)).log_softmax(-1)
            #import pdb; pdb.set_trace()
            #rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
            #rule_prob = rule_prob / rule_prob.sum(-1, keepdim=True)
            #rule_prob = rule_prob.log()


            if self.band > 0:
                #import pdb; pdb.set_trace()
                x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                diff = torch.abs(x_ids - y_ids)
                BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                band_BC_bc = (diff < self.band) & BC_bc
                band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                band = band_BC_bc | band_Bc | band_bC
                band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            elif self.band < 0:
                #import pdb; pdb.set_trace()
                #x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                #y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                #diff = torch.abs(x_ids - y_ids)
                #BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                #Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                #bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                #band_BC_bc = (diff < self.band) & BC_bc
                #band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                #band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                #band = band_BC_bc | band_Bc | band_bC
                ids = list(range(self.NT_T**2))
                import random
                random.seed(1234)
                random.shuffle(ids)
                ids = ids[(((-self.band-1)*2+1)*2*self.NT_T):]
                #rule_prob[ids] = -float('inf')
                band = torch.zeros(self.NT_T, self.NT_T).cuda()
                band = band.view(-1)
                band[ids] = -float('inf')
                #band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            #rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            rule_prob = F.log_softmax(rule_prob, -1)
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            #with torch.no_grad():
            #    rule_prob_gt = torch.matmul(nonterm_emb, ball_call_emb.view(1, self.NT_T**2, -1).transpose(-1, -2)) + ball_call_emb_bias.view(1, 1, -1) # bsz, NT, NT_T^2
            #    rule_prob_gt = F.log_softmax(rule_prob_gt, -1)
            #return (rule_prob, rule_prob_gt, ball_call_emb_bias, ball_call_emb_features, nonterm_emb_features)
            if use_eff:
                #return (rule_prob, a_emb_features, b_c_emb_features, b_c_emb_bias, None)
                return (rule_prob, b_c_emb_bias, b_c_emb_features, a_emb_features, None)
            else:
                return (rule_prob, None, None, None, None) # <- latest version

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        #import pdb; pdb.set_trace()
        #bc_emb_bias, bc_emb_features, nonterm_emb_features = rules_ll[2:]
        bc_emb_bias, bc_emb_features, nonterm_emb_features = None, None, None
        self.rules_ll = rules_ll[0][0]
        if False and (not self.relu):
            return (terms_ll, rules_ll[:2], roots_ll, bc_emb_bias, bc_emb_features, nonterm_emb_features), kl 
        else:
            #return (terms_ll, rules_ll[0], roots_ll), kl 
            return (terms_ll, rules_ll[0], roots_ll, rules_ll[1], rules_ll[2], rules_ll[3]), kl 

class CPCFGProjbBandSoftmaxallNoshareNoanti(torch.nn.Module):
    def __init__(self, V, NT, T, *args, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256,
                 num_features=0, 
                 relu=False,
                 norm_square=0,
                 unit_norm=False,
                 temperature=1,
                 **kwargs): 
        super(CPCFGProjbBandSoftmaxallNoshareNoanti, self).__init__()
        assert z_dim >= 0
        self.num_features = num_features
        self.relu = relu
        self.unit_norm = unit_norm
        self.temperature = temperature
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.genbmm = False
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.share_term = kwargs.get("share_term", False)
        self.share_rule = kwargs.get("share_rule", False)
        self.share_root = kwargs.get("share_root", False)
        self.wo_enc_emb = kwargs.get("wo_enc_emb", False)

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.nonterm_emb_softmax = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        self.a_emb = nn.Parameter(torch.randn(NT, num_features))
        self.b_c_emb = nn.Parameter(torch.randn(NT**2, num_features))

        rule_dim = s_dim if self.share_rule else s_dim + z_dim
        self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if self.share_root else s_dim + z_dim
        self.root_mlp = nn.Sequential(nn.Linear(root_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))

        #projection_matrix = get_2d_array(num_features, rule_dim).transpose(0,1)
        if temperature != 1:
            projection_matrix = projection_matrix * math.sqrt(temperature)
        #self.projection_matrix_anti = nn.Parameter(projection_matrix)
        #self.projection_matrix_anti.requires_grad = False


        if z_dim > 0:
            i_dim = w_dim
            self.enc_emb = lambda x: x 
            if not self.wo_enc_emb:
                self.enc_emb = nn.Embedding(V, w_dim)
                self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                    bidirectional=True, num_layers=1, batch_first=True)
                i_dim = h_dim * 2
            self.enc_out = nn.Linear(i_dim, z_dim * 2)

        term_dim = s_dim if self.share_term else s_dim + z_dim
        self.term_mlp = nn.Sequential(nn.Linear(term_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        self._initialize()
        if self.unit_norm:
            self._normalize()

    def _normalize(self):
        assert self.unit_norm
        self.rule_mlp.weight.data.div_(self.rule_mlp.weight.data.norm(dim=-1, keepdim=True))
        self.nonterm_emb.data.div_(self.nonterm_emb.data.norm(dim=-1, keepdim=True))
        #import pdb; pdb.set_trace()

    def _initialize(self):
        #import pdb; pdb.set_trace()
        for n, p in self.named_parameters():
            print (n)
            if p.dim() > 1 and ('projection_matrix' not in n):
                torch.nn.init.xavier_uniform_(p)
        #import pdb; pdb.set_trace()

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    def forward(self, x, lengths, *args, txt=None, txt_lengths=None, use_mean=False, **kwargs):
        """ x, lengths: word ids; txt, txt_lengths: sub-word ids """
        b, n = x.shape[:2]
        batch_size = b 
        if self.z_dim > 0:
            max_pooling = kwargs.get("max_pooling", True)
            enforce_sorted = kwargs.get("enforce_sorted", False)
            item = (x, lengths, txt, txt_lengths) + args if txt is not None else x
            mean, lvar = self.enc(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
        else:
            z = torch.zeros(b, 1).to(x.device)
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0 and not self.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0 and not self.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            nonterm_emb_softmax = self.nonterm_emb_softmax.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            if self.z_dim > 0 and not self.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim # bsz, NT, z
                )
                assert False
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1) # bsz, NT, H+z

            #import pdb; pdb.set_trace()
            #rule_prob_1 = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            ball_call_emb = self.rule_mlp.weight.unsqueeze(0) # 1, NT_T^2, H+z
            return_log = True
            if self.relu:
                return_log = False
            if self.unit_norm:
                nonterm_emb = nonterm_emb / nonterm_emb.norm(dim=-1, keepdim=True)
                ball_call_emb = ball_call_emb / ball_call_emb.norm(dim=-1, keepdim=True)

            ball_call_emb_bias = self.rule_mlp.bias.view(1, 1, self.NT_T, self.NT_T) # 1, 1, NT_T^2
            bterm_cterm_emb_bias = ball_call_emb_bias[:, :, self.NT:, self.NT:].contiguous().view(1, 1, -1) # 1, 1, T^2
            b_c_emb_bias = ball_call_emb_bias[:, :, :self.NT, :self.NT].contiguous().view(1, 1, -1) # 1, 1, T^2
            bterm_c_emb_bias = ball_call_emb_bias[:, :, self.NT:, :self.NT].contiguous().view(1, 1, -1) # 1, 1, T^2
            b_cterm_emb_bias = ball_call_emb_bias[:, :, :self.NT, self.NT:].contiguous().view(1, 1, -1) # 1, 1, T^2

            ball_call_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)
            b_c_emb = ball_call_emb[:, :self.NT, :self.NT, :].contiguous().view(1, self.NT**2, -1) # 1, NT^2, H
            #bterm_c_emb = ball_call_emb[:, self.NT:, :self.NT, :].contiguous().view(1, self.T*self.NT, -1) # 1, T*NT, H
            #b_cterm_emb = ball_call_emb[:, :self.NT, self.NT:, :].contiguous().view(1, self.NT*self.T, -1) # 1, NT*T, H

            #import pdb; pdb.set_trace()

##### softmax

            bterm_cterm_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)[:, self.NT:, self.NT:, :].contiguous().view(1, self.T**2, -1).transpose(-1, -2).contiguous() # 1, H, T^2
            bterm_cterm_rule_prob = torch.matmul(nonterm_emb_softmax, bterm_cterm_emb) + bterm_cterm_emb_bias # bsz, NT, T^2
            bterm_cterm_rule_prob = bterm_cterm_rule_prob.view(b, self.NT, self.T, self.T)

            bterm_c_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)[:, self.NT:, :self.NT, :].contiguous().view(1, self.T*self.NT, -1).transpose(-1, -2).contiguous() # 1, H, T*NT
            bterm_c_rule_prob = torch.matmul(nonterm_emb_softmax, bterm_c_emb) + bterm_c_emb_bias # bsz, NT, T*NT
            bterm_c_rule_prob = bterm_c_rule_prob.view(b, self.NT, self.T, self.NT)

            b_cterm_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)[:, :self.NT:, self.NT:, :].contiguous().view(1, self.NT*self.T, -1).transpose(-1, -2).contiguous() # 1, H, NT*T
            b_cterm_rule_prob = torch.matmul(nonterm_emb_softmax, b_cterm_emb) + b_cterm_emb_bias # bsz, NT, NT*T
            b_cterm_rule_prob = b_cterm_rule_prob.view(b, self.NT, self.NT, self.T)


            #import pdb; pdb.set_trace()



            #TODO a_emb_features = kernel(nonterm_emb, self.projection_matrix_anti, is_query=True, eps=0.0001, return_log=return_log, relu=self.relu) # bsz, NT, r
            a_emb_features = self.a_emb.unsqueeze(0).expand(b, -1, -1)

            #bc_emb_features = kernel(bc_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT_T^2, r
            b_c_emb_features = self.b_c_emb.unsqueeze(0)
            #TODO b_c_emb_features = kernel(b_c_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT^2, r
            #bterm_c_emb_features = kernel(bterm_c_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, T*NT, r
            #b_cterm_emb_features = kernel(b_cterm_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT*T, r
            #import pdb; pdb.set_trace()
# bsz, NT, 1, r
# bsz, 1, NT_T^2, r

# bsz, NT, NT_T^2

            #import pdb; pdb.set_trace()
            if not self.relu:
                if not self.genbmm:
                    #rule_prob = (
                    #         nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT_T^2, r
                    #         ).logsumexp(-1) # bsz, NT, NT_T^2
                    b_c_rule_prob = (
                             a_emb_features[:,:,None,:] + b_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT^2, r
                             ).logsumexp(-1) # bsz, NT, NT^2
                    b_c_rule_prob = b_c_rule_prob + b_c_emb_bias
                    b_c_rule_prob = b_c_rule_prob.view(b, self.NT, self.NT, self.NT)
                    #bterm_c_rule_prob = (
                    #         a_emb_features[:,:,None,:] + bterm_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, T*NT, r
                    #         ).logsumexp(-1) # bsz, NT, T*NT
                    #bterm_c_rule_prob = bterm_c_rule_prob + bterm_c_emb_bias
                    #bterm_c_rule_prob = bterm_c_rule_prob.view(b, self.NT, self.T, self.NT)
                    #b_cterm_rule_prob = (
                    #         a_emb_features[:,:,None,:] + b_cterm_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT*T, r
                    #         ).logsumexp(-1) # bsz, NT, NT*T
                    #b_cterm_rule_prob = b_cterm_rule_prob + b_cterm_emb_bias
                    #b_cterm_rule_prob = b_cterm_rule_prob.view(b, self.NT, self.NT, self.T)

                    rule_prob = torch.cat (   (torch.cat ((b_c_rule_prob, b_cterm_rule_prob), -1), torch.cat ((bterm_c_rule_prob, bterm_cterm_rule_prob), -1)), -2)


                    rule_prob = rule_prob.view(b, self.NT, self.NT_T**2)
                    #rule_prob[:, :, self.NT:, self.NT:] = a_berm_cterm
                    #rule_prob = rule_prob + bc_emb_bias#.log_softmax(-1) # bsz, NT, NT_T^2
                else:
# bsz, NT, r   1, NT^2, r
                    b_c_rule_prob = genbmm.logbmm(a_emb_features, b_c_emb_features.transpose(-1, -2).contiguous())
#TODO: fix bug here, should expand
                    #b_c_rule_prob = (
                    #         a_emb_features[:,:,None,:] + b_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT^2, r
                    #         ).logsumexp(-1) # bsz, NT, NT^2
                    b_c_rule_prob = b_c_rule_prob + b_c_emb_bias
                    b_c_rule_prob = b_c_rule_prob.view(b, self.NT, self.NT, self.NT)
                    #bterm_c_rule_prob = genbmm.logbmm(a_emb_features, bterm_c_emb_features.transpose(-1, -2).contiguous())
                    #bterm_c_rule_prob = (
                    #         a_emb_features[:,:,None,:] + bterm_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, T*NT, r
                    #         ).logsumexp(-1) # bsz, NT, T*NT
                    #bterm_c_rule_prob = bterm_c_rule_prob + bterm_c_emb_bias
                    #bterm_c_rule_prob = bterm_c_rule_prob.view(b, self.NT, self.T, self.NT)
                    #b_cterm_rule_prob = genbmm.logbmm(a_emb_features, b_cterm_emb_features.transpose(-1, -2).contiguous())
                    #b_cterm_rule_prob = (
                    #         a_emb_features[:,:,None,:] + b_cterm_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT*T, r
                    #         ).logsumexp(-1) # bsz, NT, NT*T
                    #b_cterm_rule_prob = b_cterm_rule_prob + b_cterm_emb_bias
                    #b_cterm_rule_prob = b_cterm_rule_prob.view(b, self.NT, self.NT, self.T)

                    rule_prob = torch.cat (   (torch.cat ((b_c_rule_prob, b_cterm_rule_prob), -1), torch.cat ((bterm_c_rule_prob, bterm_cterm_rule_prob), -1)), -2)


                    rule_prob = rule_prob.view(b, self.NT, self.NT_T**2)
                    #assert False
                    #bc_emb_features_2 = bc_emb_features.expand(b, -1, -1).transpose(-1, -2).contiguous() # b, r, NT_T^2    # b, NT, r
                    #rule_prob = genbmm.logbmm(nonterm_emb_features, bc_emb_features_2) + bc_emb_bias  # b, NT, NT_T^2
            else:
                rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
                rule_prob_log = rule_prob.log() + bc_emb_bias
                #rule_prob = rule_prob_log.log_softmax(-1)
            #rule_prob2 = ((
            #         nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] + bc_emb_bias.unsqueeze(-1)
            #         ).logsumexp(-1)).log_softmax(-1)
            #import pdb; pdb.set_trace()
            #rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
            #rule_prob = rule_prob / rule_prob.sum(-1, keepdim=True)
            #rule_prob = rule_prob.log()


            if self.band > 0:
                #import pdb; pdb.set_trace()
                x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                diff = torch.abs(x_ids - y_ids)
                BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                band_BC_bc = (diff < self.band) & BC_bc
                band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                band = band_BC_bc | band_Bc | band_bC
                band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            elif self.band < 0:
                #import pdb; pdb.set_trace()
                #x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                #y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                #diff = torch.abs(x_ids - y_ids)
                #BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                #Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                #bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                #band_BC_bc = (diff < self.band) & BC_bc
                #band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                #band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                #band = band_BC_bc | band_Bc | band_bC
                ids = list(range(self.NT_T**2))
                import random
                random.seed(1234)
                random.shuffle(ids)
                ids = ids[(((-self.band-1)*2+1)*2*self.NT_T):]
                #rule_prob[ids] = -float('inf')
                band = torch.zeros(self.NT_T, self.NT_T).cuda()
                band = band.view(-1)
                band[ids] = -float('inf')
                #band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            #rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            rule_prob = F.log_softmax(rule_prob, -1)
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            #with torch.no_grad():
            #    rule_prob_gt = torch.matmul(nonterm_emb, ball_call_emb.view(1, self.NT_T**2, -1).transpose(-1, -2)) + ball_call_emb_bias.view(1, 1, -1) # bsz, NT, NT_T^2
            #    rule_prob_gt = F.log_softmax(rule_prob_gt, -1)
            #return (rule_prob, rule_prob_gt, ball_call_emb_bias, ball_call_emb_features, nonterm_emb_features)
            if use_eff:
                #return (rule_prob, a_emb_features, b_c_emb_features, b_c_emb_bias, None)
                return (rule_prob, b_c_emb_bias, b_c_emb_features, a_emb_features, None)
            else:
                return (rule_prob, None, None, None, None) # <- latest version

        """ 
        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        bc_emb_bias, bc_emb_features, nonterm_emb_features = rules_ll[2:]
        self.rules_ll = rules_ll[0][0]
        if False and (not self.relu):
            return (terms_ll, rules_ll[:2], roots_ll, bc_emb_bias, bc_emb_features, nonterm_emb_features), kl 
        else:
            return (terms_ll, rules_ll[0], roots_ll), kl

        """

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        #import pdb; pdb.set_trace()
        #bc_emb_bias, bc_emb_features, nonterm_emb_features = rules_ll[2:]
        bc_emb_bias, bc_emb_features, nonterm_emb_features = None, None, None
        self.rules_ll = rules_ll[0][0]
        if False and (not self.relu):
            return (terms_ll, rules_ll[:2], roots_ll, bc_emb_bias, bc_emb_features, nonterm_emb_features), kl
        elif not use_eff:
            return (terms_ll, rules_ll[0], roots_ll), kl 
        else:
            return (terms_ll, rules_ll[0], roots_ll, rules_ll[1], rules_ll[2], rules_ll[3]), kl 



class CPCFGProjbBandSoftmaxallNoshareNoantiMixture(torch.nn.Module):
    def __init__(self, V, NT, T, *args, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256,
                 num_features=0, 
                 relu=False,
                 norm_square=0,
                 unit_norm=False,
                 temperature=1,
                 **kwargs): 
        super(CPCFGProjbBandSoftmaxallNoshareNoantiMixture, self).__init__()
        assert z_dim >= 0
        self.num_features = num_features
        self.relu = relu
        self.unit_norm = unit_norm
        self.temperature = temperature
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.genbmm = False
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.share_term = kwargs.get("share_term", False)
        self.share_rule = kwargs.get("share_rule", False)
        self.share_root = kwargs.get("share_root", False)
        self.wo_enc_emb = kwargs.get("wo_enc_emb", False)

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.mixture_weights = nn.Parameter(torch.randn(NT, 4))
        self.nonterm_emb_softmax = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim if self.share_rule else s_dim + z_dim
        self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if self.share_root else s_dim + z_dim
        self.root_mlp = nn.Sequential(nn.Linear(root_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))

        projection_matrix = get_2d_array(num_features, rule_dim).transpose(0,1)
        if temperature != 1:
            projection_matrix = projection_matrix * math.sqrt(temperature)
        self.projection_matrix_anti = nn.Parameter(projection_matrix)
        #self.projection_matrix_anti.requires_grad = False


        if z_dim > 0:
            i_dim = w_dim
            self.enc_emb = lambda x: x 
            if not self.wo_enc_emb:
                self.enc_emb = nn.Embedding(V, w_dim)
                self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                    bidirectional=True, num_layers=1, batch_first=True)
                i_dim = h_dim * 2
            self.enc_out = nn.Linear(i_dim, z_dim * 2)

        term_dim = s_dim if self.share_term else s_dim + z_dim
        self.term_mlp = nn.Sequential(nn.Linear(term_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        self._initialize()
        if self.unit_norm:
            self._normalize()

    def _normalize(self):
        assert self.unit_norm
        self.rule_mlp.weight.data.div_(self.rule_mlp.weight.data.norm(dim=-1, keepdim=True))
        self.nonterm_emb.data.div_(self.nonterm_emb.data.norm(dim=-1, keepdim=True))
        #import pdb; pdb.set_trace()

    def _initialize(self):
        #import pdb; pdb.set_trace()
        for n, p in self.named_parameters():
            print (n)
            if p.dim() > 1 and ('projection_matrix' not in n):
                torch.nn.init.xavier_uniform_(p)
        #import pdb; pdb.set_trace()

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    def forward(self, x, lengths, *args, txt=None, txt_lengths=None, use_mean=False, **kwargs):
        """ x, lengths: word ids; txt, txt_lengths: sub-word ids """
        b, n = x.shape[:2]
        batch_size = b 
        if self.z_dim > 0:
            max_pooling = kwargs.get("max_pooling", True)
            enforce_sorted = kwargs.get("enforce_sorted", False)
            item = (x, lengths, txt, txt_lengths) + args if txt is not None else x
            mean, lvar = self.enc(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
        else:
            z = torch.zeros(b, 1).cuda()
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0 and not self.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0 and not self.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            nonterm_emb_softmax = self.nonterm_emb_softmax.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            if self.z_dim > 0 and not self.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim # bsz, NT, z
                )
                assert False
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1) # bsz, NT, H+z

            #import pdb; pdb.set_trace()
            #rule_prob_1 = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            ball_call_emb = self.rule_mlp.weight.unsqueeze(0) # 1, NT_T^2, H+z
            return_log = True
            if self.relu:
                return_log = False
            if self.unit_norm:
                nonterm_emb = nonterm_emb / nonterm_emb.norm(dim=-1, keepdim=True)
                ball_call_emb = ball_call_emb / ball_call_emb.norm(dim=-1, keepdim=True)

            ball_call_emb_bias = self.rule_mlp.bias.view(1, 1, self.NT_T, self.NT_T) # 1, 1, NT_T^2
            bterm_cterm_emb_bias = ball_call_emb_bias[:, :, self.NT:, self.NT:].contiguous().view(1, 1, -1) # 1, 1, T^2
            b_c_emb_bias = ball_call_emb_bias[:, :, :self.NT, :self.NT].contiguous().view(1, 1, -1) # 1, 1, T^2
            bterm_c_emb_bias = ball_call_emb_bias[:, :, self.NT:, :self.NT].contiguous().view(1, 1, -1) # 1, 1, T^2
            b_cterm_emb_bias = ball_call_emb_bias[:, :, :self.NT, self.NT:].contiguous().view(1, 1, -1) # 1, 1, T^2

            ball_call_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)
            b_c_emb = ball_call_emb[:, :self.NT, :self.NT, :].contiguous().view(1, self.NT**2, -1) # 1, NT^2, H
            #bterm_c_emb = ball_call_emb[:, self.NT:, :self.NT, :].contiguous().view(1, self.T*self.NT, -1) # 1, T*NT, H
            #b_cterm_emb = ball_call_emb[:, :self.NT, self.NT:, :].contiguous().view(1, self.NT*self.T, -1) # 1, NT*T, H

            #import pdb; pdb.set_trace()

##### softmax
            mixture_logits = self.mixture_weights.log_softmax(-1).unsqueeze(0) # 1, NT, 4

            bterm_cterm_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)[:, self.NT:, self.NT:, :].contiguous().view(1, self.T**2, -1).transpose(-1, -2).contiguous() # 1, H, T^2
            bterm_cterm_rule_prob = torch.matmul(nonterm_emb_softmax, bterm_cterm_emb) + bterm_cterm_emb_bias # bsz, NT, T^2
            bterm_cterm_rule_prob = bterm_cterm_rule_prob.log_softmax(-1) + mixture_logits[:, :, 3:4]
            bterm_cterm_rule_prob = bterm_cterm_rule_prob.view(b, self.NT, self.T, self.T)

            bterm_c_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)[:, self.NT:, :self.NT, :].contiguous().view(1, self.T*self.NT, -1).transpose(-1, -2).contiguous() # 1, H, T*NT
            bterm_c_rule_prob = torch.matmul(nonterm_emb_softmax, bterm_c_emb) + bterm_c_emb_bias # bsz, NT, T*NT
            bterm_c_rule_prob = bterm_c_rule_prob.log_softmax(-1) + mixture_logits[:, :, 2:3]
            bterm_c_rule_prob = bterm_c_rule_prob.view(b, self.NT, self.T, self.NT)

            b_cterm_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)[:, :self.NT:, self.NT:, :].contiguous().view(1, self.NT*self.T, -1).transpose(-1, -2).contiguous() # 1, H, NT*T
            b_cterm_rule_prob = torch.matmul(nonterm_emb_softmax, b_cterm_emb) + b_cterm_emb_bias # bsz, NT, NT*T
            b_cterm_rule_prob = b_cterm_rule_prob.log_softmax(-1) + mixture_logits[:, :, 1:2]
            b_cterm_rule_prob = b_cterm_rule_prob.view(b, self.NT, self.NT, self.T)



            a_emb_features = kernel(nonterm_emb, self.projection_matrix_anti, is_query=True, eps=0.0001, return_log=return_log, relu=self.relu) # bsz, NT, r

            #bc_emb_features = kernel(bc_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT_T^2, r
            b_c_emb_features = kernel(b_c_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT^2, r
            #bterm_c_emb_features = kernel(bterm_c_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, T*NT, r
            #b_cterm_emb_features = kernel(b_cterm_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT*T, r
            #import pdb; pdb.set_trace()
# bsz, NT, 1, r
# bsz, 1, NT_T^2, r

# bsz, NT, NT_T^2

            #import pdb; pdb.set_trace()
            if not self.relu:
                if not self.genbmm:
                    #rule_prob = (
                    #         nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT_T^2, r
                    #         ).logsumexp(-1) # bsz, NT, NT_T^2
                    b_c_rule_prob = (
                             a_emb_features[:,:,None,:] + b_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT^2, r
                             ).logsumexp(-1) # bsz, NT, NT^2
                    b_c_rule_prob = b_c_rule_prob + b_c_emb_bias
                    b_c_rule_prob = b_c_rule_prob.log_softmax(-1) + mixture_logits[:, :, 0:1]
                    b_c_rule_prob = b_c_rule_prob.view(b, self.NT, self.NT, self.NT)
                    #bterm_c_rule_prob = (
                    #         a_emb_features[:,:,None,:] + bterm_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, T*NT, r
                    #         ).logsumexp(-1) # bsz, NT, T*NT
                    #bterm_c_rule_prob = bterm_c_rule_prob + bterm_c_emb_bias
                    #bterm_c_rule_prob = bterm_c_rule_prob.view(b, self.NT, self.T, self.NT)
                    #b_cterm_rule_prob = (
                    #         a_emb_features[:,:,None,:] + b_cterm_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT*T, r
                    #         ).logsumexp(-1) # bsz, NT, NT*T
                    #b_cterm_rule_prob = b_cterm_rule_prob + b_cterm_emb_bias
                    #b_cterm_rule_prob = b_cterm_rule_prob.view(b, self.NT, self.NT, self.T)

                    rule_prob = torch.cat (   (torch.cat ((b_c_rule_prob, b_cterm_rule_prob), -1), torch.cat ((bterm_c_rule_prob, bterm_cterm_rule_prob), -1)), -2)


                    rule_prob = rule_prob.view(b, self.NT, self.NT_T**2)
                    #rule_prob[:, :, self.NT:, self.NT:] = a_berm_cterm
                    #rule_prob = rule_prob + bc_emb_bias#.log_softmax(-1) # bsz, NT, NT_T^2
                else:
# bsz, NT, r   1, NT^2, r
                    b_c_rule_prob = genbmm.logbmm(a_emb_features, b_c_emb_features.transpose(-1, -2).contiguous())
                    #b_c_rule_prob = (
                    #         a_emb_features[:,:,None,:] + b_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT^2, r
                    #         ).logsumexp(-1) # bsz, NT, NT^2
                    b_c_rule_prob = b_c_rule_prob + b_c_emb_bias
                    b_c_rule_prob = b_c_rule_prob.log_softmax(-1) + mixture_logits[:, :, 0:1]
                    b_c_rule_prob = b_c_rule_prob.view(b, self.NT, self.NT, self.NT)
                    #bterm_c_rule_prob = genbmm.logbmm(a_emb_features, bterm_c_emb_features.transpose(-1, -2).contiguous())
                    #bterm_c_rule_prob = (
                    #         a_emb_features[:,:,None,:] + bterm_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, T*NT, r
                    #         ).logsumexp(-1) # bsz, NT, T*NT
                    #bterm_c_rule_prob = bterm_c_rule_prob + bterm_c_emb_bias
                    #bterm_c_rule_prob = bterm_c_rule_prob.view(b, self.NT, self.T, self.NT)
                    #b_cterm_rule_prob = genbmm.logbmm(a_emb_features, b_cterm_emb_features.transpose(-1, -2).contiguous())
                    #b_cterm_rule_prob = (
                    #         a_emb_features[:,:,None,:] + b_cterm_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT*T, r
                    #         ).logsumexp(-1) # bsz, NT, NT*T
                    #b_cterm_rule_prob = b_cterm_rule_prob + b_cterm_emb_bias
                    #b_cterm_rule_prob = b_cterm_rule_prob.view(b, self.NT, self.NT, self.T)

                    rule_prob = torch.cat (   (torch.cat ((b_c_rule_prob, b_cterm_rule_prob), -1), torch.cat ((bterm_c_rule_prob, bterm_cterm_rule_prob), -1)), -2)


                    rule_prob = rule_prob.view(b, self.NT, self.NT_T**2)
                    #assert False
                    #bc_emb_features_2 = bc_emb_features.expand(b, -1, -1).transpose(-1, -2).contiguous() # b, r, NT_T^2    # b, NT, r
                    #rule_prob = genbmm.logbmm(nonterm_emb_features, bc_emb_features_2) + bc_emb_bias  # b, NT, NT_T^2
            else:
                rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
                rule_prob_log = rule_prob.log() + bc_emb_bias
                #rule_prob = rule_prob_log.log_softmax(-1)
            #rule_prob2 = ((
            #         nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] + bc_emb_bias.unsqueeze(-1)
            #         ).logsumexp(-1)).log_softmax(-1)
            #import pdb; pdb.set_trace()
            #rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
            #rule_prob = rule_prob / rule_prob.sum(-1, keepdim=True)
            #rule_prob = rule_prob.log()


            if self.band > 0:
                #import pdb; pdb.set_trace()
                x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                diff = torch.abs(x_ids - y_ids)
                BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                band_BC_bc = (diff < self.band) & BC_bc
                band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                band = band_BC_bc | band_Bc | band_bC
                band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            elif self.band < 0:
                #import pdb; pdb.set_trace()
                #x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                #y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                #diff = torch.abs(x_ids - y_ids)
                #BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                #Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                #bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                #band_BC_bc = (diff < self.band) & BC_bc
                #band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                #band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                #band = band_BC_bc | band_Bc | band_bC
                ids = list(range(self.NT_T**2))
                import random
                random.seed(1234)
                random.shuffle(ids)
                ids = ids[(((-self.band-1)*2+1)*2*self.NT_T):]
                #rule_prob[ids] = -float('inf')
                band = torch.zeros(self.NT_T, self.NT_T).cuda()
                band = band.view(-1)
                band[ids] = -float('inf')
                #band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            #rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            #import pdb; pdb.set_trace()
            #rule_prob = F.log_softmax(rule_prob, -1)
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            #with torch.no_grad():
            #    rule_prob_gt = torch.matmul(nonterm_emb, ball_call_emb.view(1, self.NT_T**2, -1).transpose(-1, -2)) + ball_call_emb_bias.view(1, 1, -1) # bsz, NT, NT_T^2
            #    rule_prob_gt = F.log_softmax(rule_prob_gt, -1)
            #return (rule_prob, rule_prob_gt, ball_call_emb_bias, ball_call_emb_features, nonterm_emb_features)
            return (rule_prob, None, None, None, None)

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        bc_emb_bias, bc_emb_features, nonterm_emb_features = rules_ll[2:]
        self.rules_ll = rules_ll[0][0]
        if False and (not self.relu):
            return (terms_ll, rules_ll[:2], roots_ll, bc_emb_bias, bc_emb_features, nonterm_emb_features), kl 
        else:
            return (terms_ll, rules_ll[0], roots_ll), kl 

class CPCFGProjbBandSoftmaxallNoshareNoantiOrig(torch.nn.Module):
    def __init__(self, V, NT, T, *args, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256,
                 num_features=0, 
                 relu=False,
                 norm_square=0,
                 unit_norm=False,
                 temperature=1,
                 **kwargs): 
        super(CPCFGProjbBandSoftmaxallNoshareNoantiOrig, self).__init__()
        assert z_dim >= 0
        self.num_features = num_features
        self.relu = relu
        self.unit_norm = unit_norm
        self.temperature = temperature
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.genbmm = False
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.share_term = kwargs.get("share_term", False)
        self.share_rule = kwargs.get("share_rule", False)
        self.share_root = kwargs.get("share_root", False)
        self.wo_enc_emb = kwargs.get("wo_enc_emb", False)

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.nonterm_emb_softmax = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim if self.share_rule else s_dim + z_dim
        self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if self.share_root else s_dim + z_dim
        self.root_mlp = nn.Sequential(nn.Linear(root_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))

        projection_matrix = get_2d_array(num_features, rule_dim).transpose(0,1)
        if temperature != 1:
            projection_matrix = projection_matrix * math.sqrt(temperature)
        self.projection_matrix_anti = nn.Parameter(projection_matrix)
        #self.projection_matrix_anti.requires_grad = False


        if z_dim > 0:
            i_dim = w_dim
            self.enc_emb = lambda x: x 
            if not self.wo_enc_emb:
                self.enc_emb = nn.Embedding(V, w_dim)
                self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                    bidirectional=True, num_layers=1, batch_first=True)
                i_dim = h_dim * 2
            self.enc_out = nn.Linear(i_dim, z_dim * 2)

        term_dim = s_dim if self.share_term else s_dim + z_dim
        self.term_mlp = nn.Sequential(nn.Linear(term_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        self._initialize()
        if self.unit_norm:
            self._normalize()

    def _normalize(self):
        assert self.unit_norm
        self.rule_mlp.weight.data.div_(self.rule_mlp.weight.data.norm(dim=-1, keepdim=True))
        self.nonterm_emb.data.div_(self.nonterm_emb.data.norm(dim=-1, keepdim=True))
        #import pdb; pdb.set_trace()

    def _initialize(self):
        #import pdb; pdb.set_trace()
        for n, p in self.named_parameters():
            print (n)
            if p.dim() > 1 and ('projection_matrix' not in n):
                torch.nn.init.xavier_uniform_(p)
        #import pdb; pdb.set_trace()

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    def forward(self, x, lengths, *args, txt=None, txt_lengths=None, use_mean=False, **kwargs):
        """ x, lengths: word ids; txt, txt_lengths: sub-word ids """
        b, n = x.shape[:2]
        batch_size = b 
        if self.z_dim > 0:
            max_pooling = kwargs.get("max_pooling", True)
            enforce_sorted = kwargs.get("enforce_sorted", False)
            item = (x, lengths, txt, txt_lengths) + args if txt is not None else x
            mean, lvar = self.enc(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
        else:
            z = torch.zeros(b, 1).cuda()
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0 and not self.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0 and not self.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            nonterm_emb_softmax = self.nonterm_emb_softmax.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            if self.z_dim > 0 and not self.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim # bsz, NT, z
                )
                assert False
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1) # bsz, NT, H+z

            #import pdb; pdb.set_trace()
            #rule_prob_1 = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            ball_call_emb = self.rule_mlp.weight.unsqueeze(0) # 1, NT_T^2, H+z
            return_log = True
            if self.relu:
                return_log = False
            if self.unit_norm:
                nonterm_emb = nonterm_emb / nonterm_emb.norm(dim=-1, keepdim=True)
                ball_call_emb = ball_call_emb / ball_call_emb.norm(dim=-1, keepdim=True)

            ball_call_emb_bias = self.rule_mlp.bias.view(1, 1, self.NT_T, self.NT_T) # 1, 1, NT_T^2
            bterm_cterm_emb_bias = ball_call_emb_bias[:, :, self.NT:, self.NT:].contiguous().view(1, 1, -1) # 1, 1, T^2
            b_c_emb_bias = ball_call_emb_bias[:, :, :self.NT, :self.NT].contiguous().view(1, 1, -1) # 1, 1, T^2
            bterm_c_emb_bias = ball_call_emb_bias[:, :, self.NT:, :self.NT].contiguous().view(1, 1, -1) # 1, 1, T^2
            b_cterm_emb_bias = ball_call_emb_bias[:, :, :self.NT, self.NT:].contiguous().view(1, 1, -1) # 1, 1, T^2

            ball_call_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)
            b_c_emb = ball_call_emb[:, :self.NT, :self.NT, :].contiguous().view(1, self.NT**2, -1) # 1, NT^2, H
            #bterm_c_emb = ball_call_emb[:, self.NT:, :self.NT, :].contiguous().view(1, self.T*self.NT, -1) # 1, T*NT, H
            #b_cterm_emb = ball_call_emb[:, :self.NT, self.NT:, :].contiguous().view(1, self.NT*self.T, -1) # 1, NT*T, H

            #import pdb; pdb.set_trace()

##### softmax

            bterm_cterm_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)[:, self.NT:, self.NT:, :].contiguous().view(1, self.T**2, -1).transpose(-1, -2).contiguous() # 1, H, T^2
            bterm_cterm_rule_prob = torch.matmul(nonterm_emb_softmax, bterm_cterm_emb) + bterm_cterm_emb_bias # bsz, NT, T^2
            bterm_cterm_rule_prob = bterm_cterm_rule_prob.view(b, self.NT, self.T, self.T)

            bterm_c_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)[:, self.NT:, :self.NT, :].contiguous().view(1, self.T*self.NT, -1).transpose(-1, -2).contiguous() # 1, H, T*NT
            bterm_c_rule_prob = torch.matmul(nonterm_emb_softmax, bterm_c_emb) + bterm_c_emb_bias # bsz, NT, T*NT
            bterm_c_rule_prob = bterm_c_rule_prob.view(b, self.NT, self.T, self.NT)

            b_cterm_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)[:, :self.NT:, self.NT:, :].contiguous().view(1, self.NT*self.T, -1).transpose(-1, -2).contiguous() # 1, H, NT*T
            b_cterm_rule_prob = torch.matmul(nonterm_emb_softmax, b_cterm_emb) + b_cterm_emb_bias # bsz, NT, NT*T
            b_cterm_rule_prob = b_cterm_rule_prob.view(b, self.NT, self.NT, self.T)


            b_c_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)[:, :self.NT:, :self.NT, :].contiguous().view(1, self.NT*self.NT, -1).transpose(-1, -2).contiguous() # 1, H, NT*NT
            b_c_rule_prob = torch.matmul(nonterm_emb, b_c_emb) + b_c_emb_bias # bsz, NT, NT*NT
            b_c_rule_prob = b_c_rule_prob.view(b, self.NT, self.NT, self.NT)

            #a_emb_features = kernel(nonterm_emb, self.projection_matrix_anti, is_query=True, eps=0.0001, return_log=return_log, relu=self.relu) # bsz, NT, r

            #bc_emb_features = kernel(bc_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT_T^2, r
            #b_c_emb_features = kernel(b_c_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT^2, r
            #bterm_c_emb_features = kernel(bterm_c_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, T*NT, r
            #b_cterm_emb_features = kernel(b_cterm_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT*T, r
            #import pdb; pdb.set_trace()
# bsz, NT, 1, r
# bsz, 1, NT_T^2, r

# bsz, NT, NT_T^2

            #import pdb; pdb.set_trace()
            if not self.relu:
                if not self.genbmm:
                    #rule_prob = (
                    #         nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT_T^2, r
                    #         ).logsumexp(-1) # bsz, NT, NT_T^2
                    #b_c_rule_prob = (
                    #         a_emb_features[:,:,None,:] + b_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT^2, r
                    #         ).logsumexp(-1) # bsz, NT, NT^2
                    #b_c_rule_prob = b_c_rule_prob + b_c_emb_bias
                    #b_c_rule_prob = b_c_rule_prob.view(b, self.NT, self.NT, self.NT)
                    #bterm_c_rule_prob = (
                    #         a_emb_features[:,:,None,:] + bterm_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, T*NT, r
                    #         ).logsumexp(-1) # bsz, NT, T*NT
                    #bterm_c_rule_prob = bterm_c_rule_prob + bterm_c_emb_bias
                    #bterm_c_rule_prob = bterm_c_rule_prob.view(b, self.NT, self.T, self.NT)
                    #b_cterm_rule_prob = (
                    #         a_emb_features[:,:,None,:] + b_cterm_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT*T, r
                    #         ).logsumexp(-1) # bsz, NT, NT*T
                    #b_cterm_rule_prob = b_cterm_rule_prob + b_cterm_emb_bias
                    #b_cterm_rule_prob = b_cterm_rule_prob.view(b, self.NT, self.NT, self.T)

                    rule_prob = torch.cat (   (torch.cat ((b_c_rule_prob, b_cterm_rule_prob), -1), torch.cat ((bterm_c_rule_prob, bterm_cterm_rule_prob), -1)), -2)


                    rule_prob = rule_prob.view(b, self.NT, self.NT_T**2)
                    #rule_prob[:, :, self.NT:, self.NT:] = a_berm_cterm
                    #rule_prob = rule_prob + bc_emb_bias#.log_softmax(-1) # bsz, NT, NT_T^2
                else:
# bsz, NT, r   1, NT^2, r
                    #b_c_rule_prob = genbmm.logbmm(a_emb_features, b_c_emb_features.transpose(-1, -2).contiguous())
                    #b_c_rule_prob = (
                    #         a_emb_features[:,:,None,:] + b_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT^2, r
                    #         ).logsumexp(-1) # bsz, NT, NT^2
                    #b_c_rule_prob = b_c_rule_prob + b_c_emb_bias
                    #b_c_rule_prob = b_c_rule_prob.view(b, self.NT, self.NT, self.NT)
                    #bterm_c_rule_prob = genbmm.logbmm(a_emb_features, bterm_c_emb_features.transpose(-1, -2).contiguous())
                    #bterm_c_rule_prob = (
                    #         a_emb_features[:,:,None,:] + bterm_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, T*NT, r
                    #         ).logsumexp(-1) # bsz, NT, T*NT
                    #bterm_c_rule_prob = bterm_c_rule_prob + bterm_c_emb_bias
                    #bterm_c_rule_prob = bterm_c_rule_prob.view(b, self.NT, self.T, self.NT)
                    #b_cterm_rule_prob = genbmm.logbmm(a_emb_features, b_cterm_emb_features.transpose(-1, -2).contiguous())
                    #b_cterm_rule_prob = (
                    #         a_emb_features[:,:,None,:] + b_cterm_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT*T, r
                    #         ).logsumexp(-1) # bsz, NT, NT*T
                    #b_cterm_rule_prob = b_cterm_rule_prob + b_cterm_emb_bias
                    #b_cterm_rule_prob = b_cterm_rule_prob.view(b, self.NT, self.NT, self.T)

                    rule_prob = torch.cat (   (torch.cat ((b_c_rule_prob, b_cterm_rule_prob), -1), torch.cat ((bterm_c_rule_prob, bterm_cterm_rule_prob), -1)), -2)


                    rule_prob = rule_prob.view(b, self.NT, self.NT_T**2)
                    #assert False
                    #bc_emb_features_2 = bc_emb_features.expand(b, -1, -1).transpose(-1, -2).contiguous() # b, r, NT_T^2    # b, NT, r
                    #rule_prob = genbmm.logbmm(nonterm_emb_features, bc_emb_features_2) + bc_emb_bias  # b, NT, NT_T^2
            else:
                rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
                rule_prob_log = rule_prob.log() + bc_emb_bias
                #rule_prob = rule_prob_log.log_softmax(-1)
            #rule_prob2 = ((
            #         nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] + bc_emb_bias.unsqueeze(-1)
            #         ).logsumexp(-1)).log_softmax(-1)
            #import pdb; pdb.set_trace()
            #rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
            #rule_prob = rule_prob / rule_prob.sum(-1, keepdim=True)
            #rule_prob = rule_prob.log()


            if self.band > 0:
                #import pdb; pdb.set_trace()
                x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                diff = torch.abs(x_ids - y_ids)
                BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                band_BC_bc = (diff < self.band) & BC_bc
                band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                band = band_BC_bc | band_Bc | band_bC
                band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            elif self.band < 0:
                #import pdb; pdb.set_trace()
                #x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                #y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                #diff = torch.abs(x_ids - y_ids)
                #BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                #Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                #bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                #band_BC_bc = (diff < self.band) & BC_bc
                #band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                #band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                #band = band_BC_bc | band_Bc | band_bC
                ids = list(range(self.NT_T**2))
                import random
                random.seed(1234)
                random.shuffle(ids)
                ids = ids[(((-self.band-1)*2+1)*2*self.NT_T):]
                #rule_prob[ids] = -float('inf')
                band = torch.zeros(self.NT_T, self.NT_T).cuda()
                band = band.view(-1)
                band[ids] = -float('inf')
                #band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            #rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            rule_prob = F.log_softmax(rule_prob, -1)
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            #with torch.no_grad():
            #    rule_prob_gt = torch.matmul(nonterm_emb, ball_call_emb.view(1, self.NT_T**2, -1).transpose(-1, -2)) + ball_call_emb_bias.view(1, 1, -1) # bsz, NT, NT_T^2
            #    rule_prob_gt = F.log_softmax(rule_prob_gt, -1)
            #return (rule_prob, rule_prob_gt, ball_call_emb_bias, ball_call_emb_features, nonterm_emb_features)
            return (rule_prob, None, None, None, None)

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        bc_emb_bias, bc_emb_features, nonterm_emb_features = rules_ll[2:]
        self.rules_ll = rules_ll[0][0]
        if False and (not self.relu):
            return (terms_ll, rules_ll[:2], roots_ll, bc_emb_bias, bc_emb_features, nonterm_emb_features), kl 
        else:
            return (terms_ll, rules_ll[0], roots_ll), kl 

class CPCFGProjbBandSoftmaxallNoshareNoantiThird(torch.nn.Module):
    def __init__(self, V, NT, T, *args, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256,
                 num_features=0, 
                 relu=False,
                 norm_square=0,
                 unit_norm=False,
                 temperature=1,
                 **kwargs): 
        super(CPCFGProjbBandSoftmaxallNoshareNoantiThird, self).__init__()
        assert z_dim >= 0
        self.num_features = num_features
        self.relu = relu
        self.unit_norm = unit_norm
        self.temperature = temperature
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.genbmm = False
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.share_term = kwargs.get("share_term", False)
        self.share_rule = kwargs.get("share_rule", False)
        self.share_root = kwargs.get("share_root", False)
        self.wo_enc_emb = kwargs.get("wo_enc_emb", False)

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.nonterm_emb_softmax = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim if self.share_rule else s_dim + z_dim
        self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if self.share_root else s_dim + z_dim
        self.root_mlp = nn.Sequential(nn.Linear(root_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))

        projection_matrix = get_2d_array(num_features, rule_dim).transpose(0,1)
        if temperature != 1:
            projection_matrix = projection_matrix * math.sqrt(temperature)
        self.projection_matrix_anti = nn.Parameter(projection_matrix)
        #self.projection_matrix_anti.requires_grad = False


        if z_dim > 0:
            i_dim = w_dim
            self.enc_emb = lambda x: x 
            if not self.wo_enc_emb:
                self.enc_emb = nn.Embedding(V, w_dim)
                self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                    bidirectional=True, num_layers=1, batch_first=True)
                i_dim = h_dim * 2
            self.enc_out = nn.Linear(i_dim, z_dim * 2)

        term_dim = s_dim if self.share_term else s_dim + z_dim
        self.term_mlp = nn.Sequential(nn.Linear(term_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        self._initialize()
        if self.unit_norm:
            self._normalize()

    def _normalize(self):
        assert self.unit_norm
        self.rule_mlp.weight.data.div_(self.rule_mlp.weight.data.norm(dim=-1, keepdim=True))
        self.nonterm_emb.data.div_(self.nonterm_emb.data.norm(dim=-1, keepdim=True))
        #import pdb; pdb.set_trace()

    def _initialize(self):
        #import pdb; pdb.set_trace()
        for n, p in self.named_parameters():
            print (n)
            if p.dim() > 1 and ('projection_matrix' not in n):
                torch.nn.init.xavier_uniform_(p)
        #import pdb; pdb.set_trace()

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    def forward(self, x, lengths, *args, txt=None, txt_lengths=None, use_mean=False, **kwargs):
        """ x, lengths: word ids; txt, txt_lengths: sub-word ids """
        b, n = x.shape[:2]
        batch_size = b 
        if self.z_dim > 0:
            max_pooling = kwargs.get("max_pooling", True)
            enforce_sorted = kwargs.get("enforce_sorted", False)
            item = (x, lengths, txt, txt_lengths) + args if txt is not None else x
            mean, lvar = self.enc(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
        else:
            z = torch.zeros(b, 1).to(x.device)
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0 and not self.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0 and not self.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            nonterm_emb_softmax = self.nonterm_emb_softmax.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            if self.z_dim > 0 and not self.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim # bsz, NT, z
                )
                assert False
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1) # bsz, NT, H+z

            #import pdb; pdb.set_trace()
            #rule_prob_1 = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            ball_call_emb = self.rule_mlp.weight.unsqueeze(0) # 1, NT_T^2, H+z
            return_log = True
            if self.relu:
                return_log = False
            if self.unit_norm:
                nonterm_emb = nonterm_emb / nonterm_emb.norm(dim=-1, keepdim=True)
                ball_call_emb = ball_call_emb / ball_call_emb.norm(dim=-1, keepdim=True)

            ball_call_emb_bias = self.rule_mlp.bias.view(1, 1, self.NT_T, self.NT_T) # 1, 1, NT_T^2
            bterm_cterm_emb_bias = ball_call_emb_bias[:, :, self.NT:, self.NT:].contiguous().view(1, 1, -1) # 1, 1, T^2
            b_c_emb_bias = ball_call_emb_bias[:, :, :self.NT, :self.NT].contiguous().view(1, 1, -1) # 1, 1, T^2
            bterm_c_emb_bias = ball_call_emb_bias[:, :, self.NT:, :self.NT].contiguous().view(1, 1, -1) # 1, 1, T^2
            b_cterm_emb_bias = ball_call_emb_bias[:, :, :self.NT, self.NT:].contiguous().view(1, 1, -1) # 1, 1, T^2

            ball_call_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)
            b_c_emb = ball_call_emb[:, :self.NT, :self.NT, :].contiguous().view(1, self.NT**2, -1) # 1, NT^2, H
            bterm_c_emb = ball_call_emb[:, self.NT:, :self.NT, :].contiguous().view(1, self.T*self.NT, -1) # 1, T*NT, H
            b_cterm_emb = ball_call_emb[:, :self.NT, self.NT:, :].contiguous().view(1, self.NT*self.T, -1) # 1, NT*T, H

            #import pdb; pdb.set_trace()

##### softmax

            bterm_cterm_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)[:, self.NT:, self.NT:, :].contiguous().view(1, self.T**2, -1).transpose(-1, -2).contiguous() # 1, H, T^2
            bterm_cterm_rule_prob = torch.matmul(nonterm_emb_softmax, bterm_cterm_emb) + bterm_cterm_emb_bias # bsz, NT, T^2
            bterm_cterm_rule_prob = bterm_cterm_rule_prob.view(b, self.NT, self.T, self.T)

            #bterm_c_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)[:, self.NT:, :self.NT, :].contiguous().view(1, self.T*self.NT, -1).transpose(-1, -2).contiguous() # 1, H, T*NT
            #bterm_c_rule_prob = torch.matmul(nonterm_emb_softmax, bterm_c_emb) + bterm_c_emb_bias # bsz, NT, T*NT
            #bterm_c_rule_prob = bterm_c_rule_prob.view(b, self.NT, self.T, self.NT)

            #b_cterm_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)[:, :self.NT:, self.NT:, :].contiguous().view(1, self.NT*self.T, -1).transpose(-1, -2).contiguous() # 1, H, NT*T
            #b_cterm_rule_prob = torch.matmul(nonterm_emb_softmax, b_cterm_emb) + b_cterm_emb_bias # bsz, NT, NT*T
            #b_cterm_rule_prob = b_cterm_rule_prob.view(b, self.NT, self.NT, self.T)



            a_emb_features = kernel(nonterm_emb, self.projection_matrix_anti, is_query=True, eps=0.0001, return_log=return_log, relu=self.relu) # bsz, NT, r

            #bc_emb_features = kernel(bc_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT_T^2, r
            b_c_emb_features = kernel(b_c_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT^2, r
            bterm_c_emb_features = kernel(bterm_c_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, T*NT, r
            b_cterm_emb_features = kernel(b_cterm_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT*T, r
            #import pdb; pdb.set_trace()
# bsz, NT, 1, r
# bsz, 1, NT_T^2, r

# bsz, NT, NT_T^2

            #import pdb; pdb.set_trace()
            if not self.relu:
                if not self.genbmm:
                    #rule_prob = (
                    #         nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT_T^2, r
                    #         ).logsumexp(-1) # bsz, NT, NT_T^2
                    b_c_rule_prob = (
                             a_emb_features[:,:,None,:] + b_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT^2, r
                             ).logsumexp(-1) # bsz, NT, NT^2
                    b_c_rule_prob = b_c_rule_prob + b_c_emb_bias
                    b_c_rule_prob = b_c_rule_prob.view(b, self.NT, self.NT, self.NT)
                    bterm_c_rule_prob = (
                             a_emb_features[:,:,None,:] + bterm_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, T*NT, r
                             ).logsumexp(-1) # bsz, NT, T*NT
                    bterm_c_rule_prob = bterm_c_rule_prob + bterm_c_emb_bias
                    bterm_c_rule_prob = bterm_c_rule_prob.view(b, self.NT, self.T, self.NT)
                    b_cterm_rule_prob = (
                             a_emb_features[:,:,None,:] + b_cterm_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT*T, r
                             ).logsumexp(-1) # bsz, NT, NT*T
                    b_cterm_rule_prob = b_cterm_rule_prob + b_cterm_emb_bias
                    b_cterm_rule_prob = b_cterm_rule_prob.view(b, self.NT, self.NT, self.T)

                    rule_prob = torch.cat (   (torch.cat ((b_c_rule_prob, b_cterm_rule_prob), -1), torch.cat ((bterm_c_rule_prob, bterm_cterm_rule_prob), -1)), -2)


                    rule_prob = rule_prob.view(b, self.NT, self.NT_T**2)
                    #rule_prob[:, :, self.NT:, self.NT:] = a_berm_cterm
                    #rule_prob = rule_prob + bc_emb_bias#.log_softmax(-1) # bsz, NT, NT_T^2
                else:
# bsz, NT, r   1, NT^2, r
                    b_c_rule_prob = genbmm.logbmm(a_emb_features, b_c_emb_features.transpose(-1, -2).contiguous())
                    #b_c_rule_prob = (
                    #         a_emb_features[:,:,None,:] + b_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT^2, r
                    #         ).logsumexp(-1) # bsz, NT, NT^2
                    b_c_rule_prob = b_c_rule_prob + b_c_emb_bias
                    b_c_rule_prob = b_c_rule_prob.view(b, self.NT, self.NT, self.NT)
                    bterm_c_rule_prob = genbmm.logbmm(a_emb_features, bterm_c_emb_features.transpose(-1, -2).contiguous())
                    #bterm_c_rule_prob = (
                    #         a_emb_features[:,:,None,:] + bterm_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, T*NT, r
                    #         ).logsumexp(-1) # bsz, NT, T*NT
                    bterm_c_rule_prob = bterm_c_rule_prob + bterm_c_emb_bias
                    bterm_c_rule_prob = bterm_c_rule_prob.view(b, self.NT, self.T, self.NT)
                    b_cterm_rule_prob = genbmm.logbmm(a_emb_features, b_cterm_emb_features.transpose(-1, -2).contiguous())
                    #b_cterm_rule_prob = (
                    #         a_emb_features[:,:,None,:] + b_cterm_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT*T, r
                    #         ).logsumexp(-1) # bsz, NT, NT*T
                    b_cterm_rule_prob = b_cterm_rule_prob + b_cterm_emb_bias
                    b_cterm_rule_prob = b_cterm_rule_prob.view(b, self.NT, self.NT, self.T)

                    rule_prob = torch.cat (   (torch.cat ((b_c_rule_prob, b_cterm_rule_prob), -1), torch.cat ((bterm_c_rule_prob, bterm_cterm_rule_prob), -1)), -2)


                    rule_prob = rule_prob.view(b, self.NT, self.NT_T**2)
                    #assert False
                    #bc_emb_features_2 = bc_emb_features.expand(b, -1, -1).transpose(-1, -2).contiguous() # b, r, NT_T^2    # b, NT, r
                    #rule_prob = genbmm.logbmm(nonterm_emb_features, bc_emb_features_2) + bc_emb_bias  # b, NT, NT_T^2
            else:
                rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
                rule_prob_log = rule_prob.log() + bc_emb_bias
                #rule_prob = rule_prob_log.log_softmax(-1)
            #rule_prob2 = ((
            #         nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] + bc_emb_bias.unsqueeze(-1)
            #         ).logsumexp(-1)).log_softmax(-1)
            #import pdb; pdb.set_trace()
            #rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
            #rule_prob = rule_prob / rule_prob.sum(-1, keepdim=True)
            #rule_prob = rule_prob.log()


            if self.band > 0:
                #import pdb; pdb.set_trace()
                x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                diff = torch.abs(x_ids - y_ids)
                BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                band_BC_bc = (diff < self.band) & BC_bc
                band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                band = band_BC_bc | band_Bc | band_bC
                band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            elif self.band < 0:
                #import pdb; pdb.set_trace()
                #x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                #y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                #diff = torch.abs(x_ids - y_ids)
                #BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                #Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                #bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                #band_BC_bc = (diff < self.band) & BC_bc
                #band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                #band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                #band = band_BC_bc | band_Bc | band_bC
                ids = list(range(self.NT_T**2))
                import random
                random.seed(1234)
                random.shuffle(ids)
                ids = ids[(((-self.band-1)*2+1)*2*self.NT_T):]
                #rule_prob[ids] = -float('inf')
                band = torch.zeros(self.NT_T, self.NT_T).cuda()
                band = band.view(-1)
                band[ids] = -float('inf')
                #band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            #rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            rule_prob = F.log_softmax(rule_prob, -1)
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            #with torch.no_grad():
            #    rule_prob_gt = torch.matmul(nonterm_emb, ball_call_emb.view(1, self.NT_T**2, -1).transpose(-1, -2)) + ball_call_emb_bias.view(1, 1, -1) # bsz, NT, NT_T^2
            #    rule_prob_gt = F.log_softmax(rule_prob_gt, -1)
            #return (rule_prob, rule_prob_gt, ball_call_emb_bias, ball_call_emb_features, nonterm_emb_features)
            return (rule_prob, None, None, None, None)

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        bc_emb_bias, bc_emb_features, nonterm_emb_features = rules_ll[2:]
        self.rules_ll = rules_ll[0][0]
        if False and (not self.relu):
            return (terms_ll, rules_ll[:2], roots_ll, bc_emb_bias, bc_emb_features, nonterm_emb_features), kl 
        else:
            return (terms_ll, rules_ll[0], roots_ll), kl 

class CPCFGProjbBandSoftmaxallNoshareNoantiFourth(torch.nn.Module):
    def __init__(self, V, NT, T, *args, 
                 h_dim = 512,
                 w_dim = 512,
                 z_dim = 64,
                 s_dim = 256,
                 num_features=0, 
                 relu=False,
                 norm_square=0,
                 unit_norm=False,
                 temperature=1,
                 **kwargs): 
        super(CPCFGProjbBandSoftmaxallNoshareNoantiFourth, self).__init__()
        assert z_dim >= 0
        self.num_features = num_features
        self.relu = relu
        self.unit_norm = unit_norm
        self.temperature = temperature
        self.NT_T = NT + T
        self.NT = NT
        self.T = T
        self.genbmm = False
        self.z_dim = z_dim
        self.s_dim = s_dim

        self.share_term = kwargs.get("share_term", False)
        self.share_rule = kwargs.get("share_rule", False)
        self.share_root = kwargs.get("share_root", False)
        self.wo_enc_emb = kwargs.get("wo_enc_emb", False)

        self.term_emb = nn.Parameter(torch.randn(T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(NT, s_dim))
        self.nonterm_emb_softmax = nn.Parameter(torch.randn(NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))

        rule_dim = s_dim if self.share_rule else s_dim + z_dim
        self.rule_mlp = nn.Linear(rule_dim, self.NT_T ** 2)
        root_dim = s_dim if self.share_root else s_dim + z_dim
        self.root_mlp = nn.Sequential(nn.Linear(root_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, NT))

        projection_matrix = get_2d_array(num_features, rule_dim).transpose(0,1)
        if temperature != 1:
            projection_matrix = projection_matrix * math.sqrt(temperature)
        self.projection_matrix_anti = nn.Parameter(projection_matrix)
        #self.projection_matrix_anti.requires_grad = False


        if z_dim > 0:
            i_dim = w_dim
            self.enc_emb = lambda x: x 
            if not self.wo_enc_emb:
                self.enc_emb = nn.Embedding(V, w_dim)
                self.enc_rnn = nn.LSTM(w_dim, h_dim, 
                    bidirectional=True, num_layers=1, batch_first=True)
                i_dim = h_dim * 2
            self.enc_out = nn.Linear(i_dim, z_dim * 2)

        term_dim = s_dim if self.share_term else s_dim + z_dim
        self.term_mlp = nn.Sequential(nn.Linear(term_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      ResLayer(s_dim, s_dim),
                                      nn.Linear(s_dim, V))
        self._initialize()
        if self.unit_norm:
            self._normalize()

    def _normalize(self):
        assert self.unit_norm
        self.rule_mlp.weight.data.div_(self.rule_mlp.weight.data.norm(dim=-1, keepdim=True))
        self.nonterm_emb.data.div_(self.nonterm_emb.data.norm(dim=-1, keepdim=True))
        #import pdb; pdb.set_trace()

    def _initialize(self):
        #import pdb; pdb.set_trace()
        for n, p in self.named_parameters():
            print (n)
            if p.dim() > 1 and ('projection_matrix' not in n):
                torch.nn.init.xavier_uniform_(p)
        #import pdb; pdb.set_trace()

    def update_state_dict(self, new_state, strict=True):
        self.load_state_dict(new_state, strict=strict) 

    def kl(self, mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

    def enc(self, x, lengths, max_pooling=True, enforce_sorted=False):
        x_embbed = self.enc_emb(x)
        x_packed = pack_padded_sequence(
            x_embbed, lengths, batch_first=True, enforce_sorted=enforce_sorted
        )
        h_packed, _ = self.enc_rnn(x_packed)
        if max_pooling:
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
        else:
            padding_value = 0
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.sum(1)[0] / lengths.unsqueze(-1)
        out = self.enc_out(h)
        mean = out[:, : self.z_dim]
        lvar = out[:, self.z_dim :]
        return mean, lvar

    def forward(self, x, lengths, *args, txt=None, txt_lengths=None, use_mean=False, **kwargs):
        """ x, lengths: word ids; txt, txt_lengths: sub-word ids """
        b, n = x.shape[:2]
        batch_size = b 
        if self.z_dim > 0:
            max_pooling = kwargs.get("max_pooling", True)
            enforce_sorted = kwargs.get("enforce_sorted", False)
            item = (x, lengths, txt, txt_lengths) + args if txt is not None else x
            mean, lvar = self.enc(
                item, lengths, max_pooling=max_pooling, enforce_sorted=enforce_sorted
            )
            z = mean
            if not use_mean:
                z = mean.new(b, mean.size(1)).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
        else:
            z = torch.zeros(b, 1).to(x.device)
            kl = None
        self.z = z

        def roots():
            root_emb = self.root_emb.expand(b, self.s_dim)
            if self.z_dim > 0 and not self.share_root:
                root_emb = torch.cat([root_emb, self.z], -1)
            root_prob = F.log_softmax(self.root_mlp(root_emb), -1)
            return root_prob
        
        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.s_dim
            ) 
            if self.z_dim > 0 and not self.share_term:
                #z_expand = self.z.unsqueeze(1).unsqueeze(2).expand(
                #    b, n, self.T, self.z_dim
                #) # it indeed makes a difference, weird.
                z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
                z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
                term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = F.log_softmax(self.term_mlp(term_emb), -1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            nonterm_emb_softmax = self.nonterm_emb_softmax.unsqueeze(0).expand(
                b, self.NT, self.s_dim # bsz, NT, H
            )
            if self.z_dim > 0 and not self.share_rule:
                z_expand = self.z.unsqueeze(1).expand(
                    b, self.NT, self.z_dim # bsz, NT, z
                )
                assert False
                nonterm_emb = torch.cat([nonterm_emb, z_expand], -1) # bsz, NT, H+z

            #import pdb; pdb.set_trace()
            #rule_prob_1 = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            ball_call_emb = self.rule_mlp.weight.unsqueeze(0) # 1, NT_T^2, H+z
            return_log = True
            if self.relu:
                return_log = False
            if self.unit_norm:
                nonterm_emb = nonterm_emb / nonterm_emb.norm(dim=-1, keepdim=True)
                ball_call_emb = ball_call_emb / ball_call_emb.norm(dim=-1, keepdim=True)

            ball_call_emb_bias = self.rule_mlp.bias.view(1, 1, self.NT_T, self.NT_T) # 1, 1, NT_T^2
            bterm_cterm_emb_bias = ball_call_emb_bias[:, :, self.NT:, self.NT:].contiguous().view(1, 1, -1) # 1, 1, T^2
            b_c_emb_bias = ball_call_emb_bias[:, :, :self.NT, :self.NT].contiguous().view(1, 1, -1) # 1, 1, T^2
            bterm_c_emb_bias = ball_call_emb_bias[:, :, self.NT:, :self.NT].contiguous().view(1, 1, -1) # 1, 1, T^2
            b_cterm_emb_bias = ball_call_emb_bias[:, :, :self.NT, self.NT:].contiguous().view(1, 1, -1) # 1, 1, T^2

            ball_call_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)
            b_c_emb = ball_call_emb[:, :self.NT, :self.NT, :].contiguous().view(1, self.NT**2, -1) # 1, NT^2, H
            bterm_c_emb = ball_call_emb[:, self.NT:, :self.NT, :].contiguous().view(1, self.T*self.NT, -1) # 1, T*NT, H
            b_cterm_emb = ball_call_emb[:, :self.NT, self.NT:, :].contiguous().view(1, self.NT*self.T, -1) # 1, NT*T, H
            bterm_cterm_emb = ball_call_emb[:, self.NT:, self.NT:, :].contiguous().view(1, self.T*self.T, -1) # 1, NT*T, H

            #import pdb; pdb.set_trace()

##### softmax

            #bterm_cterm_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)[:, self.NT:, self.NT:, :].contiguous().view(1, self.T**2, -1).transpose(-1, -2).contiguous() # 1, H, T^2
            #bterm_cterm_rule_prob = torch.matmul(nonterm_emb_softmax, bterm_cterm_emb) + bterm_cterm_emb_bias # bsz, NT, T^2
            #bterm_cterm_rule_prob = bterm_cterm_rule_prob.view(b, self.NT, self.T, self.T)

            #bterm_c_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)[:, self.NT:, :self.NT, :].contiguous().view(1, self.T*self.NT, -1).transpose(-1, -2).contiguous() # 1, H, T*NT
            #bterm_c_rule_prob = torch.matmul(nonterm_emb_softmax, bterm_c_emb) + bterm_c_emb_bias # bsz, NT, T*NT
            #bterm_c_rule_prob = bterm_c_rule_prob.view(b, self.NT, self.T, self.NT)

            #b_cterm_emb = ball_call_emb.view(1, self.NT_T, self.NT_T, -1)[:, :self.NT:, self.NT:, :].contiguous().view(1, self.NT*self.T, -1).transpose(-1, -2).contiguous() # 1, H, NT*T
            #b_cterm_rule_prob = torch.matmul(nonterm_emb_softmax, b_cterm_emb) + b_cterm_emb_bias # bsz, NT, NT*T
            #b_cterm_rule_prob = b_cterm_rule_prob.view(b, self.NT, self.NT, self.T)



            a_emb_features = kernel(nonterm_emb, self.projection_matrix_anti, is_query=True, eps=0.0001, return_log=return_log, relu=self.relu) # bsz, NT, r

            #bc_emb_features = kernel(bc_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT_T^2, r
            b_c_emb_features = kernel(b_c_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT^2, r
            bterm_c_emb_features = kernel(bterm_c_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, T*NT, r
            b_cterm_emb_features = kernel(b_cterm_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT*T, r
            bterm_cterm_emb_features = kernel(bterm_cterm_emb, self.projection_matrix_anti, is_query=False, eps=0.0001, return_log=return_log, relu=self.relu) # 1, NT*T, r
            #import pdb; pdb.set_trace()
# bsz, NT, 1, r
# bsz, 1, NT_T^2, r

# bsz, NT, NT_T^2

            #import pdb; pdb.set_trace()
            if not self.relu:
                if not self.genbmm:
                    #rule_prob = (
                    #         nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT_T^2, r
                    #         ).logsumexp(-1) # bsz, NT, NT_T^2
                    b_c_rule_prob = (
                             a_emb_features[:,:,None,:] + b_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT^2, r
                             ).logsumexp(-1) # bsz, NT, NT^2
                    b_c_rule_prob = b_c_rule_prob + b_c_emb_bias
                    b_c_rule_prob = b_c_rule_prob.view(b, self.NT, self.NT, self.NT)
                    bterm_c_rule_prob = (
                             a_emb_features[:,:,None,:] + bterm_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, T*NT, r
                             ).logsumexp(-1) # bsz, NT, T*NT
                    bterm_c_rule_prob = bterm_c_rule_prob + bterm_c_emb_bias
                    bterm_c_rule_prob = bterm_c_rule_prob.view(b, self.NT, self.T, self.NT)
                    b_cterm_rule_prob = (
                             a_emb_features[:,:,None,:] + b_cterm_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT*T, r
                             ).logsumexp(-1) # bsz, NT, NT*T
                    b_cterm_rule_prob = b_cterm_rule_prob + b_cterm_emb_bias
                    b_cterm_rule_prob = b_cterm_rule_prob.view(b, self.NT, self.NT, self.T)
                    bterm_cterm_rule_prob = (
                             a_emb_features[:,:,None,:] + bterm_cterm_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT*T, r
                             ).logsumexp(-1) # bsz, NT, NT*T
                    bterm_cterm_rule_prob = bterm_cterm_rule_prob + bterm_cterm_emb_bias
                    bterm_cterm_rule_prob = bterm_cterm_rule_prob.view(b, self.NT, self.T, self.T)

                    rule_prob = torch.cat (   (torch.cat ((b_c_rule_prob, b_cterm_rule_prob), -1), torch.cat ((bterm_c_rule_prob, bterm_cterm_rule_prob), -1)), -2)


                    rule_prob = rule_prob.view(b, self.NT, self.NT_T**2)
                    #rule_prob[:, :, self.NT:, self.NT:] = a_berm_cterm
                    #rule_prob = rule_prob + bc_emb_bias#.log_softmax(-1) # bsz, NT, NT_T^2
                else:
# bsz, NT, r   1, NT^2, r
                    b_c_rule_prob = genbmm.logbmm(a_emb_features, b_c_emb_features.transpose(-1, -2).contiguous())
                    #b_c_rule_prob = (
                    #         a_emb_features[:,:,None,:] + b_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT^2, r
                    #         ).logsumexp(-1) # bsz, NT, NT^2
                    b_c_rule_prob = b_c_rule_prob + b_c_emb_bias
                    b_c_rule_prob = b_c_rule_prob.view(b, self.NT, self.NT, self.NT)
                    bterm_c_rule_prob = genbmm.logbmm(a_emb_features, bterm_c_emb_features.transpose(-1, -2).contiguous())
                    #bterm_c_rule_prob = (
                    #         a_emb_features[:,:,None,:] + bterm_c_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, T*NT, r
                    #         ).logsumexp(-1) # bsz, NT, T*NT
                    bterm_c_rule_prob = bterm_c_rule_prob + bterm_c_emb_bias
                    bterm_c_rule_prob = bterm_c_rule_prob.view(b, self.NT, self.T, self.NT)
                    b_cterm_rule_prob = genbmm.logbmm(a_emb_features, b_cterm_emb_features.transpose(-1, -2).contiguous())
                    #b_cterm_rule_prob = (
                    #         a_emb_features[:,:,None,:] + b_cterm_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT*T, r
                    #         ).logsumexp(-1) # bsz, NT, NT*T
                    b_cterm_rule_prob = b_cterm_rule_prob + b_cterm_emb_bias
                    b_cterm_rule_prob = b_cterm_rule_prob.view(b, self.NT, self.NT, self.T)
                    bterm_cterm_rule_prob = genbmm.logbmm(a_emb_features, bterm_cterm_emb_features.transpose(-1, -2).contiguous())
                    #b_cterm_rule_prob = (
                    #         a_emb_features[:,:,None,:] + b_cterm_emb_features[:,None,:,:] # bsz, NT, 1, r    1, 1, NT*T, r
                    #         ).logsumexp(-1) # bsz, NT, NT*T
                    bterm_cterm_rule_prob = bterm_cterm_rule_prob + bterm_cterm_emb_bias
                    bterm_cterm_rule_prob = bterm_cterm_rule_prob.view(b, self.NT, self.T, self.T)

                    rule_prob = torch.cat (   (torch.cat ((b_c_rule_prob, b_cterm_rule_prob), -1), torch.cat ((bterm_c_rule_prob, bterm_cterm_rule_prob), -1)), -2)


                    rule_prob = rule_prob.view(b, self.NT, self.NT_T**2)
                    #assert False
                    #bc_emb_features_2 = bc_emb_features.expand(b, -1, -1).transpose(-1, -2).contiguous() # b, r, NT_T^2    # b, NT, r
                    #rule_prob = genbmm.logbmm(nonterm_emb_features, bc_emb_features_2) + bc_emb_bias  # b, NT, NT_T^2
            else:
                assert False
                rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
                rule_prob_log = rule_prob.log() + bc_emb_bias
                #rule_prob = rule_prob_log.log_softmax(-1)
            #rule_prob2 = ((
            #         nonterm_emb_features[:,:,None,:] + bc_emb_features[:,None,:,:] + bc_emb_bias.unsqueeze(-1)
            #         ).logsumexp(-1)).log_softmax(-1)
            #import pdb; pdb.set_trace()
            #rule_prob = torch.matmul(nonterm_emb_features, bc_emb_features.transpose(-1, -2)) # bsz, NT, NT_T^2
            #rule_prob = rule_prob / rule_prob.sum(-1, keepdim=True)
            #rule_prob = rule_prob.log()


            if self.band > 0:
                #import pdb; pdb.set_trace()
                x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                diff = torch.abs(x_ids - y_ids)
                BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                band_BC_bc = (diff < self.band) & BC_bc
                band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                band = band_BC_bc | band_Bc | band_bC
                band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            elif self.band < 0:
                #import pdb; pdb.set_trace()
                #x_ids = torch.arange(self.NT_T).view(-1, 1).repeat(1, self.NT_T).cuda()
                #y_ids = torch.arange(self.NT_T).view(1, -1).repeat(self.NT_T, 1).cuda()
                #diff = torch.abs(x_ids - y_ids)
                #BC_bc = ((x_ids < self.NT) & (y_ids < self.NT)) | ((x_ids >= self.NT) & (y_ids >= self.NT))
                #Bc = ((x_ids < self.NT) & (y_ids >= self.NT)) 
                #bC = ((x_ids >= self.NT) & (y_ids < self.NT))
                #band_BC_bc = (diff < self.band) & BC_bc
                #band_Bc = (torch.abs(x_ids * self.T/self.NT - (y_ids-self.NT)) < self.band) & Bc
                #band_bC = (torch.abs((x_ids-self.NT) * self.NT/self.T - y_ids) < self.band) & bC
                #band = band_BC_bc | band_Bc | band_bC
                ids = list(range(self.NT_T**2))
                import random
                random.seed(1234)
                random.shuffle(ids)
                ids = ids[(((-self.band-1)*2+1)*2*self.NT_T):]
                #rule_prob[ids] = -float('inf')
                band = torch.zeros(self.NT_T, self.NT_T).cuda()
                band = band.view(-1)
                band[ids] = -float('inf')
                #band = band.float().log()
                rule_prob = rule_prob + band.view(1, 1, -1)
            #rule_prob = F.log_softmax(self.rule_mlp(nonterm_emb), -1) # bsz, NT, NT_T**2
            rule_prob = F.log_softmax(rule_prob, -1)
            rule_prob = rule_prob.view(b, self.NT, self.NT_T, self.NT_T)
            #with torch.no_grad():
            #    rule_prob_gt = torch.matmul(nonterm_emb, ball_call_emb.view(1, self.NT_T**2, -1).transpose(-1, -2)) + ball_call_emb_bias.view(1, 1, -1) # bsz, NT, NT_T^2
            #    rule_prob_gt = F.log_softmax(rule_prob_gt, -1)
            #return (rule_prob, rule_prob_gt, ball_call_emb_bias, ball_call_emb_features, nonterm_emb_features)
            return (rule_prob, None, None, None, None)

        roots_ll, terms_ll, rules_ll = roots(), terms(), rules()
        bc_emb_bias, bc_emb_features, nonterm_emb_features = rules_ll[2:]
        self.rules_ll = rules_ll[0][0]
        if False and (not self.relu):
            return (terms_ll, rules_ll[:2], roots_ll, bc_emb_bias, bc_emb_features, nonterm_emb_features), kl 
        else:
            return (terms_ll, rules_ll[0], roots_ll), kl 
