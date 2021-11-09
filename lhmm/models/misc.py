import torch as th
import torch.nn as nn

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

# post-LN
class ResidualLayerOld(nn.Module):
    def __init__(
        self, in_dim = 100,
        out_dim = 100,
        dropout = 0.,
        # unused args
        do_norm = True,
        pre_norm = True,
        do_res = True,
    ):
        super(ResidualLayerOld, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = self.dropout(x)
        #x = self.dropout(self.lin1(x).relu())
        return self.layer_norm(self.dropout(self.lin2(x).relu()) + x)

class ResidualLayerNoNorm(nn.Module):
    def __init__(
        self, in_dim = 100,
        out_dim = 100,
        dropout = 0.,
        # unused args
        do_norm = True,
        pre_norm = True,
        do_res = True,
    ):
        super(ResidualLayerNoNorm, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        #x = x.tanh()
        x = self.dropout(x)
        #x = self.dropout(self.lin1(x).relu())
        return self.dropout(self.lin2(x).relu()) + x
        #return self.dropout(self.lin2(x).tanh()) + x


# pre-LN
class ResidualLayer(nn.Module):
    def __init__(
        self, in_dim = 100,
        out_dim = 100,
    ):
        super(ResidualLayer, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.layer_norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        x = self.lin1(self.layer_norm(x)).relu()
        return self.lin2(x).relu() + x

class ResidualLayerOpt(nn.Module):
    def __init__(
        self, in_dim = 100,
        out_dim = 100,
        dropout = 0.,
        do_norm = True,
        pre_norm = True, # otherwise post_norm
        do_res = True,
    ):
        super(ResidualLayerOpt, self).__init__()

        self.do_norm = do_norm
        self.pre_norm = pre_norm
        self.do_res = do_res

        self.dropout = nn.Dropout(dropout)

        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        if self.do_norm:
            self.layer_norm = nn.LayerNorm(out_dim)
        if self.do_norm and self.pre_norm:
            self.prelayer_norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        y = self.prelayer_norm(x) if self.do_norm and self.pre_norm else x
        y = self.lin2(self.dropout(self.lin1(y).relu())).relu()
        if self.do_res:
            y = y + x
        return self.layer_norm(y) if self.do_norm else y


class ResidualLayerBuilder(nn.Module):
    def __init__(
        self,
        in_dim = 100,
        out_dim = 100,
        dropout = 0.,
        build_string = "",
    ):
        """
        Key
        d: Dense = nn.Linear
        n: Norm = nn.LayerNorm
        p: PReLU
        r: ReLU
        +: Residual
        """
        super(ResidualLayerBuilder, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.linears = []
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        y = self.prelayer_norm(x) if self.do_norm and self.pre_norm else x
        y = self.lin2(self.dropout(self.lin1(y).relu())).relu()
        if self.do_res:
            y = y + x
        return self.layer_norm(y) if self.do_norm else y


class LogDropout(nn.Module):
    def __init__(
        self, p,
    ):
        super(LogDropout, self).__init__()
        self.p = p

    def forward(self, x, column_dropout=False):
        if self.training and self.p > 0:
            if column_dropout:
                annihilate_mask = th.empty_like(x).fill_(self.p).bernoulli().bool()
            else:
                annihilate_mask = (th.empty(x.shape[-1])
                    .fill_(self.p).bernoulli().bool()[None].expand(x.shape)
                )
            return x.masked_fill(annihilate_mask, float("-inf"))
        else:
            return x

class LogDropoutM(nn.Module):
    def __init__(
        self, p,
    ):
        super(LogDropoutM, self).__init__()
        self.p = p

    def forward(self, x, annihilate_mask=None):
        if self.training and self.p > 0 and annihilate_mask is None:
            return x
            #annihilate_mask = th.empty_like(x).fill_(self.p).bernoulli().bool()
            #return x.masked_fill(annihilate_mask, float("-inf"))
        elif self.training and self.p > 0 and annihilate_mask is not None:
            while annihilate_mask.dim() < x.dim():
                annihilate_mask = annihilate_mask.unsqueeze(0)
            annihilate_mask = annihilate_mask.expand(x.shape)
            #return x.masked_fill(annihilate_mask, float("-inf"))
            return x.masked_fill(annihilate_mask, -1e5)
        else:
            return x


