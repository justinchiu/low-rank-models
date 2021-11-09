import torch
from .helpers import _Struct


class SemiMarkov(_Struct):
    """
    edge : b x N x K x C x C semimarkov potentials
    """

    def _check_potentials(self, edge, lengths=None):
        batch, N_1, K, C, C2 = edge.shape
        edge = self.semiring.convert(edge)
        N = N_1 + 1
        if lengths is None:
            lengths = torch.LongTensor([N] * batch)
        assert max(lengths) <= N, "Length longer than edge scores"
        assert max(lengths) == N, "At least one in batch must be length N"
        assert C == C2, "Transition shape doesn't match"
        return edge, batch, N, K, C, lengths

    def _dp(self, edge, lengths=None, force_grad=False, cache=True):
        #import pdb; pdb.set_trace()
        scores_left, scores_right, scores_1_C, scores_C_1, scores_1_1 = edge # 1, b, N, K, C, r
        semiring = self.semiring
        ssize = semiring.size()
        #import pdb; pdb.set_trace()
        #edge, batch, N, K, C, lengths = self._check_potentials(edge, lengths)
        batch = scores_left.size(0)
        N_1 = scores_left.size(1)
        N = N_1 + 1
        K = scores_left.size(2)
        C = scores_left.size(3) + 1
        scores_left = scores_left.unsqueeze(0)
        scores_right = scores_right.unsqueeze(0)
        scores_1_C = scores_1_C.unsqueeze(0) # 1, b, N - 1, K, 1, C-1
        #scores_C_1 = scores_C_1.unsqueeze(0)
        scores_1_1 = scores_1_1.unsqueeze(0)

        #spans = self._make_chart(N - 1, (batch, K, C, C), scores_left, force_grad)
        alpha = self._make_chart(N, (batch, K, C), scores_left, force_grad)
        beta = self._make_chart(N, (batch, C), scores_left, force_grad)
        semiring.one_(beta[0].data)
        #import pdb; pdb.set_trace()

# times: +
# sum: logsumexp
        for n in range(1, N):
# matmul 1, b, 1, 1, C x 1, b, K, C, C
    #    1, b, K, C, C
#        exp(beta) x exp(edge)
            #spans = semiring.times(
            #    beta[n - 1].view(ssize, batch, 1, 1, C),
            #    edge[:, :, n - 1].view(ssize, batch, K, C, C),
            #)
            scores_left_n = scores_left[:, :, n-1] # 1, b, K, C-1, r
            scores_right_n = scores_right[:, :, n-1] # 1, b, K, C-1, r
            scores_1_C_n = scores_1_C[:, :, n-1] # 1, b, K, 1, C-1
            scores_1_1_n = scores_1_1[:, :, n-1] # 1, b, K, 1, 1

            #transitions = (scores_left_n.unsqueeze(-2) + scores_right_n.unsqueeze(-3)).logsumexp(dim=-1) # 1, b, K, C-1, 1, r + 1, b, K, 1, C-1, r -> 1, b, K, C-1, C-1
            #print (transitions.exp().sum(-1))

            beta_n = beta[n - 1].view(ssize, batch, 1, 1, C) # 1, b, 1, 1, C
            beta_n_1 = beta_n[:, :, :, :, :C-1].squeeze(-2).unsqueeze(-1) # 1, b, 1, C-1, 1
            beta_1 =  beta_n[:, :, :, :, C-1:] # 1, b, 1, 1, 1
            beta_n_1_add_scores_right_n = (beta_n_1 + scores_left_n).logsumexp(-2, keepdim=True) # 1, b, K, 1, r
            #alpha[n - 1][:, :, :, :C-1] = torch.stack( ( (beta_n_1_add_scores_right_n + scores_right_n).logsumexp(-1), (beta_1 + scores_1_C_n).squeeze(-2)), dim=0).logsumexp(dim=0) # 1, b, K, C-1
            alpha[n - 1][:, :, :, :C-1] = (beta_n_1_add_scores_right_n + scores_right_n).logsumexp(-1) # 1, b, K, C-1
            #alpha[n - 1][:, :, :, :C-1] += (beta_1 + scores_1_C_n).squeeze(-2)
            alpha[n - 1][:, :, :, C-1:] = torch.stack( ( (beta_n_1.transpose(-1,-2) + scores_1_C_n).logsumexp(-1),  (beta_1 + scores_1_1_n).squeeze(-2)), dim=0).logsumexp(dim=0)
# 1, b, K, C, r
            #alpha[n - 1][:] = semiring.sum(spans) # 1, b, K, C
            t = max(n - K, -1)
            f1 = torch.arange(n - 1, t, -1)
            f2 = torch.arange(1, len(f1) + 1)
            beta[n][:] = semiring.sum(
                torch.stack([alpha[a][:, :, b] for a, b in zip(f1, f2)], dim=1), dim=1
            )
        v = semiring.sum(
            torch.stack([beta[l - 1][:, i] for i, l in enumerate(lengths)], dim=1),
            dim=2,
        )
        v = semiring.unconvert(v)
        #return v, spans, beta
        return v, None, beta

    @staticmethod
    def _rand():
        b = torch.randint(2, 4, (1,))
        N = torch.randint(2, 4, (1,))
        K = torch.randint(2, 4, (1,))
        C = torch.randint(2, 4, (1,))
        return torch.rand(b, N, K, C, C), (b.item(), (N + 1).item())

    def _arrange_marginals(self, marg):
        return torch.stack(marg, dim=2)

    @staticmethod
    def to_parts(sequence, extra, lengths=None):
        """
        Convert a sequence representation to edges

        Parameters:
            sequence : b x N  long tensors in [-1, 0, C-1]
            C : number of states
            lengths: b long tensor of N values
        Returns:
            edge : b x (N-1) x K x C x C semimarkov potentials
                        (t x z_t x z_{t-1})
        """
        C, K = extra
        batch, N = sequence.shape
        labels = torch.zeros(batch, N - 1, K, C, C).long()
        if lengths is None:
            lengths = torch.LongTensor([N] * batch)

        for b in range(batch):
            last = None
            c = None
            for n in range(0, N):
                if sequence[b, n] == -1:
                    assert n != 0
                    continue
                else:
                    new_c = sequence[b, n]
                    if n != 0:
                        labels[b, last, n - last, new_c, c] = 1
                    last = n
                    c = new_c
        return labels

    @staticmethod
    def from_parts(edge):
        """
        Convert a edges to a sequence representation.

        Parameters:
            edge : b x (N-1) x K x C x C semimarkov potentials
                    (t x z_t x z_{t-1})
        Returns:
            sequence : b x N  long tensors in [-1, 0, C-1]

        """
        batch, N_1, K, C, _ = edge.shape
        N = N_1 + 1
        labels = torch.zeros(batch, N).long().fill_(-1)
        on = edge.nonzero()
        for i in range(on.shape[0]):
            if on[i][1] == 0:
                labels[on[i][0], on[i][1]] = on[i][4]
            labels[on[i][0], on[i][1] + on[i][2]] = on[i][3]
        # print(edge.nonzero(), labels)
        return labels, (C, K)

    # Tests
    def enumerate(self, edge):
        semiring = self.semiring
        ssize = semiring.size()
        batch, N, K, C, _ = edge.shape
        edge = semiring.convert(edge)
        chains = {}
        chains[0] = [
            ([(c, 0)], semiring.one_(torch.zeros(ssize, batch))) for c in range(C)
        ]

        for n in range(1, N + 1):
            chains[n] = []
            for k in range(1, K):
                if n - k not in chains:
                    continue
                for chain, score in chains[n - k]:
                    for c in range(C):
                        chains[n].append(
                            (
                                chain + [(c, k)],
                                semiring.mul(
                                    score, edge[:, :, n - k, k, c, chain[-1][0]]
                                ),
                            )
                        )
        ls = [s for (_, s) in chains[N]]
        return semiring.unconvert(semiring.sum(torch.stack(ls, dim=1), dim=1)), ls
