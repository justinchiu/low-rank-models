import torch
from .helpers import _Struct, Chart

A, B = 0, 1


class CKY(_Struct):
    def _dp(self, scores, lengths=None, force_grad=False, cache=True):

        semiring = self.semiring

        # Checks
        terms, rules, roots = scores[:3]
        rules.requires_grad_(True)
        ssize = semiring.size()
        batch, N, T = terms.shape
        _, NT, _, _ = rules.shape
        S = NT + T

        terms, rules, roots = (
            semiring.convert(terms).requires_grad_(True),
            semiring.convert(rules).requires_grad_(True),
            semiring.convert(roots).requires_grad_(True),
        )
        if lengths is None:
            lengths = torch.LongTensor([N] * batch)

        # Charts
        beta = [
            Chart((batch, N, N, NT), rules, semiring, cache=cache) for _ in range(2)
        ]
        span = [None for _ in range(N)]
        v = (ssize, batch)
        term_use = terms + 0.0

        # Split into NT/T groups
        NTs = slice(0, NT)
        Ts = slice(NT, S)
        rules = rules.view(ssize, batch, 1, NT, S, S)

        def arr(a, b):
            return rules[..., a, b].contiguous().view(*v + (NT, -1)).transpose(-2, -1)

        matmul = semiring.matmul
        times = semiring.times
        #import pdb; pdb.set_trace()
       # a = times(torch.ones(5), torch.ones(5))
        X_Y_Z = arr(NTs, NTs) # 1, b, NT^2, NT
        X_Y1_Z = arr(Ts, NTs)
        X_Y_Z1 = arr(NTs, Ts) # 1, b, NT*T, NT
        X_Y1_Z1 = arr(Ts, Ts) # 1, b, T^2, NT
        flag = False
        if len(scores) == 6:
          flag = True

          # NT-> NT,NT
          bc_emb_bias, bc_emb_features, nonterm_emb_features = scores[3:]
          bc_emb_bias = bc_emb_bias.view(1, 1, NT, NT)
          Yall_Zall_bias = bc_emb_bias.view(1, 1, 1, -1) # 1, 1, 1, NT^2
          Yall_Zall_features = bc_emb_features.view(1, 1, NT, NT, -1).contiguous().view(1, 1, 1, NT**2, -1) # 1, 1, 1, NT^2, r
          #Y1_Z1_bias = bc_emb_bias[:, :, Ts, Ts].contiguous().view(1, 1, 1, -1) # 1, 1, 1, T^2
          #Y_Z1_bias = bc_emb_bias[:, :, NTs, Ts].contiguous().view(1, 1, 1, -1) # 1, 1, 1, NT*T
          #Y1_Z_bias = bc_emb_bias[:, :, Ts, NTs].contiguous().view(1, 1, 1, -1) # 1, 1, 1, T*NT
          Y_Z_bias = bc_emb_bias[:, :, NTs, NTs].contiguous().view(1, 1, 1, -1) # 1, 1, 1, NT^2
          #Y1_Z1_features = bc_emb_features.view(1, 1, S, S, -1)[:, :, Ts, Ts].contiguous().view(1, 1, 1, T**2, -1) # 1, 1, 1, T^2, r
          #Y_Z1_features = bc_emb_features.view(1, 1, S, S, -1)[:, :, NTs, Ts].contiguous().view(1, 1, 1, NT*T, -1) # 1, 1, 1, NT*T, r
          #Y1_Z_features = bc_emb_features.view(1, 1, S, S, -1)[:, :, Ts, NTs].contiguous().view(1, 1, 1, T*NT, -1) # 1, 1, 1, T*NT, r
          Y_Z_features = bc_emb_features.view(1, 1, NT, NT, -1)[:, :, NTs, NTs].contiguous().view(1, 1, 1, NT*NT, -1) # 1, 1, 1, NT*NT, r
          nonterm_emb_features = nonterm_emb_features.unsqueeze(0).unsqueeze(2) # 1, b, 1, NT, r


          """
          bc_emb_bias = bc_emb_bias.view(1, 1, S, S)
          Yall_Zall_bias = bc_emb_bias.view(1, 1, 1, -1) # 1, 1, 1, S^2
          Yall_Zall_features = bc_emb_features.view(1, 1, S, S, -1).contiguous().view(1, 1, 1, S**2, -1) # 1, 1, 1, S^2, r
          Y1_Z1_bias = bc_emb_bias[:, :, Ts, Ts].contiguous().view(1, 1, 1, -1) # 1, 1, 1, T^2
          Y_Z1_bias = bc_emb_bias[:, :, NTs, Ts].contiguous().view(1, 1, 1, -1) # 1, 1, 1, NT*T
          Y1_Z_bias = bc_emb_bias[:, :, Ts, NTs].contiguous().view(1, 1, 1, -1) # 1, 1, 1, T*NT
          Y_Z_bias = bc_emb_bias[:, :, NTs, NTs].contiguous().view(1, 1, 1, -1) # 1, 1, 1, NT^2
          Y1_Z1_features = bc_emb_features.view(1, 1, S, S, -1)[:, :, Ts, Ts].contiguous().view(1, 1, 1, T**2, -1) # 1, 1, 1, T^2, r
          Y_Z1_features = bc_emb_features.view(1, 1, S, S, -1)[:, :, NTs, Ts].contiguous().view(1, 1, 1, NT*T, -1) # 1, 1, 1, NT*T, r
          Y1_Z_features = bc_emb_features.view(1, 1, S, S, -1)[:, :, Ts, NTs].contiguous().view(1, 1, 1, T*NT, -1) # 1, 1, 1, T*NT, r
          Y_Z_features = bc_emb_features.view(1, 1, S, S, -1)[:, :, NTs, NTs].contiguous().view(1, 1, 1, NT*NT, -1) # 1, 1, 1, NT*NT, r
          nonterm_emb_features = nonterm_emb_features.unsqueeze(0).unsqueeze(2) # 1, b, 1, NT, r
          import pdb; pdb.set_trace()
          """

# bc_emb_bias: 1, 1, S^2
# bc_emb_features: 1, S^2, r
# nonterm_emb_features: bsz, NT, r

        span[0] = term_use
        if flag:
          denominator = Yall_Zall_bias.unsqueeze(-1) + Yall_Zall_features # 1, 1, 1, S^2, r
          denominator = torch.logsumexp(denominator, -2) # 1, 1, 1, r
          denominator = nonterm_emb_features.squeeze(2) + denominator # 1, b, NT, r
          denominator = torch.logsumexp(denominator, -1).unsqueeze(2) # 1, b, 1, NT
        for w in range(1, N):
            all_span = []
            v2 = v + (N - w, -1)

            Y = beta[A][: N - w, :w, :] # 1, b, N-w, w, NT
            Z = beta[B][w:, N - w :, :] # 1, b, N-w, w, NT
            if flag:
              # NT x NT
              Y_times_Z = matmul(Y.transpose(-2, -1), Z).view(*v2) # 1, b, N-w, NT^2
              Y_times_Z = Y_times_Z + Y_Z_bias # 1, b, N-w, NT^2
              X1_efficient = Y_times_Z.unsqueeze(-1) + Y_Z_features # 1, b, N-w, NT^2, r
              X1_efficient = torch.logsumexp(X1_efficient, -2)  # 1, b, N-w, r
              X1_efficient = X1_efficient.unsqueeze(3) # 1, b, N-w, 1, r
              X1_efficient = X1_efficient + nonterm_emb_features # 1, b, N-w, NT, r
              X1_efficient = torch.logsumexp(X1_efficient, -1) # 1, b, N-w, NT
              X1_efficient = X1_efficient - denominator # 1, b, N-w, NT
              X1 = X1_efficient
              #import pdb; pdb.set_trace()
            else:
              X1 = matmul(matmul(Y.transpose(-2, -1), Z).view(*v2), X_Y_Z) # NT, NT
            #import pdb; pdb.set_trace()
            all_span.append(X1)

            Y_term = term_use[..., : N - w, :, None] # 1, b, N-w, T, 1
            Z_term = term_use[..., w:, None, :] # 1, b, N-w, 1, T

            Y = Y[..., -1, :].unsqueeze(-1)  # 1, b, N-w, NT, 1
            if flag and False:
              Y_times_Z_term = times(Y, Z_term).view(*v2) # 1, b, N-w, NT * T
              Y_times_Z_term = Y_times_Z_term + Y_Z1_bias # 1, b, N-w, NT*T
              X2_efficient = Y_times_Z_term.unsqueeze(-1) + Y_Z1_features # 1, b, N-w, NT*T, r
              X2_efficient = torch.logsumexp(X2_efficient, -2)  # 1, b, N-w, r
              X2_efficient = X2_efficient.unsqueeze(3) # 1, b, N-w, 1, r
              X2_efficient = X2_efficient + nonterm_emb_features # 1, b, N-w, NT, r
              X2_efficient = torch.logsumexp(X2_efficient, -1) # 1, b, N-w, NT
              X2_efficient = X2_efficient - denominator # 1, b, N-w, NT
              X2 = X2_efficient
            else:
              X2 = matmul(times(Y, Z_term).view(*v2), X_Y_Z1) # 1, b, N-w, NT
            #import pdb; pdb.set_trace()

            Z = Z[..., 0, :].unsqueeze(-2)
            if flag and False:
              Y_term_times_Z = times(Y_term, Z).view(*v2) # 1, b, N-w, T*NT
              Y_term_times_Z = Y_term_times_Z + Y1_Z_bias # 1, b, N-w, T*NT
              X3_efficient = Y_term_times_Z.unsqueeze(-1) + Y1_Z_features # 1, b, N-w, T*NT, r
              X3_efficient = torch.logsumexp(X3_efficient, -2)  # 1, b, N-w, r
              X3_efficient = X3_efficient.unsqueeze(3) # 1, b, N-w, 1, r
              X3_efficient = X3_efficient + nonterm_emb_features # 1, b, N-w, NT, r
              X3_efficient = torch.logsumexp(X3_efficient, -1) # 1, b, N-w, NT
              X3_efficient = X3_efficient - denominator # 1, b, N-w, NT
              X3 = X3_efficient
            else:
              X3 = matmul(times(Y_term, Z).view(*v2), X_Y1_Z)
            #import pdb; pdb.set_trace()
            all_span += [X2, X3]

            if w == 1:

                if flag and False:
                  Y_term_times_Z_term = times(Y_term, Z_term).view(*v2) # 1, b, N-1, T^2, log potentials
                  Y_term_times_Z_term = Y_term_times_Z_term + Y1_Z1_bias # 1, b, N-1, T^2
                  X4_efficient = Y_term_times_Z_term.unsqueeze(-1) + Y1_Z1_features # 1, b, N-1, T^2, r
                  X4_efficient = torch.logsumexp(X4_efficient, -2)  # 1, b, N-1, r
                  X4_efficient = X4_efficient.unsqueeze(3) # 1, b, N-1, 1, r
                  X4_efficient = X4_efficient + nonterm_emb_features # 1, b, N-1, NT, r
                  X4_efficient = torch.logsumexp(X4_efficient, -1) # 1, b, N-1, NT
                #denominator = Yall_Zall_bias.unsqueeze(-1) + Yall_Zall_features # 1, 1, 1, S^2, r
                #denominator = torch.logsumexp(denominator, -2) # 1, 1, 1, r
                #denominator = nonterm_emb_features.squeeze(2) + denominator # 1, b, NT, r
                #denominator = torch.logsumexp(denominator, -1).unsqueeze(2) # 1, b, 1, NT
                  X4_efficient = X4_efficient - denominator

                #import pdb; pdb.set_trace()
                #tmp = Y1_Z1_features + Y1_Z1_bias.unsqueeze(-1) + nonterm_emb_features.transpose(-2, -3) # 1, b, NT, T^2, r
                #tmp = torch.logsumexp(tmp, -1).log_softmax(-1) # 1, b, NT, T^2
                #tmp = tmp.transpose(-1, -2)
                  X4 = X4_efficient
                else:
                  X4 = matmul(times(Y_term, Z_term).view(*v2), X_Y1_Z1)
                all_span.append(X4)

            span[w] = semiring.sum(torch.stack(all_span, dim=-1))
            beta[A][: N - w, w, :] = span[w]
            beta[B][w:N, N - w - 1, :] = span[w]

        final = beta[A][0, :, NTs]
        top = torch.stack([final[:, i, l - 1] for i, l in enumerate(lengths)], dim=1)
        log_Z = semiring.dot(top, roots)
        return log_Z, (term_use, rules, roots, span), beta
    
    def inside(self, scores, lengths=None, _autograd=True, _raw=False):
        terms, rules, _ = scores
        batch, N, T = terms.shape
        _, NT, _, _ = rules.shape

        v, (_, _, _, spans), _ = self._dp(scores, lengths)
        K = NT #if NT > T else T
        spans_marg = torch.zeros(
            batch, N, N, K, dtype=scores[1].dtype, device=scores[1].device
        )
        spans = spans[1:]
        for w in range(len(spans)):
            spans_marg[:, w, : N - w - 1] = self.semiring.unconvert(spans[w])

        return self.semiring.unconvert(v), spans_marg

    def inside_bp(self, scores, lengths=None, _autograd=True, _raw=False):
        terms, rules, _ = scores
        batch, N, T = terms.shape
        _, NT, _, _ = rules.shape

        v, (_, _, _, spans), _ = self._dp(
            scores, lengths=lengths, force_grad=True, cache=False
        )
        
        inputs = tuple(spans)
        obj = self.semiring.unconvert(v).sum(dim=0)
        marg = torch.autograd.grad(
            obj, inputs, create_graph=True, only_inputs=True, allow_unused=False
        )

        K = NT #if NT > T else T
        spans_marg = torch.zeros(
            batch, int(N * (N - 1) / 2), K, dtype=scores[1].dtype, device=scores[1].device
        ) 
        beg_idx = 0 
        spans = spans[1:]
        for w in range(len(spans)):
            end_idx = beg_idx + N - w - 1 
            mask = (marg[w + 1] != 0).float()
            x = spans[w] * mask 
            spans_marg[:, beg_idx : end_idx] = self.semiring.unconvert(x)
            beg_idx = end_idx
        spans_marg = torch.masked_fill(spans_marg, spans_marg == 0, -1e5)
        return self.semiring.unconvert(v), spans_marg

    def inside_im(self, scores, lengths=None, _autograd=True, _raw=False):
        terms, rules, _ = scores
        batch, N, T = terms.shape
        _, NT, _, _ = rules.shape

        v, (_, _, _, spans), _ = self._dp(
            scores, lengths=lengths, force_grad=True, cache=False
        )
        
        inputs = tuple(spans)
        obj = self.semiring.unconvert(v).sum(dim=0)
        marg = torch.autograd.grad(
            obj, inputs, create_graph=True, only_inputs=True, allow_unused=False
        )

        K = NT #if NT > T else T
        spans_marg = torch.zeros(
            batch, int(N * (N - 1) / 2), K, dtype=scores[1].dtype, device=scores[1].device
        ) 
        beg_idx = 0 
        spans = spans[1:]
        for w in range(len(spans)):
            end_idx = beg_idx + N - w - 1 
            x = marg[w + 1]
            spans_marg[:, beg_idx : end_idx] = self.semiring.unconvert(x)
            beg_idx = end_idx
        return self.semiring.unconvert(v), spans_marg

    def sum_discarded(self, scores, lengths=None, _autograd=True, _raw=False):
        """ this one is buggy 'cause gradients are cached during bp """
        terms, rules, _ = scores
        batch, N, T = terms.shape
        _, NT, _, _ = rules.shape

        v, (_, _, _, spans), _ = self._dp(scores, lengths)
        
        inputs = tuple(spans)
        obj = self.semiring.unconvert(v).sum(dim=0)
        marg = torch.autograd.grad(
            obj, inputs, create_graph=True, only_inputs=True, allow_unused=False,
        )

        K = NT #if NT > T else T
        spans_marg = torch.zeros(
            batch, int(N * (N - 1) / 2), K, dtype=scores[1].dtype, device=scores[1].device
        ) 
        beg_idx = 0 
        spans = spans[1:]
        for w in range(len(spans)):
            end_idx = beg_idx + N - w - 1 
            mask = (marg[w + 1] != 0).float()
            x = spans[w] * mask 
            spans_marg[:, beg_idx : end_idx] = self.semiring.unconvert(x)
            beg_idx = end_idx
        spans_marg = torch.masked_fill(spans_marg, spans_marg == 0, -1e5)
        """
        K = NT #if NT > T else T
        spans_marg = torch.zeros(
            batch, N, N, K, dtype=scores[1].dtype, device=scores[1].device
        )
        spans = spans[1:]
        for w in range(len(spans)):
            mask = (marg[w + 1] != 0).float()
            x = spans[w] * mask 
            spans_marg[:, w, : N - w - 1] = self.semiring.unconvert(x)
        spans_marg = torch.masked_fill(spans_marg, spans_marg == 0, -1e5)
        """
        """
        K = NT #if NT > T else T
        spans_marg = torch.zeros(
            batch, N, N, K, dtype=scores[1].dtype, device=scores[1].device
        )
        spans = spans[1:]
        for w in range(len(spans)):
            spans_marg[:, w, : N - w - 1] = self.semiring.unconvert(spans[w])
        """
        return self.semiring.unconvert(v), spans_marg

    def marginals(self, scores, lengths=None, _autograd=True, _raw=False):
        """
        Compute the marginals of a CFG using CKY.

        Parameters:
            terms : b x n x T
            rules : b x NT x (NT+T) x (NT+T)
            root:   b x NT

        Returns:
            v: b tensor of total sum
            spans: bxNxT terms, (bxNTx(NT+S)x(NT+S)) rules, bxNT roots

        """
        terms, rules, roots = scores
        batch, N, T = terms.shape
        _, NT, _, _ = rules.shape

        v, (term_use, rule_use, root_use, spans), alpha = self._dp(
            scores, lengths=lengths, force_grad=True, cache=False
        )
        lprob = True 
        def marginal(obj, inputs):
            obj = self.semiring.unconvert(obj).sum(dim=0)
            marg = torch.autograd.grad(
                obj, inputs, create_graph=True, only_inputs=True, allow_unused=False,
            )
            K = NT if NT > T else T
            spans_marg = torch.zeros(
                batch, N, N, K, dtype=scores[1].dtype, device=scores[1].device
            )
            span_ls = marg[3:]
            for w in range(len(span_ls)):
                span_lprob = span_ls[w]
                if lprob:
                    span_lprob = span_lprob * spans[w]     
                x = span_lprob.sum(dim=0, keepdim=True)
                spans_marg[:, w, : N - w, :x.size(-1)] = self.semiring.unconvert(x)

            rule_marg = self.semiring.unconvert(marg[0]).squeeze(1)
            root_marg = self.semiring.unconvert(marg[1])
            term_marg = self.semiring.unconvert(marg[2])

            assert term_marg.shape == (batch, N, T)
            assert root_marg.shape == (batch, NT)
            assert rule_marg.shape == (batch, NT, NT + T, NT + T)
            return (term_marg, rule_marg, root_marg, spans_marg)

        inputs = (rule_use, root_use, term_use) + tuple(spans)
        if _raw:
            paths = []
            for k in range(v.shape[0]):
                obj = v[k : k + 1]
                marg = marginal(obj, inputs)
                paths.append(marg[-1])
            paths = torch.stack(paths, 0)
            obj = v.sum(dim=0, keepdim=True)
            term_marg, rule_marg, root_marg, _ = marginal(obj, inputs)
            return term_marg, rule_marg, root_marg, paths
        else:
            return marginal(v, inputs)

    def kbest(self, scores, lengths=None, _autograd=True, _raw=False):
        """
        Compute the marginals of a CFG using CKY.

        Parameters:
            terms : b x n x T
            rules : b x NT x (NT+T) x (NT+T)
            root:   b x NT

        Returns:
            v: b tensor of total sum
            spans: bxNxT terms, (bxNTx(NT+S)x(NT+S)) rules, bxNT roots

        """
        terms, rules, roots = scores
        batch, N, T = terms.shape
        _, NT, _, _ = rules.shape

        v, (term_use, rule_use, root_use, spans), alpha = self._dp(
            scores, lengths=lengths, force_grad=True, cache=False
        )
        lprob = True 
        def marginal(obj, inputs):
            obj = self.semiring.unconvert(obj).sum(dim=0)
            marg = torch.autograd.grad(
                obj, inputs, create_graph=True, only_inputs=True, allow_unused=False,
            )
            K = NT if NT > T else T
            spans_marg = torch.zeros(
                batch, N, N, K, dtype=scores[1].dtype, device=scores[1].device
            )
            span_ls = marg[3:]
            for w in range(len(span_ls)):
                span_lprob = span_ls[w]
                if lprob:
                    span_lprob = span_lprob * spans[w]     
                x = span_lprob.sum(dim=0, keepdim=True)
                spans_marg[:, w, : N - w, :x.size(-1)] = self.semiring.unconvert(x)

            rule_marg = self.semiring.unconvert(marg[0]).squeeze(1)
            root_marg = self.semiring.unconvert(marg[1])
            term_marg = self.semiring.unconvert(marg[2])

            assert term_marg.shape == (batch, N, T)
            assert root_marg.shape == (batch, NT)
            assert rule_marg.shape == (batch, NT, NT + T, NT + T)
            return (term_marg, rule_marg, root_marg, spans_marg)

        inputs = (rule_use, root_use, term_use) + tuple(spans)
        if _raw:
            paths = []
            for k in range(v.shape[0]):
                obj = v[k : k + 1]
                marg = marginal(obj, inputs)
                paths.append(marg[-1])
            paths = torch.stack(paths, 0)
            obj = v.sum(dim=0, keepdim=True)
            term_marg, rule_marg, root_marg, _ = marginal(obj, inputs)
            return term_marg, rule_marg, root_marg, paths
        else:
            return marginal(v, inputs)

    def score(self, potentials, parts):
        terms, rules, roots = potentials[:3]
        m_term, m_rule, m_root = parts[:3]
        b = m_term.shape[0]
        return (
            m_term.mul(terms).view(b, -1).sum(-1)
            + m_rule.mul(rules).view(b, -1).sum(-1)
            + m_root.mul(roots).view(b, -1).sum(-1)
        )

    @staticmethod
    def to_parts(spans, extra, lengths=None):
        NT, T = extra

        batch, N, N, S = spans.shape
        assert S == NT + T
        terms = torch.zeros(batch, N, T)
        rules = torch.zeros(batch, NT, S, S)
        roots = torch.zeros(batch, NT)
        for b in range(batch):
            roots[b, :] = spans[b, 0, lengths[b] - 1, :NT]
            terms[b, : lengths[b]] = spans[
                b, torch.arange(lengths[b]), torch.arange(lengths[b]), NT:
            ]
            cover = spans[b].nonzero()
            left = {i: [] for i in range(N)}
            right = {i: [] for i in range(N)}
            for i in range(cover.shape[0]):
                i, j, A = cover[i].tolist()
                left[i].append((A, j, j - i + 1))
                right[j].append((A, i, j - i + 1))
            for i in range(cover.shape[0]):
                i, j, A = cover[i].tolist()
                B = None
                for B_p, k, a_span in left[i]:
                    for C_p, k_2, b_span in right[j]:
                        if k_2 == k + 1 and a_span + b_span == j - i + 1:
                            B, C = B_p, C_p
                            break
                if j > i:
                    assert B is not None, "%s" % ((i, j, left[i], right[j], cover),)
                    rules[b, A, B, C] += 1
        return terms, rules, roots

    @staticmethod
    def from_parts(chart):
        terms, rules, roots = chart
        batch, N, N, NT, S, S = rules.shape
        assert terms.shape[1] == N

        spans = torch.zeros(batch, N, N, S, dtype=rules.dtype, device=rules.device)
        rules = rules.sum(dim=-1).sum(dim=-1)
        for n in range(N):
            spans[:, torch.arange(N - n - 1), torch.arange(n + 1, N), :NT] = rules[
                :, n, torch.arange(N - n - 1)
            ]
        spans[:, torch.arange(N), torch.arange(N), NT:] = terms
        return spans, (NT, S - NT)

    @staticmethod
    def _intermediary(spans):
        batch, N = spans.shape[:2]
        splits = {}
        cover = spans.nonzero()
        left, right = {}, {}
        for k in range(cover.shape[0]):
            b, i, j, A = cover[k].tolist()
            left.setdefault((b, i), [])
            right.setdefault((b, j), [])
            left[b, i].append((A, j, j - i + 1))
            right[b, j].append((A, i, j - i + 1))

        for x in range(cover.shape[0]):
            b, i, j, A = cover[x].tolist()
            if i == j:
                continue
            b_final = None
            c_final = None
            k_final = None
            for B_p, k, a_span in left.get((b, i), []):
                if k > j:
                    continue
                for C_p, k_2, b_span in right.get((b, j), []):
                    if k_2 == k + 1 and a_span + b_span == j - i + 1:
                        k_final = k
                        b_final = B_p
                        c_final = C_p
                        break
                if b_final is not None:
                    break
            assert k_final is not None, "%s %s %s %s" % (b, i, j, spans[b].nonzero())
            splits[(b, i, j)] = k_final, b_final, c_final
        return splits

    @classmethod
    def to_networkx(cls, spans):
        cur = 0
        N = spans.shape[1]
        n_nodes = int(spans.sum().item())
        cover = spans.nonzero().cpu()
        order = torch.argsort(cover[:, 2] - cover[:, 1])
        left = {}
        right = {}
        ordered = cover[order]
        label = ordered[:, 3]
        a = []
        b = []
        topo = [[] for _ in range(N)]
        for n in ordered:
            batch, i, j, _ = n.tolist()
            # G.add_node(cur, label=A)
            if i - j != 0:
                a.append(left[(batch, i)][0])
                a.append(right[(batch, j)][0])
                b.append(cur)
                b.append(cur)
                order = max(left[(batch, i)][1], right[(batch, j)][1]) + 1
            else:
                order = 0
            left[(batch, i)] = (cur, order)
            right[(batch, j)] = (cur, order)
            topo[order].append(cur)
            cur += 1
        indices = left
        return (n_nodes, a, b, label), indices, topo

    ###### Test

    def enumerate(self, scores):
        terms, rules, roots = scores
        semiring = self.semiring
        batch, N, T = terms.shape
        _, NT, _, _ = rules.shape

        def enumerate(x, start, end):
            if start + 1 == end:
                yield (terms[:, start, x - NT], [(start, x - NT)])
            else:
                for w in range(start + 1, end):
                    for y in range(NT) if w != start + 1 else range(NT, NT + T):
                        for z in range(NT) if w != end - 1 else range(NT, NT + T):
                            for m1, y1 in enumerate(y, start, w):
                                for m2, z1 in enumerate(z, w, end):
                                    yield (
                                        semiring.times(
                                            semiring.times(m1, m2), rules[:, x, y, z]
                                        ),
                                        [(x, start, w, end)] + y1 + z1,
                                    )

        ls = []
        for nt in range(NT):
            ls += [semiring.times(s, roots[:, nt]) for s, _ in enumerate(nt, 0, N)]
        return semiring.sum(torch.stack(ls, dim=-1)), None

    @staticmethod
    def _rand():
        batch = torch.randint(2, 5, (1,))
        N = torch.randint(2, 5, (1,))
        NT = torch.randint(2, 5, (1,))
        T = torch.randint(2, 5, (1,))
        terms = torch.rand(batch, N, T)
        rules = torch.rand(batch, NT, (NT + T), (NT + T))
        roots = torch.rand(batch, NT)
        return (terms, rules, roots), (batch.item(), N.item())
