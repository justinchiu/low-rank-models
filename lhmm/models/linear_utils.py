import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from genbmm import logbmm

def nonnegative_softmax_kernel_feature_creator(
    data: torch.Tensor,
    projection_matrix: torch.Tensor,
    is_query: bool,
    eps: float=0.0001,
    log = False,
    no_shift = False,
):
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
    #ratio = 1.0

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
    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, -1) # (bsz, len) ||x||^2
    diag_data = diag_data / 2.0
    diag_data = diag_data.unsqueeze(-1) # bsz, len, 1

    if log:
        if no_shift:
            return data_dash - diag_data
            #return math.log(ratio) + data_dash - diag_data + math.log(eps)

        if is_query:
            # test
            stuff = math.log(ratio) + data_dash - diag_data
            return stuff - stuff.max(dim=-1, keepdim=True)[0].detach()
            # /test
            # looks like the above is fine and equivalent to no_shift
            return (math.log(ratio) + data_dash - diag_data
                - torch.max(data_dash, dim=-1, keepdim=True)[0].detach()
                #- torch.max(data_dash, dim=-1, keepdim=True)[0]
                #+ math.log(eps)
            )
        else:
            # test
            stuff = math.log(ratio) + data_dash - diag_data
            return stuff - stuff.max().detach()
            # /test
            # looks like the above is fine and equivalent to no_shift
            return (math.log(ratio) + data_dash - diag_data
                - torch.max(data_dash).detach()
                #- torch.max(data_dash)
                #+ math.log(eps)
            )

    # Compute exp(wx - ||x||^2/2)  
    # (Lemma 1, SM(x, y) = E_{w~N(0,I)} exp(wx - ||x||^2/2) exp(wy - ||y||^2/2))
    if is_query:
        # for each query, we can independently scale to avoid overflows
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.max(data_dash, dim=-1, keepdim=True)[0]) + eps)
    else:
        # for keys, we need to use the same normalizer to avoid overflows
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    return data_dash

def relu_softmax_kernel_feature_creator(
    data: torch.Tensor,
    projection_matrix: torch.Tensor,
    eps: float=0.0001,
):
    """
    Constructs relu kernel features for fast softmax attention.
    Args:
      data: input for which features are computes
      projection_matrix: random matrix used to compute features
      is_query: predicate indicating whether input data corresponds to queries or
        keys
      eps: numerical stabilizer.
    Returns:
      Random features for fast softmax attention.
    """
    #ratio = 1.0 / math.sqrt(projection_matrix.shape[0])
    #ratio = 1.0

    bsz = data.size(0)
   
    projection = projection_matrix.unsqueeze(0).expand(bsz, -1, -1)

    # Compute wx
    # data:       bsz, len, D
    # projection: bsz, D, #features
    data_dash = torch.bmm(
        data,
        projection
    ) # bsz, len, #features

    # relu
    # only on torch==1.7
    #return torch.maximum(data_dash, eps).log()
    return F.threshold(data_dash, eps, eps).log()

def exp_softmax_kernel_feature_creator(
    data: torch.Tensor,
    projection_matrix: torch.Tensor,
    eps: float=0.0001,
):
    """
    Constructs exp kernel features for fast softmax attention.
    Args:
      data: input for which features are computes
      projection_matrix: random matrix used to compute features
      is_query: predicate indicating whether input data corresponds to queries or
        keys
      eps: numerical stabilizer.
    Returns:
      Random features for fast softmax attention.
    """
    bsz = data.size(0)
   
    projection = projection_matrix.unsqueeze(0).expand(bsz, -1, -1)

    # Compute wx
    # data:       bsz, len, D
    # projection: bsz, D, #features
    data_dash = torch.bmm(
        data,
        projection
    ) # bsz, len, #features

    return data_dash + eps

def get_2d_array(nb_rows, nb_columns, scaling=0):
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []
    for _ in range(nb_full_blocks):
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

def mylogbmm(x, y):
    expand = x[:,:,None,:] + y[:,None,:,:]
    return expand.logsumexp(-1)

def project_logits(
    query, key, projection_matrix,
    eps=0.0001, rff_method="log", no_shift=False,
    fast=True,
):
    kernel = nonnegative_softmax_kernel_feature_creator

    if rff_method == "exp":
        kernel = exp_softmax_kernel_feature_creator
        log_query_features = kernel(
            query, projection_matrix, eps=eps,
        )
        log_key_features = kernel(
            key, projection_matrix, eps=eps,
        )
        return checkpoint(mylogbmm, log_query_features, log_key_features)

    elif rff_method == "log":
        # log space
        log_query_features = kernel(
            query, projection_matrix, is_query=True, eps=eps, log=True,
            no_shift = no_shift,
        )
        log_key_features = kernel(
            key, projection_matrix, is_query=False, eps=eps, log=True,
            no_shift = no_shift,
        )
        #import pdb; pdb.set_trace()
        # slow and memory...would like log-bmm
        # bxz x src x tgt x dim

        if fast:
            return checkpoint(mylogbmm, log_query_features, log_key_features)
        else:
            return logbmm(log_query_features, log_key_features.transpose(-1, -2).contiguous())
        # use tvm logbmm

    elif rff_method == "relu":
        kernel = relu_softmax_kernel_feature_creator
        log_query_features = kernel(
            query, projection_matrix, eps=eps,
        )
        log_key_features = kernel(
            key, projection_matrix, eps=eps,
        )
        return checkpoint(mylogbmm, log_query_features, log_key_features)

    else:
        raise ValueError(f"Invalid rff_method: {rff_method}")


def logbbmv(x, cA, K):
    K1K = 2 * K + 1
    padded_x = F.pad(x, (K, K), value=float("-inf"))
    unfolded_x = padded_x.unfold(-1, K1K, 1)
    return (unfolded_x + cA).logsumexp(-1)

def bbmv(x, cA, K):
    K1K = 2 * K + 1
    padded_x = F.pad(x, (K, K), value=0)
    unfolded_x = padded_x.unfold(-1, K1K, 1)
    result = torch.einsum("bzk,zk->bz", unfolded_x, cA)
    return result
