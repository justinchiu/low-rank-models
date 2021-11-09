#!/usr/bin/env python3
import sys
import os
import math
import wandb
wandb.init(project="pcfg")

import argparse
import json
import random
import shutil
import copy

import torch
from torch import cuda
import numpy as np
import time
import logging
from data import Dataset
from PCFG import PCFG
from utils import *
from models import CompPCFG
from model_fast import CPCFG, CPCFG2, CPCFG3, CPCFGFixProj, CPCFGProj, CPCFGProjDebug, CPCFGProj2, CPCFGProjb, CPCFGProjDebugBand, CPCFGProjbBand, CPCFGProjbBandNoanti, CPCFGProjbBandNonorm, CPCFGProjbBandSoftmax, CPCFGProjbBandSoftmaxNoshare, CPCFGProjbBandSoftmaxNoshareNoanti, CPCFGProjbBandSoftmaxallNoshareNoanti, CPCFGProjbBandSoftmaxallNoshareNoantiMixture, CPCFGProjbBandSoftmaxallNoshareNoantiOrig, CPCFGProjbBandSoftmaxallNoshareNoantiThird, CPCFGProjbBandSoftmaxallNoshareNoantiFourth
from torch.nn.init import xavier_uniform_
from torch_struct import SentCFG


def regularize(params, device, args, grad=False):
    rule_H = torch.tensor(0.0, device=device, requires_grad=grad)
    rule_D = torch.tensor(0.0, device=device, requires_grad=grad)
    rule_C = torch.tensor(0.0, device=device, requires_grad=grad)
    if params is not None and (args.reg_h_alpha != 0. or 
        args.reg_d_alpha > 0. or args.reg_c_alpha > 0.):
        rule_lprobs = params[1]
        batch_size = rule_lprobs.size(0)
        rule_lprobs = rule_lprobs.view(batch_size, args.nt_states, -1)
        rule_probs = rule_lprobs.exp()
        if args.reg_h_alpha != 0.: # minimize entropy
            rule_H = -(rule_probs * rule_lprobs).sum(-1).mean(-1)
            rule_H = rule_H * args.reg_h_alpha
        if args.reg_d_alpha > 0.: # maximize prob. dist. among rules of diff. LHS 
            rule_D = torch.matmul(rule_probs, rule_probs.transpose(-1, -2))
            I = torch.arange(args.nt_states, device=device).long()
            rule_D[:, I, I] = 0 
            rule_D = rule_D.sqrt().sum(-1).mean(-1) / 2
            rule_D = rule_D * args.reg_d_alpha
        if args.reg_c_alpha > 0.: # rule prob. should be consistent across diff. PCFG 
            rule_probs = rule_probs.transpose(0, 1)
            rule_C = torch.matmul(rule_probs, rule_probs.transpose(-1, -2))
            I = torch.arange(batch_size, device=device).long()
            rule_C[:, I, I] = 0 
            rule_C = rule_C.sqrt().sum(-1).mean(-2) / 2
            rule_C = rule_C * args.reg_c_alpha
    return rule_H, rule_D, rule_C

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--train_file', default='data/ptb-train.pkl')
parser.add_argument('--val_file', default='data/ptb-val.pkl')
parser.add_argument('--save_path', default='compound-pcfg.pt', help='where to save the model')
parser.add_argument('--model_type', default='4th', type=str, help='model name (1st/2nd)')
parser.add_argument('--infer_fast', default=True, type=bool, help='model name (1st/2nd)')
parser.add_argument('--model_init', default=None, type=str, help='load a model')
parser.add_argument('--perturb_it', default=False, type=str, help='perturb parameter values')
parser.add_argument('--data_random', default=False, type=bool, help='sync randomness for data fetching')
parser.add_argument('--keep_random', default=False, type=bool, help='same randomness for data fetching')
parser.add_argument('--no_argmax', default=False, type=bool, help='same randomness for data fetching')
# Model options
# Generative model parameters
parser.add_argument('--z_dim', default=64, type=int, help='latent dimension')
parser.add_argument('--t_states', default=60, type=int, help='number of preterminal states')
parser.add_argument('--nt_states', default=30, type=int, help='number of nonterminal states')
parser.add_argument('--state_dim', default=256, type=int, help='symbol embedding dimension')
# Inference network parameters
parser.add_argument('--h_dim', default=512, type=int, help='hidden dim for variational LSTM')
parser.add_argument('--w_dim', default=512, type=int, help='embedding dim for variational LSTM')
#
parser.add_argument('--share_term', default=False, type=bool, help='share preterminal rules')
parser.add_argument('--share_rule', default=False, type=bool, help='share binary rules')
parser.add_argument('--share_root', default=False, type=bool, help='share binary rules')
parser.add_argument('--wo_enc_emb', default=False, type=bool, help='how to encode z')
parser.add_argument('--use_mean', default=False, type=bool, help='how to use z')
# Optimization options
parser.add_argument('--num_epochs', default=10, type=int, help='number of training epochs')
parser.add_argument('--lr', default=0.001, type=float, help='starting learning rate')
parser.add_argument('--max_grad_norm', default=3, type=float, help='gradient clipping parameter')
parser.add_argument('--max_length', default=30, type=float, help='max sentence length cutoff start')
parser.add_argument('--len_incr', default=1, type=int, help='increment max length each epoch')
parser.add_argument('--final_max_length', default=40, type=int, help='final max length cutoff')
parser.add_argument('--beta1', default=0.75, type=float, help='beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for adam')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=1213, type=int, help='random seed')
parser.add_argument('--print_every', type=int, default=1000, help='print stats after N batches')
# Regularization strength
parser.add_argument('--reg_h_alpha', default=0., type=float, help='entropy of PCFG')
parser.add_argument('--reg_d_alpha', default=0., type=float, help='prob. dist. of rules of diff. LHS')
parser.add_argument('--reg_c_alpha', default=0., type=float, help='prob. dist. of rules of the same LHS')
parser.add_argument('--num_features', default=0, type=int, help='prob. dist. of rules of the same LHS')
parser.add_argument('--band', default=0, type=int, help='prob. dist. of rules of the same LHS')
parser.add_argument('--accumulate', default=1, type=int, help='prob. dist. of rules of the same LHS')
parser.add_argument('--relu', default=False, type=bool, help='how to use z')
parser.add_argument('--unit_norm', default=False, type=bool, help='how to use z')
parser.add_argument('--temperature', default=1.0, type=float, help='prob. dist. of rules of the same LHS')

def main(args, print):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  train_data = Dataset(args.train_file)
  val_data = Dataset(args.val_file)  
  train_sents = train_data.batch_size.sum()
  vocab_size = int(train_data.vocab_size)    
  max_len = max(val_data.sents.size(1), train_data.sents.size(1))
  print('Train: %d sents / %d batches, Val: %d sents / %d batches' % 
        (train_data.sents.size(0), len(train_data), val_data.sents.size(0), len(val_data)))
  print('Vocab size: %d, Max Sent Len: %d' % (vocab_size, max_len))
  print('Save Path {}'.format(args.save_path))
  cuda.set_device(args.gpu)
  
  if args.model_type == '4th':
      model = CPCFG
  elif args.model_type == '5th':
      model = CPCFG2
  elif args.model_type == '6th':
      model = CPCFG3
  elif args.model_type == '7th':
      model = CPCFGFixProj
  elif args.model_type == '8th':
      model = CPCFGProj
  elif args.model_type == '9th':
      model = CPCFGProjDebug
  elif args.model_type == '10th':
      model = CPCFGProj2
  elif args.model_type == '11th':
      model = CPCFGProjb
  elif args.model_type == '12th':
      model = CPCFGProjDebugBand
  elif args.model_type == '13th':
      model = CPCFGProjbBand
  elif args.model_type == '14th':
      model = CPCFGProjbBandNoanti
  elif args.model_type == '15th':
      model = CPCFGProjbBandNonorm
  elif args.model_type == '16th':
      model = CPCFGProjbBandSoftmax
  elif args.model_type == '17th':
      model = CPCFGProjbBandSoftmaxNoshare
  elif args.model_type == '18th':
      model = CPCFGProjbBandSoftmaxNoshareNoanti
  elif args.model_type == '19th':
      model = CPCFGProjbBandSoftmaxallNoshareNoanti
  elif args.model_type == '20th':
      model = CPCFGProjbBandSoftmaxallNoshareNoantiMixture
  elif args.model_type == '21th':
      model = CPCFGProjbBandSoftmaxallNoshareNoantiOrig
  elif args.model_type == '29th':
      model = CPCFGProjbBandSoftmaxallNoshareNoantiThird
  elif args.model_type == '39th':
      model = CPCFGProjbBandSoftmaxallNoshareNoantiFourth
  else:
      raise NameError("Invalid parser type: {}".format(opt.parser_type)) 
  wandb.run.name = f'{args.model_type}:{args.num_features}:{args.lr}:{args.num_epochs}:{args.relu}:{args.unit_norm}:{args.temperature}'
  wandb.run.save()
  config = wandb.config
  config.lr = args.lr
  config.model_type = args.model_type
  config.num_features = args.num_features
  config.num_epochs = args.num_epochs
  config.relu = args.relu
  config.unit_norm = args.unit_norm
  config.temperature = args.temperature
  kwarg = {
      "share_term": getattr(args, "share_term", False),
      "share_rule": getattr(args, "share_rule", False),
      "share_root": getattr(args, "share_root", False),
      "wo_enc_emb": getattr(args, "wo_enc_emb", False),
  }
  epoch = 0
  if True:
    model = model(vocab_size, args.nt_states, args.t_states, 
                  h_dim = args.h_dim,
                  w_dim = args.w_dim,
                  z_dim = args.z_dim,
                  s_dim = args.state_dim, 
                  num_features=args.num_features, 
                  relu=args.relu,
                  unit_norm=args.unit_norm,
                  temperature=args.temperature,
                  **kwarg)
    model.band = args.band
    optimizer_state_dict = None
    if args.model_init:
        checkpoint = torch.load(args.model_init, map_location='cpu')
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        init_model = checkpoint["model"]
        optimizer_state_dict = checkpoint['optimizer']

        def perturb_params(old_model, new_model):
            old_state_dict = old_model.state_dict()
            new_state_dict = new_model.state_dict()
            for k, v1 in new_model.named_parameters():
                v2 = old_state_dict[k]
                v0 = v2.data * v1.data
                v1.data.copy_(v0) 
        if not args.perturb_it:
            print("override parser's params.")
            model.load_state_dict(init_model.state_dict())
        else:
            print("perturb parser's params.")
            perturb_params(init_model, model)
  else:
    assert False
    from torch_struct.networks import CompPCFG as model 
    model = model(vocab = vocab_size,
                  state_dim = args.state_dim,
                  t_states = args.t_states,
                  nt_states = args.nt_states,
                  h_dim = args.h_dim,
                  w_dim = args.w_dim,
                  z_dim = args.z_dim)
    for name, param in model.named_parameters():    
      if param.dim() > 1:
        xavier_uniform_(param)

  pcfg = None if args.infer_fast else PCFG(args.nt_states, args.t_states)

  fsave = args.save_path + "/{}.pt".format(0)
  print('Saving {} checkpoint to {}'.format(0, fsave))
  checkpoint = {
    'args': args.__dict__,
    'model': model.cpu(),
    'word2idx': train_data.word2idx,
    'idx2word': train_data.idx2word
  }
  torch.save(checkpoint, fsave)


  print("model architecture")
  print(model)
  model.train()
  model.cuda()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas = (args.beta1, args.beta2))
  if optimizer_state_dict is not None:
      print ('loading optmizer state dict')
      optimizer.load_state_dict(optimizer_state_dict)
  best_val_ppl = 1e5
  best_val_f1 = 0

  if args.data_random: # reset data randomness
    seed = 1847 if args.keep_random else args.seed
    print("random seed {} for data shuffling.".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)

  while epoch < args.num_epochs:
    start_time = time.time()
    epoch += 1  
    print('Starting epoch %d' % epoch)
    train_nll = 0.
    train_kl = 0.
    total_h, total_d, total_c = 0., 0., 0.
    num_sents = 0.
    num_words = 0.
    # ADDED
    ###num_sents_added = 0.
    ###topk = 50
    ###total_entropy = 0.
    ###total_entropy_gt = 0.
    ###total_entropy_baseline = 0.
    ###total_topk = torch.zeros(topk).cuda()
    ###total_topk_gt = torch.zeros(topk).cuda()
    ###total_topk_baseline = torch.zeros(topk).cuda()
    ###total_diff_kl = 0.
    ###total_diff_kl_rev = 0.
    ###total_diff_tv = 0.
    ###total_diff_log_space = 0.
    ###total_diff_kl_baseline = 0.
    ###total_diff_kl_rev_baseline = 0.
    ###total_diff_tv_baseline = 0.
    ###total_diff_log_space_baseline = 0.
    # ADDED
    all_stats = [[0., 0., 0.]]
    b = 0
    times = []
    for i in np.random.permutation(len(train_data)):
      #break
      b += 1
      #if b > 5:
      #  break
      #import pdb; pdb.set_trace()
      sents_full, length, batch_size_full, _, gold_spans_full, gold_binary_trees_full, _ = train_data[i]
      
      if length > args.max_length or length == 1: #length filter based on curriculum 
        continue

      accumulate = args.accumulate
      if length >= 25 and args.nt_states == 60:
        accumulate = max(accumulate, 2)

      batch_size_inner = max(1, batch_size_full // accumulate)
      num_inner_loop = int(math.ceil(batch_size_full / batch_size_inner))

      torch.cuda.synchronize()
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)

      start_event.record()

      optimizer.zero_grad()
      for inner_loop in range(num_inner_loop):
        start = inner_loop * batch_size_inner
        end = min(batch_size_full, (inner_loop + 1) * batch_size_inner)
        sents = sents_full[start:end]
        batch_size = sents.size(0)
        gold_spans = gold_spans_full[start:end]
        gold_binary_trees = gold_binary_trees_full[start:end]
        sents = sents.cuda()
        lengths = torch.tensor([length] * batch_size, device=sents.device).long() 
        params, kl = model(sents, lengths)
        # ADDED
        #import pdb; pdb.set_trace()
        if len(params) == 6:
          term_log, rule_probs_log, root_log, bc_emb_bias, bc_emb_features, nonterm_emb_features = params
        else:
          term_log, rule_probs_log, root_log = params
          bc_emb_bias, bc_emb_features, nonterm_emb_features = None, None, None
        if isinstance(rule_probs_log, tuple):
          rule_probs_log, rule_probs_gt_log = rule_probs_log
          params = (term_log, rule_probs_log, root_log)
        else:
          rule_probs_gt_log = None


        #if bc_emb_bias is not None:
          #params = params + (bc_emb_bias, bc_emb_features, nonterm_emb_features)
        if args.infer_fast: 

          dist = SentCFG(params, lengths=lengths)
          if not args.no_argmax:
            spans = dist.argmax[-1]
            argmax_spans, tree = extract_parses(spans, lengths.tolist(), inc=0)
          #import pdb; pdb.set_trace()
          nll = -dist.partition
        else:
          assert False
          log_Z = pcfg._inside(*params)
          nll = -log_Z
          with torch.no_grad():
            max_score, binary_matrix, argmax_spans = pcfg._viterbi(*params)

        rule_H, rule_D, rule_C = regularize(params, sents.device, args)
        reg = (rule_H + rule_D - rule_C).mean()
        total_h += rule_H.sum().item()
        total_d += rule_D.sum().item()
        total_c += rule_C.sum().item()

        kl = torch.zeros_like(nll) if kl is None else kl
        #import pdb; pdb.set_trace()
        ((nll + kl + reg).sum()/batch_size_full).backward()
        ###with torch.no_grad():
          #ADD2
          ###rule_probs_log = rule_probs_log.view(-1, model.NT, model.NT_T**2)
          ###rule_probs = rule_probs_log.exp()
          ###entropy = - (rule_probs_log * rule_probs).sum().item()
          ###total_entropy += entropy
          ###rule_probs_baseline = rule_probs.new(rule_probs.shape).fill_(1)
          ###rule_probs_baseline = rule_probs_baseline / rule_probs_baseline.sum(-1, keepdim=True)
          ###rule_probs_baseline_log = rule_probs_baseline.log()
          ###entropy_baseline = - (rule_probs_baseline_log * rule_probs_baseline).sum().item()
          ###total_entropy_baseline += entropy_baseline
          ###rule_probs_topk, _ = torch.topk(rule_probs, topk, -1) # b, NT, topk
          ###total_topk += rule_probs_topk.sum(0).sum(0)
          ###rule_probs_topk_baseline, _ = torch.topk(rule_probs_baseline, topk, -1) # b, NT, topk
          ###total_topk_baseline += rule_probs_topk_baseline.sum(0).sum(0)
          ###if rule_probs_gt_log is not None:
          ###  rule_probs_gt = rule_probs_gt_log.exp()
          ###  rule_probs_topk_gt, _ = torch.topk(rule_probs_gt, topk, -1) # b, NT, topk
          ###  total_topk_gt += rule_probs_topk_gt.sum(0).sum(0)
          ###  entropy = - (rule_probs_gt_log * rule_probs_gt).sum().item()
          ###  total_entropy_gt += entropy
          ###  def compute_kl(attn_q, attn_p):
          ###      """computes KL(q||p)"""
          ###      return (-attn_q * attn_p.log() + attn_q * attn_q.log()).sum(-1)
          ###  total_diff_kl += compute_kl(rule_probs, rule_probs_gt).sum().item()
          ###  total_diff_kl_rev += compute_kl(rule_probs_gt, rule_probs).sum().item()
          ###  total_diff_tv += torch.abs(rule_probs_gt - rule_probs).sum().item()
          ###  total_diff_log_space = torch.abs(rule_probs_gt_log - rule_probs_log).sum().item()

          ###  total_diff_kl_baseline += compute_kl(rule_probs_baseline, rule_probs_gt).sum().item()
          ###  total_diff_kl_rev_baseline += compute_kl(rule_probs_gt, rule_probs_baseline).sum().item()
          ###  total_diff_tv_baseline += torch.abs(rule_probs_gt - rule_probs_baseline).sum().item()
          ###  total_diff_log_space_baseline = torch.abs(rule_probs_gt_log - rule_probs_baseline_log).sum().item()
          # ADDED
        train_nll += nll.sum().item()
        train_kl += kl.sum().item()
        num_sents += batch_size
        #num_sents_added += batch_size
        num_words += batch_size * (length + 1) # we implicitly generate </s> so we explicitly count it
        if not args.no_argmax:
          for bb in range(batch_size):
            span_b = [(a[0], a[1]) for a in argmax_spans[bb] if a[0] != a[1]] #ignore labels
            span_b_set = set(span_b[:-1])
            update_stats(span_b_set, [set(gold_spans[bb][:-1])], all_stats)
      torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)      
      optimizer.step()

      end_event.record()
      torch.cuda.synchronize()
      times.append(start_event.elapsed_time(end_event))

      if len(times) > 1000:
          print(f"{len(times)}: {np.mean(times):.5f}")
          print("exiting after 1k iterations")
          sys.exit()


      if args.unit_norm:
          model._normalize()
      #import pdb; pdb.set_trace()
      if b % args.print_every == 0:
        all_f1 = get_f1(all_stats)
        param_norm = sum([p.norm()**2 for p in model.parameters()]).item()**0.5
        gparam_norm = sum([p.grad.norm()**2 for p in model.parameters() 
                           if p.grad is not None]).item()**0.5
        log_str = 'Epoch: %d, Batch: %d/%d, |Param|: %.6f, |GParam|: %.2f,  LR: %.4f, ' + \
                  'H: %.3f, D: %.3f, C: %.3f, ' + \
                  'ReconPPL: %.2f, KL: %.4f, PPLBound: %.2f, ValPPL: %.2f, ValF1: %.2f, ' + \
                  'CorpusF1: %.2f, Throughput: %.2f examples/sec'
        print(log_str %
              (epoch, b, len(train_data), param_norm, gparam_norm, args.lr, 
               total_h / num_sents, total_d / num_sents, total_c / num_sents,
               np.exp(train_nll / num_words), train_kl /num_sents, 
               np.exp((train_nll + train_kl)/num_words), best_val_ppl, best_val_f1, 
               all_f1[0], num_sents / (time.time() - start_time)))
       # import pdb; pdb.set_trace()
        d = {'corpus F1': all_f1[0], 'val f1': best_val_f1, 'val ppl': best_val_ppl, 'reconPPL': np.exp(train_nll / num_words), 'kl': train_kl /num_sents, 'ppl bound': np.exp((train_nll + train_kl)/num_words)}
        ###d['total_entropy'] = total_entropy/num_sents_added/model.NT
        ###print (f'total_entropy = {total_entropy/num_sents_added/model.NT}')
        ###if rule_probs_gt_log is not None:
        ###  print (f'total_entropy_gt = {total_entropy_gt/num_sents_added/model.NT}')
        ###  d['total_entropy_gt'] = total_entropy_gt/num_sents_added/model.NT
        ###print (f'total_entropy_baseline = {total_entropy_baseline/num_sents_added/model.NT}')
        ###d['total_entropy_baseline'] = total_entropy_baseline/num_sents_added/model.NT
        ###print (f'total_topk = {(total_topk/num_sents_added/model.NT).cpu().tolist()}')
        ###d['total_topk'] = (total_topk/num_sents_added/model.NT).cpu().tolist()
        ###if rule_probs_gt_log is not None:
        ###  print (f'total_topk_gt = {(total_topk_gt/num_sents_added/model.NT).cpu().tolist()}')
        ###  d['total_topk_gt'] = (total_topk_gt/num_sents_added/model.NT).cpu().tolist()
        ###print (f'total_topk_baseline = {(total_topk_baseline/num_sents_added/model.NT).cpu().tolist()}')
        ###d['total_topk_baseline'] = (total_topk_baseline/num_sents_added/model.NT).cpu().tolist()
        ###if rule_probs_gt_log is not None:
        ###  print (f'total_diff_kl = {total_diff_kl/num_sents_added/model.NT}')
        ###  d['total_diff_kl'] = total_diff_kl/num_sents_added/model.NT
        ###  print (f'total_diff_kl_rev = {total_diff_kl_rev/num_sents_added/model.NT}')
        ###  d['total_diff_kl_rev'] = total_diff_kl_rev/num_sents_added/model.NT
        ###  print (f'total_diff_tv = {total_diff_tv/num_sents_added/model.NT}')
        ###  d['total_diff_tv'] = total_diff_tv/num_sents_added/model.NT
        ###  print (f'total_diff_log_space = {total_diff_log_space/num_sents_added/model.NT}')
        ###  d['total_diff_log_space'] = total_diff_log_space/num_sents_added/model.NT
        ###  print (f'total_diff_kl_baseline = {total_diff_kl_baseline/num_sents_added/model.NT}')
        ###  d['total_diff_kl_baseline'] = total_diff_kl_baseline/num_sents_added/model.NT
        ###  print (f'total_diff_kl_rev_baseline = {total_diff_kl_rev_baseline/num_sents_added/model.NT}')
        ###  d['total_diff_kl_rev_baseline'] = total_diff_kl_rev_baseline/num_sents_added/model.NT
        ###  print (f'total_diff_tv_baseline = {total_diff_tv_baseline/num_sents_added/model.NT}')
        ###  d['total_diff_tv_baseline'] = total_diff_tv_baseline/num_sents_added/model.NT
        ###  print (f'total_diff_log_space_baseline = {total_diff_log_space_baseline/num_sents_added/model.NT}')
        ###  d['total_diff_log_space_baseline'] = total_diff_log_space_baseline/num_sents_added/model.NT
        wandb.log(d)
        # ADDED
        ###num_sents_added = 0.
        ###topk = 50
        ###total_entropy = 0.
        ###total_entropy_gt = 0.
        ###total_entropy_baseline = 0.
        ###total_topk = torch.zeros(topk).cuda()
        ###total_topk_gt = torch.zeros(topk).cuda()
        ###total_topk_baseline = torch.zeros(topk).cuda()
        ###total_diff_kl = 0.
        ###total_diff_kl_rev = 0.
        ###total_diff_tv = 0.
        ###total_diff_log_space = 0.
        ###total_diff_kl_baseline = 0.
        ###total_diff_kl_rev_baseline = 0.
        ###total_diff_tv_baseline = 0.
        ###total_diff_log_space_baseline = 0.
        # print an example parse
        if not args.infer_fast: 
          tree = get_tree_from_binary_matrix(binary_matrix[0], length)
          action = get_actions(tree)
        else:
          if not args.no_argmax:
            action = get_actions(tree[0])
        sent_str = [train_data.idx2word[word_idx] for word_idx in list(sents[0].cpu().numpy())]
        if not args.no_argmax:
          print("Pred Tree: %s" % get_tree(action, sent_str))
        print("Gold Tree: %s" % get_tree(gold_binary_trees[0], sent_str))
    
    args.max_length = min(args.final_max_length, args.max_length + args.len_incr)
    print('--------------------------------')
    print('Checking validation perf...')    
    #import pdb; pdb.set_trace()
    val_ppl, val_f1 = eval(val_data, model, print, pcfg)
    print('--------------------------------')

    fsave = args.save_path + "/{}.pt".format(epoch)
    print('Saving {} checkpoint to {}'.format(epoch, fsave))
    checkpoint = {
      'args': args.__dict__,
      'model': model.cpu(),
      'word2idx': train_data.word2idx,
      'idx2word': train_data.idx2word,
      'epoch': epoch,
      'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, fsave)
    model.cuda()
    rules_ll = model.rules_ll
    torch.save(rules_ll, f'{args.save_path}/rules{epoch}.pt')

    if val_ppl < best_val_ppl:
      best_val_ppl = val_ppl
      best_val_f1 = val_f1
      checkpoint = {
        'args': args.__dict__,
        'model': model.cpu(),
        'word2idx': train_data.word2idx,
        'idx2word': train_data.idx2word,
        'epoch': epoch,
        'optimizer': optimizer.state_dict()
      }
      print('Saving the best checkpoint to %s' % args.save_path + "/best.pt")
      torch.save(checkpoint, args.save_path + "/best.pt")
      model.cuda()

def eval(data, model, print, pcfg):
  model.eval()
  num_sents = 0
  num_words = 0
  total_nll = 0.
  total_kl = 0.
  total_h, total_d, total_c = 0., 0., 0.
  corpus_f1 = [0., 0., 0.] 
  sent_f1 = [] 
  with torch.no_grad():
    for i in range(len(data)):
      #if i > 5:
      #  break
      sents_full, length, batch_size_full, _, gold_spans_full, gold_binary_trees_full, _ = data[i]
      if length == 1:
        continue
      #if length < 30:
      #   print ('skipped')
      #   continue
      print (length)
      if length > 50:
        print (f'skipped {length}')
        continue
      #print ('b') 
      #print (batch_size_full)
      accumulate = args.accumulate
      if length >= 30 and args.nt_states >= 100:
          accumulate = max(accumulate, 4)
          print ('genbmm')
          model.genbmm=True
      else:
          model.genbmm=False
      if length >= 30 and args.nt_states == 60:
          accumulate = max(accumulate, 2)
      if length >= 40 and args.nt_states == 60:
          accumulate = max(accumulate, 4)
      batch_size_inner = max(1, batch_size_full // accumulate)
      num_inner_loop = int(math.ceil(batch_size_full / batch_size_inner))

      for inner_loop in range(num_inner_loop):
        start = inner_loop * batch_size_inner
        end = min(batch_size_full, (inner_loop + 1) * batch_size_inner)
        sents = sents_full[start:end]
        batch_size = sents.size(0)
        gold_spans = gold_spans_full[start:end]
        gold_binary_trees = gold_binary_trees_full[start:end]
        sents = sents.cuda()
        # note that for unsuperised parsing, we should do model(sents, argmax=True, use_mean = True)
        # but we don't for eval since we want a valid upper bound on PPL for early stopping
        # see eval.py for proper MAP inference
        lengths = torch.tensor([length] * batch_size, device=sents.device).long() 
        params, kl = model(sents, lengths)
        if len(params) == 6:
          term_log, rule_probs_log, root_log, bc_emb_bias, bc_emb_features, nonterm_emb_features = params
        else:
          term_log, rule_probs_log, root_log = params
          bc_emb_bias, bc_emb_features, nonterm_emb_features = None, None, None
        if isinstance(rule_probs_log, tuple):
            rule_probs_log, rule_probs_gt_log = rule_probs_log
            params = (term_log, rule_probs_log, root_log)
        if bc_emb_bias is not None:
          params = params + (bc_emb_bias, bc_emb_features, nonterm_emb_features)
        if args.infer_fast: 
          dist = SentCFG(params, lengths=lengths)
          if not args.no_argmax:
            spans = dist.argmax[-1]
            argmax_spans, tree = extract_parses(spans, lengths.tolist(), inc=0)
          nll = -dist.partition
        else:
          assert False
          log_Z = pcfg._inside(*params)
          nll = -log_Z
          with torch.no_grad():
            max_score, binary_matrix, argmax_spans = pcfg._viterbi(*params)

        kl = torch.zeros_like(nll) if kl is None else kl
        total_nll += nll.sum().item()
        total_kl  += kl.sum().item()

        rule_H, rule_D, rule_C = regularize(params, sents.device, args)
        total_h += rule_H.sum().item()
        total_d += rule_D.sum().item()
        total_c += rule_C.sum().item()

        num_sents += batch_size
        num_words += batch_size*(length +1) # we implicitly generate </s> so we explicitly count it
        if not args.no_argmax:
          for b in range(batch_size):
            span_b = [(a[0], a[1]) for a in argmax_spans[b] if a[0] != a[1]] #ignore labels
            span_b_set = set(span_b[:-1])        
            gold_b_set = set(gold_spans[b][:-1])
            tp, fp, fn = get_stats(span_b_set, gold_b_set) 
            corpus_f1[0] += tp
            corpus_f1[1] += fp
            corpus_f1[2] += fn
            # sent-level F1 is based on L83-89 from https://github.com/yikangshen/PRPN/test_phrase_grammar.py

            model_out = span_b_set
            std_out = gold_b_set
            overlap = model_out.intersection(std_out)
            prec = float(len(overlap)) / (len(model_out) + 1e-8)
            reca = float(len(overlap)) / (len(std_out) + 1e-8)
            if len(std_out) == 0:
              reca = 1. 
              if len(model_out) == 0:
                prec = 1.
            f1 = 2 * prec * reca / (prec + reca + 1e-8)
            sent_f1.append(f1)
          #break
  if not args.no_argmax:
    tp, fp, fn = corpus_f1  
    prec = tp / max(tp + fp, 1e-6)
    recall = tp / max(tp + fn, 1e-6)
    #prec = (tp + 1e-6) / (tp + fp + 2e-6)
    #recall = (tp + 1e-6) / (tp + fn + 2e-6)
    corpus_f1 = 2*prec*recall/(prec+recall) if prec+recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1))
  else:
    sent_f1 = 0
  recon_ppl = np.exp(total_nll / num_words)
  ppl_elbo = np.exp((total_nll + total_kl)/num_words) 
  kl = total_kl /num_sents

  h, d, c = total_h / num_sents, total_d / num_sents, total_c / num_sents

  print('ReconPPL: %.2f, KL: %.4f, PPL (Upper Bound): %.2f, H: %.3f, D: %.3f, C: %.3f' %
        (recon_ppl, kl, ppl_elbo, h, d, c))
  if not args.no_argmax:
    print('Corpus F1: %.2f, Sentence F1: %.2f' %
          (corpus_f1*100, sent_f1*100))
    print (f'==corpus precision: {prec}, corpus recall: {recall}')
  model.train()
  model.genbmm=False
  return ppl_elbo, sent_f1*100

if __name__ == '__main__':
  args = parser.parse_args()
  if os.path.exists(args.save_path):
      print(f'Warning: the folder {args.save_path} exists.')
  else:
      print('Creating {}'.format(args.save_path))
      os.mkdir(args.save_path)
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  handler = logging.FileHandler(os.path.join(args.save_path, 'train.log'), 'w')
  handler.setLevel(logging.INFO)
  handler.setFormatter(formatter)
  logger.addHandler(handler)
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  console.setFormatter(formatter)
  logger.addHandler(console)
  logger.propagate = False
  logger.info('cuda:{}@{}'.format(args.gpu, os.uname().nodename))
  logger.info(args)
  main(args, logger.info)
