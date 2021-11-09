
import time as timep

import sys

import math
import time

from pathlib import Path

import numpy as np

import torch as th
from torch.nn.utils.clip_grad import clip_grad_norm_

from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

import torchtext
from datasets.lm import PennTreebank
from datasets.data import BucketIterator, BPTTIterator

from args import get_args

from utils import set_seed, get_config, get_name, get_mask_lengths
from utils import Pack
from utils import plot_counts, print_gpu_mem

import wandb


valid_schedules = ["reducelronplateau"]

WANDB_STEP = -1

BEST_VALID = -math.inf
PREV_SAVE = None

def max_diff(log_p):
    P = log_p.exp()
    n = P.shape[0]
    xy = [(x,y) for x in range(n) for y in range(x, n)]
    diff = th.tensor([(P[x] - P[y]).abs().max() for x,y in xy])
    print(f"Max diff < 0.01: {(diff < 0.01).sum()} / {diff.shape[0]}")

def update_best_valid(
    valid_losses, valid_n, model, optimizer, scheduler, name,
):
    global WANDB_STEP
    global BEST_VALID
    global PREV_SAVE
    if valid_losses.evidence > BEST_VALID:
        # do not save on dryruns
        if not wandb.run._settings._offline:
            save_f = f"wandb_checkpoints/{name}/{WANDB_STEP}_{-valid_losses.evidence / valid_n:.2f}.pth"
            print(f"Saving model to {save_f}")
            Path(save_f).parent.mkdir(parents=True, exist_ok=True)
            th.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": model.config,
            }, save_f)
            if PREV_SAVE is not None:
                Path(PREV_SAVE).unlink()
            PREV_SAVE = save_f

        BEST_VALID = valid_losses.evidence
        wandb.run.summary["best_valid_ppl"] = math.exp(-BEST_VALID / valid_n)
        wandb.run.summary["best_valid_loss"] = BEST_VALID / valid_n


def report(losses, n, prefix, start_time=None):
    loss = losses.evidence
    elbo = losses.elbo
    # cap loss otherwise overflow
    #loss = loss if loss > -1e7 else -1e7
    str_list = [
        f"{prefix}: log_prob = {loss:.2f}",
        f"xent(word) = {-loss / n:.2f}",
        f"ppl = {math.exp(-loss / n):.2f}",
    ]
    if elbo is not None:
        str_list.append(f"elbo = {elbo / n:.2f}")
    total_time = None
    if start_time is not None:
        total_time = time.time() - start_time
        str_list.append(f"total_time = {total_time:.2f}s")
    print(" | ".join(str_list))
    return total_time

def count_params(model):
    return (
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

def eval_loop(
    args, V, iter, model,
):
    total_ll = 0
    total_elbo = 0
    n = 0
    lpz, last_states = None, None
    with th.no_grad():
        for i, batch in enumerate(iter):
            model.train(False)
            if hasattr(model, "noise_scale"):
                model.noise_scale = 0
            mask, lengths, n_tokens = get_mask_lengths(batch.text, V)
            if args.iterator != "bptt":
                lpz, last_states = None, None
            losses, lpz, _ = model.score(
                batch.text,
                lpz=lpz, last_states = last_states,
                mask=mask, lengths=lengths,
            )
            total_ll += losses.evidence.detach()
            if losses.elbo is not None:
                total_elbo += losses.elbo.detach()
            n += n_tokens
    return Pack(evidence = total_ll, elbo = total_elbo), n



def fast_eval_loop(
    args, V, iter, model,
):
    total_ll = 0
    total_elbo = 0
    n = 0

    with th.no_grad():
        model.train(False)
        lpz = None
        start, transition, emission = model.compute_parameters(model.word2state)

        if hasattr(model, "temp"):
            print(f"Model temp {model.temp.item():.2f}")

        # entropy
        # assert that transition and emission are well-formed
        myt = model.transition(print_max=True)
        #myt = model.transition()
        bigt = myt.logsumexp(-1).abs().max()
        assert bigt < 1e-4, f"{bigt}"
        bige = emission.logsumexp(-1).abs().max()
        assert bige < 1e-4, f"{bige}"
        # log entropy of transition and emission

        He = -(emission.exp() * emission).sum()
        Ht = -(myt.exp() * myt).sum()
        print(f"Total transition entropy {Ht:.2f} || Total emission entropy {He.sum():.2f}")



        word2state = model.word2state
        for i, batch in enumerate(iter):
            if hasattr(model, "noise_scale"):
                model.noise_scale = 0

            text = batch.text

            mask, lengths, n_tokens = get_mask_lengths(text, V)
            N, T = text.shape

            if lpz is not None and args.iterator == "bptt":
                #start = (lpz[:,:,None] + transition[last_states,:]).logsumexp(1)
                raise NotImplementedError()

            losses, lpz = model.compute_loss(
                text, start, transition, emission, word2state,
                mask=mask, lengths=lengths)

            if word2state is not None:
                idx = th.arange(N, device=model.device)
                last_words = text[idx, lengths-1]
                last_states = model.word2state[last_words]

            total_ll += losses.evidence.detach()
            if losses.elbo is not None:
                total_elbo += losses.elbo.detach()
            n += n_tokens

    return Pack(evidence = total_ll, elbo = total_elbo), n

def collect_counts_loop(
    args, V, iter, model,
):
    from models.analysis_utils import CountCollector
    count_collector = CountCollector(model)

    model.train(False)
    lpz = None

    word2state = model.word2state
    for i, batch in enumerate(iter):
        if hasattr(model, "noise_scale"):
            model.noise_scale = 0

        text = batch.text

        mask, lengths, n_tokens = get_mask_lengths(text, V)
        N, T = text.shape

        if lpz is not None and args.iterator == "bptt":
            #start = (lpz[:,:,None] + transition[last_states,:]).logsumexp(1)
            raise NotImplementedError()

        if word2state is not None:
            idx = th.arange(N, device=model.device)
            last_words = text[idx, lengths-1]
            last_states = model.word2state[last_words]

        count_collector.collect_counts(text, mask, lengths)
    return count_collector

def train_loop(
    args, V, iter, model,
    parameters, optimizer, scheduler,
    valid_iter=None,
    verbose=False,
):
    global WANDB_STEP

    total_ll = 0
    total_elbo = 0
    n = 0
    # check is performed at end of epoch outside loop as well
    checkpoint = len(iter) // (args.num_checks - 1)
    with th.enable_grad():
        lpz = None
        last_states = None
        for i, batch in enumerate(iter):
            model.train(True)
            WANDB_STEP += 1
            optimizer.zero_grad()

            text = batch.text
            if args.iterator == "bucket":
                lpz = None
                last_states = None

            mask, lengths, n_tokens = get_mask_lengths(text, V)
            if model.timing:
                start_forward = timep.time()


            if hasattr(args, "eff") and args.eff:
                losses, _, _= model.score_rff(
                    text, lpz=lpz, last_states=last_states, mask=mask, lengths=lengths)
            else:
                losses, lpz, last_states = model.score(
                    text, lpz=lpz, last_states=last_states, mask=mask, lengths=lengths)

            if model.timing:
                print(f"forward time: {timep.time() - start_forward}")
            total_ll += losses.evidence.detach()
            if losses.elbo is not None:
                total_elbo += losses.elbo.detach()
            n += n_tokens

            loss = -losses.loss / n_tokens
            if model.timing:
                start_backward = timep.time()
            loss.backward()

            if model.timing:
                print(f"backward time: {timep.time() - start_backward}")
            gradnorm = clip_grad_norm_(parameters, args.clip)
            if args.schedule not in valid_schedules:
                # sched before opt since we want step = 1?
                # this is how huggingface does it
                scheduler.step()
            optimizer.step()
            wandb.log({
                "running_training_loss": total_ll / n,
                "running_training_ppl": math.exp(min(-total_ll / n, 700)),
                "running_training_elbo": total_elbo / n,
                "gradnorm": gradnorm,
            }, step=WANDB_STEP)
            if model.timing:
                print_gpu_mem()

            if model.timing:
                print(f"gradnorm {i}: {gradnorm} || sur {loss} || ev {losses.evidence}")
                import pdb; pdb.set_trace()

            if verbose and i % args.report_every == args.report_every - 1:
                report(
                    Pack(evidence = total_ll, elbo = total_elbo),
                    n,
                    f"Train batch {i}",
                )

            if valid_iter is not None and i % checkpoint == checkpoint-1:
                v_start_time = time.time()
                eval_fn = fast_eval_loop
                valid_losses, valid_n  = eval_fn(
                    args, V, valid_iter, model,
                )
                report(valid_losses, valid_n, "Valid eval", v_start_time)
                wandb.log({
                    "valid_loss": valid_losses.evidence / valid_n,
                    "valid_ppl": math.exp(-valid_losses.evidence / valid_n),
                }, step=WANDB_STEP)

                update_best_valid(
                    valid_losses, valid_n, model, optimizer, scheduler, args.name)

                wandb.log({
                    "lr": optimizer.param_groups[0]["lr"],
                }, step=WANDB_STEP)
                scheduler.step(valid_losses.evidence)


    return Pack(evidence = total_ll, elbo = total_elbo), n


def main():
    global WANDB_STEP
    args = get_args()

    set_seed(args.seed)

    device = th.device("cpu" if args.devid < 0 else f"cuda:{args.devid}")
    args.device = device
    aux_device = th.device("cpu" if args.aux_devid < 0 else f"cuda:{args.aux_devid}")
    args.aux_device = aux_device

    TEXT = torchtext.data.Field(batch_first = True)

    Dataset = PennTreebank

    train, valid, test = Dataset.splits(
        TEXT,
        newline_eos = True,
    )

    TEXT.build_vocab(train)
    V = TEXT.vocab

    def batch_size_tokens(new, count, sofar):
        return max(len(new.text), sofar)
    def batch_size_sents(new, count, sofar):
        return count

    if args.iterator == "bucket":
        train_iter, valid_iter, test_iter = BucketIterator.splits(
            (train, valid, test),
            batch_sizes = [args.bsz, args.eval_bsz, args.eval_bsz],
            device = device,
            sort_key = lambda x: len(x.text),
            batch_size_fn = batch_size_tokens if args.bsz_fn == "tokens" else batch_size_sents,
        )
    elif args.iterator == "bptt":
        train_iter, valid_iter, test_iter = BPTTIterator.splits(
            (train, valid, test),
            batch_sizes = [args.bsz, args.eval_bsz, args.eval_bsz],
            device = device,
            bptt_len = args.bptt,
            sort = False,
        )
    else:
        raise ValueError(f"Invalid iterator {args.iterator}")

    if args.no_shuffle_train:
        train_iter.shuffle = False

    name = get_name(args)
    import tempfile
    wandb.init(project="linear-hmm", name=name, config=args, dir=tempfile.mkdtemp())
    args.name = name

    print(" ".join(sys.argv))
    print(args)

    model = None
    if args.model == "blhmm":
        from models.blhmmlm import BLHmmLm
        model = BLHmmLm(V, args)
    elif args.model == "bandedhmm":
        from models.banded_hmmlm import BandedHmmLm
        model = BandedHmmLm(V, args)
    elif args.model == "sblhmm":
        from models.sblhmmlm import SblHmmLm
        model = SblHmmLm(V, args)
    else:
        raise ValueError("Invalid model type")
    model.to(device)
    print(model)
    num_params, num_trainable_params = count_params(model)
    print(f"Num params, trainable: {num_params:,}, {num_trainable_params:,}")
    wandb.run.summary["num_params"] = num_params

    # load frozen transition
    if args.frozen_pretrained_transition is not None:
        past_chp = th.load(args.frozen_pretrained_transition)
        from models.sblhmmlm import SblHmmLm
        past_model = SblHmmLm(V, past_chp["args"])
        past_model.load_state_dict(past_chp["model"])
        transition = past_model.transition().detach().clone()
        model.set_frozen_transition(transition)
        # cleanup
        del past_model
        del past_chp

    if args.eval_only:
        # make sure this is uncommented
        if args.eval_only != "none":
            model.load_state_dict(th.load(args.eval_only)["model"])
        from utils import dump_transition, dump_svd
        dump_transition(model)
        dump_svd(model)

        v_start_time = time.time()
        eval_fn = fast_eval_loop
        valid_losses, valid_n = eval_fn(
            args, V, valid_iter, model,
        )
        report(valid_losses, valid_n, f"Valid perf", v_start_time)

        # count states
        valid_counts = collect_counts_loop(
            args, V, valid_iter, model,
        )
        valid_counts.print_counts()

        t_start_time = time.time()
        test_losses, test_n = eval_fn(
            args, V, test_iter, model,
        )
        report(test_losses, test_n, f"Test perf", t_start_time)

        sys.exit()
        # EXIT FROM EVAL_ONLY

    parameters = list(model.parameters())
    if args.optimizer == "adamw":
        optimizer = AdamW(
            parameters,
            lr = args.lr,
            betas = (args.beta1, args.beta2),
            weight_decay = args.wd,
        )
    elif args.optimizer == "sgd":
        optimizer = SGD(
            parameters,
            lr = args.lr,
        )
    if args.schedule == "reducelronplateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor = 1. / args.decay,
            patience = args.patience,
            verbose = True,
            mode = "max",
        )
    elif args.schedule == "noam":
        warmup_steps = args.warmup_steps
        def get_lr(step):
            scale = warmup_steps ** 0.5 * min(step ** (-0.5), step * warmup_steps ** (-1.5))
            return args.lr * scale
        scheduler = LambdaLR(
            optimizer,
            get_lr,
            last_epoch=-1,
            verbse = True,
        )
    else:
        raise ValueError("Invalid schedule options")

    # training loop, factor out later if necessary
    for e in range(args.num_epochs):
        start_time = time.time()
        if args.log_counts > 0 and args.keep_counts > 0:
            # reset at START of epoch
            model.state_counts.fill_(0)
        train_losses, train_n = train_loop(
            args, V, train_iter, model,
            parameters, optimizer, scheduler,
            valid_iter = valid_iter if not args.overfit else None,
            verbose = True,
        )
        total_time = report(train_losses, train_n, f"Train epoch {e}", start_time)

        v_start_time = time.time()
        eval_fn = fast_eval_loop
        valid_losses, valid_n  = eval_fn(args, V, valid_iter, model)
        report(valid_losses, valid_n, f"Valid epoch {e}", v_start_time)

        if args.schedule in valid_schedules:
            scheduler.step(
                valid_losses.evidence if not args.overfit else train_losses.evidence)

        update_best_valid(
            valid_losses, valid_n, model, optimizer, scheduler, args.name)

        wandb.log({
            "train_loss": train_losses.evidence / train_n,
            "train_ppl": math.exp(-train_losses.evidence / train_n),
            "epoch_time": total_time,
            "valid_loss": valid_losses.evidence / valid_n,
            "valid_ppl": math.exp(-valid_losses.evidence / valid_n),
            "best_valid_loss": BEST_VALID / valid_n,
            "best_valid_ppl": math.exp(-BEST_VALID / valid_n),
            "epoch": e,
        }, step=WANDB_STEP)

    # won't use best model. Rerun with eval_only
    t_start_time = time.time()
    test_losses, test_n = eval_fn(
        args, V, test_iter, model,
    )
    report(test_losses, test_n, f"Test perf", t_start_time)

if __name__ == "__main__":
    print(" ".join(sys.argv))
    main()
