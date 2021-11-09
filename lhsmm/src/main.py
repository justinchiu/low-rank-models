import os
import argparse
import pickle
import pprint
import sys
from collections import OrderedDict
import json
import torch

import numpy as np

from models.model import make_data_loader
from data.breakfast import BreakfastCorpus
from data.corpus import Datasplit
from data.crosstask import CrosstaskCorpus
from models.framewise import FramewiseGaussianMixture, FramewiseDiscriminative, FramewiseBaseline
from models.sequential import SequentialDiscriminative, SequentialCanonicalBaseline, SequentialPredictConstraints, SequentialGroundTruth
from models.model import Model, add_training_args
from models.semimarkov.semimarkov import SemiMarkovModel
from utils.logger import logger

STAT_KEYS = [
    'mof', 'mof_non_bg', 'step_recall_non_bg', 'mean_normed_levenshtein',
    'center_step_recall_non_bg', 'f1', 'f1_non_bg', 'pred_background', 'iou_multi_non_bg',
    'predicted_label_types_per_video', 'predicted_label_types_non_bg_per_video',
    'predicted_segments_per_video', 'predicted_segments_non_bg_per_video',
    'multiple_gt_labels',
]
DISPLAY_STAT_KEYS = [
    'f1', 'f1_non_bg', 'center_step_recall_non_bg', 'mean_normed_levenshtein',
    'pred_background', 'iou_multi_non_bg',
    'predicted_label_types_per_video', 'predicted_label_types_non_bg_per_video',
    'predicted_segments_per_video', 'predicted_segments_non_bg_per_video',
    'mof', 'mof_non_bg',
    'multiple_gt_labels',
]

CLASSIFIERS = {
    'framewise_discriminative': FramewiseDiscriminative,
    'framewise_gaussian_mixture': FramewiseGaussianMixture,
    'framewise_baseline': FramewiseBaseline,
    'semimarkov': SemiMarkovModel,
    'sequential_discriminative': SequentialDiscriminative,
    'sequential_canonical_baseline': SequentialCanonicalBaseline,
    'sequential_predict_constraints': SequentialPredictConstraints,
    'sequential_ground_truth': SequentialGroundTruth,
}


def add_serialization_args(parser):
    group = parser.add_argument_group('serialization')
    group.add_argument('--model_output_path')
    group.add_argument('--model_input_path')
    group.add_argument('--prediction_output_path')


def add_misc_args(parser):
    group = parser.add_argument_group('miscellaneous')
    group.add_argument('--compare_to_prediction_folder', help='root folder containing *_pred.npy and *_true.npy prediction files (for comparison)')
    group.add_argument('--compare_only',
                       action='store_true',
                       help="skip everything to do with models and just evaluate these serialized predictions")
    group.add_argument('--compare_load_splits_from_predictions', action='store_true')


def add_data_args(parser):
    group = parser.add_argument_group('data')
    group.add_argument('--dataset', choices=['crosstask', 'breakfast'], default='crosstask')
    group.add_argument('--features', choices=['raw', 'pca'], default='pca')
    group.add_argument('--feature_downscale', type=float, default=1.0)
    group.add_argument('--feature_permutation_seed', type=int)
    group.add_argument('--batch_size', type=int, default=5)
    group.add_argument('--remove_background', action='store_true')
    group.add_argument('--pca_components_per_group', type=int, default=100)
    group.add_argument('--pca_no_background', action='store_true')

    group.add_argument('--mix_tasks', action='store_true', help='train on all tasks simultaneously')

    group.add_argument('--frame_subsample', type=int, default=1, help="interval to subsample frames at (e.g. 10 takes every 10th frame)")

    group.add_argument('--task_specific_steps', action='store_true', help="")
    group.add_argument('--annotate_background_with_previous', action='store_true', help="")

    group.add_argument('--no_merge_classes', action='store_true', help="")

    group.add_argument('--force_optimal_assignment', action='store_true', help="force optimal assignment to maximize MoF")

    group.add_argument('--no_cache_features', action='store_true', help="")

    group.add_argument('--crosstask_feature_groups',
                       choices=['i3d', 'resnet', 'audio', 'narration'],
                       nargs='+', default=['i3d', 'resnet', 'audio'])
    group.add_argument('--crosstask_training_data', choices=['primary', 'related'], nargs='+', default=['primary'])

    group.add_argument('--crosstask_cross_validation', action='store_true')
    # group.add_argument('--crosstask_cross_validation_n_train', type=int, default=30)
    group.add_argument('--crosstask_cross_validation_seed', type=int)


def add_classifier_args(parser):
    group = parser.add_argument_group('classifier')
    group.add_argument('--classifier', choices=CLASSIFIERS.keys(), required=True)
    group.add_argument('--training', choices=['supervised', 'unsupervised'], default='supervised')
    group.add_argument('--cuda', action='store_true')
    for name, cls in CLASSIFIERS.items():
        cls.add_args(parser)

def write_predictions(test_data, predictions_by_video, output_path):
    # TODO: unuglify this
    for video, pred in predictions_by_video.items():
        labels = []
        task = test_data._tasks_by_video[video]
        for index in pred:
            if index in test_data._corpus._background_indices:
                label = "<BKG>"
            else:
                label = test_data._corpus.index2label[index].replace(' ', '_')
            labels.append('{}:{}'.format(task, label))
        with open(os.path.join(output_path, video), 'w') as f:
            f.write('### Recognized sequence: ###\n')
            f.write('\n') # TODO
            f.write('### Score: ###\n')
            f.write('\n') # TODO
            f.write('### Frame level recognition: ###\n')
            f.write(' '.join(labels))

def test(args, model: Model, test_data: Datasplit, test_data_name: str, verbose=True, prediction_output_path=None):
    #import pdb; pdb.set_trace()
#def fit(self, train_data: Datasplit, use_labels: bool, callback_fn=None):
        #self.model.train()
        #self.model.flatten_parameters()
        #if use_labels:
        #    assert not self.args.sm_constrain_transitions
        #initialize = True
        #if use_labels and self.args.sm_supervised_method in ['closed-form', 'closed-then-gradient']:
        #    self.fit_supervised(train_data)
        #    if self.args.sm_supervised_method == 'closed-then-gradient':
        #        initialize = False
        #        callback_fn(-1, {})
        #    else:
        #        return
        #if self.args.sm_init_non_projection_parameters_from:
        #    initialize = False
        #    if callback_fn:
        #        callback_fn(-1, {})
        #optimizer, scheduler = make_optimizer(self.args, self.model.parameters())
        #big_loader = make_data_loader(self.args, train_data, batch_by_task=False, shuffle=True, batch_size=100)
    #big_loader = make_data_loader(model.args, train_data, batch_by_task=False, shuffle=True, batch_size=100)
    #samp = next(iter(big_loader))
    #big_features = samp['features']
    #big_lengths = samp['lengths']
    #if model.args.cuda:
    #    big_features = big_features.cuda()
    #    big_lengths = big_lengths.cuda()

        #if initialize:
        #    self.model.initialize_gaussian(big_features, big_lengths)

        #loader = make_data_loader(self.args, train_data, batch_by_task=True, shuffle=True, batch_size=self.args.batch_size)
    loader = make_data_loader(model.args, test_data, batch_by_task=True, shuffle=True, batch_size=model.args.batch_size)

    C = model.n_classes
    K = model.args.sm_max_span_length

    for epoch in range(1):
        #start_time = time.time()
        # call here since we may set eval in callback_fn
        #self.model.train()
        losses = []
        multi_batch_losses = []
        nlls = []
        kls = []
        log_dets = []
        num_frames = 0
        num_videos = 0
        train_nll = 0
        train_kl = 0
        train_log_det = 0
        # for batch_ix, batch in enumerate(tqdm.tqdm(loader, ncols=80)):
        for batch_ix, batch in enumerate(loader):
            #if self.args.train_limit and batch_ix >= self.args.train_limit:
            #    break
            tasks = batch['task_name']
            videos = batch['video_name']
            features = batch['features']
            task_indices = batch['task_indices']
            lengths = batch['lengths']

            #if 'train' in self.args.sm_constrain_with_narration:
            #    assert all_equal(tasks)
            #    constraints_expanded = self.expand_constraints(
            #        train_data, tasks[0], task_indices[0], 1 - batch['constraints']
            #    )
            #    constraints_expanded *= self.args.sm_constrain_narration_weight
            #else:
            #    constraints_expanded = None
            constraints_expanded = None

            num_frames += lengths.sum().item()
            num_videos += len(lengths)

            # assert len( task_indices) == self.n_classes, "remove_background and multi-task fit() not implemented"

            #if self.args.cuda:
            if model.args.cuda:
                features = features.cuda()
                lengths = lengths.cuda()
                if constraints_expanded is not None:
                    constraints_expanded = constraints_expanded.cuda()

            #if use_labels:
            #    labels = batch['gt_single']
            #    if self.args.cuda:
            #        labels = labels.cuda()
            #    spans = semimarkov_utils.labels_to_spans(labels, max_k=K)
            #    use_mean_z = True
            #else:
            #    spans = None
            #    use_mean_z = False
            spans = None
            use_mean_z = False

            #addl_allowed_ends = self.make_additional_allowed_ends(tasks, lengths)
            addl_allowed_ends = model.make_additional_allowed_ends(tasks, lengths)
            #import pdb; pdb.set_trace()
            task_indices_orig = task_indices
            task_indices = [torch.LongTensor(
                list(range(model.n_classes)),
            ).to(task_indices_orig[0].device) for _ in range(len(task_indices_orig))]

            ll, log_det = model.model.log_likelihood(features,
                                             lengths,
                                             valid_classes_per_instance=task_indices,
                                             spans=spans,
                                             add_eos=True,
                                             use_mean_z=use_mean_z,
                                             additional_allowed_ends_per_instance=addl_allowed_ends,
                                             constraints=constraints_expanded)
            nll = -ll
            #kl = self.model.kl.mean()
            kl = model.model.kl.mean()
            #if use_labels:
            #    this_loss = nll - log_det
            #else:
            #    this_loss = nll - log_det + kl
            this_loss = nll - log_det + kl
            multi_batch_losses.append(this_loss)
            nlls.append(nll.item())
            kls.append(kl.item())
            log_dets.append(log_det.item())

            train_nll += (nll.item() * len(videos))
            train_kl += (kl.item() * len(videos))
            train_log_det += (log_det.item() * len(videos))

            losses.append(this_loss.item())

            #if len(multi_batch_losses) >= self.args.batch_accumulation:
            #    loss = sum(multi_batch_losses) / len(multi_batch_losses)
            #    loss = loss.item()
            #    #loss.backward()
            #    multi_batch_losses = []

                #if self.args.print_every and (batch_ix % self.args.print_every == 0):
                #    param_norm = sum([p.norm()**2 for p in self.model.parameters()
                #                      if p.requires_grad]).item()**0.5
                #    gparam_norm = sum([p.grad.norm()**2 for p in self.model.parameters()
                #                       if p.requires_grad and p.grad is not None]).item()**0.5
                #    log_str = 'Epoch: %02d, Batch: %03d/%03d, |Param|: %.6f, |GParam|: %.2f, lr: %.2E, ' + \
                #              'loss: %.4f, recon: %.4f, kl: %.4f, log_det: %.4f, recon_bound: %.2f, Throughput: %.2f vid / sec'
                #    print(log_str %
                #          (epoch, batch_ix, len(loader), param_norm, gparam_norm,
                #           optimizer.param_groups[0]["lr"],
                #           (train_nll + train_kl + train_log_det) / num_videos, # loss
                #           train_nll / num_frames, # recon
                #           train_kl / num_frames, # kl
                #           train_log_det / num_videos, # log_det
                #           (train_nll + train_kl) / num_frames, # recon_bound
                #          num_videos / (time.time() - start_time))) # Throughput
                #if self.args.max_grad_norm is not None:
                #    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                #optimizer.step()
                #self.model.zero_grad()
            #break
        #ORIG: train_loss = np.mean(losses)
        train_loss = train_nll / num_videos
        #stats = {'main': train_loss}
        return train_loss
        #if scheduler is not None:
        #    scheduler.step(train_loss)
        #callback_fn(epoch, {'train_loss': train_loss,
        #                    'train_nll_frame_avg': train_nll / num_frames,
        #                    'train_kl_vid_avg': train_kl / num_videos,
        #                    'train_recon_bound': (train_nll + train_kl) / num_frames})
#ORIG: def test(args, model: Model, test_data: Datasplit, test_data_name: str, verbose=True, prediction_output_path=None):
#ORIG:     if args.training == 'supervised':
#ORIG:         optimal_assignment = False
#ORIG:     else:
#ORIG:         assert args.training == 'unsupervised'
#ORIG:         # if we're constraining the transitions to be the canonical order in the semimarkov, we don't need oracle reassignment
#ORIG:         optimal_assignment = not (args.classifier == 'semimarkov' and args.sm_constrain_transitions)
#ORIG:         if 'train' in args.sm_constrain_with_narration or 'test' in args.sm_constrain_with_narration:
#ORIG:             optimal_assignment = False
#ORIG:     if args.force_optimal_assignment:
#ORIG:         optimal_assignment = True
#ORIG:     if model is not None:
#ORIG:         predictions_by_video = model.predict(test_data)
#ORIG:         prediction_function = lambda video: predictions_by_video[video.name]
#ORIG:     else:
#ORIG:         prediction_function = None
#ORIG:     # print('prediction_output_path: {}'.format(prediction_output_path))
#ORIG:     if prediction_output_path is not None:
#ORIG:         assert model is not None
#ORIG:         write_predictions(test_data, predictions_by_video, prediction_output_path)
#ORIG:     stats = test_data.accuracy_corpus(
#ORIG:         optimal_assignment,
#ORIG:         prediction_function,
#ORIG:         prefix=test_data_name,
#ORIG:         verbose=verbose,
#ORIG:         compare_to_folder=args.compare_to_prediction_folder if not test_data_name.startswith('train') else None
#ORIG:     )
#ORIG:     return stats


def make_model_path(path, split_name):
    if path.endswith('.pkl'):
        return path
    else:
        # is directory
        return os.path.join(path, '{}.pkl'.format(split_name))


def train(args, train_data: Datasplit, dev_data: Datasplit, split_name, verbose=False, train_sub_data=None):
    #import pdb; pdb.set_trace()
    model = CLASSIFIERS[args.classifier].from_args(args, train_data)

    if args.training == 'supervised':
        use_labels = True
        early_stopping_on_dev = True
    else:
        assert args.training == 'unsupervised'
        use_labels = False
        #ORIG: early_stopping_on_dev = False
        early_stopping_on_dev = True

    def evaluate_on_data(data, name):
        #import pdb; pdb.set_trace()
        with torch.no_grad():
            stats_by_name = test(args, model, data, name, verbose=verbose)

        # all_mof = np.array([stats['mof'] for stats in stats_by_name.values()])
        # sum_mof = all_mof.sum(axis=0)
        #
        # all_mof_non_bg = np.array([stats['mof_non_bg'] for stats in stats_by_name.values()])
        # sum_mof_non_bg = all_mof_non_bg.sum(axis=0)
        #
        # all_step_recall_non_bg = np.array([stats['step_recall_non_bg'] for stats in stats_by_name.values()])
        # sum_step_recall_non_bg = all_step_recall_non_bg.sum(axis=0)
        #
        # all_leven = np.array([stats['mean_normed_levenshtein'] for stats in stats_by_name.values()])
        # sum_leven = all_leven.sum(axis=0)

        #ORIG: d = {}
        #ORIG: for key in STAT_KEYS:
        #ORIG:     all_stats = np.array([stats[key] for stats in stats_by_name.values()])
        #ORIG:     sum_stats = all_stats.sum(axis=0)
        #ORIG:     d['{}_{}'.format(name, key)] = float(sum_stats[0]) / sum_stats[1]
        #ORIG: return d
        return {'dev_mof': stats_by_name}

        # return {
        #     '{}_mof'.format(name): float(sum_mof[0]) / sum_mof[1],
        #     '{}_mof_non_bg'.format(name): float(sum_mof_non_bg[0]) / sum_mof_non_bg[1],
        #     '{}_step_recall_non_bg'.format(name): float(sum_step_recall_non_bg[0]) / sum_step_recall_non_bg[1],
        #     '{}_mean_normed_levenshtein'.format(name): float(sum_leven[0]) / sum_leven[1],
        # }

    models_by_epoch = {}
    dev_mof_by_epoch = {}
    stats_by_epoch = {}

    def callback_fn(epoch, stats):
        #import pdb; pdb.set_trace()
        stats_by_epoch[epoch] = stats
        if train_sub_data is not None:
            train_name = 'train_subset'
            train_stats = evaluate_on_data(train_sub_data, train_name)
        else:
            assert False
            train_name = 'train'
            train_stats = evaluate_on_data(train_data, train_name)
        split_stats = [train_stats]
        if epoch == -1 or epoch % args.dev_decode_frequency == 0:
            dev_stats = evaluate_on_data(dev_data, 'dev')
            split_stats.append(dev_stats)
        else:
            dev_stats = None
        log_str = '{}\tepoch {:2d}'.format(split_name, epoch)
        for stat, value in stats.items():
            if isinstance(value, float):
                log_str += '\t{} {:.4f}'.format(stat, value)
            else:
                log_str += '\t{} {}'.format(stat, value)
        # log_str += '\t{} '.format(train_name)
        for i, stats in enumerate(split_stats):
            log_str += '\n'
            for name, val in sorted(stats.items()):
                log_str += '{}: {} {:.4f}'.format(i, name, val)
        # log_str += '\t{} mof {:.4f}\tdev mof {:.4f}'.format(train_name, train_mof, dev_mof)
        logger.debug(log_str)
        models_by_epoch[epoch] = pickle.dumps(model)

        if dev_stats is not None:
            dev_mof_by_epoch[epoch] = dev_stats['dev_mof']

        if args.model_output_path and epoch % 5 == 0:
            os.makedirs(args.model_output_path, exist_ok=True)
            model_fname = os.path.join(args.model_output_path, '{}_epoch-{}.pkl'.format(split_name, epoch))
            print("writing model to {}".format(model_fname))
            with open(model_fname, 'wb') as f:
                pickle.dump(model, f)

    model.fit(train_data, use_labels=use_labels, callback_fn=callback_fn)

    #import pdb; pdb.set_trace()

    if early_stopping_on_dev and dev_mof_by_epoch:
        #best_dev_epoch, best_dev_mof = max(dev_mof_by_epoch.items(), key=lambda t: t[1])
        best_dev_epoch, best_dev_mof = min(dev_mof_by_epoch.items(), key=lambda t: t[1])
        logger.debug("best dev mov {:.4f} in epoch {}".format(best_dev_mof, best_dev_epoch))
        best_model = pickle.loads(models_by_epoch[best_dev_epoch])
    elif stats_by_epoch and 'train_loss' in next(iter(stats_by_epoch.values())):
        best_epoch, best_train_stats = min(stats_by_epoch.items(), key=lambda t: t[1]['train_loss'])
        logger.debug("best train loss {:.4f} in epoch {}".format(best_train_stats['train_loss'], best_epoch))
        best_model = pickle.loads(models_by_epoch[best_epoch])
    else:
        best_model = model

    if args.model_output_path:
        os.makedirs(args.model_output_path, exist_ok=True)
        model_fname = make_model_path(args.model_output_path, split_name)
        print("writing model to {}".format(model_fname))
        with open(model_fname, 'wb') as f:
            pickle.dump(best_model, f)

    return best_model


def make_data_splits(args):
    # split_name -> (train_data, test_data)
    splits = OrderedDict()

    if args.dataset == 'crosstask':
        features_contain_background = True
        if args.features == 'pca':
            max_components = 200
            assert args.pca_components_per_group <= max_components
            features_contain_background = not args.pca_no_background
            feature_root = 'data/crosstask/crosstask_processed/crosstask_primary_pca-{}_{}-bkg_by-task'.format(
                max_components,
                "no" if args.pca_no_background else "with",
            )
            dimensions_per_feature_group = {
                feature_group: args.pca_components_per_group
                for feature_group in args.crosstask_feature_groups
            }
        else:
            feature_root = 'data/crosstask/crosstask_features'
            dimensions_per_feature_group = None

        corpus = CrosstaskCorpus(
            release_root="data/crosstask/crosstask_release",
            feature_root=feature_root,
            dimensions_per_feature_group=dimensions_per_feature_group,
            features_contain_background=features_contain_background,
            task_specific_steps=args.task_specific_steps,
            annotate_background_with_previous=args.annotate_background_with_previous,
            use_secondary='related' in args.crosstask_training_data,
            constraints_root='data/crosstask/crosstask_constraints',
            load_constraints=True,
        )
        corpus._cache_features = True
        if args.no_cache_features:
            corpus._cache_features = False
        train_task_sets = args.crosstask_training_data


        test_task_sets = ['primary']
        task_ids = sorted([task_id for task_set in sorted(set(train_task_sets) | set(test_task_sets))
                           for task_id in CrosstaskCorpus.TASK_IDS_BY_SET[task_set]])
        if args.crosstask_cross_validation:
            assert False
            if train_task_sets != ['primary']:
                raise NotImplementedError("cross validation with related tasks")
            split_names_and_full = [
                ('cv_train_{}'.format(args.crosstask_cross_validation_seed), True, train_task_sets),
                ('cv_train_{}'.format(args.crosstask_cross_validation_seed), False, train_task_sets),
                ('cv_test_{}'.format(args.crosstask_cross_validation_seed), True, train_task_sets),
            ]
        else:
            split_names_and_full = [
                ('train', True, train_task_sets),
                ('train', False, test_task_sets),
                ('val', True, test_task_sets)
            ]
        if args.compare_load_splits_from_predictions:
            assert args.compare_to_prediction_folder
            assert args.compare_only
            assert not args.crosstask_cross_validation, "just pass --compare_to_prediction_folder, --compare_only, and --compare_load_splits_from_predictions"
            with open(os.path.join(args.compare_to_prediction_folder, 'y_pred.json'), 'rb') as f:
                preds_by_task_and_video = json.load(f)
            val_videos_override = []
            for task, data in preds_by_task_and_video.items():
                val_videos_override.extend(data.keys())
            print("loaded predictions for {} videos; using as the validation set".format(len(val_videos_override)))
        else:
            val_videos_override = None

            # TODO: here
        if args.mix_tasks:
            splits['all'] = tuple(
                corpus.get_datasplit(remove_background=args.remove_background,
                                     task_sets=task_sets,
                                     task_ids=task_ids,
                                     split=split,
                                     full=full,
                                     subsample=args.frame_subsample,
                                     feature_downscale=args.feature_downscale,
                                     val_videos_override=val_videos_override,
                                     feature_permutation_seed=args.feature_permutation_seed,
                                     )
                for split, full, task_sets in split_names_and_full
            )
            train_videos = set(p[1] for p in splits['all'][0]._tasks_and_video_names)
            test_videos = set(p[1] for p in splits['all'][2]._tasks_and_video_names)
            assert not(train_videos & test_videos),\
                "overlap in train and test videos: {}".format(train_videos & test_videos)
        else:
            for task_id in task_ids:
                splits['{}_val'.format(task_id)] = tuple(
                    corpus.get_datasplit(remove_background=args.remove_background,
                                         task_sets=task_sets,
                                         task_ids=[task_id],
                                         split=split,
                                         full=full,
                                         subsample=args.frame_subsample,
                                         feature_downscale=args.feature_downscale,
                                         val_videos_override=val_videos_override,
                                         feature_permutation_seed=args.feature_permutation_seed,
                                         )
                    for split, full, task_sets in split_names_and_full
                )
    elif args.dataset == 'breakfast':
        assert not args.annotate_background_with_previous
        if args.features == 'pca':
            max_components = 64
            assert args.pca_components_per_group == max_components
            features_contain_background = not args.pca_no_background
            assert features_contain_background # not implemented!
            feature_root = 'data/breakfast/breakfast_processed/breakfast_pca-{}_{}-bkg_by-task'.format(
                max_components,
                "no" if args.pca_no_background else "with",
            )
        else:
            feature_root = 'data/breakfast/reduced_fv_64'
        corpus = BreakfastCorpus(mapping_file='data/breakfast/mapping.txt',
                                 feature_root=feature_root,
                                 label_root='data/breakfast/BreakfastII_15fps_qvga_sync',
                                 task_specific_steps=args.task_specific_steps)
        corpus._cache_features = True

        all_splits = list(sorted(BreakfastCorpus.DATASPLITS.keys()))
        for heldout_split in all_splits:
            splits[heldout_split] = (
                corpus.get_datasplit(remove_background=args.remove_background,
                                     splits=[sp for sp in all_splits if sp != heldout_split],
                                     full=True,
                                     subsample=args.frame_subsample,
                                     feature_downscale=args.feature_downscale,
                                     feature_permutation_seed=args.feature_permutation_seed,
                                     ),
                corpus.get_datasplit(remove_background=args.remove_background,
                                     splits=[sp for sp in all_splits if sp != heldout_split],
                                     full=True,
                                     subsample=args.frame_subsample,
                                     feature_downscale=args.feature_downscale,
                                     feature_permutation_seed=args.feature_permutation_seed,
                                     ), # has issue with some tasks being dropped if we pass full=False
                corpus.get_datasplit(remove_background=args.remove_background,
                                     splits=[heldout_split],
                                     full=True,
                                     subsample=args.frame_subsample,
                                     feature_downscale=args.feature_downscale,
                                     feature_permutation_seed=args.feature_permutation_seed,
                                     ),
            )
    else:
        raise NotImplementedError("invalid dataset {}".format(args.dataset))

    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    add_serialization_args(parser)
    add_data_args(parser)
    add_classifier_args(parser)
    add_training_args(parser)
    add_misc_args(parser)
    args = parser.parse_args()

    print(' '.join(sys.argv))

    pprint.pprint(vars(args))

    stats_by_split_and_task = {}

    stats_by_split_by_task = {}

    for split_name, (train_data, train_sub_data, test_data) in make_data_splits(args).items():
        print(split_name)
        if args.compare_only:
            assert args.compare_to_prediction_folder
            model = None
        else:
            if args.model_input_path:
                model_path = make_model_path(args.model_input_path, split_name)
                print("loading model from {}".format(model_path))
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                if vars(args) != vars(model.args):
                    print("warning: command line args and serialized model args differ:")
                    cmd_d = vars(args)
                    ser_d = vars(model.args)
                    for key in set(cmd_d) | set(ser_d):
                        if key == 'model_input_path' or key == 'model_output_path':
                            continue
                        if key not in ser_d or key not in cmd_d or ser_d[key] != cmd_d[key]:
                            print("{}: {} != {}".format(key, cmd_d.get(key, "<NP>"), ser_d.get(key, "<NP>")))

                    print("setting model args to serialized args")
                model.args = args
                try:
                    model.model.eval()
                    if args.cuda:
                        model.model.cuda()
                    else:
                        model.model.cpu()
                except Exception as e:
                    print(e)

            else:
                model = train(args, train_data, test_data, split_name, train_sub_data=train_sub_data)

        print('split_name: {}'.format(split_name))
        # prediction_output_path = args.prediction_output_path if 'val' in split_name else None
        prediction_output_path = args.prediction_output_path

        stats_by_task = test(args, model, test_data, split_name, prediction_output_path=prediction_output_path)
        stats_by_split_by_task[split_name] = {}
        for task, stats in stats_by_task.items():
            stats_by_split_and_task["{}_{}".format(split_name, task)] = stats
            stats_by_split_by_task[split_name][task] = stats
        print()


    def divide(d):
        divided = {}
        for key, vals in d.items():
            assert len(vals) == 2
            divided[key] = float(vals[0]) / vals[1]
        return divided


    print()
    pprint.pprint(stats_by_split_and_task)

    print()
    pprint.pprint({k: divide(d) for k, d in stats_by_split_and_task.items()})

    summed_across_tasks = {}
    divided_averaged_across_tasks = {}

    sum_within_split_averaged_across_splits = {}

    for key in next(iter(stats_by_split_and_task.values())):
        arrs = np.array([d[key] for d in stats_by_split_and_task.values()])
        summed_across_tasks[key] = np.sum(arrs, axis=0)

        divided_averaged_across_tasks[key] = np.mean([
            divide(d)[key] for d in stats_by_split_and_task.values()
        ])

    print()

    summed_across_tasks_divided = divide(summed_across_tasks)

    print("summed across tasks:")
    pprint.pprint(summed_across_tasks_divided)
    print()
    print("averaged across tasks:")
    pprint.pprint(divided_averaged_across_tasks)
    print()
    # print("averaged across splits:")
    # pprint.pprint(sum_within_split_averaged_across_splits)

    stat_dict = divided_averaged_across_tasks

    print(', '.join(STAT_KEYS))
    print(', '.join('{:.4f}'.format(stat_dict[key]) for key in STAT_KEYS))

    print(', '.join(DISPLAY_STAT_KEYS))
    print(', '.join('{:.4f}'.format(stat_dict[key]) for key in DISPLAY_STAT_KEYS))

    if any(stat.startswith('compare_') for stat in stat_dict):
        compare_keys = ['comparison_{}'.format(key) for key in DISPLAY_STAT_KEYS]
        print(', '.join(compare_keys))
        print(', '.join('{:.4f}'.format(stat_dict[key]) for key in compare_keys))
